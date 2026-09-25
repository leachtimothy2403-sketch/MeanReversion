"""open_exit_search (10-h version): staged, resumable search for handoff steps 3 + 5. Pre-registered 2026-09-25, see README.md.

  prep    per-instrument caches + instrument scales (results/scales.json)
  stage0  ablation: live opening trade + legs (LIVE_COMP) vs one-knob variants, every instrument / split (no selection)
  TRACKS  short = search window 2024-09-16..2026-06-30 (as flex_search);
          long  = 2016-09-15..2026-06-30, must work in BOTH regimes (pre-2024 and 2024+); instruments with 2016+ data
    s1    random configs per instrument; daily R over the track window; fitness per fold f (4 folds, see HOLD-OUT)
    s2    union of each fold's top configs; one-step neighbours; robust_f = min(fit_f, mean neighbour fit_f)
    s3    each fold's top-k3 robust configs + baselines on every split; checks A (held-out of the folds that selected
          the config), B (2016-23 stress, short track only), C (final holdout); ALL = A & B & C
  stage4  FTMO 2-step on one $100k account: ALL-passing configs of both tracks (<= k4 per instrument and track) alone and
          in 2-4 instrument combos; uniform risk x breaker x near sizing, plus mixed per-strategy risk for combos
          (second, report-only pool 'AC' = A & C, i.e. may fail the 2016-23 stress)
  stage5  FundedNext Futures Flex 100K for the NQ configs (ALL and AC pools) + NQ baselines
HOLD-OUT  fold f holds out, in every month of the track window, its ((month# + f) mod 4 + 1)-th Monday-week; fold 0 of the
          short track = the flex_search held-out weeks. A config is selected per fold on that fold's TRAIN days only
          and checked on that fold's held-out days (no leakage).
Usage: py -3 oe_search.py all --workers 4
"""
import os, sys, json, time, math, argparse, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
import oe_space as SP
from oe_data import build
from oe_engine import sim

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.environ.get("OE_RESULTS", "results")); os.makedirs(RES, exist_ok=True)
def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True); open(os.path.join(RES, "run.log"), "a").write(msg + "\n")

TRACKS = {"short": SP.SEARCH, "long": ("2016-09-15", "2026-06-30")}
REGIME_SPLIT = "2024-01-01"
NF = 4
def _bd(a, b): return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(a, b)]
def heldout(a, b, f):
    d = pd.Series(pd.bdate_range(a, b)); out = set()
    for mi, (_, g) in enumerate(d.groupby([d.dt.year, d.dt.month])):
        wk = g.dt.to_period("W-SUN"); weeks = list(dict.fromkeys(wk))
        out |= set(g[wk == weeks[min((mi + f) % 4, len(weeks) - 1)]].dt.strftime("%Y-%m-%d"))
    return out
SEQ = {"final": _bd(*SP.FINAL), "stress": _bd(*SP.STRESS)}
for tk, (a_, b_) in TRACKS.items():
    win = _bd(a_, b_); SEQ[f"{tk}_win"] = win
    for f in range(NF):
        h = heldout(a_, b_, f); SEQ[f"{tk}_held{f}"] = sorted(h); SEQ[f"{tk}_train{f}"] = [d for d in win if d not in h]

# ------------------------------------------------------------------ per-process data
G = {}; SCALE = {}
def data(name):
    if name in G: return G[name]
    r = build(name); g = {k: r[k] for k in r}; days = [str(x) for x in r["days"]]; g["days"] = days
    pos = {d: i for i, d in enumerate(days)}
    close = pd.Series(r["rth_close"].astype(float))
    ma = close.rolling(100).mean().shift(1); prev = close.shift(1)
    g["ma_on"] = {0: np.ones(len(days), bool), 100: ((prev > ma) | ma.isna()).to_numpy()}
    g["gap"] = (r["blocks"][:, 0, 0].astype(float) - prev.to_numpy())
    B = r["blocks"]; drng = pd.Series(np.nanmax(B[:, :, 1], axis=1) - np.nanmin(B[:, :, 2], axis=1)).astype(float)
    g["volr"] = (drng.shift(1) / drng.shift(1).rolling(20, min_periods=10).median().shift(1)).to_numpy()
    g["dow"] = pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    rng = pd.Series(r["rng120"].astype(float)); sd = np.array(days)
    med = float(np.median(rng[(sd >= SP.SEARCH[0]) & (sd <= SP.SEARCH[1])]))
    g["dyn"] = (rng.rolling(60, min_periods=20).median().shift(1) / med).fillna(1.0).clip(0.2, 3.0).to_numpy()
    g["med"] = med
    g["idx"] = {k: np.array([pos[d] for d in v if d in pos], int) for k, v in SEQ.items()}
    g["old"] = np.array([d < REGIME_SPLIT for d in days])
    G[name] = g
    return g

def load_scales():
    if not SCALE: SCALE.update(json.load(open(os.path.join(RES, "scales.json"))))
    return SCALE

def day_ok(P, g, i):
    if not g["ma_on"][P["ma"]][i]: return False
    if P["dow"] == "mon" and g["dow"][i] == 0: return False
    if P["dow"] == "fri" and g["dow"][i] == 4: return False
    if P["volf"] != "off":
        v = g["volr"][i]
        if not np.isfinite(v): return True
        if P["volf"] == "skip_low" and v < 0.7: return False
        if P["volf"] == "skip_high" and v > 1.5: return False
        if P["volf"] == "mid" and not (0.7 <= v <= 1.5): return False
    return True

def day_trades(name, P, idx):
    g = data(name); c = SP.INST[name]["cost"]; sc = load_scales()[name]; out = {}
    for i in idx:
        if not day_ok(P, g, i): continue
        u = sc * (g["dyn"][i] if P["dyn"] else 1.0)
        out[int(i)] = sim(g["blocks"][i], P, u, c, g["gap"][i])
    return out

def vecs(name, P, idx):
    """daily R and trade count aligned to idx"""
    tr = day_trades(name, P, idx); pos = {int(i): k for k, i in enumerate(idx)}
    R = np.zeros(len(idx)); N = np.zeros(len(idx))
    for i, L in tr.items():
        R[pos[i]] = sum(t["R"] for t in L); N[pos[i]] = len(L)
    return R, N

def _t(x): sd = x.std(); return float(x.mean() / sd * math.sqrt(len(x))) if len(x) > 1 and sd > 0 else 0.0

def fitness(track, R, N, mask, old):
    """short: t-stat of daily R on the fold's TRAIN days, >= 1 trade / 4 days, both halves > 0.
    long : min(t pre-2024, t 2024+) on the fold's TRAIN days; each regime needs >= 1 trade / 4 days and R > 0."""
    if track == "short":
        x = R[mask]; n = N[mask].sum()
        if len(x) == 0 or n < 0.25 * len(x): return -99.0
        h = len(x) // 2
        if x[:h].sum() <= 0 or x[h:].sum() <= 0: return -99.0
        return round(_t(x), 3)
    ts = []
    for m in (mask & old, mask & ~old):
        x = R[m]; n = N[m].sum()
        if len(x) < 100 or n < 0.25 * len(x) or x.sum() <= 0: return -99.0
        ts.append(_t(x))
    return round(min(ts), 3)

MK = {}
def masks(name, track):
    if (name, track) in MK: return MK[(name, track)]
    MK[(name, track)] = _masks(name, track); return MK[(name, track)]

def _masks(name, track):
    g = data(name); win = g["idx"][f"{track}_win"]; wset = {int(i): k for k, i in enumerate(win)}
    ms = []
    for f in range(NF):
        m = np.zeros(len(win), bool); m[[wset[int(i)] for i in g["idx"][f"{track}_train{f}"]]] = True; ms.append(m)
    return win, ms, g["old"][win]

def fits(name, track, P):
    win, ms, old = masks(name, track); R, N = vecs(name, P, win)
    return [fitness(track, R, N, m, old) for m in ms], R, N

def metrics(Rall, Nall, idx):
    if len(idx) == 0: return dict(days=0, n=0, Rday=None, t=None, maxDD=None)
    x = Rall[idx]; eq = np.cumsum(x)
    return dict(days=len(idx), n=int(Nall[idx].sum()), Rday=round(float(x.mean()), 4), t=round(_t(x), 3),
                maxDD=round(float((eq - np.maximum.accumulate(eq)).min()), 2))

def track_insts(a, track):
    if track == "short": return a.inst
    return [n for n in a.inst if len(data(n)["idx"]["stress"]) >= 1000]

def done(fn, a):
    if os.path.exists(fn) and not a.force:
        log(f"  {os.path.basename(fn)} exists - skipped (use --force to redo)"); return True
    return False

# ------------------------------------------------------------------ prep
def prep(a):
    sc = {}
    for n in a.inst_all:
        t = time.time(); g = data(n)
        log(f"prep {n}: {len(g['days'])} days {g['days'][0]}..{g['days'][-1]}, median 2h range (search) {g['med']:.1f}; days "
            f"short train0/held0 {len(g['idx']['short_train0'])}/{len(g['idx']['short_held0'])}, long train0/held0 "
            f"{len(g['idx']['long_train0'])}/{len(g['idx']['long_held0'])}, final {len(g['idx']['final'])}, stress {len(g['idx']['stress'])} ({time.time() - t:.0f}s)")
    ref = data("NDX100")["med"]
    for n in a.inst_all: sc[n] = SP.FIXED_SCALE.get(n, round(data(n)["med"] / ref, 4))
    json.dump(sc, open(os.path.join(RES, "scales.json"), "w"), indent=1); log("scales (x NDX points):", sc)

# ------------------------------------------------------------------ stage 0
def variants():
    L = SP.LIVE; V = [("LIVE_COMP", L), ("LIVE_COMP+MA100", dict(L, ma=100)), ("entry_tf5", dict(L, entry_tf=5)),
                      ("vwap_off", dict(L, vwap=False)), ("legs_off", dict(L, legs=False)), ("dyn_scale", dict(L, dyn=True)),
                      ("legs_same_mgmt_be1", dict(L, legs_mgmt="same", be_at=1.0))]
    V += [(f"time_stop_{x}", dict(L, time_stop=x)) for x in (60, 90, 180, 240, 999)]
    V += [(f"be_at_{x}", dict(L, be_at=x)) for x in (0.5, 1.0, 1.5, 2.0)]
    V += [(f"partial_{x}", dict(L, partial=x)) for x in (1.0, 1.5, 2.0, 3.0)]
    V += [(f"trail_{x}", dict(L, trail=x)) for x in (1.0, 1.5, 2.0, 3.0)]
    V += [("runner_noTP_120", dict(L, big_tp=0, std_tp=0)),
          ("runner_noTP_trail1.5_close", dict(L, big_tp=0, std_tp=0, trail=1.5, time_stop=999)),
          ("runner_noTP_be1_trail2_close", dict(L, big_tp=0, std_tp=0, be_at=1.0, trail=2.0, time_stop=999))]
    V += [(f"pullback_{f}_{w}", dict(L, entry_mode="pb", pb_frac=f, pb_wait=w)) for f in (0.3, 0.5) for w in (5, 15)]
    V += [(f"gap_{m}_{x}", dict(L, gap=m, gap_min=x)) for m in ("with", "against") for x in (0, 50)]
    V += [(f"body_min_{x}", dict(L, body_min=x)) for x in (0.3, 0.5)]
    V += [(f"volf_{x}", dict(L, volf=x)) for x in ("skip_low", "skip_high", "mid")]
    V += [(f"skip_{x}", dict(L, dow=x)) for x in ("mon", "fri")]
    return V

SPLITS0 = ["short_train0", "short_held0", "short_win", "final", "stress"]
def all_splits(name, P, extra=()):
    g = data(name); Rall, Nall = vecs(name, P, np.arange(len(g["days"])))
    row = {}
    for s in list(SPLITS0) + list(extra):
        for k, v in metrics(Rall, Nall, g["idx"][s]).items(): row[f"{s}_{k}"] = v
    old = g["old"]
    for lab, m in (("pre2024", old), ("since2024", ~old)):
        ix = np.flatnonzero(m & (np.array(g["days"]) <= TRACKS["long"][1])); row[f"{lab}_Rday"] = round(float(Rall[ix].mean()), 4) if len(ix) else None
    ys = pd.Series(Rall).groupby(pd.Series([d[:4] for d in g["days"]])).sum()
    for y, v in ys.items(): row[f"R_{y}"] = round(float(v), 1)
    return row, Rall, Nall

def _s0(job):
    name, vn, P = job; load_scales()
    row, _, _ = all_splits(name, P)
    return dict(inst=name, variant=vn, params=json.dumps(P), **row)

def stage0(a):
    fn = os.path.join(RES, "stage0_ablation.csv")
    if done(fn, a): return
    jobs = [(n, vn, P) for n in a.inst_all for vn, P in variants()]
    log(f"stage0: {len(jobs)} ablation runs")
    with Pool(a.workers) as p: rows = p.map(_s0, jobs, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(fn, index=False)
    cols = ["variant", "short_train0_Rday", "short_held0_Rday", "final_Rday", "pre2024_Rday", "since2024_Rday", "short_win_maxDD", "stress_maxDD"]
    for n in a.inst_all: log(f"stage0 {n}:\n" + df[df.inst == n][cols].to_string(index=False))

# ------------------------------------------------------------------ stage 1
def _s1(job):
    track, name, i0, i1, seed = job; load_scales(); out = []
    for i in range(i0, i1):
        P = SP.sample(np.random.default_rng([seed, i]))
        fs, R, N = fits(name, track, P)
        out.append(dict(inst=name, idx=i, params=json.dumps(P), **{f"fit{f}": fs[f] for f in range(NF)},
                        win_Rday=round(float(R.mean()), 4), win_n=int(N.sum())))
    return out

def s1(a, track):
    D = os.path.join(RES, track); os.makedirs(D, exist_ok=True)
    n1 = a.n1 if track == "short" else a.n1_long; jobs = []
    for k, n in enumerate(track_insts(a, track)):
        fn = os.path.join(D, f"stage1_{n}.csv"); dn = set(pd.read_csv(fn).idx) if os.path.exists(fn) else set()
        seed = (5000 if track == "short" else 9000) + k * 17
        jobs += [(track, n, s, min(s + 100, n1), seed) for s in range(0, n1, 100)
                 if not all(i in dn for i in range(s, min(s + 100, n1)))]
    log(f"[{track}] stage1: {len(jobs)} chunks of 100 configs to do ({n1} per instrument: {track_insts(a, track)})")
    t1 = time.time()
    with Pool(a.workers) as p:
        for k, rows in enumerate(p.imap_unordered(_s1, jobs)):
            df = pd.DataFrame(rows); n = df.inst.iloc[0]; fn = os.path.join(D, f"stage1_{n}.csv")
            df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
            if k % 50 == 0:
                el = time.time() - t1; eta = el / (k + 1) * (len(jobs) - k - 1)
                log(f"  [{track}] stage1 chunk {k + 1}/{len(jobs)}, elapsed {el / 60:.0f} min, ETA this stage {eta / 3600:.1f} h")

# ------------------------------------------------------------------ stage 2
def _s2(job):
    track, name, P, nn, seed = job; load_scales()
    fb, _, _ = fits(name, track, P)
    nb = [fits(name, track, Q)[0] for Q in SP.neighbours(P, np.random.default_rng(seed), nn)]
    row = dict(inst=name, params=json.dumps(P))
    for f in range(NF):
        m = float(np.mean([max(x[f], -1.0) for x in nb])) if nb else -1.0
        row[f"fit{f}"] = fb[f]; row[f"nb{f}"] = round(m, 3); row[f"robust{f}"] = round(min(fb[f], m), 3)
    return row

def s2(a, track):
    D = os.path.join(RES, track); fn = os.path.join(D, "stage2.csv")
    if done(fn, a): return
    top2 = a.top2 if track == "short" else a.top2_long; nn = a.nn if track == "short" else a.nn_long; jobs = []
    for n in track_insts(a, track):
        s1_ = pd.read_csv(os.path.join(D, f"stage1_{n}.csv")).drop_duplicates("idx"); pick = set()
        for f in range(NF):
            pick |= set(s1_[s1_[f"fit{f}"] > 0].sort_values(f"fit{f}", ascending=False).head(top2).params)
        log(f"[{track}] stage2 {n}: {len(s1_)} configs, passing filter per fold {[int((s1_[f'fit{f}'] > 0).sum()) for f in range(NF)]}, "
            f"union of fold tops {len(pick)} x {nn} neighbours")
        jobs += [(track, n, json.loads(p), nn, 11 + k) for k, p in enumerate(sorted(pick))]
    rows = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(_s2, jobs, chunksize=2)):
            rows.append(r)
            if k % 200 == 0: log(f"  [{track}] stage2 {k}/{len(jobs)}")
    pd.DataFrame(rows).to_csv(fn, index=False)

# ------------------------------------------------------------------ stage 3
def _s3(job):
    track, name, tag, P, folds, rob = job; load_scales()
    extra = [f"{track}_{s}{f}" for f in range(NF) for s in ("train", "held")] + [f"{track}_win"]
    row, Rall, Nall = all_splits(name, P, extra)
    g = data(name)
    for f in range(NF):   # held-out split by regime (used by the long-track check)
        ix = g["idx"][f"{track}_held{f}"]
        for lab, m in (("pre", g["old"][ix]), ("post", ~g["old"][ix])):
            row[f"{track}_held{f}_{lab}_Rday"] = round(float(Rall[ix[m]].mean()), 4) if m.any() else None
    return dict(track=track, inst=name, tag=tag, sel_folds=folds, robust_max=rob, params=json.dumps(P), **row)

def s3(a, track):
    D = os.path.join(RES, track); fn = os.path.join(D, "stage3_report.csv")
    if done(fn, a): return
    s2_ = pd.read_csv(os.path.join(D, "stage2.csv")); jobs = []
    for n in track_insts(a, track):
        d = s2_[s2_.inst == n]; sel = {}
        for f in range(NF):
            for p in d[d[f"robust{f}"] > 0].sort_values(f"robust{f}", ascending=False).head(a.k3).params: sel.setdefault(p, []).append(f)
        for k, (p, fs) in enumerate(sel.items()):
            rob = float(d[d.params == p][[f"robust{f}" for f in range(NF)]].max(axis=1).iloc[0])
            jobs.append((track, n, f"{track[0].upper()}{k}", json.loads(p), ",".join(map(str, fs)), rob))
        jobs += [(track, n, b, P, "", None) for b, P in SP.BASELINES.items()]
    log(f"[{track}] stage3: {len(jobs)} configs on every split")
    with Pool(a.workers) as p: rows = p.map(_s3, jobs, chunksize=1)
    df = pd.DataFrame(rows); out = []
    for n, d in df.groupby("inst"):
        d = d.copy(); isb = d.tag.isin(list(SP.BASELINES)); base = d[isb]
        def A_f(r, f):
            tr, he = r[f"{track}_train{f}_Rday"], r[f"{track}_held{f}_Rday"]
            ok = he is not None and tr is not None and he > 0 and he >= 0.4 * tr
            if track == "long":
                ok = ok and (r[f"{track}_held{f}_pre_Rday"] or 0) >= 0 and (r[f"{track}_held{f}_post_Rday"] or 0) >= 0
            return bool(ok)
        for f in range(NF): d[f"A{f}"] = [A_f(r, f) for r in d.to_dict("records")]
        d["A_held"] = [all(r[f"A{int(f)}"] for f in r["sel_folds"].split(",")) if r["sel_folds"] else False for r in d.to_dict("records")]
        d["A_4folds"] = d[[f"A{f}" for f in range(NF)]].sum(axis=1)
        if track == "short":
            b_st = base.stress_Rday.max() if base.stress_days.max() >= 250 else None
            d["B_stress"] = (d.stress_days >= 250) & (d.stress_Rday >= 0) & (d.stress_Rday >= (b_st if b_st is not None else 0))
        else:
            d["B_stress"] = True   # 2016-23 is inside the long track's training data: evidence = held-out weeks of both regimes
        d["C_final"] = d.final_Rday.isna() | (d.final_Rday >= 0)
        d["ALL"] = d.A_held & d.B_stress & d.C_final & ~isb
        d["AC"] = d.A_held & d.C_final & ~isb
        out.append(d)
    df = pd.concat(out); df.to_csv(fn, index=False)
    cols = ["tag", "sel_folds", "robust_max", f"{track}_win_Rday", "A_4folds", "A_held", "final_Rday", "pre2024_Rday", "since2024_Rday", "stress_maxDD", "B_stress", "ALL"]
    for n, d in df.groupby("inst"):
        nb = d[~d.tag.isin(list(SP.BASELINES))]
        log(f"[{track}] stage3 {n}: {int(d.ALL.sum())} of {len(nb)} candidates pass ALL, {int(d.AC.sum())} pass A&C\n"
            + nb.sort_values(["ALL", "robust_max"], ascending=False)[cols].head(10).to_string(index=False)
            + "\n" + d[d.tag.isin(list(SP.BASELINES))][cols].to_string(index=False))

# ------------------------------------------------------------------ stage 4: FTMO 2-step, one account
def events(name, P, sidx=0):
    """{date: [(entry_key, exit_key, R, mae_R, tp_R, strategy_index, stop_points)]}; keys = UTC minute of the day."""
    g = data(name); out = {}
    for i, L in day_trades(name, P, range(len(g["days"]))).items():
        om = int(g["open_utc_min"][i])
        for t in L:
            out.setdefault(g["days"][i], []).append((om + t["eb"] + 1, om + t["xb"] + 1, t["R"], t["mae_R"], t["tp_R"], sidx, t["slp"]))
    return out

def ftmo_day(ev, bal, risks, breaker, smart, goal, need_days):
    F = SP.FTMO; day0 = bal; lim = max(day0 - F["dll"], F["floor"]); realized = 0.0; opn = []; traded = False; nent = 0
    why = lambda: "daily" if day0 - F["dll"] >= F["floor"] else "max"
    def exits(key, bal, realized):
        nonlocal opn
        opn.sort(key=lambda x: x[0]); keep = []
        for x in opn:
            if key is None or x[0] <= key:
                if bal - x[2] <= lim: return bal, realized, why()
                bal += x[1]; realized += x[1]
                if bal <= lim: return bal, realized, why()
            else: keep.append(x)
        opn = keep; return bal, realized, None
    for e in ev:
        bal, realized, w = exits(e[0], bal, realized)
        if w: return bal, w, traded
        if breaker is not None and realized <= -breaker: continue
        s = risks[e[5]]
        if smart == "near":
            if bal >= goal:
                if need_days <= 0 or nent >= 1: continue
                s = F["min_size"]
            elif e[4] is not None and e[4] > 0:
                s = min(s, max(F["min_size"], (goal - bal) / e[4]))
        opn.append((e[1], e[2] * s, e[3] * s)); traded = True; nent += 1
    bal, realized, w = exits(None, bal, realized)
    return bal, w, traded

def weekly_starts(seq, H):
    wk = pd.to_datetime(pd.Series(seq)).dt.to_period("W-SUN")
    return [k for k in range(len(seq)) if (k == 0 or wk.iloc[k] != wk.iloc[k - 1]) and k + H <= len(seq)]

def ftmo_horizon(ev, seq, risks, brk, smart):
    F = SP.FTMO; H = F["H"]; P = Fl = 0; dd = []; starts = weekly_starts(seq, H)
    for s0 in starts:
        phase, bal, goal, td, res = 1, F["start"], F["start"] + F["t1"], 0, None
        for k in range(s0, s0 + H):
            bal, w, tr = ftmo_day(ev.get(seq[k], []), bal, risks, brk, smart, goal, F["min_days"] - td); td += int(tr)
            if w: res = "F"; break
            if bal >= goal and td >= F["min_days"]:
                if phase == 1: phase, bal, goal, td = 2, F["start"], F["start"] + F["t2"], 0
                else: res = "P"; dd.append(k - s0 + 1); break
        P += res == "P"; Fl += res == "F"
    n = len(starts)
    return dict(n=n, pass60=round(P / n, 3) if n else None, fail60=round(Fl / n, 3) if n else None, med=float(np.median(dd)) if dd else None)

EV = {}
def _init4(ev): EV.update(ev)
def merged(combo):
    ev = {}
    for k, c in enumerate(combo):
        for d, L in EV[c].items(): ev.setdefault(d, []).extend([x[:5] + (k,) + x[6:] for x in L])
    for d in ev: ev[d].sort()
    return ev

def _s4(job):
    combo, settings, full = job; ev = merged(combo); out = []
    for risks, brk, sm in settings:
        row = dict(combo=" | ".join(combo), n_strats=len(combo), risks="/".join(str(int(r)) for r in risks), breaker=brk, smart=sm)
        for k, v in ftmo_horizon(ev, SEQ["short_win"], risks, brk, sm).items(): row[f"search_{k}"] = v
        if full:
            for k, v in ftmo_horizon(ev, SEQ["stress"], risks, brk, sm).items(): row[f"stress_{k}"] = v
            for s in ("short_held0", "final"):
                sq = [d for d in SEQ[s] if d <= "2026-08-21"]
                row[f"{s}_usd_day"] = round(sum(x[2] * risks[x[5]] for d in sq for x in ev.get(d, [])) / len(sq), 1)
        out.append(row)
    return out

def pool_candidates(a, pool, insts):
    cand = {}
    for track in TRACKS:
        fn = os.path.join(RES, track, "stage3_report.csv")
        if not os.path.exists(fn): continue
        s3_ = pd.read_csv(fn)
        for n in insts:
            d = s3_[(s3_.inst == n) & s3_[pool]].sort_values("robust_max", ascending=False).head(a.k4)
            for p, t, r in zip(d.params, d.tag, d.robust_max): cand[f"{n}:{t}"] = (n, json.loads(p), r)
    return cand

def stage4(a):
    for pool in ("ALL", "AC"): _stage4(a, pool)

def _stage4(a, pool):
    sfx = "" if pool == "ALL" else "_AC"; fn = os.path.join(RES, f"stage4_report{sfx}.csv")
    if done(fn, a): return
    cand = pool_candidates(a, pool, [x for x in a.inst if x in SP.FTMO_INST])
    for b, P in SP.BASELINES.items(): cand[f"NDX100:{b}"] = ("NDX100", P, -1.0)
    log(f"stage4[{pool}]: {len(cand)} strategies: {list(cand)}")
    ev = {c: events(n, P) for c, (n, P, _) in cand.items()}
    inst_of = {c: c.split(":")[0] for c in cand}; names = list(cand)
    isbase = lambda c: c.split(":")[1] in SP.BASELINES
    best1 = {}
    for c in names:
        if isbase(c): continue
        i = inst_of[c]
        if i not in best1 or cand[c][2] > cand[best1[i]][2]: best1[i] = c
    combos = [(c,) for c in names]
    combos += [cb for cb in itertools.combinations(names, 2) if inst_of[cb[0]] != inst_of[cb[1]] and not all(isbase(c) for c in cb)]
    tops = list(best1.values()) + ["NDX100:LIVE_COMP+MA100"]
    for r in (3, 4):
        combos += [cb for cb in itertools.combinations(tops, r) if len({inst_of[c] for c in cb}) == r]
    combos = list(dict.fromkeys(combos))
    uni = [(r, b, s) for r in SP.RISK for b in SP.BREAKER for s in SP.SMART]
    jobs = []
    for cb in combos:
        st = [((r,) * len(cb), b, s) for r, b, s in uni]
        if len(cb) > 1:
            st += [(rs, 2000, "near") for rs in itertools.product((500, 1000), repeat=len(cb)) if len(set(rs)) > 1]
        jobs.append((cb, st, False))
    log(f"stage4[{pool}]: {len(combos)} combos, {sum(len(j[1]) for j in jobs)} challenge simulations, ranked on search-window pass-within-{SP.FTMO['H']}")
    rows = []
    with Pool(a.workers, initializer=_init4, initargs=(ev,)) as p:
        for k, r in enumerate(p.imap_unordered(_s4, jobs, chunksize=2)):
            rows += r
            if k % 200 == 0: log(f"  stage4[{pool}] combo {k}/{len(jobs)}")
    df = pd.DataFrame(rows).sort_values(["search_pass60", "search_fail60"], ascending=[False, True])
    df.to_csv(os.path.join(RES, f"stage4_search{sfx}.csv"), index=False)
    top = df.drop_duplicates("combo").head(a.top4)
    base = df[df.combo.isin([f"NDX100:{b}" for b in SP.BASELINES])].drop_duplicates("combo")
    def st(r):
        rs = tuple(float(x) for x in str(r.risks).split("/"))
        return (rs, None if pd.isna(r.breaker) else r.breaker, None if pd.isna(r.smart) else r.smart)
    jobs = [(tuple(r.combo.split(" | ")), [st(r)], True) for r in pd.concat([top, base]).itertuples()]
    rows = []
    with Pool(a.workers, initializer=_init4, initargs=(ev,)) as p:
        for r in p.imap_unordered(_s4, jobs): rows += r
    rep = pd.DataFrame(rows).drop_duplicates(["combo", "risks", "breaker", "smart"]).sort_values("search_pass60", ascending=False)
    rep.to_csv(fn, index=False)
    cols = ["combo", "risks", "breaker", "smart", "search_pass60", "search_fail60", "search_med", "stress_pass60", "stress_fail60", "short_held0_usd_day", "final_usd_day"]
    log(f"stage4[{pool}] report:\n" + rep[cols].head(40).to_string(index=False))

# ------------------------------------------------------------------ stage 5: FundedNext Futures Flex 100K (NQ)
FLEX = dict(start=100000.0, target=5000.0, mll=2500.0, lock=100100.0, consistency=0.40, pv=2.0, maxc=50, H=60)
def flex_day(ev, bal, floor, risk, breaker):
    F = FLEX; realized = 0.0; opn = []
    def exits(key, bal, realized):
        nonlocal opn
        opn.sort(key=lambda x: x[0]); keep = []
        for x in opn:
            if key is None or x[0] <= key:
                if bal - x[2] <= floor: return bal, realized, True
                bal += x[1]; realized += x[1]
                if bal <= floor: return bal, realized, True
            else: keep.append(x)
        opn = keep; return bal, realized, False
    for e in ev:
        bal, realized, dead = exits(e[0], bal, realized)
        if dead: return bal, True
        if breaker is not None and realized <= -breaker: continue
        n = max(1, min(F["maxc"], math.floor(risk / (e[6] * F["pv"]) + 1e-9)))
        opn.append((e[1], e[2] * e[6] * F["pv"] * n, e[3] * e[6] * F["pv"] * n))
    bal, realized, dead = exits(None, bal, realized)
    return bal, dead

def flex_cohorts(ev, seq, risk, brk):
    F = FLEX; wk = pd.to_datetime(pd.Series(seq)).dt.to_period("W-SUN")
    starts = [k for k in range(len(seq)) if k == 0 or wk.iloc[k] != wk.iloc[k - 1]]
    res = []
    for s0 in starts:
        bal = F["start"]; peak = bal; floor = bal - F["mll"]; best = 0.0; out = None
        for k in range(s0, min(len(seq), s0 + 250)):   # cohorts capped at 250 trading days (then 'open')
            b0 = bal; bal, dead = flex_day(ev.get(seq[k], []), bal, floor, risk, brk)
            if dead: out = ("F", k - s0 + 1); break
            best = max(best, bal - b0)
            if bal - F["start"] >= max(F["target"], best / F["consistency"]): out = ("P", k - s0 + 1); break
            peak = max(peak, bal); floor = min(peak - F["mll"], F["lock"])
        res.append((out or ("O", len(seq) - s0), len(seq) - s0))
    P = [d for (o, d), _ in res if o == "P"]; Fl = [1 for (o, _), _ in res if o == "F"]; resolved = len(P) + len(Fl)
    elig = [(o, d) for (o, d), left in res if left >= F["H"]]
    p60 = sum(1 for o, d in elig if o == "P" and d <= F["H"]); f60 = sum(1 for o, d in elig if o == "F" and d <= F["H"])
    return dict(cohorts=len(res), pass_rate=round(len(P) / resolved, 3) if resolved else None, failed=len(Fl),
                pass60=round(p60 / len(elig), 3) if elig else None, fail60=round(f60 / len(elig), 3) if elig else None,
                med=float(np.median(P)) if P else None)

def _s5(job):
    c, risk, brk = job; ev = EV[c]; row = dict(strategy=c, risk=risk, breaker=brk)
    for s in ("short_win", "stress"):
        for k, v in flex_cohorts(ev, SEQ[s], risk, brk).items(): row[f"{s}_{k}"] = v
    for s in ("short_held0", "final"):
        sq = [d for d in SEQ[s] if d <= "2026-09-15"]
        row[f"{s}_usd_day"] = round(sum(x[2] * x[6] * FLEX["pv"] * max(1, math.floor(risk / (x[6] * FLEX["pv"]) + 1e-9)) for d in sq for x in ev.get(d, [])) / len(sq), 1)
    return row

def stage5(a):
    fn = os.path.join(RES, "stage5_flex_nq.csv")
    if "NQ" not in a.inst or done(fn, a): return
    cand = {}
    for pool in ("ALL", "AC"):
        for c, (n, P, r) in pool_candidates(a, pool, ["NQ"]).items(): cand[f"{c}[{pool}]"] = (n, P)
    for b, P in SP.BASELINES.items(): cand[f"NQ:{b}"] = ("NQ", P)
    cand = {c: v for c, v in cand.items() if not (c.endswith("[AC]") and c.replace("[AC]", "[ALL]") in cand)}
    log(f"stage5 Flex (NQ): {len(cand)} strategies: {list(cand)}")
    ev = {c: events(n, P) for c, (n, P) in cand.items()}
    jobs = [(c, r, b) for c in cand for r in (250, 375, 500) for b in (500, 1000, None)]
    with Pool(a.workers, initializer=_init4, initargs=(ev,)) as p: rows = p.map(_s5, jobs)
    df = pd.DataFrame(rows).sort_values("short_win_pass60", ascending=False); df.to_csv(fn, index=False)
    log("stage5 Flex (NQ):\n" + df.head(30).to_string(index=False))

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "stage0", "s1", "s2", "s3", "stage4", "stage5", "all"])
    ap.add_argument("--track", default="short", choices=list(TRACKS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inst", default=",".join(SP.INST))
    ap.add_argument("--n1", type=int, default=50000, help="short track: stage-1 configs per instrument")
    ap.add_argument("--n1-long", type=int, default=35000, help="long track: stage-1 configs per instrument")
    ap.add_argument("--top2", type=int, default=150, help="short: top configs per fold into stage 2")
    ap.add_argument("--top2-long", type=int, default=60)
    ap.add_argument("--nn", type=int, default=16)
    ap.add_argument("--nn-long", type=int, default=12)
    ap.add_argument("--k3", type=int, default=15, help="top robust configs per fold into stage 3")
    ap.add_argument("--k4", type=int, default=3, help="stage 4/5: configs per instrument and track")
    ap.add_argument("--top4", type=int, default=80)
    ap.add_argument("--force", action="store_true", help="redo stages whose output already exists")
    a = ap.parse_args(); a.inst = [x for x in a.inst.split(",") if x]
    a.inst_all = sorted(set(a.inst) | {"NDX100"}, key=list(SP.INST).index)
    t0 = time.time(); log("start", a.stage, vars(a))
    if a.stage == "all":
        plan = [("prep", None), ("stage0", None)] + [(s, tk) for tk in TRACKS for s in ("s1", "s2", "s3")] + [("stage4", None), ("stage5", None)]
    else:
        plan = [(a.stage, a.track if a.stage in ("s1", "s2", "s3") else None)]
    for st, tk in plan:
        if st != "prep": load_scales()
        (globals()[st](a, tk) if tk else globals()[st](a)); log(f"{st}{'[' + tk + ']' if tk else ''} done, elapsed {round(time.time() - t0)} s")
    if a.stage == "all": open(os.path.join(RES, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
