"""open_exit_search: staged, resumable search (handoff steps 3 + 5). Pre-registered 2026-09-25, see README.md.

  prep    build the per-instrument caches, compute instrument scales (results/scales.json)
  stage0  ablation: the live opening trade + legs (LIVE_COMP) and one-knob exit-management / entry variants,
          every instrument, every split (no selection -> direct answer to step 3)
  stage1  random configs per instrument, scored on TRAIN days (t-stat of daily R; >= 1 trade / 4 days; both halves > 0)
  stage2  robustness: one-step neighbours, robust = min(fit, mean neighbour fit)
  stage3  top robust configs + baselines on every split + the pre-registered checks (A held-out, B stress, C final)
  stage4  FTMO 2-step (one $100k account): configs passing ALL checks, alone and in 2-3 instrument combos,
          x risk / breaker / near sizing; ranked by pass-within-60-trading-days on TRAIN weekly cohorts
Usage: py -3 oe_search.py all --workers 4 [--n1 30000] [--inst NDX100,GER40,...]
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

HELD = SP.heldout_days()
def _bd(a, b): return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(a, b)]
SEQ = {"train": [d for d in _bd(*SP.SEARCH) if d not in HELD], "held": sorted(HELD), "search": _bd(*SP.SEARCH),
       "final": _bd(*SP.FINAL), "stress": _bd(*SP.STRESS)}

# ------------------------------------------------------------------ per-process data
G = {}; SCALE = {}
def data(name):
    if name in G: return G[name]
    r = build(name); g = {k: r[k] for k in r}; days = [str(x) for x in r["days"]]; g["days"] = days
    pos = {d: i for i, d in enumerate(days)}
    close = pd.Series(r["rth_close"].astype(float))
    ma = close.rolling(100).mean().shift(1); prev = close.shift(1)
    g["ma_on"] = {0: np.ones(len(days), bool), 100: ((prev > ma) | ma.isna()).to_numpy()}
    rng = pd.Series(r["rng120"].astype(float)); sd = np.array(days)
    med = float(np.median(rng[(sd >= SP.SEARCH[0]) & (sd <= SP.SEARCH[1])]))
    g["dyn"] = (rng.rolling(60, min_periods=20).median().shift(1) / med).fillna(1.0).clip(0.2, 3.0).to_numpy()
    g["med"] = med
    g["idx"] = {k: np.array([pos[d] for d in v if d in pos], int) for k, v in SEQ.items()}
    G[name] = g
    return g

def load_scales():
    if not SCALE: SCALE.update(json.load(open(os.path.join(RES, "scales.json"))))
    return SCALE

def unit(name, P, g, i):
    return load_scales()[name] * (g["dyn"][i] if P["dyn"] else 1.0)

def day_trades(name, P, idx):
    g = data(name); on = g["ma_on"][P["ma"]]; c = SP.INST[name]["cost"]
    return {int(i): sim(g["blocks"][i], P, unit(name, P, g, i), c) for i in idx if on[i]}

def score(name, P, split):
    g = data(name); idx = g["idx"][split]
    if len(idx) == 0: return dict(days=0, n=0, R=0.0, Rday=None, t=None, pf=None, R_h1=None, R_h2=None, maxDD=None)
    tr = day_trades(name, P, idx); posn = {int(i): k for k, i in enumerate(idx)}
    daily = np.zeros(len(idx)); n = 0; w = lo = 0.0
    for i, L in tr.items():
        for t in L:
            daily[posn[i]] += t["R"]; n += 1
            if t["R"] > 0: w += t["R"]
            else: lo -= t["R"]
    h = len(idx) // 2; sd = daily.std(); eq = np.cumsum(daily)
    return dict(days=len(idx), n=n, R=round(daily.sum(), 2), Rday=round(daily.mean(), 4),
                t=round(daily.mean() / sd * math.sqrt(len(idx)), 3) if sd > 0 else 0.0,
                pf=round(w / lo, 3) if lo > 0 else None, R_h1=round(daily[:h].sum(), 2), R_h2=round(daily[h:].sum(), 2),
                maxDD=round(float((eq - np.maximum.accumulate(eq)).min()), 2))

def fitness(m):
    if m["days"] == 0 or m["n"] < 0.25 * m["days"] or m["R_h1"] <= 0 or m["R_h2"] <= 0: return -99.0
    return m["t"]

def all_splits(name, P):
    row = {}
    for s in ("train", "held", "search", "final", "stress"):
        for k, v in score(name, P, s).items(): row[f"{s}_{k}"] = v
    y = {}
    g = data(name); idx = g["idx"]["stress"]
    for i, L in day_trades(name, P, idx).items():
        yr = g["days"][i][:4]; y[yr] = y.get(yr, 0.0) + sum(t["R"] for t in L)
    for yr in sorted(y): row[f"stressR_{yr}"] = round(y[yr], 1)
    return row

# ------------------------------------------------------------------ prep
def prep(a):
    sc = {}
    for n in a.inst_all:
        t = time.time(); g = data(n)
        log(f"prep {n}: {len(g['days'])} days {g['days'][0]}..{g['days'][-1]}, median 2h range (search) {g['med']:.1f}, "
            f"train/held/final/stress days {[len(g['idx'][k]) for k in ('train', 'held', 'final', 'stress')]} ({time.time() - t:.0f}s)")
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
    return V

def _s0(job):
    name, vn, P = job; load_scales()
    return dict(inst=name, variant=vn, params=json.dumps(P), **all_splits(name, P))

def stage0(a):
    jobs = [(n, vn, P) for n in a.inst_all for vn, P in variants()]
    log(f"stage0: {len(jobs)} ablation runs")
    with Pool(a.workers) as p: rows = p.map(_s0, jobs, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "stage0_ablation.csv"), index=False)
    cols = ["variant", "train_Rday", "held_Rday", "final_Rday", "stress_Rday", "search_maxDD", "stress_maxDD"]
    for n in a.inst_all: log(f"stage0 {n}:\n" + df[df.inst == n][cols].to_string(index=False))

# ------------------------------------------------------------------ stage 1
def _s1(job):
    name, i0, i1, seed = job; load_scales(); out = []
    for i in range(i0, i1):
        P = SP.sample(np.random.default_rng([seed, i]))
        m = score(name, P, "train"); out.append(dict(inst=name, idx=i, params=json.dumps(P), fit=fitness(m), **m))
    return out

def stage1(a):
    jobs = []; done = {}
    for k, n in enumerate(a.inst):
        fn = os.path.join(RES, f"stage1_{n}.csv"); done[n] = set(pd.read_csv(fn).idx) if os.path.exists(fn) else set()
        jobs += [(n, s, min(s + 200, a.n1), 5000 + k * 17) for s in range(0, a.n1, 200)
                 if not all(i in done[n] for i in range(s, min(s + 200, a.n1)))]
    log(f"stage1: {len(jobs)} chunks of 200 configs to do ({a.n1} per instrument, {len(a.inst)} instruments)")
    with Pool(a.workers) as p:
        for k, rows in enumerate(p.imap_unordered(_s1, jobs)):
            df = pd.DataFrame(rows); n = df.inst.iloc[0]; fn = os.path.join(RES, f"stage1_{n}.csv")
            df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
            if k % 25 == 0: log(f"  stage1 chunk {k + 1}/{len(jobs)}")

# ------------------------------------------------------------------ stage 2
def _s2(job):
    name, P, nn, seed = job; load_scales()
    fb = fitness(score(name, P, "train"))
    nb = [max(fitness(score(name, Q, "train")), -1.0) for Q in SP.neighbours(P, np.random.default_rng(seed), nn)]
    return dict(inst=name, params=json.dumps(P), fit=fb, nb_mean=round(float(np.mean(nb)), 3), nb_min=round(float(np.min(nb)), 3),
                robust=round(min(fb, float(np.mean(nb))), 3))

def stage2(a):
    jobs = []
    for n in a.inst:
        s1 = pd.read_csv(os.path.join(RES, f"stage1_{n}.csv")).drop_duplicates("idx")
        top = s1[s1.fit > 0].sort_values("fit", ascending=False).head(a.top2)
        log(f"stage2 {n}: {len(s1)} stage-1 configs, {(s1.fit > 0).sum()} pass the filter, checking {len(top)} x {a.nn} neighbours")
        jobs += [(n, json.loads(p), a.nn, 11 + k) for k, p in enumerate(top.params)]
    with Pool(a.workers) as p: rows = p.map(_s2, jobs, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "stage2.csv"), index=False)
    for n in a.inst:
        d = df[df.inst == n].sort_values("robust", ascending=False)
        log(f"stage2 {n} best robust:", d.head(3)[["robust", "fit", "nb_mean"]].to_dict("records"))

# ------------------------------------------------------------------ stage 3
def _s3(job):
    name, tag, P, robust = job; load_scales()
    return dict(inst=name, tag=tag, robust=robust, params=json.dumps(P), **all_splits(name, P))

def stage3(a):
    s2 = pd.read_csv(os.path.join(RES, "stage2.csv")); jobs = []
    for n in a.inst:
        d = s2[(s2.inst == n) & (s2.robust > 0)].sort_values("robust", ascending=False).head(a.k3)
        jobs += [(n, f"cand{k}", json.loads(p), r) for k, (p, r) in enumerate(zip(d.params, d.robust))]
        jobs += [(n, b, P, None) for b, P in SP.BASELINES.items()]
    log(f"stage3: {len(jobs)} configs on every split")
    with Pool(a.workers) as p: rows = p.map(_s3, jobs, chunksize=1)
    df = pd.DataFrame(rows); out = []
    for n, d in df.groupby("inst"):
        base = d[d.tag.isin(list(SP.BASELINES))]
        b_st = base.stress_Rday.max() if base.stress_days.max() >= 250 else None
        d = d.copy()
        d["A_held"] = (d.held_Rday > 0) & (d.held_Rday >= 0.4 * d.train_Rday)
        d["B_stress"] = (d.stress_days >= 250) & (d.stress_Rday >= 0) & (d.stress_Rday >= (b_st if b_st is not None else 0))
        d["C_final"] = d.final_Rday.isna() | (d.final_Rday >= 0)
        d["D_beats_base_held"] = d.held_Rday > base.held_Rday.max()
        d["ALL"] = d.A_held & d.B_stress & d.C_final & ~d.tag.isin(list(SP.BASELINES))
        out.append(d)
    df = pd.concat(out); df.to_csv(os.path.join(RES, "stage3_report.csv"), index=False)
    cols = ["tag", "robust", "train_Rday", "held_Rday", "search_maxDD", "final_Rday", "stress_Rday", "stress_maxDD", "A_held", "B_stress", "C_final", "ALL"]
    for n, d in df.groupby("inst"):
        log(f"stage3 {n}: {int(d.ALL.sum())} of {int((~d.tag.isin(list(SP.BASELINES))).sum())} candidates pass ALL\n" + d[~d.tag.isin(list(SP.BASELINES))][cols].head(12).to_string(index=False)
            + "\n" + d[d.tag.isin(list(SP.BASELINES))][cols].to_string(index=False))

# ------------------------------------------------------------------ stage 4: FTMO 2-step, one account
def events(name, P):
    """{date: [(entry_key, exit_key, R, mae_R, tp_R)]} over all days; keys = UTC minute of the day."""
    g = data(name); out = {}
    for i, L in day_trades(name, P, range(len(g["days"]))).items():
        om = int(g["open_utc_min"][i])
        for t in L:
            out.setdefault(g["days"][i], []).append((om + t["eb"] + 1, om + t["xb"] + 1, t["R"], t["mae_R"], t["tp_R"]))
    return out

def ftmo_day(ev, bal, risk, breaker, smart, goal, need_days):
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
        s = risk
        if smart == "near":
            if bal >= goal:
                if need_days <= 0 or nent >= 1: continue
                s = F["min_size"]
            elif e[4] is not None and e[4] > 0:
                s = min(s, max(F["min_size"], (goal - bal) / e[4]))
        opn.append((e[1], e[2] * s, e[3] * s)); traded = True; nent += 1
    bal, realized, w = exits(None, bal, realized)
    return bal, w, traded

def horizon(ev, seq, risk, brk, smart):
    F = SP.FTMO; H = F["H"]
    wk = pd.to_datetime(pd.Series(seq)).dt.to_period("W-SUN")
    starts = [k for k in range(len(seq)) if (k == 0 or wk.iloc[k] != wk.iloc[k - 1]) and k + H <= len(seq)]
    P = Fl = 0; dd = []
    for s0 in starts:
        phase, bal, goal, td, res = 1, F["start"], F["start"] + F["t1"], 0, None
        for k in range(s0, s0 + H):
            bal, w, tr = ftmo_day(ev.get(seq[k], []), bal, risk, brk, smart, goal, F["min_days"] - td); td += int(tr)
            if w: res = "F"; break
            if bal >= goal and td >= F["min_days"]:
                if phase == 1: phase, bal, goal, td = 2, F["start"], F["start"] + F["t2"], 0
                else: res = "P"; dd.append(k - s0 + 1); break
        P += res == "P"; Fl += res == "F"
    n = len(starts)
    return dict(n=n, pass60=round(P / n, 3) if n else None, fail60=round(Fl / n, 3) if n else None, med=float(np.median(dd)) if dd else None)

EV = {}
def _init4(ev): EV.update(ev)
def _s4(job):
    combo, settings, full = job
    ev = {}
    for c in combo:
        for d, L in EV[c].items(): ev.setdefault(d, []).extend(L)
    for d in ev: ev[d].sort()
    out = []
    for risk, brk, sm in settings:
        row = dict(combo=" | ".join(combo), n_strats=len(combo), risk=risk, breaker=brk, smart=sm)
        for k, v in horizon(ev, SEQ["train"], risk, brk, sm).items(): row[f"train_{k}"] = v
        if full:
            for s in ("search", "stress"):
                for k, v in horizon(ev, SEQ[s], risk, brk, sm).items(): row[f"{s}_{k}"] = v
            for s in ("held", "final"):
                sq = [d for d in SEQ[s] if d <= "2026-08-21"]   # CFD data end (UK100 ends 07-31)
                row[f"{s}_usd_day"] = round(sum(x[2] for d in sq for x in ev.get(d, [])) * risk / len(sq), 1)
        out.append(row)
    return out

def stage4(a):
    """Main pool = configs passing ALL checks. Secondary pool 'AC' (report only, clearly regime-dependent) = configs
    passing A (held-out) and C (final) but not necessarily B (2016-23 stress)."""
    for pool in ("ALL", "AC"): _stage4(a, pool)

def _stage4(a, pool):
    s3 = pd.read_csv(os.path.join(RES, "stage3_report.csv")); cand = {}
    s3["AC"] = s3.A_held & s3.C_final & ~s3.tag.isin(list(SP.BASELINES))
    sfx = "" if pool == "ALL" else "_AC"
    for n in [x for x in a.inst if x in SP.FTMO_INST]:
        d = s3[(s3.inst == n) & s3[pool]].sort_values("robust", ascending=False).head(a.k4)
        for k, p in enumerate(d.params): cand[f"{n}:{d.tag.iloc[k]}"] = (n, json.loads(p))
    for b, P in SP.BASELINES.items(): cand[f"NDX100:{b}"] = ("NDX100", P)
    log(f"stage4[{pool}]: {len(cand)} strategies: {list(cand)}")
    ev = {c: events(n, P) for c, (n, P) in cand.items()}
    inst_of = {c: c.split(":")[0] for c in cand}; names = list(cand)
    combos = [(c,) for c in names]
    for r in (2, 3):
        combos += [cb for cb in itertools.combinations(names, r) if len({inst_of[c] for c in cb}) == r
                   and not all(c.endswith(tuple(SP.BASELINES)) for c in cb)]
    settings = [(r, b, s) for r in SP.RISK for b in SP.BREAKER for s in SP.SMART]
    log(f"stage4: {len(combos)} combos x {len(settings)} settings, ranked on TRAIN pass-within-{SP.FTMO['H']}")
    rows = []
    with Pool(a.workers, initializer=_init4, initargs=(ev,)) as p:
        for k, r in enumerate(p.imap_unordered(_s4, [(cb, settings, False) for cb in combos], chunksize=2)):
            rows += r
            if k % 100 == 0: log(f"  stage4 combo {k}/{len(combos)}")
    df = pd.DataFrame(rows).sort_values(["train_pass60", "train_fail60"], ascending=[False, True])
    df.to_csv(os.path.join(RES, f"stage4_train{sfx}.csv"), index=False)
    top = df.drop_duplicates("combo").head(a.top4)
    base = df[df.combo.isin([f"NDX100:{b}" for b in SP.BASELINES])].drop_duplicates("combo")
    jobs = [(tuple(r.combo.split(" | ")), [(r.risk, None if pd.isna(r.breaker) else r.breaker, None if pd.isna(r.smart) else r.smart)], True)
            for r in pd.concat([top, base]).itertuples()]
    rows = []
    with Pool(a.workers, initializer=_init4, initargs=(ev,)) as p:
        for r in p.imap_unordered(_s4, jobs): rows += r
    rep = pd.DataFrame(rows).drop_duplicates(["combo", "risk", "breaker", "smart"]).sort_values("train_pass60", ascending=False); rep.to_csv(os.path.join(RES, f"stage4_report{sfx}.csv"), index=False)
    cols = ["combo", "risk", "breaker", "smart", "train_pass60", "train_fail60", "train_med", "search_pass60", "search_fail60", "stress_pass60", "stress_fail60", "held_usd_day", "final_usd_day"]
    log(f"stage4[{pool}] report:\n" + rep[cols].head(40).to_string(index=False))

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "stage0", "stage1", "stage2", "stage3", "stage4", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inst", default=",".join(SP.INST), help="instruments to search (comma list)")
    ap.add_argument("--n1", type=int, default=30000, help="stage-1 random configs per instrument")
    ap.add_argument("--top2", type=int, default=300)
    ap.add_argument("--nn", type=int, default=16)
    ap.add_argument("--k3", type=int, default=30)
    ap.add_argument("--k4", type=int, default=2, help="stage-4 configs per instrument (those passing ALL checks)")
    ap.add_argument("--top4", type=int, default=60)
    a = ap.parse_args(); a.inst = [x for x in a.inst.split(",") if x]
    a.inst_all = sorted(set(a.inst) | {"NDX100"}, key=list(SP.INST).index)
    t0 = time.time(); log("start", a.stage, vars(a))
    for st in (["prep", "stage0", "stage1", "stage2", "stage3", "stage4"] if a.stage == "all" else [a.stage]):
        globals()[st](a); log(f"{st} done, elapsed {round(time.time() - t0)} s")
    if a.stage == "all": open(os.path.join(RES, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
