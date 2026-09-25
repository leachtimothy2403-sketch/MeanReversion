"""flex_search: staged, resumable search for a FundedNext Futures Flex $100K strategy on NQ (MNQ), 3 sessions.

  stage1  random search per session (NY / LDN / ASIA), trade-level score on TRAIN days
  stage2  robustness: re-score the top configs' one-step neighbours, keep configs on a plateau
  stage3  combine session candidates (+ "session off") with risk / breaker / profit-stop and simulate weekly Flex
          cohorts on TRAIN days; rank by pass rate
  stage4  report the top combos on the held-out weeks, the contiguous search window, the final holdout
          (2026-07-01..09-15, first look)
Usage:  python fs_search.py all --workers 4 [--n1 20000]
        python fs_search.py stage1|stage2|stage3|stage4 ...
Outputs in flex_search/results/ ; progress in flex_search/results/run.log
"""
import os, sys, json, time, argparse, itertools, math
import numpy as np, pandas as pd
from multiprocessing import Pool
from fs_data import build, ORDER
from fs_engine import sim
import fs_space as SP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True); open(os.path.join(RES, "run.log"), "a").write(msg + "\n")

# ------------------------------------------------------------------ shared data (loaded once per worker process)
G = {}
def make_data(start, end):
    r = build(start, end); g = {}
    days = list(r["days"]); g["days"] = days
    g["blocks"] = {k: r[k] for k in r if k not in ("days", "rth_close")}
    close = pd.Series(r["rth_close"])
    g["ma_on"] = {0: np.ones(len(days), bool)}
    for n in (50, 100, 200):   # state known before the day: previous NY close vs its n-day average; missing -> on
        prev = close.shift(1); ma = close.rolling(n).mean().shift(1)
        g["ma_on"][n] = ((prev > ma) | ma.isna()).to_numpy()
    return g

def load():
    if G: return G
    G.update(make_data("2024-01-02", "2026-09-15")); days = G["days"]
    ho = SP.heldout_days(days)
    G["train"] = np.array([i for i, d in enumerate(days) if SP.SEARCH[0] <= d <= SP.SEARCH[1] and d not in ho])
    G["held"] = np.array([i for i, d in enumerate(days) if d in ho])
    G["search"] = np.array([i for i, d in enumerate(days) if SP.SEARCH[0] <= d <= SP.SEARCH[1]])
    G["final"] = np.array([i for i, d in enumerate(days) if SP.FINAL[0] <= d <= SP.FINAL[1]])
    return G

def session_trades(sess, P, idx, g=None):
    """{day_index: [trade,...]} for the given day indices (respecting the MA switch)."""
    g = g or load(); blk = g["blocks"][f"{sess}_{P['offset']}"]; on = g["ma_on"][P["ma"]]
    return {i: sim(blk[i], P) for i in idx if on[i]}

def score(sess, P, idx):
    """trade-level metrics in R (cost included, size-weighted) on the given days."""
    c = SP.COST[sess]; tr = session_trades(sess, P, idx)
    daily = np.zeros(len(idx)); n = 0; wins = 0.0; losses = 0.0
    pos = {i: k for k, i in enumerate(idx)}
    for i, L in tr.items():
        for t in L:
            r = (t["pnl"] - c) / t["sl"] * t["size"]; daily[pos[i]] += r; n += 1
            if r > 0: wins += r
            else: losses -= r
    h = len(idx) // 2
    sd = daily.std()
    eq = np.cumsum(daily); dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    return dict(n=n, tpd=round(n / len(idx), 2), R=round(daily.sum(), 2), R_trade=round(daily.sum() / n, 4) if n else 0.0,
                pfR=round(wins / losses, 3) if losses > 0 else (9.99 if wins > 0 else 0.0),
                t=round(daily.mean() / sd * math.sqrt(len(idx)), 3) if sd > 0 else 0.0,
                R_h1=round(daily[:h].sum(), 2), R_h2=round(daily[h:].sum(), 2), maxDD=round(dd, 2))

def fitness(m, min_n):
    """stage-1/2 objective: t-stat of daily R, only if enough trades and both halves of TRAIN positive."""
    if m["n"] < min_n or m["R_h1"] <= 0 or m["R_h2"] <= 0: return -99.0
    return m["t"]

# ------------------------------------------------------------------ stage 1
def _s1(args):
    sess, i0, i1, seed = args
    load(); rng = np.random.default_rng(seed); out = []
    for i in range(i0, i1):
        r2 = np.random.default_rng([seed, i]); P = SP.sample(sess, r2)
        if SP.is_dead(P): continue
        m = score(sess, P, G["train"]); out.append(dict(session=sess, idx=i, params=json.dumps(P), **m))
    return out

def stage1(a):
    load()
    for sess in ORDER:
        fn = os.path.join(RES, f"stage1_{sess}.csv")
        done = set(pd.read_csv(fn).idx) if os.path.exists(fn) else set()
        chunks = [(sess, s, min(s + 250, a.n1), 1000 + ORDER.index(sess)) for s in range(0, a.n1, 250)]
        chunks = [c for c in chunks if not all(i in done for i in range(c[1], c[2]))]
        log(f"stage1 {sess}: {a.n1} configs, {len(chunks)} chunks to do")
        with Pool(a.workers) as p:
            for k, rows in enumerate(p.imap_unordered(_s1, chunks)):
                if rows:
                    df = pd.DataFrame(rows); df = df[~df.idx.isin(done)]
                    df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
                if k % 10 == 0: log(f"  {sess} chunk {k + 1}/{len(chunks)}")

# ------------------------------------------------------------------ stage 2
def _s2(args):
    sess, P, nn, seed, min_n = args
    load(); rng = np.random.default_rng(seed)
    base = score(sess, P, G["train"]); fb = fitness(base, min_n)
    nb = [fitness(score(sess, Q, G["train"]), min_n) for Q in SP.neighbours(sess, P, rng, nn)]
    nb = [max(x, -1.0) for x in nb]   # a dead neighbour counts as t=-1, not -99
    return dict(session=sess, params=json.dumps(P), fit=fb, nb_mean=round(float(np.mean(nb)), 3),
                nb_min=round(float(np.min(nb)), 3), robust=round(min(fb, float(np.mean(nb))), 3), **base)

def stage2(a):
    load(); rows = []
    for sess in ORDER:
        s1 = pd.read_csv(os.path.join(RES, f"stage1_{sess}.csv"))
        min_n = int(0.25 * len(G["train"]))            # at least ~1 trade every 4 days
        s1["fit"] = [fitness(r, min_n) for r in s1.to_dict("records")]
        top = s1.sort_values("fit", ascending=False).head(a.top2)
        top = top[top.fit > 0]
        log(f"stage2 {sess}: {len(top)} configs x {a.nn} neighbours (stage1 rows {len(s1)}, passing filter {(s1.fit > 0).sum()})")
        jobs = [(sess, json.loads(p), a.nn, 7 + k, min_n) for k, p in enumerate(top.params)]
        with Pool(a.workers) as p:
            rows += p.map(_s2, jobs)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "stage2.csv"), index=False)
    for sess in ORDER:
        d = df[df.session == sess].sort_values("robust", ascending=False)
        log(f"stage2 {sess} best robust:", d.head(3)[["robust", "fit", "nb_mean", "n", "R", "pfR"]].to_dict("records"))

# ------------------------------------------------------------------ stage 3: Flex cohorts
def merge_days(trades_by_sess, g=None):
    """{day: [(entry_key, exit_key, net_pts_per_contract, sl, net_tp_pts, size, mae_pts), ...] sorted by entry}.
    Sessions are ordered ASIA -> LDN -> NY inside a CME trading day, trades by bar."""
    g = g or load(); out = {}
    for si, s in enumerate(ORDER):
        if s not in trades_by_sess: continue
        c = SP.COST[s]
        for i, L in trades_by_sess[s].items():
            for t in L:
                out.setdefault(i, []).append((si * 10000 + t["eb"], si * 10000 + t["xb"], t["pnl"] - c, t["sl"],
                                              max(t["tpd"] - c, 0.25), t["size"], t["mae"]))
    for i in out: out[i].sort()
    return out

def ncon(risk, sl, size):
    return max(1, min(SP.MAXC, math.floor(risk / (sl * SP.PV) * size)))

def day_usd(merged, D, risk, breaker, pstop):
    """Plain per-day $ P&L (no challenge state) for reporting."""
    pnl = np.zeros(D); ntr = np.zeros(D, int)
    for i, ev in merged.items():
        taken = []
        for e in ev:
            realized = sum(x[0] for x in taken if x[1] < e[0])
            if breaker is not None and realized <= -breaker: continue
            if pstop is not None and realized >= pstop: continue
            taken.append((e[2] * SP.PV * ncon(risk, e[3], e[5]), e[1]))
        pnl[i] = sum(x[0] for x in taken); ntr[i] = len(taken)
    return pnl, ntr

def run_day(ev, bal, floor, risk, breaker, pstop, smart, small):
    """One day of a challenge. Returns (balance, failed, small). Entries in entry order; exits applied when their bar
    has passed. smart sizing (None = off; "min" | 0.25 | 0.5): (a) while below the base target, a trade is sized so its
    take-profit would just reach the $105k target when that needs fewer contracts than the normal risk; (b) once the
    balance has reached $105k (the consistency rule has raised the target), every further trade is 1 MNQ ("min") or
    that fraction of the normal size (at least 1 MNQ)."""
    F = SP.FLEX; goal = F["start"] + F["target"]; realized = 0.0; opn = []
    def exits_until(key, bal, realized, small):
        nonlocal opn
        opn.sort(key=lambda x: x[0]); keep = []
        for x in opn:
            if key is None or x[0] < key:
                if bal - x[2] <= floor: return bal, realized, small, True
                bal += x[1]; realized += x[1]
                if smart is not None and bal >= goal: small = True
            else: keep.append(x)
        opn = keep
        return bal, realized, small, False
    for e in ev:
        bal, realized, small, dead = exits_until(e[0], bal, realized, small)
        if dead: return bal, True, small
        if breaker is not None and realized <= -breaker: continue
        if pstop is not None and realized >= pstop: continue
        n = ncon(risk, e[3], e[5])
        if smart is not None:
            if small: n = 1 if smart == "min" else max(1, math.floor(n * smart))
            elif goal - bal > 0: n = min(n, max(1, math.ceil((goal - bal) / (e[4] * SP.PV))))
        opn.append((e[1], e[2] * SP.PV * n, e[6] * SP.PV * n))
    bal, realized, small, dead = exits_until(None, bal, realized, small)
    return bal, dead, small

def flex_cohorts(merged, seq, risk, breaker, pstop, smart, starts=None, g=None):
    """Weekly FundedNext Flex cohorts over the day-index sequence `seq` (one new $100k challenge per Monday-week)."""
    g = g or load(); F = SP.FLEX; days = g["days"]
    if starts is None:
        wk = pd.to_datetime(pd.Series([days[i] for i in seq])).dt.to_period("W-SUN")
        starts = [k for k in range(len(seq)) if k == 0 or wk.iloc[k] != wk.iloc[k - 1]]
    res = []
    for s0 in starts:
        bal = F["start"]; peak = bal; floor = bal - F["mll"]; best = 0.0; out = None; small = False
        for k in range(s0, len(seq)):
            b0 = bal
            bal, dead, small = run_day(merged.get(seq[k], []), bal, floor, risk, breaker, pstop, smart, small)
            if dead: out = ("FAIL", k - s0 + 1); break
            best = max(best, bal - b0); tgt = max(F["target"], best / F["consistency"])
            if bal - F["start"] >= tgt: out = ("PASS", k - s0 + 1); break
            peak = max(peak, bal); floor = min(peak - F["mll"], F["lock"])
        res.append(out or ("OPEN", len(seq) - s0))
    P = [d for o, d in res if o == "PASS"]; Fl = [d for o, d in res if o == "FAIL"]
    resolved = len(P) + len(Fl)
    return dict(cohorts=len(res), passed=len(P), failed=len(Fl), open=len(res) - resolved,
                pass_rate=round(len(P) / resolved, 3) if resolved else 0.0, med_days=float(np.median(P)) if P else None)

_C = {}
def _s3(args):
    combo, settings = args
    tb = _C["tb"]; sessions = {s: c for s, c in combo.items() if c is not None}
    merged = merge_days({s: tb[s][c] for s, c in sessions.items()})
    D = len(G["days"]); out = []
    for risk, brk, ps, smart in settings:
        pnl, ntr = day_usd(merged, D, risk, brk, ps)
        tr = flex_cohorts(merged, list(G["train"]), risk, brk, ps, smart)
        out.append(dict(combo=json.dumps(combo), risk=risk, breaker=brk, pstop=ps, smart=smart,
                        trades_per_day=round(ntr[G["train"]].mean(), 2), usd_train=round(pnl[G["train"]].sum()), **tr))
    return out

def _init3(tb):
    load(); _C["tb"] = tb

def stage3(a):
    load(); s2 = pd.read_csv(os.path.join(RES, "stage2.csv"))
    cand = {}
    for s in ORDER:
        d = s2[(s2.session == s) & (s2.robust > 0)].sort_values("robust", ascending=False).head(a.k3)
        cand[s] = [json.loads(p) for p in d.params]
        log(f"stage3 {s}: {len(cand[s])} robust candidates")
    allidx = list(range(len(G["days"])))
    tb = {s: [session_trades(s, P, allidx) for P in cand[s]] for s in ORDER}
    json.dump(cand, open(os.path.join(RES, "stage3_candidates.json"), "w"))
    rng = np.random.default_rng(3)
    opts = {s: [None] + list(range(len(cand[s]))) for s in ORDER}
    combos = [dict(zip(ORDER, c)) for c in itertools.product(*[opts[s] for s in ORDER]) if any(x is not None for x in c)]
    jobs = []
    for cb in combos:
        st = []
        for _ in range(a.r3):
            r_, b_, p_ = SP.RISK[rng.integers(len(SP.RISK))], SP.BREAKER[rng.integers(len(SP.BREAKER))], SP.PROFIT_STOP[rng.integers(len(SP.PROFIT_STOP))]
            st += [(r_, b_, p_, sm) for sm in SP.SMART]     # every setting with each smart-sizing mode
        jobs.append((cb, st))
    log(f"stage3: {len(combos)} session combos x {len(SP.SMART) * a.r3} settings = {len(SP.SMART) * a.r3 * len(combos)} challenge simulations")
    rows = []
    with Pool(a.workers, initializer=_init3, initargs=(tb,)) as p:
        for k, r in enumerate(p.imap_unordered(_s3, jobs, chunksize=4)):
            rows += r
            if k % 200 == 0: log(f"  stage3 combo {k}/{len(jobs)}")
    df = pd.DataFrame(rows).sort_values(["pass_rate", "passed"], ascending=False)
    df.to_csv(os.path.join(RES, "stage3.csv"), index=False)
    log("stage3 top 5:", df.head(5)[["combo", "risk", "breaker", "pstop", "smart", "pass_rate", "passed", "failed", "med_days"]].to_dict("records"))
    sm = df.assign(smart=df.smart.astype(str)).groupby("smart").pass_rate.agg(["mean", "max"]).round(3).to_dict()
    log("stage3 smart sizing effect (mean / max pass rate):", sm)

# ------------------------------------------------------------------ stage 4: report (first look at held-out data)
def _nan(x): return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
def _smart(x):
    x = _nan(x)
    if x is None or x == "None": return None
    return "min" if x == "min" else float(x)

def stage4(a):
    load(); cand = json.load(open(os.path.join(RES, "stage3_candidates.json")))
    s3 = pd.read_csv(os.path.join(RES, "stage3.csv"))
    s3 = s3[s3.passed >= 10].head(a.top4)
    allidx = list(range(len(G["days"]))); D = len(G["days"]); cache = {}
    def tb(s, c):
        if (s, c) not in cache: cache[(s, c)] = session_trades(s, cand[s][c], allidx)
        return cache[(s, c)]
    out = []
    for r in s3.to_dict("records"):
        combo = json.loads(r["combo"]); sessions = {s: int(c) for s, c in combo.items() if _nan(c) is not None}
        brk, ps = _nan(r["breaker"]), _nan(r["pstop"]); smart = _smart(r["smart"])
        merged = merge_days({s: tb(s, c) for s, c in sessions.items()})
        pnl, ntr = day_usd(merged, D, r["risk"], brk, ps)
        row = dict(r)
        for lab in ("train", "held", "final"): row[f"{lab}_usd_per_day"] = round(pnl[G[lab]].mean(), 1)
        cs = flex_cohorts(merged, list(G["search"]), r["risk"], brk, ps, smart); row.update({f"search_{k}": v for k, v in cs.items()})
        cf = flex_cohorts(merged, list(G["final"]), r["risk"], brk, ps, smart); row.update({f"final_{k}": v for k, v in cf.items()})
        for s, c in sessions.items():
            row[f"{s}_params"] = json.dumps(cand[s][c])
            for lab in ("train", "held", "final"):
                m = score(s, cand[s][c], G[lab]); row[f"{s}_{lab}_Rday"] = round(m["R"] / len(G[lab]), 3); row[f"{s}_{lab}_n"] = m["n"]
        out.append(row)
    # baselines: the current live NY strategy and the displacement->FV variant, same simulator, never selected
    base = dict(offset=0, window=120, exit="hard", open_trade=True, vwap=True, big_thr=15., big_sl=35., big_tp=115., std_sl=25.,
                std_tp=42., legs=True, leg_gap=15., leg_sl=35., leg_tp=115., leg_max=6, rev_mode="std", min_fv=125., min_bos=10.,
                disp_pts=20., wait_open=True, maxpos=1, stack_trig="bos", stack_gap=15., stack_size=1.0, ma=0)
    BASES = {"LIVE": base, "LIVE+MA100": dict(base, ma=100),
             "DISP_FV": dict(base, rev_mode="disp_fv", min_fv=60., maxpos=2, exit="run"),
             "DISP_FV+MA100": dict(base, rev_mode="disp_fv", min_fv=60., maxpos=2, exit="run", ma=100)}
    brows = []
    for name, P in BASES.items():
        merged = merge_days({"NY": session_trades("NY", P, allidx)})
        for risk in SP.RISK:
            for brk in (1000, None):
                for smart in SP.SMART:
                    pnl, ntr = day_usd(merged, D, risk, brk, None)
                    row = dict(baseline=name, risk=risk, breaker=brk, smart=smart)
                    for lab in ("train", "search", "final"):
                        c = flex_cohorts(merged, list(G[lab]), risk, brk, None, smart); row.update({f"{lab}_{k}": v for k, v in c.items()})
                    for lab in ("train", "held", "final"): row[f"{lab}_usd_per_day"] = round(pnl[G[lab]].mean(), 1)
                    brows.append(row)
    bdf = pd.DataFrame(brows); bdf.to_csv(os.path.join(RES, "stage4_baselines.csv"), index=False)
    log("baselines:\n" + bdf[["baseline", "risk", "breaker", "smart", "train_pass_rate", "search_pass_rate", "search_med_days", "train_usd_per_day", "held_usd_per_day", "final_usd_per_day"]].to_string(index=False))
    if a.stress:   # 2016-2023 regime stress test (report only, never used for selection)
        g2 = make_data("2016-09-15", "2023-12-29"); yrs = pd.Series(pd.to_datetime(g2["days"]).year)
        all2 = list(range(len(g2["days"])))
        for row in out:
            combo = json.loads(row["combo"]); sessions = {s: int(c) for s, c in combo.items() if _nan(c) is not None}
            merged = merge_days({s: session_trades(s, cand[s][c], all2, g2) for s, c in sessions.items()}, g2)
            brk, ps = _nan(row["breaker"]), _nan(row["pstop"])
            pnl2, _ = day_usd(merged, len(all2), row["risk"], brk, ps)
            for y, v in pd.Series(pnl2).groupby(yrs).sum().items(): row[f"stress_usd_{y}"] = round(v)
            cs = flex_cohorts(merged, all2, row["risk"], brk, ps, _smart(row["smart"]), g=g2); row.update({f"stress_{k}": v for k, v in cs.items()})
    df = pd.DataFrame(out); df.to_csv(os.path.join(RES, "stage4_report.csv"), index=False)
    log("stage4 written:", len(df), "rows")
    cols = ["combo", "risk", "breaker", "pstop", "smart", "pass_rate", "search_pass_rate", "final_passed", "final_failed", "final_open",
            "train_usd_per_day", "held_usd_per_day", "final_usd_per_day"]
    if len(df): log("\n" + df[cols].head(15).to_string(index=False))

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "stage3", "stage4", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n1", type=int, default=20000, help="stage-1 random configs per session")
    ap.add_argument("--top2", type=int, default=200, help="stage-2 configs per session to robustness-check")
    ap.add_argument("--nn", type=int, default=12, help="stage-2 neighbours per config")
    ap.add_argument("--k3", type=int, default=12, help="stage-3 candidates per session (+ 'off')")
    ap.add_argument("--r3", type=int, default=12, help="stage-3 random risk/breaker/profit-stop settings per combo")
    ap.add_argument("--top4", type=int, default=40, help="stage-4 combos to report")
    ap.add_argument("--stress", action="store_true", help="stage 4: also run the finalists over 2016-2023 (report only)")
    a = ap.parse_args()
    t0 = time.time(); log("start", a.stage, vars(a))
    for st in (["stage1", "stage2", "stage3", "stage4"] if a.stage == "all" else [a.stage]):
        globals()[st](a); log(f"{st} done, elapsed {round(time.time() - t0)} s")
    open(os.path.join(RES, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
