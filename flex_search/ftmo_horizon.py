"""Time-aware FTMO metric: share of weekly cohorts that PASS within H trading days (cohorts need >= H days of data)."""
import json, numpy as np, pandas as pd
import fs_search as FS, fs_space as SP
pd.set_option("display.width", 250)
G = FS.load(); F = SP.FTMO
def horizon(merged, seq, risk, brk, smart, H=60, g=None):
    g = g or G; days = g["days"]
    wk = pd.to_datetime(pd.Series([days[i] for i in seq])).dt.to_period("W-SUN")
    starts = [k for k in range(len(seq)) if (k == 0 or wk.iloc[k] != wk.iloc[k - 1]) and k + H <= len(seq)]
    P = Fl = 0; dd = []
    for s0 in starts:
        phase, bal, goal, td, res = 1, F["start"], F["start"] + F["t1"], 0, None
        for k in range(s0, s0 + H):
            bal, why, tr = FS.ftmo_run_day(merged.get(seq[k], []), bal, risk, brk, smart, goal, F["min_days"] - td); td += int(tr)
            if why: res = "F"; break
            if bal >= goal and td >= F["min_days"]:
                if phase == 1: phase, bal, goal, td = 2, F["start"], F["start"] + F["t2"], 0
                else: res = "P"; dd.append(k - s0 + 1); break
        P += res == "P"; Fl += res == "F"
    n = len(starts)
    return dict(n=n, pass_H=round(P / n, 3) if n else None, fail=round(Fl / n, 3) if n else None, med=float(np.median(dd)) if dd else None)
base = dict(offset=0, window=120, exit="hard", open_trade=True, vwap=True, big_thr=15., big_sl=35., big_tp=115., std_sl=25., std_tp=42.,
            legs=True, leg_gap=15., leg_sl=35., leg_tp=115., leg_max=6, rev_mode="std", min_fv=125., min_bos=10., disp_pts=20.,
            wait_open=True, maxpos=1, stack_trig="bos", stack_gap=15., stack_size=1.0, ma=0)
allidx = list(range(len(G["days"]))); rows = []
g2 = FS.make_data("2016-09-15", "2023-12-29"); all2 = list(range(len(g2["days"])))
for name, P in (("LIVE", base), ("LIVE+MA100", dict(base, ma=100))):
    m = FS.merge_days({"NY": FS.session_trades("NY", P, allidx)}); m2 = FS.merge_days({"NY": FS.session_trades("NY", P, all2, g2)}, g2)
    for risk in (500, 750, 1000, 1500, 2000):
        for brk in (2000, None):
            for sm in (None, "near"):
                a = horizon(m, list(G["search"]), risk, brk, sm); s = horizon(m2, all2, risk, brk, sm, g=g2)
                rows.append(dict(strategy=name, risk=risk, breaker=brk, smart=sm or "off", search_pass60=a["pass_H"], search_fail60=a["fail"], med_days=a["med"], stress_pass60=s["pass_H"]))
b = pd.DataFrame(rows); b.to_csv("results_ftmo2_cfd/benchmark_pass60.csv", index=False)
print(b.sort_values("search_pass60", ascending=False).head(12).to_string(index=False))
# top search candidates, same metric
cand = json.load(open("results_ftmo2_cfd/stage3_candidates.json")); r = pd.read_csv("results_ftmo2_cfd/stage4_checked.csv")
out = []
for x in r.head(30).to_dict("records"):
    combo = json.loads(x["combo"]); ss = {s: int(c) for s, c in combo.items() if c is not None}
    m = FS.merge_days({s: FS.session_trades(s, cand[s][c], allidx) for s, c in ss.items()})
    brk = None if pd.isna(x["breaker"]) else x["breaker"]; sm = None if pd.isna(x["smart"]) else x["smart"]
    a = horizon(m, list(G["search"]), x["risk"], brk, sm)
    out.append(dict(combo=x["combo"], risk=x["risk"], breaker=brk, smart=sm, search_pass60=a["pass_H"], med_days=a["med"], train_usd=x["train_usd_per_day"], held_usd=x["held_usd_per_day"], final_usd=x["final_usd_per_day"]))
o = pd.DataFrame(out).drop_duplicates(["combo", "risk", "breaker", "smart"]); o.to_csv("results_ftmo2_cfd/top_candidates_pass60.csv", index=False)
print(o.sort_values("search_pass60", ascending=False).head(12).to_string(index=False))
