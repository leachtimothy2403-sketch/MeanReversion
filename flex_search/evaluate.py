"""Apply the pre-registered checks (A held-out, B beats benchmark +5pt, C 2016-23 stress, D final-holdout flag) to
stage4_report.csv. Run with the same FS_MARKET / FS_CHALLENGE env as the search. Writes stage4_checked.csv."""
import json, numpy as np, pandas as pd
import fs_search as FS, fs_space as SP
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
G = FS.load(); R = FS.RES
b = pd.read_csv(f"{R}/stage4_baselines.csv"); b["smart"] = b.smart.fillna("off")
print("BASELINES, best search-window pass rate per strategy:")
print(b.sort_values("search_pass_rate", ascending=False).groupby("baseline").head(1)[["baseline", "risk", "breaker", "smart", "train_pass_rate", "search_pass_rate", "search_med_days", "held_usd_per_day", "final_usd_per_day", "final_passed", "final_failed", "final_open"]].to_string(index=False))
bench = b[b.baseline == "LIVE+MA100"].sort_values("search_pass_rate", ascending=False).iloc[0]
g2 = FS.make_data("2016-09-15", "2023-12-29"); all2 = list(range(len(g2["days"])))
base = dict(offset=0, window=120, exit="hard", open_trade=True, vwap=True, big_thr=15., big_sl=35., big_tp=115., std_sl=25., std_tp=42.,
            legs=True, leg_gap=15., leg_sl=35., leg_tp=115., leg_max=6, rev_mode="std", min_fv=125., min_bos=10., disp_pts=20.,
            wait_open=True, maxpos=1, stack_trig="bos", stack_gap=15., stack_size=1.0, ma=100)
m2 = FS.merge_days({"NY": FS.session_trades("NY", base, all2, g2)}, g2)
bk = None if pd.isna(bench.breaker) else bench.breaker; sm = None if bench.smart == "off" else bench.smart
bs = FS.challenge_cohorts(m2, all2, int(bench.risk), bk, None, sm, g=g2)
print(f"\nBENCHMARK LIVE+MA100 risk {bench.risk} breaker {bk} smart {sm}: search pass {bench.search_pass_rate}, final {bench.final_usd_per_day}$/day, stress pass {bs['pass_rate']}")
r = pd.read_csv(f"{R}/stage4_report.csv")
r["A_held"] = (r.held_usd_per_day > 0) & (r.held_usd_per_day >= 0.4 * r.train_usd_per_day)
r["B_beats_bench"] = r.search_pass_rate >= bench.search_pass_rate + 0.05
r["C_stress"] = r.stress_pass_rate >= bs["pass_rate"] - 0.05
r["D_final_flag"] = (r.final_usd_per_day < 0) & (bench.final_usd_per_day > 0)
r["ALL"] = r.A_held & r.B_beats_bench & r.C_stress & ~r.D_final_flag
print(f"\n{len(r)} candidates. A: {r.A_held.sum()}  B: {r.B_beats_bench.sum()}  C: {r.C_stress.sum()}  not-D: {(~r.D_final_flag).sum()}  ALL: {r.ALL.sum()}")
cols = ["combo", "risk", "breaker", "pstop", "smart", "pass_rate", "search_pass_rate", "train_usd_per_day", "held_usd_per_day", "final_usd_per_day", "stress_pass_rate", "A_held", "B_beats_bench", "C_stress", "D_final_flag"]
print(r[cols].head(25).to_string(index=False))
r.to_csv(f"{R}/stage4_checked.csv", index=False)
json.dump(dict(bench=dict(bench.astype(str)), bench_stress=bs), open(f"{R}/benchmark.json", "w"), indent=1)
