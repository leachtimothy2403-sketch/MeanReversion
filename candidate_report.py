#!/usr/bin/env python3
"""
MeanReversion — one-shot candidate report for a search survivor:

  1. Gate 2 — blind OOS holdout        (gate2_holdout.gate2_check)
  2. Gate 3 — parameter-plateau check  (plateau_check.plateau_check)
  3. A REAL-historical-replay prop-firm check: replay the actual
     historical trade sequence starting on every calendar Monday spanning
     the candidate's trade history, using the real 1-step challenge rules
     (ftmo_challenge_rules.simulate_1step — 10% target, 3% daily loss,
     trailing 10% max-loss floor). Reports BOTH:
       - an OVERALL pass rate across every weekly cohort (average-case)
       - a ROLLING WORST 24-MONTH-WINDOW pass rate
         (ftmo_challenge_rules.rolling_worst_window_pass_rate, ported
         2026-08-24 from RCTBE's `_ftmo_2step_simulator.py`), i.e. RCTBE's
         own real risk-selection metric — see RCTBE_SYSTEM_BRIEFING.md
         Section 9's "month-by-month breakdown swung wildly... no
         consistent safe stretch" finding for exactly why an average
         alone is the wrong number to trust: it can look strong while
         hiding a cluster of failures concentrated in one bad stretch
         (a COVID-2020-style regime), which is the specific failure mode
         the worst-window lens exists to catch. **Do not read "overall
         pass rate" as if it were the worst-window number** — they answer
         different questions, and only the worst-window one is comparable
         to how RCTBE itself picks a real risk level.
     This is still NOT a day-block-bootstrap Monte Carlo (RCTBE's
     `_prop_firm_analysis.py`, N_SIMS=8000, the standard RCTBE's own real
     go/no-go portfolio decisions have used) — it's cheaper, uses only
     the one realized trade sequence this candidate actually produced
     (not a resampled distribution), and cohorts overlap their neighbors
     (same non-independence caveat RCTBE's own 157-cohort version
     carries). Treat it as a first-pass sanity read, not a substitute for
     a real pooled-portfolio Monte Carlo before sizing any real risk
     against a candidate from this search.

Usage:
    py -3 candidate_report.py --asset NDX100 --rank 0
    py -3 candidate_report.py --asset SPX500 --rank 0 1 2 --risk-pct 0.0075
"""
import argparse
import json
from collections import defaultdict

import mean_reversion as mr
import gate2_holdout as g2
import plateau_check as pc
import ftmo_challenge_rules as ftmo


def _candidate_params(row: dict) -> dict:
    keys = list(mr.SPACE.keys()) + ["asset"]
    return {k: row.get(k, mr.SPACE[k][0] if k in mr.SPACE else None) for k in keys}


def historical_replay_check(row: dict, df, risk_pct: float) -> dict:
    """Every-Monday real-historical replay — see module docstring."""
    p = _candidate_params(row)
    cost = mr.COST_TABLE[p["asset"]]
    records = mr.get_trade_records(df, p, cost=cost)
    if len(records) < 30:
        return {"n_trades": len(records), "note": "too few trades for a meaningful replay"}

    by_date = defaultdict(list)
    for r in records:
        by_date[r["date"].date()].append(r["r"])
    all_days = sorted(by_date.keys())

    # Every calendar Monday spanning the trade history, not just Mondays
    # that happen to have a trade — see ftmo_challenge_rules.
    # get_mondays_full's own docstring for why that distinction matters
    # (a Monday with zero trades is still a valid, real cohort start
    # point; skipping it would silently under-sample real cohorts).
    mondays = ftmo.get_mondays_full(all_days)
    outcomes = []
    for start in mondays:
        res = ftmo.simulate_1step(dict(by_date), all_days, start, risk_pct)
        outcomes.append(res)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "PASS"]
    fail_reasons = defaultdict(int)
    for o in outcomes:
        if o["outcome"] == "FAIL":
            fail_reasons[o["reason"]] += 1

    outcome_strs = [o["outcome"] for o in outcomes]
    worst_rate, worst_start = ftmo.rolling_worst_window_pass_rate(mondays, outcome_strs)

    return {
        "n_trades": len(records),
        "n_cohorts": n,
        "overall_pass_pct": round(100 * n_pass / n, 1) if n else None,
        "overall_fail_pct": round(100 * n_fail / n, 1) if n else None,
        "overall_still_going_pct": round(100 * (n - n_pass - n_fail) / n, 1) if n else None,
        "median_days_to_pass": (sorted(pass_days)[len(pass_days) // 2] if pass_days else None),
        "fail_reasons": dict(fail_reasons),
        # RCTBE's real risk-selection metric — see module/function
        # docstrings. worst_window_pass_pct is None if no 24-month
        # window in this candidate's history had >=8 resolved cohorts
        # (too little history for a meaningful worst-window read yet).
        "worst_window_pass_pct": round(100 * worst_rate, 1) if worst_start is not None else None,
        "worst_window_start": str(worst_start) if worst_start is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--rank", type=int, nargs="+", default=[0])
    ap.add_argument("--top-json", default="meanreversion_output/top_strategies.json")
    ap.add_argument("--risk-pct", type=float, default=0.0075,
                     help="Risk per trade as a fraction of equity, e.g. 0.0075 = 0.75%%. "
                          "RCTBE's own history found this needs real calibration per "
                          "portfolio, not a default trusted blindly.")
    args = ap.parse_args()

    with open(args.top_json, encoding="utf-8") as f:
        all_rows = json.load(f)
    asset_rows = [r for r in all_rows if r["asset"] == args.asset]
    print(f"{len(asset_rows)} {args.asset} candidates in {args.top_json}\n")

    df = mr.load_asset(args.asset)
    if df is None:
        raise SystemExit(f"No mr_precomputed_{args.asset}.parquet found — run precompute.py {args.asset} first")

    for rank in args.rank:
        if rank >= len(asset_rows):
            print(f"rank {rank}: out of range")
            continue
        row = asset_rows[rank]
        print(f"{'=' * 70}\n{args.asset} rank {rank}: score={row['score']} "
              f"avg_pf={row['avg_profit_factor']} n_trades={row['n_trades_total']} "
              f"periods_passed={row['periods_passed']}/{row['periods_evaluated']}")
        print({k: row.get(k) for k in mr.SPACE.keys()})

        gate2 = g2.gate2_check(row, df)
        print(f"\n[Gate 2 — blind OOS holdout]  {gate2['verdict']}  "
              f"PF_IS={gate2.get('pf_is')} PF_OOS={gate2.get('pf_oos')} "
              f"n_OOS={gate2.get('n_trades_oos')} degradation={gate2.get('degradation')}")

        plateau = pc.plateau_check(row, df)
        print(f"[Gate 3 — plateau check]      {plateau['verdict']}  "
              f"{plateau['n_healthy']}/{plateau['n_neighbors']} neighbors healthy "
              f"({plateau['frac_healthy'] * 100:.0f}%)")

        replay = historical_replay_check(row, df, args.risk_pct)
        if "note" in replay:
            print(f"[Historical replay @ {args.risk_pct*100:.2f}% risk]  {replay['note']} "
                  f"(n_trades={replay['n_trades']})")
        else:
            print(f"[Historical replay @ {args.risk_pct*100:.2f}% risk]  "
                  f"OVERALL pass={replay['overall_pass_pct']}% fail={replay['overall_fail_pct']}% "
                  f"still_running={replay['overall_still_going_pct']}% "
                  f"median_days_to_pass={replay['median_days_to_pass']} "
                  f"({replay['n_cohorts']} weekly cohorts, {replay['n_trades']} trades) "
                  f"fail_reasons={replay['fail_reasons']}")
            wwp = replay['worst_window_pass_pct']
            print(f"    WORST 24-MONTH WINDOW pass={wwp}%" +
                  (f" (starting {replay['worst_window_start']})" if wwp is not None else
                   "  — not enough history yet for a 24-month window with >=8 resolved cohorts") +
                  "  <-- this, not the overall average above, is RCTBE's real risk-selection metric")
        print()


if __name__ == "__main__":
    main()
