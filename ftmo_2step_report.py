#!/usr/bin/env python3
"""
MeanReversion — 2-step FTMO challenge historical-replay report.

Same real-historical-replay methodology as `candidate_report.py`'s
`historical_replay_check` (every calendar Monday spanning the
candidate's trade history, real trade sequence, both an overall pass
rate and RCTBE's real rolling-worst-24-month-window pass rate — see
that file's docstring for why the worst-window number, not the
average, is the real risk-selection metric), but using
`ftmo_challenge_rules.simulate_2step` (Challenge + Verification phases,
10%/5% targets, $5k daily loss, $90k static max-loss floor, 4-trading-
day minimum per phase, Phase 2 starts the next trading day after
Phase 1 completes, in the same historical sequence — see that module's
docstring for full mechanics) instead of `simulate_1step`.

Added 2026-08-27 per Tim's request for the 2-step analysis specifically
(the original `candidate_report.py` only wired up 1-step). Not merged
into `candidate_report.py` itself to avoid changing that file's existing
output/behavior — this is an additive sibling script, same pattern as
`gate2_holdout.py`/`plateau_check.py` being separate single-purpose
scripts.

Usage:
    py -3 ftmo_2step_report.py --asset NDX100 --rank 3
    py -3 ftmo_2step_report.py --asset NDX100 --rank 3 --risk-pct 0.005
"""
import argparse
import json
from collections import defaultdict

import mean_reversion as mr
import ftmo_challenge_rules as ftmo


def _candidate_params(row: dict) -> dict:
    keys = list(mr.SPACE.keys()) + ["asset"]
    return {k: row.get(k, mr.SPACE[k][0] if k in mr.SPACE else None) for k in keys}


def historical_replay_2step(row: dict, df, risk_pct: float, max_concurrent: int = None) -> dict:
    p = _candidate_params(row)
    cost = mr.COST_TABLE[p["asset"]]
    records = mr.get_trade_records(df, p, cost=cost)
    if len(records) < 30:
        return {"n_trades": len(records), "note": "too few trades for a meaningful replay"}

    by_date = defaultdict(list)
    for r in records:
        by_date[r["date"].date()].append(r["r"])
    all_days = sorted(by_date.keys())

    span_days = (all_days[-1] - all_days[0]).days
    trades_per_day_active = len(records) / len(all_days)
    trades_per_day_calendar = len(records) / span_days if span_days else None

    mondays = ftmo.get_mondays_full(all_days)
    outcomes = []
    for start in mondays:
        res = ftmo.simulate_2step(dict(by_date), all_days, start, risk_pct, max_concurrent=max_concurrent)
        outcomes.append(res)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "PASS"]
    fail_reasons = defaultdict(int)
    fail_phase = defaultdict(int)
    for o in outcomes:
        if o["outcome"] == "FAIL":
            fail_reasons[o["reason"]] += 1
            fail_phase[o.get("phase")] += 1

    outcome_strs = [o["outcome"] for o in outcomes]
    worst_rate, worst_start = ftmo.rolling_worst_window_pass_rate(mondays, outcome_strs)

    return {
        "n_trades": len(records),
        "first_trade": str(all_days[0]), "last_trade": str(all_days[-1]),
        "trades_per_day_active_day": round(trades_per_day_active, 2),
        "trades_per_day_calendar": round(trades_per_day_calendar, 2) if trades_per_day_calendar else None,
        "n_cohorts": n,
        "overall_pass_pct": round(100 * n_pass / n, 1) if n else None,
        "overall_fail_pct": round(100 * n_fail / n, 1) if n else None,
        "overall_still_going_pct": round(100 * (n - n_pass - n_fail) / n, 1) if n else None,
        "median_days_to_pass_both_phases": (sorted(pass_days)[len(pass_days) // 2] if pass_days else None),
        "fail_reasons": dict(fail_reasons),
        "fail_by_phase": dict(fail_phase),
        "worst_window_pass_pct": round(100 * worst_rate, 1) if worst_start is not None else None,
        "worst_window_start": str(worst_start) if worst_start is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--rank", type=int, nargs="+", default=[0])
    ap.add_argument("--top-json", default="meanreversion_output/top_strategies.json")
    ap.add_argument("--risk-pct", type=float, default=0.0075)
    ap.add_argument("--max-concurrent", type=int, default=None,
                     help="Cap NEW entries taken per calendar day (both phases). "
                          "Default None = uncapped (candidate's natural entry rate).")
    args = ap.parse_args()

    with open(args.top_json, encoding="utf-8") as f:
        all_rows = json.load(f)
    asset_rows = [r for r in all_rows if r["asset"] == args.asset]

    df = mr.load_asset(args.asset)
    if df is None:
        raise SystemExit(f"No mr_precomputed_{args.asset}.parquet found — run precompute.py first")

    for rank in args.rank:
        if rank >= len(asset_rows):
            print(f"rank {rank}: out of range")
            continue
        row = asset_rows[rank]
        print(f"{'=' * 70}\n{args.asset} rank {rank}: score={row['score']} avg_pf={row['avg_profit_factor']} "
              f"n_trades={row['n_trades_total']} periods_passed={row['periods_passed']}/{row['periods_evaluated']}")
        print({k: row.get(k) for k in mr.SPACE.keys()})

        replay = historical_replay_2step(row, df, args.risk_pct, max_concurrent=args.max_concurrent)
        if "note" in replay:
            print(f"\n[2-step replay @ {args.risk_pct*100:.2f}% risk]  {replay['note']} (n_trades={replay['n_trades']})")
            continue

        print(f"\nTrade history: {replay['first_trade']} -> {replay['last_trade']}, "
              f"{replay['n_trades']} trades, {replay['trades_per_day_calendar']} trades/calendar-day "
              f"({replay['trades_per_day_active_day']} trades/active-trading-day)")
        mc_tag = f", max_concurrent={args.max_concurrent}" if args.max_concurrent is not None else ""
        print(f"\n[2-step replay @ {args.risk_pct*100:.2f}% risk/trade{mc_tag}]  "
              f"OVERALL pass={replay['overall_pass_pct']}% fail={replay['overall_fail_pct']}% "
              f"still_running={replay['overall_still_going_pct']}% "
              f"median_days_to_pass(both phases)={replay['median_days_to_pass_both_phases']} "
              f"({replay['n_cohorts']} weekly cohorts)")
        print(f"    fail_reasons={replay['fail_reasons']}  fail_by_phase={replay['fail_by_phase']}")
        wwp = replay['worst_window_pass_pct']
        print(f"    WORST 24-MONTH WINDOW pass={wwp}%" +
              (f" (starting {replay['worst_window_start']})" if wwp is not None else
               "  — not enough history yet for a 24-month window with >=8 resolved cohorts") +
              "  <-- real risk-selection metric, not the overall average above")
        print()


if __name__ == "__main__":
    main()
