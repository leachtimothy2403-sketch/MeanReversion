#!/usr/bin/env python3
"""
MeanReversion — full prop-firm review of every VPS search candidate.

Per Tim's explicit request (2026-08-27): run the complete pipeline —
Gate 2 (blind OOS holdout), Gate 3 (plateau/neighbor check), and the
2-step FTMO historical replay — against EVERY candidate the VPS's
`static_risk_full` search produced (not just the top few by raw
score), because RCTBE's own project history has repeatedly shown that
candidates which look best by raw score/period-count often fail the
prop-firm replay, and candidates that look mediocre by raw score
sometimes pass it cleanly. Screening only the top-N by score would
silently repeat that exact mistake.

Tiered for efficiency (Gate 3 and the FTMO replay are each much more
expensive than Gate 2 alone):
  1. Gate 2 on every candidate (cheap: two backtests each).
  2. Gate 3 only on Gate-2 PASSes (perturbation is ~20-30x the cost of
     one backtest).
  3. A single baseline FTMO 2-step replay (uncapped, 0.75% risk) only
     on candidates that PASS both Gate 2 and Gate 3 — this is the
     "does this even have a chance" screen. A max_concurrent x risk_pct
     sweep (the expensive, informative part) is left as a manual
     follow-up on whichever candidates look most promising here, same
     as done by hand for the two candidates checked before this script
     existed.

Usage:
    py -3 vps_full_review.py --top-json PATH1 PATH2 PATH3 --asset NDX100
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


def ftmo_baseline(row, df, risk_pct=0.0075):
    p = _candidate_params(row)
    cost = mr.COST_TABLE[p["asset"]]
    records = mr.get_trade_records(df, p, cost=cost)
    if len(records) < 30:
        return {"note": "too few trades", "n_trades": len(records)}
    by_date = defaultdict(list)
    for r in records:
        by_date[r["date"].date()].append(r["r"])
    all_days = sorted(by_date.keys())
    span_days = (all_days[-1] - all_days[0]).days
    mondays = ftmo.get_mondays_full(all_days)
    outcomes = [ftmo.simulate_2step(dict(by_date), all_days, s, risk_pct) for s in mondays]
    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    outcome_strs = [o["outcome"] for o in outcomes]
    worst_rate, worst_start = ftmo.rolling_worst_window_pass_rate(mondays, outcome_strs)
    return {
        "n_trades": len(records),
        "trades_per_day": round(len(records) / span_days, 2) if span_days else None,
        "overall_pass_pct": round(100 * n_pass / n, 1) if n else None,
        "worst_window_pass_pct": round(100 * worst_rate, 1) if worst_start is not None else None,
        "worst_window_start": str(worst_start) if worst_start is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--top-json", nargs="+", required=True)
    ap.add_argument("--risk-pct", type=float, default=0.0075)
    ap.add_argument("--checkpoint", default="vps_full_review_checkpoint.json",
                     help="Incremental progress file -- re-running with the same "
                          "--top-json set resumes from here instead of restarting "
                          "(this environment kills long-running foreground calls "
                          "at a ~10min cap, so this script must survive being cut "
                          "off and re-invoked).")
    args = ap.parse_args()

    df = mr.load_asset(args.asset)
    if df is None:
        raise SystemExit(f"No mr_precomputed_{args.asset}.parquet found")

    all_rows = []
    for path in args.top_json:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            if r.get("asset") == args.asset:
                r["_source"] = path
                all_rows.append(r)

    print(f"{len(all_rows)} total {args.asset} candidates across {len(args.top_json)} file(s)\n")

    results = []
    start_i = 0
    try:
        with open(args.checkpoint, encoding="utf-8") as f:
            results = json.load(f)
        start_i = len(results)
        print(f"Resuming from checkpoint: {start_i}/{len(all_rows)} already done\n")
    except FileNotFoundError:
        pass

    n_g2_pass = sum(1 for r in results if r.get("gate2") == "PASS")
    n_g3_checked = sum(1 for r in results if "gate3" in r)
    n_g3_pass = sum(1 for r in results if r.get("gate3") in ("PASS", "MIXED"))
    n_ftmo_checked = sum(1 for r in results if "ftmo_worst_window" in r)

    for i, row in enumerate(all_rows):
        if i < start_i:
            continue
        g2r = g2.gate2_check(row, df)
        rec = {
            "idx": i, "source": row["_source"], "direction": row.get("direction"),
            "entry_mode": row.get("entry_mode"), "static_risk_pct": row.get("static_risk_pct"),
            "rr": row.get("rr"), "score": row.get("score"), "avg_pf": row.get("avg_profit_factor"),
            "min_pf": row.get("min_profit_factor"), "n_trades": row.get("n_trades_total"),
            "periods_passed": row.get("periods_passed"),
            "gate2": g2r["verdict"], "gate2_degradation": g2r.get("degradation"),
            "gate2_n_oos": g2r.get("n_trades_oos"),
        }
        if g2r["verdict"] == "PASS":
            n_g2_pass += 1
            g3r = pc.plateau_check(row, df)
            n_g3_checked += 1
            rec["gate3"] = g3r["verdict"]
            rec["gate3_frac"] = g3r["frac_healthy"]
            if g3r["verdict"] in ("PASS", "MIXED"):
                n_g3_pass += 1
                ftmo_r = ftmo_baseline(row, df, args.risk_pct)
                n_ftmo_checked += 1
                rec["ftmo_trades_per_day"] = ftmo_r.get("trades_per_day")
                rec["ftmo_overall_pass"] = ftmo_r.get("overall_pass_pct")
                rec["ftmo_worst_window"] = ftmo_r.get("worst_window_pass_pct")
                rec["ftmo_worst_start"] = ftmo_r.get("worst_window_start")
                rec["full_params"] = row
        results.append(rec)
        print(f"  [{i+1}/{len(all_rows)}] {rec['source'].split('_')[-4] if '_' in rec['source'] else rec['source']} "
              f"{rec['direction']:<5} risk={rec['static_risk_pct']} rr={rec['rr']} score={rec['score']:.3f} "
              f"n={rec['n_trades']:<6} -> Gate2={rec['gate2']:<4}" +
              (f" Gate3={rec.get('gate3','-'):<5}" if 'gate3' in rec else "") +
              (f" FTMO_worst_window={rec.get('ftmo_worst_window')}%" if 'ftmo_worst_window' in rec else ""))
        with open(args.checkpoint, "w", encoding="utf-8") as f:
            json.dump(results, f, default=str)

    print(f"\n=== SUMMARY ===")
    print(f"{len(all_rows)} candidates checked")
    print(f"{n_g2_pass} passed Gate 2 (blind OOS)")
    print(f"{n_g3_pass}/{n_g3_checked} of those also passed/mixed Gate 3 (plateau)")
    print(f"{n_ftmo_checked} got the FTMO baseline replay")

    survivors = [r for r in results if "ftmo_worst_window" in r]
    survivors.sort(key=lambda r: (r["ftmo_worst_window"] if r["ftmo_worst_window"] is not None else -1), reverse=True)
    print(f"\n=== RANKED BY FTMO WORST-WINDOW PASS RATE (baseline, uncapped, {args.risk_pct*100:.2f}% risk) ===")
    for r in survivors:
        print(f"  worst_window={r['ftmo_worst_window']}%  overall={r['ftmo_overall_pass']}%  "
              f"trades/day={r['ftmo_trades_per_day']}  n={r['n_trades']}  "
              f"{r['direction']} risk={r['static_risk_pct']} rr={r['rr']}  score={r['score']:.3f}  "
              f"[{r['source']}]")

    with open("vps_full_review_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results (including all params for survivors) written to vps_full_review_results.json")


if __name__ == "__main__":
    main()
