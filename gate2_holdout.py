#!/usr/bin/env python3
"""
MeanReversion — Gate 2: clean blind OOS holdout.

Anchored walk-forward split:
  - IS  = periods 1-3 combined (first 60% of history, contiguous)
  - OOS = periods 4-5 combined (last 40% of history, contiguous)

A candidate PASSES Gate 2 iff, on the combined OOS block:
    PF_OOS            >= 1.05
    n_trades_OOS       >= 30
    degradation_ratio  = (PF_OOS - 1) / (PF_IS - 1)  >= 0.40

Genuinely blind ONLY if the candidate came from a search run with
MR_IS_ONLY=1 (mean_reversion.py's default) — otherwise periods 4-5 were
already visible to the search's own accept gate, and this is a re-score
of bars the search already used to select candidates, not a clean blind
test. Same caveat, same fix, as RCTBE's own `_layer2_gate2_holdout.py`
this is ported from.

Usage:
    py -3 gate2_holdout.py --asset NDX100
    py -3 gate2_holdout.py --asset SPX500 --rank 0 1 2 3 4
"""
import argparse
import json

import numpy as np

import mean_reversion as mr

PF_OOS_MIN = 1.05
N_TRADES_OOS_MIN = 30
DEGRADATION_MIN = 0.40


def _candidate_params(row: dict) -> dict:
    keys = list(mr.SPACE.keys()) + ["asset"]
    return {k: row.get(k, mr.SPACE[k][0] if k in mr.SPACE else None) for k in keys}


def gate2_check(row: dict, df) -> dict:
    p = _candidate_params(row)
    n = len(df)
    bounds = np.linspace(0, n, mr.N_PERIODS_SEARCH + 1).astype(int)
    is_df = df.iloc[bounds[0]:bounds[3]]    # periods 1-3, contiguous, combined
    oos_df = df.iloc[bounds[3]:bounds[5]]   # periods 4-5, contiguous, combined

    cost = mr.COST_TABLE[p["asset"]]

    is_sigs = mr.generate_signals(is_df, p)
    oos_sigs = mr.generate_signals(oos_df, p)
    is_bt = mr.backtest_signals(is_sigs, is_df, cost=cost, p=p)
    oos_bt = mr.backtest_signals(oos_sigs, oos_df, cost=cost, p=p)

    if is_bt is None or oos_bt is None:
        return {"verdict": "INSUFFICIENT_DATA", "is_bt": is_bt, "oos_bt": oos_bt, "degradation": None}

    pf_is, pf_oos = is_bt["profit_factor"], oos_bt["profit_factor"]
    n_oos = oos_bt["n_trades"]
    degradation = (pf_oos - 1) / (pf_is - 1) if pf_is > 1 else None

    passed = (
        pf_oos >= PF_OOS_MIN
        and n_oos >= N_TRADES_OOS_MIN
        and degradation is not None
        and degradation >= DEGRADATION_MIN
    )
    return {
        "verdict": "PASS" if passed else "FAIL",
        "pf_is": pf_is, "pf_oos": pf_oos,
        "n_trades_is": is_bt["n_trades"], "n_trades_oos": n_oos,
        "degradation": round(degradation, 3) if degradation is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--rank", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--top-json", default="meanreversion_output/top_strategies.json")
    args = ap.parse_args()

    with open(args.top_json, encoding="utf-8") as f:
        all_rows = json.load(f)
    asset_rows = [r for r in all_rows if r["asset"] == args.asset]
    print(f"{len(asset_rows)} {args.asset} candidates in {args.top_json}\n")

    df = mr.load_asset(args.asset)
    if df is None:
        raise SystemExit(f"No mr_precomputed_{args.asset}.parquet found — run precompute.py first")

    n_pass = 0
    for rank in args.rank:
        if rank >= len(asset_rows):
            continue
        row = asset_rows[rank]
        r = gate2_check(row, df)
        n_pass += (r["verdict"] == "PASS")
        print(f"[rank {rank:2d}] score={row['score']:7.3f} orig_avg_pf={row['avg_profit_factor']:.3f} "
              f"-> {r['verdict']:8s} PF_IS={r.get('pf_is')} PF_OOS={r.get('pf_oos')} "
              f"n_OOS={r.get('n_trades_oos')} degradation={r.get('degradation')}")

    print(f"\n{n_pass}/{len(args.rank)} checked candidates PASS the clean blind holdout.")


if __name__ == "__main__":
    main()
