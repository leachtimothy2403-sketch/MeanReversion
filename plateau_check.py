#!/usr/bin/env python3
"""
MeanReversion — parameter-plateau check (Gate 3).

For each candidate, perturbs one swept NUMERIC/ordered parameter at a
time by +/-1 grid step (holding everything else fixed, including asset),
re-runs backtest_multiperiod on that asset's full precomputed dataset,
and checks whether the perturbed neighbor is still "healthy" (produces a
result at all AND avg_profit_factor > 1.0). A candidate whose edge only
exists at one exact, isolated point in parameter space is much more
likely fitted to noise than one whose neighborhood is broadly healthy.

Categorical / non-ordered params (asset, direction, use_session_close,
skip_weekday) are held fixed, not perturbed.

>=70% of valid neighbor checks healthy -> PASS
>=40% -> MIXED (soft-fail, report but don't reject outright)
else   -> FAIL

Usage:
    py -3 plateau_check.py --asset NDX100 --rank 0
    py -3 plateau_check.py --asset SPX500 --rank 0 1 2
"""
import argparse
import json

import mean_reversion as mr

# "direction" excluded (categorical — flipping it tests a different
# economic hypothesis, not neighborhood robustness of this one; the
# search sweeping it already answers "does the other side also work").
PERTURBABLE = [
    "atr_window", "consolidation_bars", "consolidation_atr_mult",
    "deviation_threshold_atr", "bos_lookback_bars", "bos_confirm_bars",
    "target_fraction", "exit_horizon_bars", "k_buf", "k_cap", "k_floor",
    "session_start_h", "session_window_h",
]


def _candidate_params(row: dict) -> dict:
    keys = list(mr.SPACE.keys()) + ["asset"]
    return {k: row.get(k, mr.SPACE[k][0] if k in mr.SPACE else None) for k in keys}


def plateau_check(row: dict, df) -> dict:
    SPACE = mr.SPACE
    base = _candidate_params(row)
    neighbors = []
    for param in PERTURBABLE:
        grid = SPACE[param]
        try:
            idx = grid.index(base[param])
        except ValueError:
            continue
        for step in (-1, 1):
            j = idx + step
            if j < 0 or j >= len(grid):
                continue
            neighbor = dict(base)
            neighbor[param] = grid[j]
            if neighbor.get("k_floor", 0) >= neighbor.get("k_cap", 1):
                continue
            neighbors.append((param, grid[idx], grid[j], neighbor))

    results = []
    for param, old_val, new_val, neighbor in neighbors:
        r = mr.backtest_multiperiod(df, neighbor)
        healthy = r is not None and r["avg_profit_factor"] > 1.0
        results.append({
            "param": param, "from": old_val, "to": new_val,
            "healthy": healthy,
            "avg_pf": r["avg_profit_factor"] if r else None,
            "periods_passed": f"{r['periods_passed']}/{r['periods_evaluated']}" if r else None,
        })

    n = len(results)
    n_healthy = sum(1 for r in results if r["healthy"])
    frac = n_healthy / n if n else 0.0
    verdict = "PASS" if frac >= 0.70 else ("MIXED" if frac >= 0.40 else "FAIL")
    return {"n_neighbors": n, "n_healthy": n_healthy, "frac_healthy": round(frac, 3),
            "verdict": verdict, "detail": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--rank", type=int, nargs="+", default=[0])
    ap.add_argument("--top-json", default="meanreversion_output/top_strategies.json")
    args = ap.parse_args()

    with open(args.top_json, encoding="utf-8") as f:
        all_rows = json.load(f)
    asset_rows = [r for r in all_rows if r["asset"] == args.asset]
    print(f"{len(asset_rows)} {args.asset} candidates in {args.top_json}")

    df = mr.load_asset(args.asset)
    if df is None:
        raise SystemExit(f"No mr_precomputed_{args.asset}.parquet found — run precompute.py first")

    for rank in args.rank:
        if rank >= len(asset_rows):
            print(f"rank {rank}: out of range")
            continue
        row = asset_rows[rank]
        print(f"\n=== {args.asset} rank {rank}: score={row['score']} avg_pf={row['avg_profit_factor']} "
              f"min_pf={row['min_profit_factor']} n_trades={row['n_trades_total']} "
              f"pass={row['periods_passed']}/{row['periods_evaluated']} wfe={row.get('wfe')} ===")
        print({k: row.get(k) for k in mr.SPACE.keys()})

        result = plateau_check(row, df)
        print(f"\nPlateau check: {result['n_healthy']}/{result['n_neighbors']} neighbors healthy "
              f"({result['frac_healthy']*100:.0f}%) -> {result['verdict']}")
        for d in result["detail"]:
            status = "OK  " if d["healthy"] else "FAIL"
            print(f"  [{status}] {d['param']:24s} {d['from']} -> {str(d['to']):<6} "
                  f"avg_pf={d['avg_pf']} periods={d['periods_passed']}")


if __name__ == "__main__":
    main()
