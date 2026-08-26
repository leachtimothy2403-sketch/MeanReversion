#!/usr/bin/env python3
"""
video_strategy_backtest.py — NDX100 backtest of the prop-firm "fair
value / break of structure" strategy described in Tim's transcript,
built as a NEW variant alongside (not a replacement for) the existing
entry_mode="bos" in mean_reversion.py.

WHAT'S REUSED FROM THE EXISTING, ALREADY-VALIDATED ENGINE (unchanged):
  - Fair value: day-open anchor, updated at each confirmed consolidation
    (_fair_value_series / _consolidation_range, imported verbatim from
    mean_reversion.py). NOTE: this consolidation-reset mechanic is NOT
    described in the transcript Tim provided — it's inherited from the
    project's existing design (which itself "translates the user's
    brief" from an earlier conversation). Kept as-is because it's
    already built, tested, and presumably reflects the fuller version
    of "fair value" Tim has described previously. Flagged explicitly so
    this is a visible assumption, not a silent one.
  - Entry: break of structure (bos_mode "fractal" or "raw_wick"),
    confirmed for bos_confirm_bars, requiring price to have been
    extended >= deviation_threshold_atr from fair value within the
    trailing extension_lookback_bars window. Direct match to the
    transcript's "wait for a break of structure" / "trading toward fair
    price" description.
  - R-multiple trade replay (_replay_trade / backtest_signals), real
    NDX100 spread/commission cost (cost_table.COST_TABLE).

WHAT'S NEW / DIFFERENT — the transcript's explicit departure from the
existing entry_mode="bos": stop-loss and take-profit are NOT structural
(anchored to a swing/breakout extent, ATR-clamped) here. They are
STATIC — one fixed % of entry price, identical on every single trade
for a given parameter draw, with take-profit set to a fixed
risk:reward multiple of that same distance. This is a literal reading
of: "position sizing should not be a feeling... every single one
should be static... you tie it to the evaluation math." The transcript
gives a $2,000-max-loss / $3,000-target eval as its own example of
where a ~1:1.5 R:R comes from, so 1.5 is used as the default RR.

Two phases, per Tim's request ("reproduce the same strategy, make sure
it is profitable, then optimize its parameters"):
  1. BASELINE — one literal parameter draw representing the transcript's
     description as directly as possible, run on 2022 (bear) and 2024
     (bull) NDX100 1-min data separately (same cross-year convention
     this project has used throughout its history).
  2. SWEEP — a random search over the same entry-side knobs the
     existing bos entry_mode already sweeps, PLUS static_risk_pct and
     rr, evaluated against the same accept-gate spirit (profitable in
     BOTH years, minimum trade count per year) used throughout this
     project.

Data: real Dukascopy NDX100 1-min bars, staged from Tim's own cache via
the device bridge, run through the project's own precompute.py
(harmonization + dead-bar drop unchanged) — not synthetic data.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from cost_table import COST_TABLE
from mean_reversion import (
    _consolidation_range,
    _fair_value_series,
    backtest_signals,
    SESSION_CLOSE_UTC_HOUR,
)

ASSET = "NDX100"
COST = COST_TABLE[ASSET]
MIN_TRADES_PER_YEAR = 100  # same statistical-validity floor mean_reversion.py uses per period

# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — same fair-value/BOS entry engine as entry_mode=
#  "bos" in mean_reversion.py, but with STATIC risk/reward construction
#  instead of the structural/ATR-clamped stop.
# ══════════════════════════════════════════════════════════════════════════

def generate_signals_static(df: pd.DataFrame, p: dict) -> List[dict]:
    atr = df[f"atr_{p['atr_window']}"]
    roll_high, roll_low, is_consolidating = _consolidation_range(df, atr, p)
    fair_value = _fair_value_series(df, atr, p, roll_high, roll_low, is_consolidating)

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]

    deviation_atr = (close_s - fair_value) / atr
    threshold = p["deviation_threshold_atr"]
    extended_below = deviation_atr <= -threshold
    extended_above = deviation_atr >= threshold
    ext_lookback = p.get("extension_lookback_bars", 1)
    extended_below_recent = extended_below.rolling(ext_lookback, min_periods=1).max().astype(bool)
    extended_above_recent = extended_above.rolling(ext_lookback, min_periods=1).max().astype(bool)

    bos_lookback = p["bos_lookback_bars"]
    confirm_bars = p["bos_confirm_bars"]
    bos_mode = p.get("bos_mode", "raw_wick")

    if bos_mode == "raw_wick":
        recent_swing_high = high_s.shift(1).rolling(bos_lookback, min_periods=1).max()
        recent_swing_low = low_s.shift(1).rolling(bos_lookback, min_periods=1).min()
    else:
        recent_swing_high = df["swing_high_confirmed"].rolling(bos_lookback, min_periods=1).max()
        recent_swing_low = df["swing_low_confirmed"].rolling(bos_lookback, min_periods=1).min()

    beyond_up = close_s > recent_swing_high
    beyond_down = close_s < recent_swing_low
    beyond_up_confirmed = beyond_up.rolling(confirm_bars).sum() >= confirm_bars
    beyond_down_confirmed = beyond_down.rolling(confirm_bars).sum() >= confirm_bars
    bos_up = (beyond_up_confirmed & ~beyond_up_confirmed.shift(1, fill_value=False)).to_numpy()
    bos_down = (beyond_down_confirmed & ~beyond_down_confirmed.shift(1, fill_value=False)).to_numpy()

    buy_cand = bos_up & extended_below_recent.to_numpy()
    sell_cand = bos_down & extended_above_recent.to_numpy()

    direction = p["direction"]
    if direction == "Buy":
        sell_cand = np.zeros_like(sell_cand)
    elif direction == "Sell":
        buy_cand = np.zeros_like(buy_cand)

    candidate = buy_cand | sell_cand
    idx = df.index
    skip_wd = p.get("skip_weekday", -1)
    session_start_h = p.get("session_start_h", 0)
    session_window_h = p.get("session_window_h", 24)
    if skip_wd is not None and skip_wd >= 0:
        candidate &= (idx.weekday.to_numpy() != skip_wd)
    if session_window_h < 24:
        hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
        start = session_start_h % 24
        end = (start + session_window_h) % 24
        in_window = (hour >= start) & (hour < end) if start < end else (hour >= start) | (hour < end)
        candidate &= in_window

    cand_idx = np.where(candidate)[0]
    if len(cand_idx) == 0:
        return []

    is_buy_arr = buy_cand[cand_idx]
    entry_arr = close_s.to_numpy()[cand_idx]

    # --- STATIC risk construction (the transcript's explicit departure
    # from structural/ATR-based sizing): one fixed % of entry price,
    # identical on every trade. TP is a fixed R:R multiple of that same
    # distance. No swing/ATR anchoring, no floor/cap clamp at all.
    risk_pct = p["static_risk_pct"]
    rr = p["rr"]
    dist_arr = entry_arr * risk_pct
    sl_arr = np.where(is_buy_arr, entry_arr - dist_arr, entry_arr + dist_arr)
    tp_arr = np.where(is_buy_arr, entry_arr + rr * dist_arr, entry_arr - rr * dist_arr)

    horizon_ts_arr = idx[cand_idx] + pd.Timedelta(minutes=1) * p["exit_horizon_bars"]
    exit_idx_arr = idx.searchsorted(horizon_ts_arr, side="left")
    exit_idx_arr = np.minimum(exit_idx_arr, len(idx))

    if p.get("use_session_close"):
        entry_ts_s = pd.Series(idx[cand_idx])
        same_day_close = entry_ts_s.dt.normalize() + pd.Timedelta(hours=SESSION_CLOSE_UTC_HOUR)
        session_close_s = same_day_close.where(entry_ts_s < same_day_close, same_day_close + pd.Timedelta(days=1))
        session_close_idx = idx.searchsorted(session_close_s.to_numpy(), side="left")
        exit_idx_arr = np.minimum(exit_idx_arr, session_close_idx)

    return [
        {
            "direction": "Buy" if is_buy_arr[i] else "Sell",
            "entry": float(entry_arr[i]),
            "sl": float(sl_arr[i]),
            "tp": float(tp_arr[i]),
            "risk": float(dist_arr[i]),
            "bar_idx": int(cand_idx[i]),
            "exit_idx": int(exit_idx_arr[i]),
        }
        for i in range(len(cand_idx))
    ]


def eval_year(df: pd.DataFrame, p: dict) -> Optional[dict]:
    sigs = generate_signals_static(df, p)
    return backtest_signals(sigs, df, cost=COST, p=p)


# ══════════════════════════════════════════════════════════════════════════
#  SWEEP SPACE — same entry-side knobs mean_reversion.SPACE already
#  sweeps for entry_mode="bos", plus the new static_risk_pct / rr.
# ══════════════════════════════════════════════════════════════════════════

STATIC_SPACE = {
    "atr_window":              [14, 30, 60, 120],
    "consolidation_bars":      [5, 8, 13, 20, 30, 45],
    "consolidation_atr_mult":  [0.5, 0.75, 1.0, 1.25, 1.5],
    "deviation_threshold_atr": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "bos_mode":                ["fractal", "raw_wick"],
    "bos_lookback_bars":       [1, 2, 3, 5, 8, 13, 20, 30, 45, 60],
    "bos_confirm_bars":        [1, 2, 3, 5],
    "extension_lookback_bars": [1, 3, 5, 10, 20],
    "exit_horizon_bars":       [15, 30, 60, 90, 120, 180, 240, 360],
    "use_session_close":       [False, True],
    "direction":               ["Buy", "Sell", "Both"],
    "skip_weekday":            [-1, 0, 1, 2, 3, 4],
    "session_start_h":         [0, 4, 8, 12, 16, 20],
    "session_window_h":        [24, 16, 8, 4],
    # New vs. the existing bos entry_mode — the transcript's static
    # risk sizing tied to prop-firm eval math (~1:1.5 was its own
    # explicit example, from $2,000 max loss / $3,000 target).
    "static_risk_pct":         [0.0010, 0.0015, 0.0020, 0.0025, 0.0030,
                                  0.0040, 0.0050, 0.0075, 0.0100],
    "rr":                      [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
}


def sample_params() -> dict:
    return {k: random.choice(v) for k, v in STATIC_SPACE.items()}


BASELINE_PARAMS = {
    "atr_window": 30,
    "consolidation_bars": 13,
    "consolidation_atr_mult": 1.0,
    "deviation_threshold_atr": 2.0,
    "bos_mode": "raw_wick",       # literal "close beyond a previous candle's wick" per the transcript's plain description
    "bos_lookback_bars": 3,
    "bos_confirm_bars": 1,
    "extension_lookback_bars": 5,
    "exit_horizon_bars": 120,
    "use_session_close": False,
    "direction": "Both",
    "skip_weekday": -1,
    "session_start_h": 0,
    "session_window_h": 24,
    "static_risk_pct": 0.0025,    # 0.25% of entry price as the static stop
    "rr": 1.5,                    # the transcript's own explicit number
}


def fmt_result(label: str, r: Optional[dict]) -> str:
    if r is None:
        return f"  {label}: no result (too few trades or no signals)"
    return (f"  {label}: n={r['n_trades']:>5}  win%={r['win_rate_pct']:>5.1f}  "
            f"PF={r['profit_factor']:>5.3f}  expectancy(R)={r['expectancy_r']:>+.4f}  "
            f"total_R={r['total_r']:>+8.2f}  max_dd(R)={r['max_dd_r']:>+7.2f}  "
            f"timeout%={r['timeout_frac']*100:>5.1f}")


def main():
    print("Loading precomputed NDX100 data...")
    df_all = pd.read_parquet("mr_precomputed_NDX100.parquet")
    df_2022 = df_all[df_all.index.year == 2022]
    df_2024 = df_all[df_all.index.year == 2024]
    print(f"  2022: {len(df_2022):,} bars ({df_2022.index[0]} .. {df_2022.index[-1]})")
    print(f"  2024: {len(df_2024):,} bars ({df_2024.index[0]} .. {df_2024.index[-1]})")

    # ── Phase 1: baseline (literal transcript reproduction) ──
    print("\n" + "=" * 78)
    print("PHASE 1 — BASELINE (literal reading of the transcript's parameters)")
    print(json.dumps(BASELINE_PARAMS, indent=2))
    print("=" * 78)
    r22 = eval_year(df_2022, BASELINE_PARAMS)
    r24 = eval_year(df_2024, BASELINE_PARAMS)
    print(fmt_result("2022 (bear)", r22))
    print(fmt_result("2024 (bull)", r24))

    # Quick sensitivity on static_risk_pct alone, RR fixed at the
    # transcript's 1.5, everything else held at baseline — shows how
    # much the single "static stop size" choice alone moves the result.
    print("\n-- static_risk_pct sensitivity (rr=1.5, all else baseline) --")
    for rp in [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0060, 0.0080, 0.0100]:
        p = {**BASELINE_PARAMS, "static_risk_pct": rp}
        a = eval_year(df_2022, p)
        b = eval_year(df_2024, p)
        pf_a = a["profit_factor"] if a else float("nan")
        pf_b = b["profit_factor"] if b else float("nan")
        n_a = a["n_trades"] if a else 0
        n_b = b["n_trades"] if b else 0
        print(f"  risk_pct={rp:.4f}  2022: PF={pf_a:.3f} n={n_a:<5}  2024: PF={pf_b:.3f} n={n_b:<5}")

    # ── Phase 2: parameter sweep ──
    N_ITER = int(os.environ.get("SWEEP_ITER", 4000))
    print("\n" + "=" * 78)
    print(f"PHASE 2 — SWEEP ({N_ITER:,} random draws, both years must be evaluated)")
    print("=" * 78)

    rows = []
    accepted = []
    t0 = time.time()
    for i in range(1, N_ITER + 1):
        p = sample_params()
        r22 = eval_year(df_2022, p)
        r24 = eval_year(df_2024, p)
        if r22 is None or r24 is None:
            continue
        row = {**p,
               "n_2022": r22["n_trades"], "pf_2022": r22["profit_factor"],
               "exp_2022": r22["expectancy_r"], "dd_2022": r22["max_dd_r"],
               "n_2024": r24["n_trades"], "pf_2024": r24["profit_factor"],
               "exp_2024": r24["expectancy_r"], "dd_2024": r24["max_dd_r"]}
        rows.append(row)
        if (r22["n_trades"] >= MIN_TRADES_PER_YEAR and r24["n_trades"] >= MIN_TRADES_PER_YEAR
                and r22["profit_factor"] > 1.0 and r24["profit_factor"] > 1.0):
            row["avg_pf"] = (r22["profit_factor"] + r24["profit_factor"]) / 2
            row["min_pf"] = min(r22["profit_factor"], r24["profit_factor"])
            accepted.append(row)
        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  iter {i:>5}/{N_ITER}  tested={len(rows):>5}  accepted={len(accepted):>4}  "
                  f"rate={rate:.1f}/s  eta={((N_ITER - i) / rate)/60:.1f}min")

    pd.DataFrame(rows).to_csv("video_strategy_sweep_results.csv", index=False)
    print(f"\nSweep done: {len(rows)} evaluable draws, {len(accepted)} accepted "
          f"(PF>1.0 AND >={MIN_TRADES_PER_YEAR} trades in BOTH years).")

    accepted.sort(key=lambda r: r["min_pf"], reverse=True)
    top = accepted[:20]
    with open("video_strategy_top_candidates.json", "w") as f:
        json.dump(top, f, indent=2, default=str)

    print("\nTop candidates (by min(PF_2022, PF_2024), i.e. worst-year-first):")
    for r in top[:10]:
        print(f"  min_pf={r['min_pf']:.3f}  avg_pf={r['avg_pf']:.3f}  "
              f"2022: PF={r['pf_2022']:.3f} n={r['n_2022']} exp={r['exp_2022']:+.4f}  "
              f"2024: PF={r['pf_2024']:.3f} n={r['n_2024']} exp={r['exp_2024']:+.4f}  "
              f"risk_pct={r['static_risk_pct']} rr={r['rr']} bos_mode={r['bos_mode']} "
              f"dev_thr={r['deviation_threshold_atr']} dir={r['direction']}")

    print(f"\nWrote video_strategy_sweep_results.csv ({len(rows)} rows) and "
          f"video_strategy_top_candidates.json ({len(top)} candidates).")


if __name__ == "__main__":
    main()
