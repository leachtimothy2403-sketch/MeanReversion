#!/usr/bin/env python3
"""
MeanReversion — narrowed neighborhood search, seeded from the VPS's real
Gate2+Gate3+FTMO survivors.

Per Tim's request (2026-08-27): "lets focus the search in the similar
settings (but still some good variability)" — the follow-up to reviewing
ALL 88 VPS candidates through the full pipeline (`vps_full_review.py`).

That review found only 3 candidates that are both robust (Gate 2 blind-OOS
PASS, Gate 3 plateau PASS/MIXED) AND show real FTMO promise once risk is
tuned down (see vps_full_review_results.json idx 0/33/80, and
mean-reversion-strategy-status.md's 2026-08-27 update for the tuned
worst-window numbers). Comparing their full parameter sets side by side
shows a real shared neighborhood, not three unrelated points:

  - direction: Buy, all three (the one Gate2-pass Sell candidate, idx 47,
    FAILED Gate 3 outright — 3.6% neighbors healthy).
  - entry_mode: static_risk, all three (the space's other two modes,
    "bos"/"range_fade", are untested by this review — this narrowed
    search stays inside the family that's actually been validated).
  - consolidation_atr_mult: 1.0 exactly, all three — the TIGHTEST value
    in the full grid (0.5-1.5). Not a coincidence three independent
    search draws all landed on the grid's edge.
  - require_htf_confirm: False, all three.
  - exit_horizon_bars: 360 (the grid's MAX), all three — another
    edge-of-grid convergence, this time suggesting the grid itself may be
    truncating something that wants to go even longer. Left as the
    survivors' shared value here (can't extend past what's precomputed
    without a precompute.py change) but this is worth flagging.
  - static_risk_pct: 0.002-0.004 — all three are in the BOTTOM THIRD of
    the full 0.001-0.01 grid, and separately, every FTMO tuning session
    this project has run (this candidate set and the earlier now-retired
    Sell candidate) has found viable risk only by going LOWER than the
    raw-search winner's own risk_pct, never higher. Narrowed to the
    bottom half of the full grid.
  - Genuinely NOT shared, and left with FULL variability on purpose (this
    is the "still some good variability" half of Tim's request) so this
    search doesn't just re-confirm 3 known points: bos_mode (raw_wick
    2/3, fractal 1/3), rr (bimodal: 1.25 vs 3.0), bos_lookback_bars,
    bos_confirm_bars, extension_lookback_bars, atr_window,
    deviation_threshold_atr, consolidation_bars, use_session_close,
    k_buf/k_cap/k_floor, skip_weekday, session_start_h, session_window_h.

Mechanically: imports mean_reversion.py unchanged (same generate_signals,
same backtest_multiperiod, same Gate 2/3-compatible walk-forward split)
and only monkey-patches SPACE's ranges before calling its own main() — so
every non-search behavior (checkpointing, MR_IS_ONLY truncation, output
format, scoring) is byte-for-byte identical to a normal mean_reversion.py
run, just sampling from a narrower box.

Usage (same env-var knobs as mean_reversion.py):
    MR_OUTPUT_DIR=meanreversion_output_narrowed MR_ITERATIONS=15000 \\
    MR_ASSET_FILTER=NDX100 python3 mean_reversion_narrowed.py
"""
import mean_reversion as mr

mr.SPACE["direction"] = ["Buy"]
mr.SPACE["entry_mode"] = ["static_risk"]
mr.SPACE["consolidation_atr_mult"] = [0.75, 1.0, 1.25]
mr.SPACE["require_htf_confirm"] = [False]
mr.SPACE["exit_horizon_bars"] = [240, 360]
mr.SPACE["static_risk_pct"] = [0.0015, 0.0020, 0.0025, 0.0030, 0.0040]

# Everything else in mr.SPACE is left exactly as-is (full variability) —
# see module docstring for the list and the reasoning.

if __name__ == "__main__":
    print("=" * 70)
    print("  NARROWED SEARCH — neighborhood of the 3 real Gate2+Gate3+FTMO")
    print("  survivors from vps_full_review.py (2026-08-27)")
    print("  Fixed: direction=Buy, entry_mode=static_risk, "
          "consolidation_atr_mult in {0.75,1.0,1.25},")
    print("         require_htf_confirm=False, exit_horizon_bars in {240,360}, "
          "static_risk_pct in {0.15-0.40%}")
    print("  Free (full variability): bos_mode, rr, bos_lookback_bars, "
          "bos_confirm_bars, extension_lookback_bars,")
    print("         atr_window, deviation_threshold_atr, consolidation_bars, "
          "use_session_close, k_buf/k_cap/k_floor,")
    print("         skip_weekday, session_start_h, session_window_h")
    print("=" * 70)
    mr.main()
