#!/usr/bin/env python3
"""
MeanReversion — precompute: 1-minute OHLCV + multi-window ATR + confirmed
swing fractals + daily-open anchor.

Self-contained port of RCTBE's `_precompute_mean_reversion.py` (2026-08-24)
— this project shares RCTBE's Dukascopy data-loading conventions
(divisor-break harmonization, dead-bar drop) but doesn't import RCTBE's
code, so this repo can be cloned and run on its own (e.g. a VPS that
never checks out RCTBE at all). The harmonization logic below is copied
from RCTBE's `layer1_features.py` (`_harmonize_divisor_breaks`) rather
than imported — same "no cross-project reliance" pattern that file's own
docstring documents for ITS borrowing from OPRrsitomt5. If RCTBE finds
and fixes a new data bug in this logic, mirror the fix here too.

Mean reversion trades an intraday deviation from a "fair value" anchor
(the day's opening price, updated when price consolidates at a new
level) using 1-minute bars — no regime model, no HMM, no walk-forward
refit involved at all, so this precompute is fast (a handful of
vectorized rolling-window passes over the raw 1-minute cache), unlike a
regime-conditioned pipeline.

Restricted to index CFDs (NDX100/SPX500/US30/GER40/FRA40/UK100/JPN225) —
"trade indexes" was the explicit brief this strategy was designed
against. Widen INDEX_ASSETS below (or pass a specific asset on the
command line) to test other instrument classes.

*** DATA DEPTH CAVEAT — READ BEFORE TRUSTING A 5-PERIOD WALK-FORWARD
SPLIT ON EVERY ASSET ***
Per RCTBE's own `layer1_features.py` (`PLAUSIBLE_PRICE_RANGE` comments,
confirmed 2026-08-06 by direct multi-year Dukascopy probing), NOT all 7
index assets have the same history depth in the shared Dukascopy cache:
    NDX100, SPX500, US30, GER40, JPN225   ~10 years (full history)
    FRA40, UK100                          ~3.3 years only (2023-2026)
A 5-period walk-forward split on FRA40/UK100 divides a much shorter,
much more recent-only sample — each period covers ~8 months instead of
~2 years, with correspondingly less statistical power and no coverage of
any regime before 2023. Treat any FRA40/UK100 result with that firmly in
mind (same caveat RCTBE's own README applies to USDJPY's shorter cache).
This script does NOT skip them — the search can still surface something
worth a closer look — but a candidate found on either of these two
deserves the extra scrutiny CANDIDATE_DEPLOYMENT_CHECKLIST.md's
"temporal robustness" stage already calls for on any thin-history
candidate, more than usual.

Precomputes only what does NOT depend on any swept mean-reversion
parameter — see mean_reversion.py's own docstring for what IS computed
fresh per search iteration instead (the fair-value/consolidation logic,
deviation threshold, BOS lookback/confirm window).

Usage:
    py -3 precompute.py <ASSET>
    py -3 precompute.py ALL
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
#  Data source + divisor-harmonization — copied from RCTBE's
#  layer1_features.py, not imported (see module docstring).
# ══════════════════════════════════════════════════════════════════════════

DUKASCOPY_CACHE_ROOT = Path(
    os.environ.get("MR_DUKASCOPY_ROOT", r"C:\Users\leach\OPRrsitomt5\data\dukascopy")
)

# Index CFDs only — Dukascopy codes + plausible price ranges copied from
# RCTBE's layer1_features.py (PILOT_ASSETS / PLAUSIBLE_PRICE_RANGE),
# confirmed there by direct probing against known historical levels, not
# guessed. See the module docstring above for each asset's real history
# depth (FRA40/UK100 are ~3.3yr, not the ~10yr the other 5 have).
INDEX_ASSETS = {
    "NDX100": "USATECHIDXUSD",
    "SPX500": "USA500IDXUSD",
    "US30":   "USA30IDXUSD",
    "GER40":  "DEUIDXEUR",
    "FRA40":  "FRAIDXEUR",
    "UK100":  "GBRIDXGBP",
    "JPN225": "JPNIDXJPY",
}

PLAUSIBLE_PRICE_RANGE = {
    "NDX100": (5_000, 40_000),
    "SPX500": (1_500, 10_000),
    "US30":   (10_000, 60_000),
    "GER40":  (5_000, 35_000),
    "FRA40":  (3_000, 15_000),
    "UK100":  (3_000, 15_000),
    "JPN225": (10_000, 90_000),
}

DIVISOR_BREAK_LOG_THRESHOLD = np.log(5)  # ratio outside [0.2, 5] — not a credible real 1-min move

ATR_WINDOWS = [14, 30, 60, 120]   # minutes; swept via atr_window in mean_reversion.SPACE
K = 3                              # fractal half-window


def _harmonize_divisor_breaks(df: pd.DataFrame, plausible_range: tuple[float, float]) -> pd.DataFrame:
    """Detect and correct divisor-scale issues in OHLC — both mid-series
    discontinuities and a whole series sitting at a uniformly-wrong scale
    with no internal jump at all. Verbatim port of RCTBE's
    layer1_features._harmonize_divisor_breaks — see that file for the
    full incident writeup (a real SPX500 100x-divisor bug this exact
    check was built to catch, twice)."""
    close = df["close"]
    log_ret = np.log(close).diff()
    break_mask = log_ret.abs() > DIVISOR_BREAK_LOG_THRESHOLD

    if break_mask.any():
        log_correction = (-log_ret.where(break_mask, 0.0)).cumsum()
        factor = np.exp(log_correction)
        corrected = df.copy()
        for col in ("open", "high", "low", "close"):
            corrected[col] = df[col] * factor
    else:
        corrected = df.copy()

    median_price = corrected["close"].median()
    lo, hi = plausible_range
    if not (lo <= median_price <= hi):
        target = (lo * hi) ** 0.5
        power = round(np.log10(target / median_price))
        corrected[["open", "high", "low", "close"]] *= 10.0**power

    return corrected


def load_1m_ohlcv(duk_dir: Path, plausible_range: tuple[float, float]) -> pd.DataFrame:
    """Same harmonization/dead-bar-drop pipeline as RCTBE's
    layer1_features.load_5m_ohlcv, minus the 5-min resample step — this
    strategy trades the native 1-minute bars directly."""
    files = sorted(duk_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cached data at {duk_dir}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = _harmonize_divisor_breaks(df, plausible_range)
    # Dead-hour placeholder bars (OHLC = previous close, volume 0) — Dukascopy
    # emits these through dead trading hours instead of leaving true gaps.
    df = df[df["volume"] > 0]
    return df[["open", "high", "low", "close", "volume"]].copy()


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def precompute_asset(asset: str) -> pd.DataFrame:
    duk_code = INDEX_ASSETS[asset]
    duk_dir = DUKASCOPY_CACHE_ROOT / duk_code
    print(f"[{asset}] loading 1-min OHLCV from {duk_dir}...", flush=True)
    df = load_1m_ohlcv(duk_dir, PLAUSIBLE_PRICE_RANGE[asset])
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"[{asset}] {len(df):,} 1-min bars, {df.index[0]} .. {df.index[-1]} "
          f"(~{n_years:.1f} years)", flush=True)

    for w in ATR_WINDOWS:
        df[f"atr_{w}"] = _atr(df, w)

    idx_utc = df.index.tz_convert("UTC") if df.index.tz is not None else df.index
    df["day_id"] = idx_utc.date
    day_changed = pd.Series(df["day_id"].to_numpy(), index=df.index).ne(
        pd.Series(df["day_id"].to_numpy(), index=df.index).shift(1)
    )
    df["day_open"] = df["open"].where(day_changed).ffill()

    is_fractal_low = df["low"] == df["low"].rolling(2 * K + 1, center=True).min()
    is_fractal_high = df["high"] == df["high"].rolling(2 * K + 1, center=True).max()
    df["swing_low_confirmed"] = df["low"].where(is_fractal_low).shift(K)
    df["swing_high_confirmed"] = df["high"].where(is_fractal_high).shift(K)

    atr_cols = [f"atr_{w}" for w in ATR_WINDOWS]
    out = df.dropna(subset=atr_cols)
    out_path = f"mr_precomputed_{asset}.parquet"
    out.to_parquet(out_path)
    print(f"[{asset}] saved {len(out):,} rows to {out_path}", flush=True)
    return out


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    assets = list(INDEX_ASSETS) if arg == "ALL" else [arg]
    for asset in assets:
        precompute_asset(asset)


if __name__ == "__main__":
    main()
