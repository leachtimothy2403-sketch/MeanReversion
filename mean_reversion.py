#!/usr/bin/env python3
"""
MeanReversion — mean reversion to an intraday "fair value" anchor, on
1-minute index bars.

Self-contained port of RCTBE's `layer2_mean_reversion.py` (2026-08-24) —
same design, same validation standard, but with no import of RCTBE's own
code, so this repo can be cloned and run entirely on its own (its own
VPS deployment, its own git history). Constants that mirror a validation
STANDARD RCTBE established (walk-forward period count, minimum trade
counts, checkpoint cadence) are copied as literals below with a note —
if RCTBE's own standard changes, mirror the change here too, same
"duplicated, not imported, but kept in sync" convention RCTBE's own
`layer2_cost_table.py` documents for its own borrowing from a sister
project.

Design (translates the user's brief into swept, backtestable rules):

  - Fair value. Starts each UTC trading day at that day's first traded
    price (day_open, precomputed). Updates intraday whenever price
    CONSOLIDATES at a new level: a rising-edge event where the rolling
    high-low range over the trailing `consolidation_bars` bars first
    drops to <= `consolidation_atr_mult` * ATR — at that bar, fair value
    resets to the midpoint of that consolidation range. Both knobs swept.
    Purely causal (rolling windows only look backward); the update
    itself is a simple forward-fill of a sparse "update event" series, so
    it vectorizes over the whole history in one pass per search iteration
    — see generate_signals.

  - Distance to trade. `deviation_threshold_atr` (swept) — trade only
    once price has been at least that many ATRs away from the CURRENT
    fair value within the trailing `extension_lookback_bars` window (not
    necessarily on the exact same bar as the BOS confirmation — see
    2026-08-25 note below). Below fair value = "extended below" (looking
    to buy back toward it); above = "extended above" (looking to sell
    back toward it). ATR window itself is swept (`atr_window`, selecting
    among 4 precomputed windows) since it governs both this threshold
    and the stop distance below.

  - Entry signal: a break of structure, on one of two swept `bos_mode`s:
      "fractal"  — price closes beyond the nearest CONFIRMED swing point
                   (K=3 fractal, see precompute.py) within a trailing
                   `bos_lookback_bars` window (original design).
      "raw_wick" — price closes beyond the raw high/low (the WICK, not a
                   filtered fractal) of any bar in the trailing
                   `bos_lookback_bars` window — added 2026-08-25, see
                   note below.
    Either way: held for `bos_confirm_bars` consecutive closes (swept),
    rising-edge only (fires once per breakout episode, not every bar
    while still beyond). Entry price is that confirmation bar's close.

    2026-08-25 — frequency fix. The original design produced ~30 trades
    per ~9-month walk-forward period (roughly one every 7-10 days) on
    the top search survivors, nowhere near the "many trades a day" the
    strategy was meant to produce on 1-minute bars. Two root causes,
    fixed together:
      1. BOS and the deviation-from-fair-value condition were required
         on the EXACT SAME bar (buy_cand = bos_up & extended_below, both
         indexed identically). Since fair value can itself be sliding
         toward price throughout an extension (see the consolidation fix
         below), the deviation condition often collapsed back under
         threshold by the time a multi-bar BOS confirmation completed a
         few minutes later — two already-narrow conditions had to land
         on the identical minute. Fixed by adding `extension_lookback_
         bars` (swept): the deviation condition now only needs to have
         been true at ANY point in that trailing window, not on the BOS
         bar itself.
      2. `bos_lookback_bars`'s swept range started at 5 (minutes) and
         the "fractal" mode's swing points are themselves K=3-bar
         smoothed/lagged features — genuinely 1-minute-scale structure
         (the immediately preceding candle or two) was never explorable.
         Fixed by widening the shared grid down to 1-3 bars and adding
         the "raw_wick" bos_mode per the user's own definition: a candle
         closing beyond the wick (raw high/low, no fractal filter) of a
         previous candle IS a break of structure, evaluated on whatever
         small `bos_lookback_bars` window the search picks. Both modes
         are swept side by side (not a hard swap) so the search itself
         can show which one actually produces a validated edge, rather
         than trading one unverified assumption for another.
    See also the `MIN_TRADES_*`/score-cap note further down — the
    accept gate and scoring previously had almost no incentive to prefer
    higher-frequency setups once a candidate cleared a trivially low bar.

  - Stop-loss: structural — anchored to the raw high/low extreme of the
    SAME `bos_lookback_bars` window (the extent of the move being
    faded), buffered by `k_buf` * ATR, then clamped to
    [`k_floor`, `k_cap`] * ATR.

  - Take-profit: a swept fraction (`target_fraction`) of the distance
    from entry back to the CURRENT fair value. Degenerate targets
    (fair value already past entry the wrong way) are dropped.

  - Exit: a wall-clock horizon (`exit_horizon_bars`, minutes) computed
    from elapsed real time from the entry timestamp, not row offset —
    dropped dead-hour bars would otherwise silently drift a bar-count
    horizon. Optional session-close cap (`use_session_close`).

  - Also swept: `direction` (Buy/Sell/Both), `skip_weekday`,
    `session_start_h`/`session_window_h` (hours of trading), `bos_mode`
    and `extension_lookback_bars` (see 2026-08-25 note above).

  - `consolidation_atr_mult`'s grid ceiling was lowered from 2.0 to 1.5
    on 2026-08-25: the top two search survivors reviewed that day both
    sat at the old 2.0 ceiling, which lets the fair-value anchor reset
    on almost any brief pause — closer to continuously chasing price
    than the "day-open anchor, occasionally refreshed at a genuine
    consolidation" concept from the original brief. A chasing anchor
    also shrinks the deviation-from-fair-value reading throughout an
    extension, compounding the same-bar timing issue fixed above.

  2026-08-26 — second entry mechanism + multi-timeframe confirmation.
  After 45,000+ iterations/worker still found zero accept-gate
  candidates on real NDX100 data, overnight research
  (strategy_improvement_ideas_2026-08-26.md) empirically ruled out data/
  regime explanations and landed on: BOS's edge is real but thin, and the
  2026-08-25 frequency fix likely diluted it further. Two new swept
  mechanisms were added ALONGSIDE "bos" (not a replacement — the search
  itself decides which holds up):
    - `entry_mode="range_fade"` — adapted from RCTBE's layer2_range_
      fade.py: trade DURING an ongoing consolidation (wick into a
      `fade_zone_pct` band near a range edge, without closing beyond it)
      instead of only at BOS's rare breakout event. SL just past the
      faded edge (same k_buf/k_floor/k_cap philosophy as bos). TP =
      target_fraction of the way across the range. Adds a THIRD exit
      race condition (see _replay_trade's `regime_ok`): if the range
      stops "consolidating" mid-trade, exit immediately at that bar's
      close — the trade's premise just broke.
    - `require_htf_confirm` — cheap 5-min K=3 fractal swing filter
      (precompute.py's `_htf_swing_columns`, NOT the full HMM regime
      label RCTBE uses — deferred per the research doc's own
      recommendation to test a cheaper mechanism first): blocks a new
      trade if the opposite-direction 5-min structure broke within the
      trailing `htf_lookback_bars` (1-min bars). Applies to both entry
      modes. Needs precompute.py re-run on any asset precomputed before
      this date — generate_signals raises a clear RuntimeError, not a
      silent no-op or crash, if those columns are missing.
    - `bos_mode`/`bos_lookback_bars`/`bos_confirm_bars`/`extension_
      lookback_bars`/`deviation_threshold_atr` only affect entry_mode=
      "bos"; `fade_zone_pct` only affects entry_mode="range_fade" — see
      SPACE's inline comments.

VALIDATION STANDARD — matches the sister project's own bar, not a looser
one built for this project specifically:
  - 5-period walk-forward split (N_PERIODS_SEARCH=5), accept gate
    requires >=4/5 periods profitable (MIN_PERIODS_PASS), mirrors every
    Layer 2 strategy in RCTBE.
  - MR_IS_ONLY=1 (default ON here — see below) truncates every asset's
    data to periods 1-3 (the first 60% of history) BEFORE the search
    ever sees it, exactly like RCTBE_L2_IS_ONLY in RCTBE's
    layer2_optimizer.py. This closes a real selection leak RCTBE's own
    project history found and fixed (HANDOFF.md / RCTBE_SYSTEM_
    BRIEFING.md Section 5's "old methodology" vs "new methodology"
    split): without it, a candidate could scrape into the accept gate's
    top pool by looking good on periods 4-5 — the SAME bars Gate 2 later
    calls a "blind OOS holdout" — which isn't a genuinely blind test.
    With it on, gate2_holdout.py's later check on periods 4-5 is
    genuinely blind, not a re-score of bars the search already used to
    select candidates. Since this is a brand-new project with no
    existing "old methodology" candidate pool to stay comparable with,
    there's no reason to default to the leakier version — set
    MR_IS_ONLY=0 only if you deliberately want the old (leakier)
    behavior for a specific comparison.
  - Every candidate needs >=500 trades total (MIN_TRADES_TOTAL) and
    >=100/period (MIN_TRADES_PER_PERIOD) to be scored at all — raised
    2026-08-25 in two steps: first 100/8 -> 200/40 (a moderate first
    step, before any real-data measurement), then 200/40 -> 500/100 after
    a real 2024 NDX100 check (_real_ndx_trades_per_day.py) both (a)
    caught and fixed a bug that made bos_mode="raw_wick" fire ZERO trades
    ever on real data (its BOS comparison was structurally impossible —
    see generate_signals' raw_wick branch), and (b) confirmed that, once
    fixed, the user's explicit "2-3 trades a day" target is genuinely
    achievable: a median of ~1.5 trades/day and 34% of random param draws
    clearing >=3/day across 800 draws on one real year of NDX100 data.
    100/period is a statistical-validity floor (well below even a modest
    ~0.5-1/day candidate over a ~500-trading-day walk-forward period),
    not the target itself — the score formula's own trade-count term
    (see backtest_multiperiod) is what actually rewards approaching
    2-3/day, and its cap was raised alongside this to saturate around
    1200 trades/period (~2.4/day) to match.

Performance note: unlike a regime-conditioned strategy, nothing here
needs a per-iteration HMM refit. What IS computed fresh per iteration
(fair value, deviation, BOS detection) uses vectorized pandas rolling
ops over the full asset history, not a per-candidate Python loop.

Usage:
    py -3 mean_reversion.py
Env vars: MR_OUTPUT_DIR, MR_ITERATIONS, MR_ASSET_FILTER, MR_IS_ONLY
(default "1" — see VALIDATION STANDARD above), MR_DATA_DIR.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from cost_table import COST_TABLE
from precompute import ATR_WINDOWS, INDEX_ASSETS

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_DIR      = os.environ.get("MR_OUTPUT_DIR", "meanreversion_output")
RESULTS_CSV     = os.path.join(OUTPUT_DIR, "results.csv")
TOP_JSON        = os.path.join(OUTPUT_DIR, "top_strategies.json")
LOG_FILE        = os.path.join(OUTPUT_DIR, "run.log")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
DONE_FILE       = os.path.join(OUTPUT_DIR, "DONE")

N_ITERATIONS = int(os.environ.get("MR_ITERATIONS", 20_000))

# Blind-search selection fix — see module docstring's VALIDATION STANDARD
# section. Default ON (unlike RCTBE_L2_IS_ONLY, which defaults off there
# only because turning it on retroactively would have broken
# comparability with RCTBE's existing "old methodology" candidate pool —
# no such pool exists here).
IS_ONLY_SEARCH = os.environ.get("MR_IS_ONLY", "1").lower() in ("1", "true", "yes")

# Validation-standard constants — mirror RCTBE's layer2_optimizer.py.
# See module docstring: duplicated, not imported, but must be kept in
# sync if RCTBE's own standard ever changes.
N_PERIODS_SEARCH      = 5
MAX_WEAK_PERIODS      = 1
MIN_PERIODS_PASS      = N_PERIODS_SEARCH - MAX_WEAK_PERIODS   # 4
# Raised 2026-08-25 from 100/8, then again same day from 200/40 — see
# module docstring's VALIDATION STANDARD section. The second raise
# followed a real-data check (_real_ndx_trades_per_day.py, real 2024
# NDX100 1-min bars, not synthetic) run after fixing a bug that had made
# bos_mode="raw_wick" fire zero trades ever (see generate_signals' raw_wick
# branch comment) — once fixed, random param draws on real data hit a
# median of ~1.5 trades/day and 34% of draws cleared >=3/day, confirming
# the user's explicit "2-3 times a day" target is mechanically achievable,
# not just a synthetic artifact. A 5-period walk-forward split of ~10
# years of history gives each period roughly 500 trading days, so 100
# trades/period is still a floor well below even a modest ~0.5-1/day
# candidate — a statistical-validity minimum, not the target itself (the
# score's avg_n term below is what actually rewards approaching 2-3/day).
MIN_TRADES_TOTAL      = 500
MIN_TRADES_PER_PERIOD = 100
TOP_N                 = 100
CHECKPOINT_EVERY      = 500
CHECKPOINT_SECONDS    = 1800
MAX_PF                = 10.0
MAX_TIMEOUT_FRAC      = 0.50
SESSION_CLOSE_UTC_HOUR = 21

DATA_DIR = os.environ.get("MR_DATA_DIR", ".")

# ══════════════════════════════════════════════════════════════════════════
#  PARAMETER SPACE — every knob the brief called out as "should be
#  optimized" (SL/TP, distance from fair value to start trading, hours of
#  trading, signal to enter, fair-value update sensitivity) is swept here.
# ══════════════════════════════════════════════════════════════════════════

SPACE = {
    "atr_window":              ATR_WINDOWS,                       # feeds deviation threshold + stop distance
    "consolidation_bars":      [5, 8, 13, 20, 30, 45],             # fair-value update sensitivity (window)
    "consolidation_atr_mult":  [0.5, 0.75, 1.0, 1.25, 1.5],        # fair-value update sensitivity (tightness) — ceiling lowered from 2.0 2026-08-25, see docstring
    "deviation_threshold_atr": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],# distance from fair value to start trading — BOS entry_mode only, see docstring
    "entry_mode":              ["bos", "range_fade"],              # added 2026-08-26 — see docstring's entry-mode note
    "bos_mode":                ["fractal", "raw_wick"],            # added 2026-08-25 — see docstring's frequency-fix note. entry_mode="bos" only.
    "bos_lookback_bars":       [1, 2, 3, 5, 8, 13, 20, 30, 45, 60],# widened down to 1-3 2026-08-25 for genuine 1-minute-scale structure. entry_mode="bos" only.
    "bos_confirm_bars":        [1, 2, 3, 5],                      # signal to enter — how convincing the break must be. entry_mode="bos" only.
    "extension_lookback_bars": [1, 3, 5, 10, 20],                  # added 2026-08-25 — deviation no longer required on the exact BOS bar, see docstring. entry_mode="bos" only.
    "fade_zone_pct":           [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],# added 2026-08-26 — how close to the range edge counts as "faded". entry_mode="range_fade" only, see docstring.
    "require_htf_confirm":     [False, True],                     # added 2026-08-26 — 5-min structure must not have just broken the opposite way. Needs precompute.py's htf_swing_*_confirmed columns.
    "htf_lookback_bars":       [30, 60, 120, 240, 480],            # how far back (1-min bars) to check for an opposing 5-min break. require_htf_confirm=True only.
    "target_fraction":         [0.5, 0.75, 1.0, 1.25],             # TP: how far back toward fair value (bos) / across the range (range_fade)
    "exit_horizon_bars":       [15, 30, 60, 90, 120, 180, 240, 360],
    "use_session_close":       [False, True],
    "k_buf":                   [0.1, 0.2, 0.3, 0.5],               # SL construction (structural + ATR buffer)
    "k_cap":                   [1.5, 2.0, 2.5, 3.0, 4.0],
    "k_floor":                 [0.5, 0.75, 1.0, 1.5],
    "direction":               ["Buy", "Sell", "Both"],
    "skip_weekday":            [-1, 0, 1, 2, 3, 4],
    "session_start_h":         [0, 4, 8, 12, 16, 20],              # hours of trading
    "session_window_h":        [24, 16, 8, 4],
}


def sample_params(available_assets: List[str]) -> dict:
    local = dict(SPACE)
    local["asset"] = available_assets
    for _ in range(5_000):
        p = {k: random.choice(v) for k, v in local.items()}
        if p["k_floor"] >= p["k_cap"]:
            continue  # a no-op combo — floor never binds if it's >= cap
        return p
    raise RuntimeError("Cannot sample valid params")


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

_CACHE: dict[str, pd.DataFrame] = {}


def load_asset(asset: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, f"mr_precomputed_{asset}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════

def _consolidation_range(df: pd.DataFrame, atr: pd.Series, p: dict):
    """Rolling high/low over `consolidation_bars`, and whether that range
    is CURRENTLY tight enough (<= consolidation_atr_mult * ATR) to count
    as an active consolidation — extracted 2026-08-26 (was inline in
    `_fair_value_series` only) so `_fair_value_series`, the new
    entry_mode="range_fade" trigger, and range_fade's regime-break exit
    all agree on exactly the same definition of "currently ranging"
    instead of three independent, potentially-drifting copies."""
    consolidation_bars = p["consolidation_bars"]
    mult = p["consolidation_atr_mult"]

    roll_high = df["high"].rolling(consolidation_bars).max()
    roll_low = df["low"].rolling(consolidation_bars).min()
    roll_range = roll_high - roll_low
    is_consolidating = (roll_range <= mult * atr) & atr.notna() & roll_range.notna()
    return roll_high, roll_low, is_consolidating


def _fair_value_series(df: pd.DataFrame, atr: pd.Series, p: dict,
                        roll_high: pd.Series, roll_low: pd.Series,
                        is_consolidating: pd.Series) -> pd.Series:
    """Day-open anchor, updated at every confirmed consolidation event.
    Fully causal and vectorized — one rolling pass + one ffill, no python
    loop, despite fair value being logically path-dependent."""
    consolidating_rising = is_consolidating & ~is_consolidating.shift(1, fill_value=False)
    consolidation_level = (roll_high + roll_low) / 2.0

    day_start = df["day_open"].ne(df["day_open"].shift(1)) | (df.index == df.index[0])

    update_value = pd.Series(np.nan, index=df.index)
    update_value[day_start] = df["day_open"][day_start]
    update_value[consolidating_rising] = consolidation_level[consolidating_rising]
    return update_value.ffill()


def generate_signals(df: pd.DataFrame, p: dict) -> List[dict]:
    atr = df[f"atr_{p['atr_window']}"]
    roll_high, roll_low, is_consolidating = _consolidation_range(df, atr, p)
    fair_value = _fair_value_series(df, atr, p, roll_high, roll_low, is_consolidating)

    close_s = df["close"]
    high_s = df["high"]
    low_s = df["low"]

    entry_mode = p.get("entry_mode", "bos")

    if p.get("require_htf_confirm"):
        if "htf_swing_high_confirmed" not in df.columns or "htf_swing_low_confirmed" not in df.columns:
            raise RuntimeError(
                "require_htf_confirm=True but this asset's precomputed data "
                "is missing htf_swing_high_confirmed/htf_swing_low_confirmed "
                "— re-run precompute.py (added 2026-08-26, see "
                "_htf_swing_columns in precompute.py's docstring) before "
                "sampling this parameter."
            )

    if entry_mode == "range_fade":
        # Added 2026-08-26 — see strategy_improvement_ideas_2026-08-26.md
        # Section 3 / RCTBE's layer2_range_fade.py. Trades DURING an
        # ongoing consolidation instead of only at BOS (a rare event by
        # definition): price wicks into a `fade_zone_pct` band near
        # either edge of the CURRENT rolling consolidation range without
        # CLOSING beyond it (a close beyond the edge looks like a real
        # breakout in progress, not a rejection worth fading), and we
        # fade back toward the opposite side of the range.
        fade_pct = p["fade_zone_pct"]
        range_size = roll_high - roll_low
        upper_zone_thresh = roll_high - fade_pct * range_size
        lower_zone_thresh = roll_low + fade_pct * range_size
        has_range = range_size > 0

        in_upper_zone = is_consolidating & has_range & (high_s >= upper_zone_thresh) & (close_s < roll_high)
        in_lower_zone = is_consolidating & has_range & (low_s <= lower_zone_thresh) & (close_s > roll_low)

        # Rising-edge only — fires once per approach into the zone, not
        # every bar spent still inside it.
        sell_cand = (in_upper_zone & ~in_upper_zone.shift(1, fill_value=False)).to_numpy()
        buy_cand = (in_lower_zone & ~in_lower_zone.shift(1, fill_value=False)).to_numpy()
    else:
        deviation_atr = (close_s - fair_value) / atr
        threshold = p["deviation_threshold_atr"]
        extended_below = deviation_atr <= -threshold   # too cheap -> looking to BUY back toward fair value
        extended_above = deviation_atr >= threshold     # too rich -> looking to SELL back toward fair value

        # Extension no longer has to land on the exact same bar as the BOS
        # confirmation (added 2026-08-25 — see module docstring's frequency-
        # fix note): true if extended at ANY point in the trailing
        # extension_lookback_bars window, inclusive of the current bar.
        ext_lookback = p.get("extension_lookback_bars", 1)
        extended_below_recent = extended_below.rolling(ext_lookback, min_periods=1).max().astype(bool)
        extended_above_recent = extended_above.rolling(ext_lookback, min_periods=1).max().astype(bool)

        # --- break of structure ---
        bos_lookback = p["bos_lookback_bars"]
        confirm_bars = p["bos_confirm_bars"]
        bos_mode = p.get("bos_mode", "fractal")

        if bos_mode == "raw_wick":
            # Added 2026-08-25, per the user's own definition: a candle
            # closing beyond the WICK (raw high/low, no fractal filter) of a
            # PREVIOUS candle within the trailing bos_lookback_bars window IS
            # a break of structure — genuine 1-minute-scale structure, not a
            # K=3-bar-smoothed fractal. See module docstring.
            #
            # 2026-08-25 correctness fix: the window MUST exclude the current
            # bar (shift(1) below) — the very first version of this branch
            # rolled high_s/low_s INCLUDING the current bar, which made
            # beyond_up = close_s > recent_swing_high structurally impossible
            # on any real (internally-consistent) OHLC bar, since
            # recent_swing_high >= high_s[t] >= close_s[t] always holds once
            # the current bar's own high is in the window — i.e. raw_wick
            # could NEVER fire. Confirmed empirically: 0/401 random draws
            # produced a single raw_wick trade on real 2024 NDX100 data (see
            # _real_ndx_trades_per_day.py), while the earlier "it works" read
            # on synthetic data (selftest.py's build_synthetic) turned out to
            # be an artifact of that generator's own bug — its `close` isn't
            # clamped into [low, high], so close could exceed the bar's own
            # high there, silently satisfying the broken comparison. Real
            # bars can't do that. The fractal branch below never had this bug
            # — swing_high_confirmed/swing_low_confirmed are already
            # shift(K)'d in precompute.py, so they're inherently past-only.
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

    if p.get("require_htf_confirm"):
        # Cheap multi-timeframe confirmation (added 2026-08-26, see
        # strategy_improvement_ideas_2026-08-26.md Section 5): don't fade
        # against a higher-timeframe structure break that just happened
        # the OPPOSING way — a bearish 5-min break in the last
        # `htf_lookback_bars` (1-min bars) blocks new buys, a bullish one
        # blocks new sells. Applies to both entry_mode values.
        htf_lookback = p.get("htf_lookback_bars", 60)
        htf_swing_high = df["htf_swing_high_confirmed"]
        htf_swing_low = df["htf_swing_low_confirmed"]
        recent_htf_break_down = (close_s < htf_swing_low).rolling(htf_lookback, min_periods=1).max().astype(bool).to_numpy()
        recent_htf_break_up = (close_s > htf_swing_high).rolling(htf_lookback, min_periods=1).max().astype(bool).to_numpy()
        buy_cand = buy_cand & ~recent_htf_break_down
        sell_cand = sell_cand & ~recent_htf_break_up

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
    a_arr = atr.to_numpy()[cand_idx]
    entry_arr = close_s.to_numpy()[cand_idx]
    fv_arr = fair_value.to_numpy()[cand_idx]

    target_fraction = p["target_fraction"]

    if entry_mode == "range_fade":
        # SL: structural, just past the range edge being faded (same
        # ATR-buffered-floor-and-cap philosophy as the bos branch below).
        # TP: swept fraction across the range (0.5 = midpoint), from the
        # NEAR edge toward the FAR edge — matches target_fraction's
        # existing convention (1.0 = the far edge itself).
        range_low_arr = roll_low.to_numpy()[cand_idx]
        range_high_arr = roll_high.to_numpy()[cand_idx]
        range_size_arr = range_high_arr - range_low_arr
        sl_anchor_arr = np.where(is_buy_arr, range_low_arr, range_high_arr)
        tp_arr = np.where(
            is_buy_arr,
            range_low_arr + target_fraction * range_size_arr,
            range_high_arr - target_fraction * range_size_arr,
        )
    else:
        bos_lookback = p["bos_lookback_bars"]
        recent_low_raw = low_s.rolling(bos_lookback, min_periods=1).min().to_numpy()[cand_idx]
        recent_high_raw = high_s.rolling(bos_lookback, min_periods=1).max().to_numpy()[cand_idx]
        sl_anchor_arr = np.where(is_buy_arr, recent_low_raw, recent_high_raw)
        tp_arr = np.where(
            is_buy_arr,
            entry_arr + target_fraction * (fv_arr - entry_arr),
            entry_arr - target_fraction * (entry_arr - fv_arr),
        )

    valid_tp = np.where(is_buy_arr, tp_arr > entry_arr, tp_arr < entry_arr)

    k_buf, k_cap, k_floor = p["k_buf"], p["k_cap"], p["k_floor"]
    sl_structural_arr = np.where(is_buy_arr, sl_anchor_arr - k_buf * a_arr, sl_anchor_arr + k_buf * a_arr)
    raw_dist_arr = np.abs(entry_arr - sl_structural_arr)
    dist_arr = np.clip(raw_dist_arr, k_floor * a_arr, k_cap * a_arr)
    sl_arr = np.where(is_buy_arr, entry_arr - dist_arr, entry_arr + dist_arr)

    horizon_ts_arr = idx[cand_idx] + pd.Timedelta(minutes=1) * p["exit_horizon_bars"]
    exit_idx_arr = idx.searchsorted(horizon_ts_arr, side="left")
    exit_idx_arr = np.minimum(exit_idx_arr, len(idx))

    if p.get("use_session_close"):
        entry_ts_s = pd.Series(idx[cand_idx])
        same_day_close = entry_ts_s.dt.normalize() + pd.Timedelta(hours=SESSION_CLOSE_UTC_HOUR)
        session_close_s = same_day_close.where(entry_ts_s < same_day_close, same_day_close + pd.Timedelta(days=1))
        session_close_idx = idx.searchsorted(session_close_s.to_numpy(), side="left")
        exit_idx_arr = np.minimum(exit_idx_arr, session_close_idx)

    signals: List[dict] = [
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
        if valid_tp[i]
    ]
    return signals


# ══════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════

def _replay_trade(sig: dict, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int,
                   regime_ok: Optional[np.ndarray] = None):
    """Shared SL/TP-race replay logic for one signal. Returns (r_multiple,
    is_timeout) or None if the signal has no room to run.

    `regime_ok` (added 2026-08-26, entry_mode="range_fade" only): an
    optional per-bar boolean array, same length/alignment as
    highs/lows/closes, True while the market is still "consolidating"
    (see _consolidation_range) — i.e. the premise the range-fade trade was
    entered on still holds. When provided, adds a THIRD race condition: if
    the range breaks (regime_ok goes False) while the trade is open, exit
    immediately at that bar's CLOSE rather than riding to the normal
    horizon. Priority on same-bar ties: SL > break > TP (SL stays most
    conservative; a break is treated as more urgent than a plain target
    hit since the trade's whole premise just failed). When regime_ok is
    None (the default, and always the case for entry_mode="bos"), this
    reduces EXACTLY to the original 2-way SL/TP race below."""
    ei, e, sl, tp, risk, d = sig["bar_idx"], sig["entry"], sig["sl"], sig["tp"], sig["risk"], sig["direction"]
    end = min(sig["exit_idx"], n)
    if end <= ei + 1:
        return None
    h_win, l_win = highs[ei + 1:end], lows[ei + 1:end]
    if d == "Buy":
        sl_mask = l_win <= sl
        tp_mask = h_win >= tp
    else:
        sl_mask = h_win >= sl
        tp_mask = l_win <= tp

    if regime_ok is not None:
        break_mask = ~regime_ok[ei + 1:end]
        has_break = break_mask.any()
    else:
        break_mask = None
        has_break = False

    has_sl, has_tp = sl_mask.any(), tp_mask.any()
    if not has_sl and not has_tp and not has_break:
        ep = closes[min(end - 1, n - 1)]
        r = (ep - e) / risk if d == "Buy" else (e - ep) / risk
        return r, True

    sl_i = int(np.argmax(sl_mask)) if has_sl else n + 1
    tp_i = int(np.argmax(tp_mask)) if has_tp else n + 1
    break_i = int(np.argmax(break_mask)) if has_break else n + 1

    if sl_i <= tp_i and sl_i <= break_i:      # SL wins ties — conservative
        r = -1.0
    elif break_i <= tp_i:                     # break wins over a same/later TP
        ep = closes[ei + 1:end][break_i]
        r = (ep - e) / risk if d == "Buy" else (e - ep) / risk
    else:
        r = (tp - e) / risk if d == "Buy" else (e - tp) / risk
    return r, False


def backtest_signals(signals: List[dict], df: pd.DataFrame, cost: float = 0.0,
                      p: Optional[dict] = None) -> Optional[dict]:
    if len(signals) < MIN_TRADES_PER_PERIOD:
        return None

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)

    regime_ok = None
    if p is not None and p.get("entry_mode") == "range_fade":
        atr = df[f"atr_{p['atr_window']}"]
        _, _, is_consolidating = _consolidation_range(df, atr, p)
        regime_ok = is_consolidating.to_numpy()

    r_results: List[float] = []
    n_timeout = 0

    for sig in signals:
        out = _replay_trade(sig, highs, lows, closes, n, regime_ok=regime_ok)
        if out is None:
            continue
        pnl_r, is_timeout = out
        if is_timeout:
            n_timeout += 1
        risk = sig["risk"]
        if cost > 0 and risk > 0:
            pnl_r -= cost / risk
        r_results.append(pnl_r)

    if len(r_results) < MIN_TRADES_PER_PERIOD:
        return None

    arr = np.array(r_results)
    wins = arr[arr > 0]
    loss = arr[arr < 0]
    nt = len(arr)
    timeout_frac = n_timeout / nt

    gross_win = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(loss.sum()) if len(loss) > 0 else None
    pf = MAX_PF if gross_loss is None else min(gross_win / gross_loss, MAX_PF)
    total_r = float(arr.sum())

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    max_dd_r = float((equity - peak).min())

    return {
        "n_trades": nt,
        "win_rate_pct": round(len(wins) / nt * 100, 2) if nt else 0,
        "profit_factor": round(pf, 3),
        "total_r": round(total_r, 2),
        "expectancy_r": round(float(arr.mean()), 4),
        "max_dd_r": round(max_dd_r, 2),
        "timeout_frac": round(timeout_frac, 4),
        "n_timeout": n_timeout,
    }


def get_trade_records(df: pd.DataFrame, p: dict, cost: float = 0.0) -> List[dict]:
    """Same per-trade R-multiple computation as backtest_signals, but
    returns one record per trade (date, direction, r) — for the FTMO
    challenge simulator and Gate 2/3 checks to consume."""
    sigs = generate_signals(df, p)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    idx = df.index
    n = len(closes)

    regime_ok = None
    if p.get("entry_mode") == "range_fade":
        atr = df[f"atr_{p['atr_window']}"]
        _, _, is_consolidating = _consolidation_range(df, atr, p)
        regime_ok = is_consolidating.to_numpy()

    records = []
    for sig in sigs:
        out = _replay_trade(sig, highs, lows, closes, n, regime_ok=regime_ok)
        if out is None:
            continue
        r, _ = out
        risk = sig["risk"]
        if cost > 0 and risk > 0:
            r -= cost / risk
        records.append({"date": idx[sig["bar_idx"]], "direction": sig["direction"], "r": r})
    return records


def backtest_multiperiod(df: pd.DataFrame, p: dict) -> Optional[dict]:
    n = len(df)
    bounds = np.linspace(0, n, N_PERIODS_SEARCH + 1).astype(int)
    periods = [df.iloc[bounds[i]:bounds[i + 1]] for i in range(N_PERIODS_SEARCH)]

    cost = COST_TABLE[p["asset"]]

    results = []
    for period_df in periods:
        if len(period_df) < 500:
            results.append(None)
            continue
        sigs = generate_signals(period_df, p)
        r = backtest_signals(sigs, period_df, cost=cost, p=p)
        results.append(r)

    valid = [r for r in results if r is not None]
    if len(valid) < N_PERIODS_SEARCH - 1:
        return None

    passing = sum(1 for r in valid if r["profit_factor"] > 1.0)
    if passing < MIN_PERIODS_PASS:
        return None

    n_trades = sum(r["n_trades"] for r in valid)
    if n_trades < MIN_TRADES_TOTAL:
        return None

    total_timeout = sum(r["n_timeout"] for r in valid)
    timeout_frac = total_timeout / n_trades
    if timeout_frac > MAX_TIMEOUT_FRAC:
        return None

    pfs = [r["profit_factor"] for r in valid]
    rets = [r["total_r"] for r in valid]
    avg_pf = float(np.mean(pfs))
    min_pf = float(np.min(pfs))
    avg_ret = float(np.mean(rets))
    avg_n = n_trades / len(valid)
    ret_std = max(float(np.std(rets)), 0.01)
    consistency = 1.0 / (1.0 + ret_std / (abs(avg_ret) + 1e-9))
    calmar = avg_ret / (ret_std + 1e-9)
    pf_component = 0.4 * min(avg_pf, 5.0) + 0.6 * min(min_pf, 5.0)
    # avg_n divisor: 15.0 -> 50.0 -> 600.0, all 2026-08-25 (cap: 30/period
    # -> 100/period -> 1200/period). The first two raises were before any
    # real-data measurement; this one follows the real 2024 NDX100 check
    # (_real_ndx_trades_per_day.py) that confirmed 2-3 trades/day —
    # roughly 1000-1500/period across a ~500-trading-day walk-forward
    # period — is genuinely achievable, matching the user's explicit
    # target. 1200/period (~2.4/day) sits inside that band: the search
    # keeps getting rewarded for more trades all the way up to it instead
    # of plateauing at a frequency far below the brief's intent. See
    # module docstring / MIN_TRADES_* comment above.
    score = pf_component * (consistency ** 2) * min(avg_n / 600.0, 2.0) * max(calmar, 0)

    wfe = None
    if len(valid) == N_PERIODS_SEARCH:
        is_pfs, oos_pfs = pfs[:3], pfs[3:5]
        is_edge = float(np.mean(is_pfs)) - 1.0
        oos_edge = float(np.mean(oos_pfs)) - 1.0
        if is_edge > 0:
            wfe = round(oos_edge / is_edge, 4)

    return {
        "avg_profit_factor": round(avg_pf, 3),
        "min_profit_factor": round(min_pf, 3),
        "avg_total_r": round(avg_ret, 2),
        "avg_trades": round(avg_n, 1),
        "n_trades_total": n_trades,
        "consistency": round(consistency, 3),
        "calmar": round(calmar, 3),
        "periods_evaluated": len(valid),
        "periods_passed": passing,
        "period_pfs": [round(r["profit_factor"], 3) for r in results if r is not None],
        "timeout_frac": round(timeout_frac, 4),
        "cost_applied": cost,
        "score": round(score, 4),
        "wfe": wfe,
    }


# ══════════════════════════════════════════════════════════════════════════
#  OUTPUT / CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = None


def write_row(row: dict) -> None:
    global _CSV_FIELDS
    is_new = not os.path.exists(RESULTS_CSV)
    if _CSV_FIELDS is None:
        _CSV_FIELDS = list(row.keys())
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def save_top(top: list) -> None:
    with open(TOP_JSON, "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, default=str)


def save_checkpoint(iteration: int, top_strategies: list, n_tested: int, n_valid: int) -> None:
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "iteration": iteration, "n_tested": n_tested, "n_valid": n_valid,
            "top_strategies": top_strategies,
            "saved_at": datetime.now().isoformat(),
        }, f, default=str)
    os.replace(tmp, CHECKPOINT_FILE)


def load_checkpoint() -> Optional[dict]:
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _CSV_FIELDS
    _CSV_FIELDS = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt = load_checkpoint()
    if ckpt is not None:
        start_iteration = ckpt["iteration"] + 1
        top_strategies0 = ckpt["top_strategies"]
        n_tested0 = ckpt["n_tested"]
        n_valid0 = ckpt["n_valid"]
        resumed = True
    else:
        start_iteration = 1
        top_strategies0 = []
        n_tested0 = n_valid0 = 0
        resumed = False
        if os.path.exists(RESULTS_CSV):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = f"{OUTPUT_DIR}_stale_{stamp}"
            os.makedirs(archive, exist_ok=True)
            for fname in ("results.csv", "run.log", "top_strategies.json"):
                src = os.path.join(OUTPUT_DIR, fname)
                if os.path.exists(src):
                    os.replace(src, os.path.join(archive, fname))

    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)

    sep = "=" * 70
    print(sep)
    print("  MEAN REVERSION (FAIR VALUE / BREAK OF STRUCTURE) RANDOM SEARCH")
    print(f"  {N_ITERATIONS:,} iterations  |  {N_PERIODS_SEARCH}-period walk-forward "
          f"validation (native scoring, min {MIN_PERIODS_PASS}/{N_PERIODS_SEARCH} periods pass)")
    print(f"  MR_IS_ONLY={'ON' if IS_ONLY_SEARCH else 'OFF'} — "
          f"{'every asset truncated to periods 1-3 before the search sees it (genuinely blind Gate 2)' if IS_ONLY_SEARCH else 'search sees full history — Gate 2 later re-scores bars the accept gate already used, NOT a clean blind test'}")
    if resumed:
        print(f"  RESUMING from iteration {start_iteration:,} (checkpoint saved {ckpt['saved_at']})")
    print(sep)

    if resumed and start_iteration > N_ITERATIONS:
        _log(f"[WARN] Checkpoint already at iteration {ckpt['iteration']:,}, which is >= "
             f"this run's N_ITERATIONS target ({N_ITERATIONS:,}) - NOTHING NEW WILL BE "
             f"TESTED. Increase MR_ITERATIONS past {ckpt['iteration']:,}, or point "
             f"MR_OUTPUT_DIR at a fresh folder for this campaign.")

    asset_filter = os.environ.get("MR_ASSET_FILTER")
    wanted_assets = {a.strip() for a in asset_filter.split(",")} if asset_filter else None
    if wanted_assets:
        _log(f"[+] MR_ASSET_FILTER active — will only load: {sorted(wanted_assets)}")

    _log("\n[+] Loading precomputed mean-reversion data (1-min OHLCV + ATR + swing fractals)...")
    available_assets = []
    for asset in INDEX_ASSETS:
        if wanted_assets is not None and asset not in wanted_assets:
            continue
        df = load_asset(asset)
        if df is not None and len(df) > 5000:
            if IS_ONLY_SEARCH:
                full_len = len(df)
                bounds = np.linspace(0, full_len, N_PERIODS_SEARCH + 1).astype(int)
                df = df.iloc[bounds[0]:bounds[3]]
            _CACHE[asset] = df
            _log(f"  [OK]   {asset}: {len(df):,} bars  |  {df.index[0]} -> {df.index[-1]}"
                 f"{'  (IS-only truncated)' if IS_ONLY_SEARCH else ''}")
            available_assets.append(asset)
        else:
            _log(f"  [SKIP] {asset} — no mr_precomputed_{asset}.parquet found "
                 f"(run precompute.py first)")

    if not available_assets:
        _log("[ERROR] No precomputed mean-reversion data available.")
        sys.exit(1)

    _log(f"\n[+] Assets available: {available_assets}")
    if resumed:
        _log(f"[+] Resuming search — {start_iteration - 1:,} iterations already done "
             f"({n_valid0:,} valid so far)...\n")
    else:
        _log(f"[+] Starting {N_ITERATIONS:,} iteration search...\n")

    top_strategies: List[dict] = top_strategies0
    n_tested = n_tested0
    n_valid = n_valid0
    t_start = time.time()
    last_ckpt_time = t_start
    iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, N_ITERATIONS + 1):
            p = sample_params(available_assets)
            df = _CACHE.get(p["asset"])
            if df is None:
                continue

            result = backtest_multiperiod(df, p)
            n_tested += 1

            if result is not None:
                n_valid += 1
                row = {**p, **result, "iteration": iteration}
                write_row(row)

                top_strategies.append(row)
                top_strategies.sort(key=lambda x: x.get("score", -999), reverse=True)
                top_strategies = top_strategies[:TOP_N]

            if iteration % 200 == 0:
                elapsed = time.time() - t_start
                done = iteration - start_iteration + 1
                rate = done / elapsed if elapsed > 0 else 0
                eta_h = (N_ITERATIONS - iteration) / rate / 3600 if rate > 0 else 0
                best = top_strategies[0] if top_strategies else {}
                _log(
                    f"Iter {iteration:>6,}/{N_ITERATIONS:,} | "
                    f"Valid={n_valid:,} | Rate={rate:.1f}/s | ETA={eta_h:.1f}h | "
                    f"Best: {best.get('asset','?')} "
                    f"dev_thr={best.get('deviation_threshold_atr','?')} "
                    f"target_frac={best.get('target_fraction','?')} "
                    f"PF={best.get('avg_profit_factor','?')} "
                    f"Score={best.get('score','?')}"
                )

            now = time.time()
            if iteration % CHECKPOINT_EVERY == 0 or (now - last_ckpt_time) >= CHECKPOINT_SECONDS:
                save_top(top_strategies)
                save_checkpoint(iteration, top_strategies, n_tested, n_valid)
                last_ckpt_time = now
                _log(f"  [CKPT] iter={iteration:,} saved top {len(top_strategies)} strategies")
    finally:
        save_top(top_strategies)
        save_checkpoint(iteration, top_strategies, n_tested, n_valid)

    with open(DONE_FILE, "w") as f:
        f.write(datetime.now().isoformat())

    _log(f"\n[+] Done. {n_tested:,} tested, {n_valid:,} valid. Top strategy: "
         f"{top_strategies[0] if top_strategies else 'none'}")


if __name__ == "__main__":
    main()
