# MeanReversion

A prop-firm trading strategy: fade intraday index price back toward a
"fair value" anchor once it's deviated far enough, on 1-minute bars,
triggered by a break of structure. Built with the same
search-then-validate discipline as the sister project **RCTBE**
(`C:\Users\leach\RCTBE`) — same 5-period walk-forward accept gate, same
blind out-of-sample holdout (Gate 2), same parameter-plateau robustness
check (Gate 3), same real FTMO 1-step challenge rules — but this repo is
**fully self-contained**: nothing here imports RCTBE's code, so it can
be cloned and run on its own machine (a VPS included) with no other
project checked out alongside it. A few files are duplicated from RCTBE
on purpose (`cost_table.py`, `ftmo_challenge_rules.py`, the data-
harmonization logic inside `precompute.py`) — each says so in its own
docstring, with a note to mirror any future fix back from RCTBE if one
happens there.

## The strategy

- **Fair value** starts each UTC day at that day's first traded price
  (`day_open`). It updates intraday whenever price **consolidates** at a
  new level — a rising-edge event where the rolling high-low range over
  the trailing `consolidation_bars` bars first drops to
  `<= consolidation_atr_mult * ATR`; fair value then resets to the
  midpoint of that range. Both knobs swept.
- **Distance to trade**: only once price is at least `deviation_
  threshold_atr` ATRs away from the *current* fair value. `atr_window`
  itself is swept (14/30/60/120 min).
- **Entry — break of structure**: while extended (deviation condition
  true at any point in the trailing `extension_lookback_bars` window,
  not necessarily the exact same bar as the break itself — see below),
  wait for price to close beyond the nearest structure within a trailing
  `bos_lookback_bars` window, held for `bos_confirm_bars` consecutive
  closes. Two swept `bos_mode`s define "structure": `fractal` (the
  nearest confirmed K=3 swing point — the original design) or
  `raw_wick` (the raw high/low — the wick — of any bar in the lookback
  window, no fractal filter). Entry = that bar's close.
- **Stop-loss**: structural — the raw high/low extreme of the same
  lookback window, buffered by `k_buf` * ATR, clamped to
  `[k_floor, k_cap]` * ATR.
- **Take-profit**: `target_fraction` of the distance from entry back to
  the current fair value.
- **Exit**: a wall-clock horizon (`exit_horizon_bars`), plus an optional
  session-close cap.
- **Also swept**: `direction`, `skip_weekday`, `session_start_h`/
  `session_window_h` (hours of trading), `bos_mode`,
  `extension_lookback_bars`.

Full design rationale is in `mean_reversion.py`'s module docstring,
including a 2026-08-25 note on a frequency fix: the original design's
top search survivors were producing only ~30 trades per ~9-month period
(roughly one every 7-10 days) — nowhere near the "many trades a day" a
1-minute strategy is meant to produce. Root causes were (1) requiring
the deviation-from-fair-value condition on the *exact same bar* as the
BOS confirmation, which a chasing fair-value anchor (see the
`consolidation_atr_mult` note below) made increasingly rare, and (2)
`bos_lookback_bars` never exploring truly 1-minute-scale windows.
`extension_lookback_bars` and `bos_mode: raw_wick` fix both.

A follow-up same-day fix (also 2026-08-25) caught a real bug in the
first cut of `raw_wick`: its rolling swing-high/low window included the
*current* bar, which makes `close > recent_swing_high` structurally
impossible on any real OHLC bar (close can never exceed its own bar's
high) — so `raw_wick` fired **zero trades, ever**, on real data, even
though a synthetic-data check had (misleadingly) suggested it worked.
Fixed by excluding the current bar from the window (`shift(1)` before
the rolling max/min) — a real check against one year of 2024 NDX100
1-minute data (`_real_ndx_trades_per_day.py`, not committed — ad hoc)
confirmed the fix: 800 random param draws averaged a median of ~1.5
trades/day, with 34% of draws clearing >=3/day, confirming the "2-3
times a day" target is genuinely achievable on real price action.
`MIN_TRADES_PER_PERIOD`/`MIN_TRADES_TOTAL` and the score's trade-count
term were raised to match (see below and the module docstring) so the
search is rewarded for finding higher-frequency setups all the way up
toward that real target, not stopping at a floor set before real
frequency was ever measured.

Default asset universe (`precompute.py`'s `INDEX_ASSETS`): NDX100,
SPX500, US30, GER40, FRA40, UK100, JPN225.

**Data depth caveat — read before trusting every asset equally.** Per
RCTBE's own cache provenance notes, NDX100/SPX500/US30/GER40/JPN225 have
~10 years of 1-minute history; **FRA40 and UK100 only have ~3.3 years**
(2023-2026 in the shared Dukascopy cache). A 5-period walk-forward split
on those two divides a much shorter, more recent-only sample — treat any
FRA40/UK100 result with real caution, same as RCTBE flags for USDJPY's
shorter cache. `precompute.py` prints each asset's actual date range and
year count so this is visible every run, not just in this README.

## Validation standard (matches RCTBE, not a looser bar)

1. **Accept gate** (built into the search): profitable in >=4 of 5
   chronological walk-forward periods, >=500 trades total, >=100/period
   (raised 2026-08-25 in two steps, 100/8 -> 200/40 -> 500/100 — see
   `mean_reversion.py`'s docstring; the second raise follows a real
   trades/day measurement on 2024 NDX100 data, not a guess. This is a
   statistical-validity floor, not the 2-3/day target itself — the
   score's trade-count term is what rewards approaching that).
2. **`MR_IS_ONLY=1`** (default ON): every asset is truncated to periods
   1-3 (first 60% of history) *before the search ever sees it*. This is
   the fix RCTBE's own project history calls the "new methodology"
   (`RCTBE_SYSTEM_BRIEFING.md` Section 5) — without it, a candidate
   could get into the top pool partly because it looked good on periods
   4-5, the SAME bars Gate 2 later calls a blind holdout. With it on,
   Gate 2's later check is genuinely blind from the start. Since this is
   a fresh project with no existing "old methodology" pool to stay
   comparable with, there's no reason to default to the leakier version.
3. **Gate 2 — blind OOS holdout** (`gate2_holdout.py`): PF_OOS >= 1.05,
   n_trades_OOS >= 30, degradation_ratio >= 0.40.
4. **Gate 3 — plateau check** (`plateau_check.py`): nudge each swept
   numeric parameter ±1 grid step; >=70% of neighbors still healthy =
   PASS, >=40% = MIXED, else FAIL.
5. **Historical-replay prop-firm check** (`candidate_report.py`): every
   calendar Monday spanning the candidate's trade history replayed
   against the real 1-step challenge rules (`ftmo_challenge_rules.py`).
   Reports BOTH an overall average pass rate across all weekly cohorts
   AND a **rolling worst 24-month-window pass rate**
   (`rolling_worst_window_pass_rate`, ported from RCTBE's `_ftmo_2step_
   simulator.py`) — RCTBE's own real risk-selection metric. The two are
   not interchangeable: an average can look strong while hiding a
   cluster of failures concentrated in one bad stretch (a COVID-2020-
   style regime), which is exactly what the worst-window lens exists to
   catch instead. `candidate_report.py` labels both explicitly so this
   isn't a trap for anyone skimming the output. Still NOT the full
   day-block-bootstrap Monte Carlo RCTBE's own final go/no-go decisions
   use — see that script's own docstring for exactly what this does and
   doesn't prove.

Nothing here is "validated" by running the search alone — same as every
RCTBE strategy, a candidate is a search result until it's cleared 2-4
above, and even then a real deployment decision needs the full pooled-
portfolio Monte Carlo this repo deliberately doesn't reimplement.

## How to run it

```powershell
# 1. Precompute (fast — no HMM refit, just OHLCV + ATR + fractals)
py -3 precompute.py ALL

# 2. Sanity-check the pipeline on synthetic data (optional but cheap)
py -3 selftest.py

# 3. Run the search
py -3 mean_reversion.py
$env:MR_ITERATIONS = "100000"          # default 20,000
$env:MR_ASSET_FILTER = "NDX100,GER40"  # optional, restrict assets

# 4. Check the top candidates on a given asset
py -3 candidate_report.py --asset NDX100 --rank 0 1 2 3 4
```

Env vars: `MR_OUTPUT_DIR` (default `meanreversion_output`),
`MR_ITERATIONS` (default 20,000), `MR_ASSET_FILTER`, `MR_IS_ONLY`
(default `1`), `MR_DATA_DIR`, `MR_DUKASCOPY_ROOT` (default
`C:\Users\leach\OPRrsitomt5\data\dukascopy`, same cache RCTBE reads).
Checkpointed every 500 iterations or 30 minutes — safe to stop and
resume the search at any time.

See `VPS_DEPLOYMENT.md` for running this unattended on the VPS.

## Repo layout

| File | What it is |
|---|---|
| `precompute.py` | 1-min OHLCV + ATR + swing fractals + daily-open anchor, per asset. |
| `mean_reversion.py` | The strategy + random-search engine — `SPACE`, `generate_signals`, `backtest_signals`, `backtest_multiperiod`, checkpointing, `main()`. |
| `gate2_holdout.py` | Gate 2 — blind OOS holdout. |
| `plateau_check.py` | Gate 3 — parameter-plateau robustness check. |
| `candidate_report.py` | One-shot report: Gate 2 + Gate 3 + historical-replay prop-firm check for a given candidate. |
| `selftest.py` | Plumbing self-test on synthetic data. |
| `cost_table.py` | Per-asset spread/commission, copied from RCTBE's `layer2_cost_table.py`. |
| `ftmo_challenge_rules.py` | Real 1-step/2-step FTMO challenge simulation, copied from RCTBE's `_ftmo_challenge_rules.py`. |

## Version control

This repo tracks code only — no precomputed `.parquet` caches or search
output are committed (see `.gitignore`); precompute is fast enough to
regenerate on any machine that has the shared Dukascopy cache mounted,
unlike RCTBE's own HMM precompute (which IS committed there because a
10-15 min walk-forward refit per asset is worth caching). If a specific
`meanreversion_output/top_strategies.json` result is worth preserving
long-term, commit that one file deliberately.
