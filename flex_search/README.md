# flex_search: FundedNext Futures Flex $100K strategy search (NQ -> MNQ)

Pre-registered 2026-09-25. Staged, resumable search across three sessions, each with its own parameters:
ASIA (09:00 Tokyo), LDN (08:00 London), NY (09:30 New York). Each session can also be switched off.

**Data** `..\NASDAQFuturesData\nq_continuous_1min.parquet` (Databento NQ continuous, not in git; copy it over), or set
`MR_NQ_PARQUET`. `fs_data.py` builds a per-session cache in `cache\` on the first run.

**Splits**
- Search window 2024-09-16..2026-06-30.
- One rotating week per month is held out: month k holds out its ((k mod 4)+1)-th week.
- Final holdout 2026-07-01..2026-09-15, looked at once in stage 4.
- 2016-2023 stress test, report only.

**Stages** (`fs_search.py`):
1. Random configs per session, scored on TRAIN days. Score: t-stat of daily R; needs ≥ 1 trade per 4 days and both halves positive.
2. Robustness: the top configs' one-step neighbours are re-scored. `robust = min(score, mean neighbour score)`.
3. Combine the top robust candidates per session (plus "session off") with the challenge knobs:
   - risk per trade: $250-1,000
   - day breaker: $500 / $750 / $1,000 / $1,500 / off
   - day profit stop: none / $1,500 / $1,900
   - smart sizing (below)

   Then simulate weekly Flex cohorts on TRAIN days.
4. Report the top combos on the held-out weeks, the contiguous search window, the final holdout and 2016-2023, plus fixed baselines: LIVE, LIVE+MA100, DISP_FV, DISP_FV+MA100.

**Knobs per session** (`fs_space.py`): anchor offset, entry window, exit mode, opening trade (+ VWAP filter, big-candle
threshold, SL/TP), standard SL/TP, FVG continuation legs (gap, SL/TP, max), reversal mode (off / std / displacement->FV /
mixed), min distance from fair value, BOS buffer, displacement size, wait-for-opening-trade, max concurrent reversal
positions, stacking trigger (BOS / FVG / both), stacked size, MA switch (off/50/100/200).
Point-valued knobs are scaled per session (NY 1.0, LDN 0.4, ASIA 0.3).

**Flex rules modelled**
- $5,000 target.
- $2,500 end-of-day trailing max loss, locking at $100,100. The intraday check uses each trade's maximum adverse excursion.
- 40% consistency rule: target = max(5,000, best day / 0.4).
- No daily loss limit, no minimum days, max 50 MNQ.

Costs per MNQ contract round trip: NY 1.24 pt, LDN 1.5, ASIA 1.75.

**Smart sizing** (stage-3 knob: off / "min" / 0.25 / 0.5):
- (a) Near the $105k target, each new trade is sized so that its take-profit just reaches $105k.
- (b) After $105k, when the consistency rule has raised the target, trades are 1 MNQ ("min") or that fraction of normal size.

**Run on the VPS** `.\start_flex_search.ps1` (defaults: 4 workers, 30,000 configs per session; roughly 1.5-3 h).
When `results\DONE` exists, commit `results\` (stage-1 raw CSVs are gitignored).
