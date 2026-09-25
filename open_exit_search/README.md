# open_exit_search — handoff steps 3 + 5 (pre-registered 2026-09-25, before any results; 10-hour version)

**Component searched:** the opening/expansion trade + FVG continuation legs only. The reversal trades are left out; they are ~breakeven over 10 years.

**Step 3, exit management:**
- break-even after X×SL
- half off at X×SL
- trailing stop X×SL
- no-TP runner
- time stop (60 / 90 / 120 / 180 / 240 min / session close)
- leg management same as the opening trade or fixed

**Entry variants and filters:**
- 1- vs 5-min opening candle
- pullback limit entry (30/50/70% of the candle, valid 5/15/30 min)
- overnight gap with / against the candle (minimum size)
- candle body ≥ 30/50/70% of its range
- previous-day range filter
- skip Mondays / Fridays
- MA100 switch
- dynamic point scaling

Filters are sampled "off" half the time. The gap and body filters gate the whole day (opening trade and legs); the VWAP filter stays opening-trade-only, as in live.

**Step 5, other opens:**

| instrument | open (local) | data | cost (pts round trip) |
|---|---|---|---|
| NQ (MNQ futures) | 09:30 New York | Databento 2016-09.. | 1.24 |
| NDX100 CFD | 09:30 New York | Dukascopy 2016.. | 1.83 |
| US30 CFD | 09:30 New York | Dukascopy 2016.. (2019 mostly missing) | 2.78 |
| SPX500 CFD | 09:30 New York | Dukascopy 2016.. (x100 quote 2023-04..2026-06 fixed in the builder) | 0.60 |
| GER40 CFD | 09:00 Frankfurt | Dukascopy 2016.. | 3.39 |
| FRA40 CFD | 09:00 Paris | Dukascopy 2023-03.. only (short track only) | 1.61 |
| UK100 CFD | 08:00 London | Dukascopy 2023-03..2026-07 only (short track only) | 1.85 |

- **Costs:** CFD costs are the FTMO-demo spreads measured in RCTBE.
- **Point scaling:** point knobs are NDX points × (instrument median first-2h range / NDX100's, search window). NQ/NDX100 are unscaled.
- **Engine check:** `check_live.py` reproduces the live NDX opening+leg trades exactly.

## Two tracks, 4-fold held-out weeks
- **Short track:** 2024-09-16..2026-06-30, the same window as flex_search.
  - Fitness = t-stat of daily R on the fold's TRAIN days.
  - Needs ≥ 1 trade per 4 days and both halves > 0.
- **Long track:** 2016-09-15..2026-06-30, for instruments with 2016+ data.
  - Fitness = min(t pre-2024, t 2024+) on the fold's TRAIN days.
  - Each regime needs R > 0 and ≥ 1 trade per 4 days. The config must work in both regimes.
- **Folds:** fold f holds out the ((month# + f) mod 4 + 1)-th Monday-week of every month. Short fold 0 = the flex_search held-out weeks.
  - Each fold selects on its own TRAIN days and is checked only on its own held-out days, so there is no leakage.
  - Long-track held-out ≈ 520 days per fold, short ≈ 91.
- **Final:** 2026-07-01 to the end of data. **Stress:** 2016-09-15..2023-12-29.

## Stages (`oe_search.py all`)
0. **Ablation.** LIVE_COMP vs ~50 one-knob variants, every instrument, every split: `results/stage0_ablation.csv`. This is the direct answer to step 3.
1. **s1.** Random configs per instrument (short 50k, long 35k); fitness for all 4 folds from one daily-R pass.
2. **s2.** Union of each fold's top configs (short 150, long 60 per fold); one-step neighbours (16 / 12); `robust_f = min(fit_f, mean neighbour fit_f)`.
3. **s3.** Each fold's top 15 robust configs + baselines on every split. Checks:
   - **A:** in every fold that selected the config, held-out R/day > 0 and ≥ 40% of that fold's TRAIN R/day. Long track: also held-out R/day ≥ 0 in both regimes.
   - **B (short only):** 2016–23 stress R/day ≥ 0 and ≥ the best baseline's (needs ≥ 250 stress days). The long track has 2016–23 inside its training data, so its out-of-sample evidence is the held-out weeks of both regimes + final.
   - **C:** final R/day ≥ 0.
   - **ALL** = A & B & C. **AC** = A & C (report-only pool, regime-dependent).
4. **FTMO 2-step on one $100k account.** ALL-passing configs of both tracks (≤ 3 per instrument and track) + NDX100 baselines.
   - Combos: singles; pairs of different instruments; 3- and 4-instrument combos of the best config per instrument.
   - Settings: uniform risk $500/750/1000/1500 × $2k breaker on/off × near sizing on/off, plus mixed per-strategy risk ($500 / $1,000 each, breaker + near) for combos.
   - Ranked on pass-within-60 of weekly cohorts on the short search window (in-sample for short-track configs, which already passed the checks). The top 80 + baselines are reported on 2016–23 cohorts and held-out / final $ per day. The same is repeated for the AC pool (`*_AC.csv`).
5. **FundedNext Futures Flex 100K for NQ.** NQ configs (ALL and AC pools, both tracks) + NQ baselines.
   - $250 / $375 / $500 per trade (whole MNQ), breaker $500 / $1,000 / off.
   - Rules: $5k target with the 40% consistency rule, $2.5k EOD trailing loss locking at $100,100, intraday check with MAE.
   - Cohorts capped at 250 days. `results/stage5_flex_nq.csv`.

**Note:** the live component itself fails check A on NDX100 short fold 0 (held-out weeks negative, as in the earlier FTMO2 search). Treat A as a filter against in-sample luck, not as proof.

## Run
- **Laptop (done 2026-09-25):** `CFDData\build_dukas_1min.py <NAME> <year>` then `combine` for US30, SPX500, GER40, FRA40, UK100.
- **VPS:**
  1. `git pull`.
  2. Copy the five `CFDData\*_dukascopy_1min.parquet` files.
  3. `.\start_open_exit_search.ps1` (`-Workers N` if more cores).
  4. The log shows an ETA per stage-1 run. Everything is resumable.
  5. When `results\DONE` exists, commit `results\`.
