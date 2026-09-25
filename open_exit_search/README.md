# open_exit_search — handoff steps 3 + 5 (pre-registered 2026-09-25, before any results)

**Component searched:** the opening/expansion trade + FVG continuation legs only. The reversal trades are left out; they are ~breakeven over 10 years.

**Step 3, exit management of the opening trade:**
- break-even after X×SL
- half off at X×SL
- trailing stop X×SL
- no-TP runner
- time stop (60 / 90 / 120 / 180 / 240 min / session close)
- 5-min opening candle vs 1-min
- same management for the FVG legs or not

**Step 5, other opens:** the same mechanic at other cash opens.

| instrument | open (local) | data | cost (pts round trip) |
|---|---|---|---|
| NQ (MNQ futures) | 09:30 New York | Databento 2016-09.. | 1.24 |
| NDX100 CFD | 09:30 New York | Dukascopy 2016.. | 1.83 |
| US30 CFD | 09:30 New York | Dukascopy 2016.. (2019 mostly missing) | 2.78 |
| SPX500 CFD | 09:30 New York | Dukascopy 2016.. | 0.60 |
| GER40 CFD | 09:00 Frankfurt | Dukascopy 2016.. | 3.39 |
| FRA40 CFD | 09:00 Paris | Dukascopy 2023-03.. only | 1.61 |
| UK100 CFD | 08:00 London | Dukascopy 2023-03..2026-07 only | 1.85 |

CFD costs are the FTMO-demo spreads measured in RCTBE (`layer2_cost_table.py`).
- **Point knobs** are NDX points. For other instruments they are multiplied by the ratio of the instrument's median first-2h range to NDX100's on the search window (`results/scales.json`). NQ/NDX100 use the live points unchanged.
- **Knob `dyn`:** also multiplies by the trailing-60-day median range / its search-window median (known before the day).
- **Engine check:** `check_live.py` reproduces the live NDX opening+leg trades of `../rctbe_combo_experiment/cfd_fv125_trades_10y.csv` exactly. The only differences are 3 days missing from the older RTH file.

## Splits (same as flex_search)
- **Search:** 2024-09-16..2026-06-30. One rotating Monday-week per month is held out (same business-day weeks for every instrument).
- **Final:** 2026-07-01 to the end of data.
- **Stress:** 2016-09-15..2023-12-29, report only. FRA40 and UK100 have no stress data, so they cannot pass check B.

## Stages (`oe_search.py all`)
0. **Ablation (no selection).** `LIVE_COMP` (live opening trade + legs) and one-knob variants, on every instrument and split: `results/stage0_ablation.csv`. This is the direct answer to step 3.
1. **Random search.** Configs per instrument, scored on TRAIN days.
   - Score = t-stat of daily R.
   - Needs ≥ 1 trade per 4 days and both TRAIN halves positive.
2. **Robustness.** One-step neighbours of the top configs are re-scored; `robust = min(fit, mean neighbour fit)`.
3. **Top 30 robust per instrument + baselines on every split.** Pre-registered checks:
   - **A** held-out R/day > 0 and ≥ 40% of TRAIN R/day
   - **B** stress (2016–23) R/day ≥ 0 and ≥ the best baseline's stress R/day (needs ≥ 250 stress days)
   - **C** final R/day ≥ 0
   - **ALL** = A & B & C (**D**, beating the baseline on the held-out weeks, is reported only)
4. **FTMO 2-step on one $100k account.** Run on configs passing ALL (≤ 2 per CFD instrument) plus the NDX100 baselines.
   - Tested alone and in 2–3-instrument combos.
   - Settings: risk $500/750/1000/1500 per R, $2k account day breaker on/off, near sizing on/off.
   - Rules: daily loss on equity incl. MAE, static $90k floor, 4 min days.
   - Ranked on pass-within-60-trading-days of TRAIN weekly cohorts. The top 60 + baselines are reported on search / stress cohorts and held-out / final $ per day.
   - A second, report-only pool (`stage4_*_AC.csv`) uses configs passing A and C but not necessarily B, i.e. edges that only exist in the 2024+ regime. Treat anything from it as regime-dependent.

**Note:** the live baseline itself fails check A on NDX100 (held-out weeks negative, as in the earlier FTMO2 search). The held-out set is only ~91 days, so treat A as a filter against in-sample luck, not as proof.

## Run
**Laptop, once:** build the CFD files with `CFDData\build_dukas_1min.py <NAME> <year>` then `combine`. Already done 2026-09-25 for US30, SPX500, GER40, FRA40, UK100.

**VPS:**
1. `git pull`.
2. Copy `CFDData\{us30,spx500,ger40,fra40,uk100}_dukascopy_1min.parquet` to the same paths.
3. `.\start_open_exit_search.ps1`
4. Wait until `results\DONE` exists (about 2–3 h on 4 workers).
5. Commit `results\`.
