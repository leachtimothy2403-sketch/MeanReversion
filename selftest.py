#!/usr/bin/env python3
"""
MeanReversion — plumbing self-test, on SYNTHETIC data (a deliberately
mean-reverting random walk, reset toward a day-open anchor). This does
NOT tell you anything about whether the strategy has a real edge on real
markets — it only confirms the pipeline (sample_params -> generate_
signals -> backtest_signals/backtest_multiperiod -> Gate 2 -> Gate 3 ->
historical-replay check) runs end to end without exceptions on this
machine's Python/pandas/numpy versions, before spending real time on
precompute.py against the real Dukascopy cache.

Usage:
    py -3 selftest.py
"""
import random

import numpy as np
import pandas as pd

import precompute as pre
import mean_reversion as mr
import gate2_holdout as g2
import plateau_check as pc
from candidate_report import historical_replay_check


def build_synthetic(n_days: int = 60, bars_per_day: int = 800, seed: int = 0) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows, idx = [], []
    t0 = pd.Timestamp("2024-01-02", tz="UTC")
    price = 15000.0
    for d in range(n_days):
        day_start = t0 + pd.Timedelta(days=d)
        if day_start.weekday() >= 5:
            continue
        day_open = price
        for b in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=b)
            shock = np.random.normal(0, 4.0)
            if np.random.rand() < 0.003:
                shock += np.random.choice([-1, 1]) * np.random.uniform(30, 80)
            pull = (day_open - price) * 0.01   # deliberate mean-reversion bias, see module docstring
            o = price
            move = shock + pull
            c = price + move
            # 2026-08-25 fix: high/low must be derived from BOTH open and
            # close (then widened by independent wick noise), not from
            # open alone with close drawn separately afterward — the
            # earlier version let close land outside [low, high], an
            # internally-inconsistent OHLC bar no real market ever
            # produces. That silently made bos_mode="raw_wick" look like
            # it fired plausibly on this synthetic data when the real
            # underlying comparison (close beyond a bar's own wick) is
            # structurally impossible on genuine OHLC — see
            # mean_reversion.py's raw_wick branch comment for the full
            # story and the real-data check that caught it.
            h = max(o, c) + abs(np.random.normal(0, 2.0))
            l = min(o, c) - abs(np.random.normal(0, 2.0))
            price = c
            rows.append((o, h, l, c, 1000 + np.random.rand() * 100))
            idx.append(ts)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))

    for w in pre.ATR_WINDOWS:
        df[f"atr_{w}"] = pre._atr(df, w)
    day_changed = pd.Series(df.index.date, index=df.index).ne(pd.Series(df.index.date, index=df.index).shift(1))
    df["day_open"] = df["open"].where(day_changed).ffill()
    K = pre.K
    is_fl = df["low"] == df["low"].rolling(2 * K + 1, center=True).min()
    is_fh = df["high"] == df["high"].rolling(2 * K + 1, center=True).max()
    df["swing_low_confirmed"] = df["low"].where(is_fl).shift(K)
    df["swing_high_confirmed"] = df["high"].where(is_fh).shift(K)
    return df.dropna(subset=[f"atr_{w}" for w in pre.ATR_WINDOWS])


def main():
    print("Building synthetic dataset...")
    df = build_synthetic()
    print(f"  {len(df):,} bars, {df.index[0]} .. {df.index[-1]}\n")

    mr.COST_TABLE["TESTIDX"] = 1.0

    print("Stage 1/4 — sampling 300 random parameter draws through generate_signals "
          "+ backtest_signals + backtest_multiperiod + get_trade_records...")
    n_ok, n_exc = 0, 0
    for trial in range(300):
        p = mr.sample_params(["TESTIDX"])
        p["asset"] = "TESTIDX"
        try:
            sigs = mr.generate_signals(df, p)
            mr.backtest_signals(sigs, df, cost=1.0)
            mr.get_trade_records(df, p, cost=1.0)
            mr.backtest_multiperiod(df, p)
            n_ok += 1
        except Exception as e:
            n_exc += 1
            print(f"  [FAIL] trial {trial}: {type(e).__name__}: {e}\n    params={p}")
    print(f"  {n_ok}/300 ran cleanly, {n_exc} raised an exception.\n")
    if n_exc:
        raise SystemExit(f"{n_exc} trials raised — fix before running a real search.")

    print("Stage 2/4 — finding one candidate that clears the accept gate "
          f"(profitable in >=4/5 periods, >={mr.MIN_TRADES_TOTAL} trades total, "
          f">={mr.MIN_TRADES_PER_PERIOD}/period)...")
    best = None
    for trial in range(500):
        p = mr.sample_params(["TESTIDX"])
        p["asset"] = "TESTIDX"
        full = mr.backtest_multiperiod(df, p)
        if full is not None and (best is None or full["score"] > best[1]["score"]):
            best = (p, full)
    if best is None:
        print("  No candidate cleared the accept gate in 500 samples on this synthetic "
              "dataset — not necessarily a bug (synthetic data is small/short), but "
              "Stages 3-4 below need one to run. Try increasing n_days in build_synthetic().")
        return
    p, full = best
    row = {**p, **full}
    print(f"  Found: score={full['score']} avg_pf={full['avg_profit_factor']} "
          f"n_trades={full['n_trades_total']}\n")

    print("Stage 3/4 — Gate 2 (blind OOS holdout) + Gate 3 (plateau check)...")
    gate2 = g2.gate2_check(row, df)
    plateau = pc.plateau_check(row, df)
    print(f"  Gate 2: {gate2['verdict']}   Gate 3: {plateau['verdict']} "
          f"({plateau['n_healthy']}/{plateau['n_neighbors']} neighbors healthy)\n")

    print("Stage 4/4 — every-Monday historical-replay prop-firm check...")
    replay = historical_replay_check(row, df, risk_pct=0.0075)
    print(f"  {replay}\n")

    print("ALL STAGES RAN WITHOUT EXCEPTION — the pipeline is wired correctly.")
    print("(Results above are from SYNTHETIC, deliberately mean-reverting data — they")
    print(" say nothing about whether a real edge exists. Run precompute.py against")
    print(" real Dukascopy data next.)")


if __name__ == "__main__":
    main()
