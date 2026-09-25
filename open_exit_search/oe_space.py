"""open_exit_search: instruments, costs, splits, knob grids, baselines. PRE-REGISTERED 2026-09-25 before any results.
Handoff steps 3 (exit management of the opening/expansion trade) and 5 (transplant opening trade + FVG legs to other opens).
Component searched = opening trade + FVG continuation legs ONLY (no reversal trades: ~breakeven over 10y)."""
import pandas as pd

# ---- instruments ----------------------------------------------------------------------------------------------
# cost = round-trip points (NQ: MNQ commission+slippage as in flex_search; CFDs: FTMO-demo measured spread,
# RCTBE layer2_cost_table). NQ/NDX100 use the live point values unchanged (scale 1.0); others are scaled by the
# ratio of their median first-120-min range to NDX100's on the search window (computed in oe_search.scales()).
INST = {
    "NQ":     dict(file="../NASDAQFuturesData/nq_continuous_1min.parquet", tz="America/New_York", open=(9, 30), close=(16, 0), cost=1.24),
    "NDX100": dict(file="../CFDData/ndx100_dukascopy_1min.parquet", tz="America/New_York", open=(9, 30), close=(16, 0), cost=1.83),
    "US30":   dict(file="../CFDData/us30_dukascopy_1min.parquet",   tz="America/New_York", open=(9, 30), close=(16, 0), cost=2.78),
    "SPX500": dict(file="../CFDData/spx500_dukascopy_1min.parquet", tz="America/New_York", open=(9, 30), close=(16, 0), cost=0.60),
    "GER40":  dict(file="../CFDData/ger40_dukascopy_1min.parquet",  tz="Europe/Berlin",    open=(9, 0),  close=(17, 30), cost=3.39),
    "FRA40":  dict(file="../CFDData/fra40_dukascopy_1min.parquet",  tz="Europe/Paris",     open=(9, 0),  close=(17, 30), cost=1.61),
    "UK100":  dict(file="../CFDData/uk100_dukascopy_1min.parquet",  tz="Europe/London",    open=(8, 0),  close=(16, 30), cost=1.85),
}
FIXED_SCALE = {"NQ": 1.0, "NDX100": 1.0}
FTMO_INST = ["NDX100", "US30", "SPX500", "GER40", "FRA40", "UK100"]   # CFDs tradable on one FTMO account (NQ = futures)

# ---- splits (same as flex_search) -------------------------------------------------------------------------------
SEARCH = ("2024-09-16", "2026-06-30")
FINAL = ("2026-07-01", "2026-09-15")      # each instrument to the end of its data (CFDs ~08-24, UK100 07-31)
STRESS = ("2016-09-15", "2023-12-29")     # report only; FRA40 / UK100 data start 2023-03 (no real stress test)
def heldout_days():
    """One rotating Monday-week per month inside SEARCH (month k holds out its ((k mod 4)+1)-th week), on the
    business-day calendar so every instrument shares the same held-out weeks."""
    d = pd.Series(pd.bdate_range(*SEARCH)); out = set()
    for mi, (_, g) in enumerate(d.groupby([d.dt.year, d.dt.month])):
        wk = g.dt.to_period("W-SUN"); weeks = list(dict.fromkeys(wk))
        out |= set(g[wk == weeks[min(mi % 4, len(weeks) - 1)]].dt.strftime("%Y-%m-%d"))
    return out

# ---- knobs ----------------------------------------------------------------------------------------------------
# PTS knobs are NDX-points (x instrument scale, x optional dynamic scale). 0 for a TP = no take-profit (runner).
# MULT knobs are multiples of the trade's own stop distance. time_stop: minutes after the open at whose bar close
# everything still open is closed (999 = session close). entry_tf: minutes in the opening candle (1 = live).
PTS = dict(big_thr=[10, 15, 20, 25], big_sl=[25, 30, 35, 40, 50], big_tp=[0, 60, 80, 100, 115, 130, 150, 200],
           std_sl=[15, 20, 25, 30, 35], std_tp=[0, 30, 42, 50, 60, 80], leg_gap=[10, 15, 20, 25], leg_sl=[25, 35, 45],
           leg_tp=[0, 60, 90, 115, 140], gap_min=[0, 20, 50, 100])
MULT = dict(be_at=[0, 0.5, 1.0, 1.5, 2.0], partial=[0, 1.0, 1.5, 2.0, 3.0], trail=[0, 1.0, 1.5, 2.0, 3.0])
CAT = dict(entry_tf=[1, 5], vwap=[True, False], time_stop=[60, 90, 120, 180, 240, 999], legs=[True, False],
           leg_max=[1, 2, 4, 6], legs_mgmt=["fixed", "same"], leg_window=[30, 60, 120], ma=[0, 100], dyn=[False, True],
           entry_mode=["close", "pb"], pb_frac=[0.3, 0.5, 0.7], pb_wait=[5, 15, 30], gap=["off", "with", "against"],
           body_min=[0, 0.3, 0.5, 0.7], volf=["off", "skip_low", "skip_high", "mid"], dow=["none", "mon", "fri"])
# entry/filter knobs added 2026-09-25 (10-h version). volf: previous session range / median of the 20 before it
# (skip_low: skip < 0.7, skip_high: skip > 1.5, mid: trade only 0.7..1.5). dow: skip Mondays / Fridays.
# Filters are sampled 'off' with probability 0.5 (else uniform over the other values) so random configs are not
# dominated by heavily filtered, low-frequency variants.
FILTER_OFF = dict(entry_mode="close", gap="off", body_min=0, volf="off", dow="none")
ORDERED = set(PTS) | set(MULT) | {"time_stop", "leg_max", "leg_window", "pb_frac", "pb_wait", "body_min"}
KNOBS = list(CAT) + list(PTS) + list(MULT)
def grid(k): return CAT.get(k) or PTS.get(k) or MULT[k]
def sample(rng):
    P = {}
    for k in KNOBS:
        g = grid(k)
        if k in FILTER_OFF:
            others = [x for x in g if x != FILTER_OFF[k]]
            P[k] = FILTER_OFF[k] if rng.random() < 0.5 else others[rng.integers(len(others))]
        else:
            P[k] = g[rng.integers(len(g))]
    return P
def neighbours(P, rng, n):
    out = []; keys = [k for k in KNOBS if len(grid(k)) > 1]
    for _ in range(n * 4):
        k = keys[rng.integers(len(keys))]; g = grid(k); i = g.index(P[k])
        opts = [j for j in (i - 1, i + 1) if 0 <= j < len(g)] if k in ORDERED else [j for j in range(len(g)) if j != i]
        Q = dict(P); Q[k] = g[opts[rng.integers(len(opts))]]
        if Q != P and Q not in out: out.append(Q)
        if len(out) >= n: break
    return out

# live opening trade + FVG legs, exactly as the live bot (without its reversal trades)
LIVE = dict(entry_tf=1, vwap=True, time_stop=120, legs=True, leg_max=6, legs_mgmt="fixed", leg_window=120, ma=0, dyn=False,
            big_thr=15, big_sl=35, big_tp=115, std_sl=25, std_tp=42, leg_gap=15, leg_sl=35, leg_tp=115, be_at=0, partial=0, trail=0,
            entry_mode="close", pb_frac=0.5, pb_wait=15, gap="off", gap_min=0, body_min=0, volf="off", dow="none")
BASELINES = {"LIVE_COMP": LIVE, "LIVE_COMP+MA100": dict(LIVE, ma=100)}

# ---- FTMO 2-step (stage 4) --------------------------------------------------------------------------------------
FTMO = dict(start=100000.0, t1=10000.0, t2=5000.0, dll=5000.0, floor=90000.0, min_days=4, H=60, min_size=10.0)
RISK = [500, 750, 1000, 1500]      # $ per 1R, per trade, every strategy in the combo
BREAKER = [2000, None]             # account level: no new entries once the day's realized P&L <= -$2,000
SMART = [None, "near"]
