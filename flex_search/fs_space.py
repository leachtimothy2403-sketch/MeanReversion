"""Search space, splits, costs and challenge rules for flex_search. Pre-registered 2026-09-25 (before any results)."""
import numpy as np, pandas as pd

# ---- splits ---------------------------------------------------------------------------------------------------
SEARCH = ("2024-09-16", "2026-06-30")     # selection happens only on these days MINUS the held-out weeks
FINAL = ("2026-07-01", "2026-09-15")      # final holdout, evaluated once in stage 4
def heldout_days(days):
    """One rotating week per calendar month inside SEARCH: month #m (from Sep 2024) holds out its ((m % 4)+1)-th
    Monday-starting week (clipped to the month's last week). Returns a set of day strings."""
    d = pd.to_datetime(pd.Series([x for x in days if SEARCH[0] <= x <= SEARCH[1]]))
    out = set()
    for mi, (_, g) in enumerate(d.groupby([d.dt.year, d.dt.month])):
        wk = g.dt.to_period("W-SUN")
        weeks = list(dict.fromkeys(wk))
        pick = weeks[min(mi % 4, len(weeks) - 1)]
        out |= set(g[wk == pick].dt.strftime("%Y-%m-%d"))
    return out

# ---- costs (points per contract, round trip, incl. commission + slippage estimate) -----------------------------
COST = {"NY": 1.24, "LDN": 1.5, "ASIA": 1.75}
PV = 2.0          # MNQ $ per point
MAXC = 50         # FundedNext Flex 100K: 5 minis / 50 micros

# ---- FundedNext Futures Flex $100K (fundednext.com/futures/flex, Sept 2026) ------------------------------------
FLEX = dict(start=100000.0, target=5000.0, mll=2500.0, lock=100100.0, consistency=0.40)

# ---- per-session knob grids ------------------------------------------------------------------------------------
# point-denominated knobs are NY-base values scaled by the session's typical range (median 2h range NY 201 /
# LDN 75 / ASIA 61 points on 2024-26 data) -> scale 1.0 / 0.4 / 0.3, rounded to the 0.25 tick.
SCALE = {"NY": 1.0, "LDN": 0.4, "ASIA": 0.3}
PTS = dict(big_thr=[10, 15, 20, 25], big_sl=[25, 30, 35, 40, 50], big_tp=[60, 80, 100, 115, 130, 150],
           std_sl=[15, 20, 25, 30, 35], std_tp=[30, 35, 42, 50, 60], leg_gap=[10, 15, 20, 25], leg_sl=[25, 35, 45],
           leg_tp=[60, 90, 115, 140], min_fv=[40, 60, 80, 100, 125, 150, 200], min_bos=[0, 5, 10, 15, 20],
           disp_pts=[15, 20, 25, 30, 40], stack_gap=[10, 15, 20])
CAT = dict(offset=[0, 30], window=[60, 90, 120, 150], exit=["hard", "run"], open_trade=[True, False], vwap=[True, False],
           legs=[True, False], leg_max=[1, 2, 4, 6], rev_mode=["off", "std", "disp_fv", "mixed"], wait_open=[True, False],
           maxpos=[1, 2, 3], stack_trig=["bos", "fvg", "both"], stack_size=[1.0, 0.5], ma=[0, 50, 100, 200])
KNOBS = list(CAT) + list(PTS)

def grid(session, k):
    if k in CAT: return CAT[k]
    s = SCALE[session]
    return [round(v * s * 4) / 4 for v in PTS[k]]

def sample(session, rng):
    return {k: grid(session, k)[rng.integers(len(grid(session, k)))] for k in KNOBS}

def neighbours(session, P, rng, n):
    """n random one-knob, one-step neighbours of P."""
    out = []
    keys = [k for k in KNOBS if len(grid(session, k)) > 1]
    for _ in range(n * 3):
        k = keys[rng.integers(len(keys))]; g = grid(session, k); i = g.index(P[k])
        opts = [j for j in (i - 1, i + 1) if 0 <= j < len(g)] if k in PTS or k in ("window", "leg_max", "maxpos", "ma") else [j for j in range(len(g)) if j != i]
        Q = dict(P); Q[k] = g[opts[rng.integers(len(opts))]]
        if Q != P and Q not in out: out.append(Q)
        if len(out) >= n: break
    return out

def is_dead(P):
    """config that can never trade"""
    return (not P["open_trade"]) and (not P["legs"]) and P["rev_mode"] == "off"

# ---- challenge-level knobs (stage 3) ---------------------------------------------------------------------------
RISK = [250, 375, 500, 750, 1000]          # $ risk per full-size trade -> MNQ contracts = floor(risk / (sl * 2))
BREAKER = [500, 750, 1000, 1500, None]     # stop new entries once the day's realized loss reaches this
PROFIT_STOP = [None, 1500, 1900]           # stop new entries once the day's realized profit reaches this (consistency)
SMART = [None, "min", 0.25, 0.5]            # near-target sizing + reduced size after $105k (see fs_search.run_day)
