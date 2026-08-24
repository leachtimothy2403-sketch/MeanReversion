"""
MeanReversion — trading costs, used to make the backtest reflect real
spread/commission rather than a frictionless fill.

Copied verbatim from the sister project RCTBE's `layer2_cost_table.py`
(itself copied from OPRrsitomt5\\opr_cost_table.py, FTMO-Demo, live-
measured 2026-07-24/2026-08-02) — RCTBE's copy is the source of truth for
this table; if it's updated there, mirror the change here too.
Duplicated rather than imported directly so this project stays fully
self-contained (its own repo, its own VPS deployment) and never depends
on RCTBE being checked out on whatever machine runs it — the exact same
"no cross-project reliance" pattern RCTBE's own file documents for its
own borrowing from OPRrsitomt5.

Only the 7 index assets in INDEX_ASSETS (precompute.py) are actually
used by this project today; the rest of the table is kept for provenance
and in case the asset universe is widened later, same as RCTBE's own
full table.
"""
from typing import Dict

COST_TABLE: Dict[str, float] = {
    "XAUUSD": 0.53,
    "NDX100": 1.83,
    "SPX500": 0.60,
    "EURUSD": 0.00007,
    # Added 2026-08-06 alongside the 4 new pilot assets, same source/method.
    "US30":   2.78,
    "GER40":  3.39,
    "GBPUSD": 0.00010,
    "USDJPY": 0.02,  # not broker-measured — conservative estimate per OPRrsitomt5's own table, flag if a candidate emerges here
    # Added 2026-08-06 alongside the remaining 10 cached assets, same source.
    "AUDUSD": 0.00010,
    "NZDUSD": 0.00011,
    "USDCAD": 0.00011,
    "USDCHF": 0.00014,
    "XAGUSD": 0.055,
    "UKOIL":  0.063,
    "USOIL":  0.071,
    "FRA40":  1.61,
    "UK100":  1.85,
    "JPN225": 10.00,
    # Added 2026-08-06 - first stock/crypto CFDs in the project. UNVERIFIED:
    # OPRrsitomt5 never traded these instrument types, so unlike everything
    # else in this table there is no real broker-measured cost to copy -
    # these are rough estimates only (~0.1% of price for the stocks, a
    # typical-order-of-magnitude BTC CFD spread), not measured. Flag loudly
    # if a candidate emerges on any of these before trusting its economics -
    # get a real measured spread first, same as USDJPY's caveat below but
    # more so, since that at least had a same-instrument-class precedent.
    "AAPL":   0.30,
    "XOM":    0.15,
    "WMT":    0.10,
    "DIS":    0.10,
    "BTCUSD": 40.00,
}
