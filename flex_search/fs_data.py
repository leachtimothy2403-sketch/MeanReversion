"""flex_search data prep: NQ 1-min (Databento continuous) -> per-trading-day, per-session bar arrays.

Sessions (anchor = local exchange open, DST-aware; anchor offset 0 or 30 min is a search knob):
  ASIA  09:00 Asia/Tokyo        (19:00/20:00 ET the evening before -> belongs to the NEXT CME trading day)
  LDN   08:00 Europe/London     (02:00-04:00 ET)
  NY    09:30 America/New_York
CME trading day D runs 18:00 ET on D-1 .. 17:00 ET on D. Within a day the order is ASIA -> LDN -> NY.
Each session array holds MAXB bars from the anchor (entries use the first `window` bars; 'run' exits may use the rest),
truncated at the next session's anchor+0 or 17:00 ET, whichever is first (NaN padded).
Cache: flex_search/cache/sessions_<start>_<end>.npz  (regenerable, gitignored).

Data path (env FS_MARKET selects): nq -> MR_NQ_PARQUET or ../NASDAQFuturesData/nq_continuous_1min.parquet;
cfd -> MR_CFD_PARQUET or ../CFDData/ndx100_dukascopy_1min.parquet (built by CFDData/build_cfd_24h.py).
"""
import os, numpy as np, pandas as pd
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
NYZ = ZoneInfo("America/New_York")
SESS = {"ASIA": ("Asia/Tokyo", 9, 0), "LDN": ("Europe/London", 8, 0), "NY": ("America/New_York", 9, 30)}
ORDER = ["ASIA", "LDN", "NY"]
OFFSETS = (0, 30)
MAXB = 360  # 6h of 1-min bars after the anchor (cap)

MARKET = os.environ.get("FS_MARKET", "nq")   # "nq" (Databento NQ futures) or "cfd" (Dukascopy NDX100 CFD)

def data_path():
    if MARKET == "cfd":
        return os.environ.get("MR_CFD_PARQUET", os.path.join(HERE, "..", "CFDData", "ndx100_dukascopy_1min.parquet"))
    return os.environ.get("MR_NQ_PARQUET", os.path.join(HERE, "..", "NASDAQFuturesData", "nq_continuous_1min.parquet"))

def build(start="2024-01-02", end="2026-09-15"):
    cache = os.path.join(HERE, "cache"); os.makedirs(cache, exist_ok=True)
    fn = os.path.join(cache, ("" if MARKET == "nq" else MARKET + "_") + f"sessions_{start}_{end}.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True); return {k: z[k] for k in z.files}
    df = pd.read_parquet(data_path(), columns=["open", "high", "low", "close", "volume"]).sort_index()
    df = df[~df.index.duplicated()]
    df = df[(df.index >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2)) & (df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    idx = df.index; arr = df[["open", "high", "low", "close", "volume"]].to_numpy("float64")
    pos = pd.Series(np.arange(len(idx)), index=idx)
    days = [d for d in pd.bdate_range(start, end)]
    out_days, rth_close = [], []
    blocks = {(s, o): [] for s in ORDER for o in OFFSETS}
    for D in days:
        dn = D.date()
        day_end = pd.Timestamp(dn.year, dn.month, dn.day, 17, 0, tz=NYZ).tz_convert("UTC")
        anchors = {}
        for s in ORDER:
            tz, hh, mm = SESS[s]
            if s == "ASIA":   # Tokyo morning of calendar day D == evening of D-1 in New York
                a = pd.Timestamp(dn.year, dn.month, dn.day, hh, mm, tz=ZoneInfo(tz))
            else:
                a = pd.Timestamp(dn.year, dn.month, dn.day, hh, mm, tz=ZoneInfo(tz))
            anchors[s] = a.tz_convert("UTC")
        # require the NY cash session to exist (full-ish RTH) for the day to count
        ny0 = anchors["NY"]
        i0 = idx.searchsorted(ny0);
        if i0 >= len(idx) or idx[i0] != ny0: continue
        i_close = idx.searchsorted(pd.Timestamp(dn.year, dn.month, dn.day, 16, 0, tz=NYZ).tz_convert("UTC")) - 1
        if i_close - i0 < 300: continue
        ok = True; day_blocks = {}
        for si, s in enumerate(ORDER):
            nxt = anchors[ORDER[si + 1]] if si + 1 < len(ORDER) else day_end
            for o in OFFSETS:
                a = anchors[s] + pd.Timedelta(minutes=o)
                stop = min(nxt, a + pd.Timedelta(minutes=MAXB)) if s != "NY" else min(day_end, a + pd.Timedelta(minutes=MAXB))
                j0, j1 = idx.searchsorted(a), idx.searchsorted(stop)
                blk = np.full((MAXB, 5), np.nan)
                if j0 < len(idx) and idx[j0] == a and j1 > j0:
                    seg = arr[j0:j1]; mins = ((idx[j0:j1] - a).total_seconds() // 60).astype(int)
                    m = mins < MAXB; blk[mins[m]] = seg[m]
                    # forward-fill gaps (no trade in that minute): flat bar at prior close, zero volume
                    for k in range(1, MAXB):
                        if np.isnan(blk[k, 0]) and not np.isnan(blk[k - 1, 3]) and k < (stop - a).total_seconds() // 60:
                            blk[k, :4] = blk[k - 1, 3]; blk[k, 4] = 0.0
                day_blocks[(s, o)] = blk
        for k, b in day_blocks.items(): blocks[k].append(b)
        out_days.append(str(dn)); rth_close.append(arr[i_close, 3])
    res = {"days": np.array(out_days), "rth_close": np.array(rth_close)}
    for (s, o), L in blocks.items(): res[f"{s}_{o}"] = np.stack(L).astype("float32")
    np.savez_compressed(fn, **res)
    return res

if __name__ == "__main__":
    import sys, time
    t = time.time(); r = build(*sys.argv[1:3]) if len(sys.argv) > 2 else build()
    print(len(r["days"]), r["days"][0], r["days"][-1], {k: r[k].shape for k in r if k not in ("days", "rth_close")}, round(time.time() - t), "s")
