"""Per-instrument day blocks: 1-min bars from the local cash open to the local close (max 510 bars), NaN padded,
in-session gaps forward-filled as flat zero-volume bars. Cache: cache/oe_<INST>.npz (regenerable, gitignored).
Env override for a data file: OE_<INST>_PARQUET."""
import os, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from oe_space import INST
HERE = os.path.dirname(os.path.abspath(__file__))
MAXB = 510

def path(name):
    return os.environ.get(f"OE_{name}_PARQUET", os.path.normpath(os.path.join(HERE, INST[name]["file"])))

def build(name, start="2016-01-01", end="2026-09-30"):
    fn = os.path.join(HERE, "cache", f"oe_{name}_{os.path.getsize(path(name))}.npz")   # new source file -> new cache
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True); return {k: z[k] for k in z.files}
    cfg = INST[name]; tz = ZoneInfo(cfg["tz"])
    df = pd.read_parquet(path(name), columns=["open", "high", "low", "close", "volume"]).sort_index()
    df = df[~df.index.duplicated()]
    idx = df.index; arr = df.to_numpy("float64")
    days, blocks, rclose, omin = [], [], [], []
    for D in pd.bdate_range(start, end):
        dn = D.date()
        a = pd.Timestamp(dn.year, dn.month, dn.day, *cfg["open"], tz=tz).tz_convert("UTC")
        z = pd.Timestamp(dn.year, dn.month, dn.day, *cfg["close"], tz=tz).tz_convert("UTC")
        L = min(MAXB, int((z - a).total_seconds() // 60))
        j0 = idx.searchsorted(a); j1 = idx.searchsorted(z)
        if j0 >= len(idx) or idx[j0] != a or (j1 - j0) < 0.75 * L: continue
        blk = np.full((MAXB, 5), np.nan)
        mins = ((idx[j0:j1] - a).total_seconds() // 60).astype(int); m = mins < L
        blk[mins[m]] = arr[j0:j1][m]
        for k in range(1, L):
            if np.isnan(blk[k, 0]): blk[k, :4] = blk[k - 1, 3]; blk[k, 4] = 0.0
        days.append(str(dn)); blocks.append(blk.astype("float32")); rclose.append(blk[L - 1, 3])
        omin.append(a.hour * 60 + a.minute)
    B = np.stack(blocks)
    rng120 = np.nanmax(B[:, :120, 1], axis=1) - np.nanmin(B[:, :120, 2], axis=1)
    res = dict(days=np.array(days), blocks=B, rth_close=np.array(rclose), open_utc_min=np.array(omin), rng120=rng120)
    np.savez_compressed(fn, **res)
    return res

if __name__ == "__main__":
    import sys, time
    for n in (sys.argv[1:] or list(INST)):
        t = time.time(); r = build(n)
        print(n, len(r["days"]), r["days"][0], r["days"][-1], "median 2h range", round(float(np.median(r["rng120"][-400:])), 1), round(time.time() - t), "s", flush=True)
