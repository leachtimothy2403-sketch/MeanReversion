"""Build one 1-min parquet per index CFD from the Dukascopy daily cache (for open_exit_search).
Usage: python build_dukas_1min.py <NAME> <year|combine>
NAME -> Dukascopy symbol: US30=USA30IDXUSD, SPX500=USA500IDXUSD, GER40=DEUIDXEUR, FRA40=FRAIDXEUR, UK100=GBRIDXGBP
Output: CFDData/<name>_dukascopy_1min.parquet (UTC index, 05:00-21:30 UTC only, bars with volume > 0)."""
import sys, glob, os, pandas as pd
SYM = dict(US30="USA30IDXUSD", SPX500="USA500IDXUSD", GER40="DEUIDXEUR", FRA40="FRAIDXEUR", UK100="GBRIDXGBP", NDX100="USATECHIDXUSD")
name, arg = sys.argv[1], sys.argv[2]
SRC = os.path.join(os.path.expanduser(os.environ.get("MR_DUKAS_ROOT", "~/mnt/dukascopy")), SYM[name])
OUT = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(OUT, "parts_" + name.lower()); os.makedirs(P, exist_ok=True)
if arg == "combine":
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(P, "*.parquet")))]).sort_index()
    df = df[~df.index.duplicated()]
    if name == "SPX500":   # Dukascopy USA500IDXUSD is quoted x100 from 2023-04-10 to 2026-06-26 -> back to index points
        m = df["close"] > 20000; df.loc[m, ["open", "high", "low", "close"]] = df.loc[m, ["open", "high", "low", "close"]] / 100.0
        print("SPX500: rescaled", int(m.sum()), "x100 bars")
    df.to_parquet(os.path.join(OUT, f"{name.lower()}_dukascopy_1min.parquet")); print("combined", name, len(df), df.index.min(), df.index.max())
else:
    fr = []
    for f in sorted(glob.glob(os.path.join(SRC, f"{arg}*.parquet"))):
        d = pd.read_parquet(f); d = d[d["volume"] > 0][["open", "high", "low", "close", "volume"]]
        m = d.index.hour * 60 + d.index.minute; fr.append(d[(m >= 300) & (m <= 1290)])
    if not fr: print(name, arg, "no files"); sys.exit(0)
    df = pd.concat(fr).sort_index(); df = df[~df.index.duplicated()]
    df.astype("float32").to_parquet(os.path.join(P, f"{arg}.parquet")); print(name, arg, len(fr), "files", len(df), "rows")
