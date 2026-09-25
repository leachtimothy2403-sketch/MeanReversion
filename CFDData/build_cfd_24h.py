"""Build one 24h 1-min parquet for NDX100 CFD (Dukascopy USATECHIDXUSD daily cache) for flex_search --market cfd.
Usage: python build_cfd_24h.py <year|all> ; 'combine' merges parts -> ndx100_dukascopy_1min.parquet"""
import sys, glob, os, pandas as pd
SRC = os.path.expanduser(os.environ.get("MR_DUKAS_DIR", "~/mnt/OPRrsitomt5/data/dukascopy/USATECHIDXUSD"))
OUT = os.path.dirname(os.path.abspath(__file__))
arg = sys.argv[1]
if arg == "combine":
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, "parts", "*.parquet")))]).sort_index()
    df = df[~df.index.duplicated()]
    df.to_parquet(os.path.join(OUT, "ndx100_dukascopy_1min.parquet")); print("combined", len(df), df.index.min(), df.index.max())
else:
    fs = sorted(glob.glob(os.path.join(SRC, f"{arg}*.parquet")))
    fr = []
    for f in fs:
        d = pd.read_parquet(f); fr.append(d[d["volume"] > 0][["open", "high", "low", "close", "volume"]])
    df = pd.concat(fr).sort_index(); df = df[~df.index.duplicated()]
    df.astype("float32").to_parquet(os.path.join(OUT, "parts", f"{arg}.parquet")); print(arg, len(fs), "files", len(df), "rows")
