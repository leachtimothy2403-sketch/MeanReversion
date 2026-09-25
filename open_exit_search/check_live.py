"""Sanity: LIVE_COMP on NDX100 must reproduce rctbe_combo_experiment/cfd_fv125_trades_10y.csv open+leg trades."""
import numpy as np, pandas as pd
from oe_data import build; from oe_engine import sim; from oe_space import LIVE
g = build("NDX100"); rows = []
for i, d in enumerate(g["days"]):
    for t in sim(g["blocks"][i], LIVE, 1.0, 1.83): rows.append(dict(day=d, kind=t["kind"], R=t["R"]))
T = pd.DataFrame(rows); T["y"] = T.day.str[:4]
ref = pd.read_csv("../rctbe_combo_experiment/cfd_fv125_trades_10y.csv"); ref = ref[ref.kind != "rev"]; ref["y"] = ref.day.str[:4]
print("engine:", T.groupby("kind").R.agg(["size", "sum"]).round(1).to_dict())
print("ref   :", ref.groupby("kind").R.agg(["size", "sum"]).round(1).to_dict())
print(pd.DataFrame({"eng": T.groupby("y").R.sum(), "ref": ref.groupby("y").R.sum()}).round(1).T.to_string())
