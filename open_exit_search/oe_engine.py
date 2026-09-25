"""Opening/expansion trade + FVG continuation legs with exit management, one day block -> trades.

Opening trade: the first `entry_tf`-minute candle (open of bar 0, close of bar tf-1, high/low over it); direction =
its colour (doji -> no trade); optional VWAP filter (close on the right side of the candle's typical price);
big candle (range > big_thr, x2.2 for 5-min candles) -> big_sl/big_tp else std_sl/std_tp; entry at the candle close.
FVG legs (unchanged live rule): from bar max(2, tf), every bar that prints a fresh 3-candle FVG in the opening
direction with gap >= leg_gap adds a leg at its close; the streak ends on the first bar without one; max leg_max
legs, entries only before leg_window minutes. Legs trade even when the VWAP filter rejects the opening trade (live).
Exit management (opening trade; legs too when legs_mgmt == "same", multiples of the leg's own stop):
  be_at    move the stop to entry once the best price has moved be_at x SL in favour
  partial  take half off at partial x SL
  trail    stop trails the best price by trail x SL
  TP 0     no take-profit (runner)
  time_stop close everything at the close of bar time_stop-1 (999 = session close)
Per bar: stop first, then TP, then partial (conservative); stop updates use the bar's extreme and apply from the
next bar. R = (realized points - cost) / initial stop points, size-weighted; mae_R = worst adverse excursion of the
full position / stop; tp_R = net R if the full TP is hit (None without TP)."""
import numpy as np

def sim(b, P, u, cost):
    c = b[:, 3]
    lv = np.flatnonzero(~np.isnan(c))
    if len(lv) < 30 or lv[0] != 0: return []
    lastb = int(lv[-1]); T = lastb + 1 if P["time_stop"] == 999 else min(P["time_stop"], lastb + 1)
    last = T - 1; tf = P["entry_tf"]
    if last <= tf: return []
    o, h, l = b[:, 0].astype(float), b[:, 1].astype(float), b[:, 2].astype(float); c = c.astype(float)
    ec = c[tf - 1]; eh = h[:tf].max(); el = l[:tf].min()
    od = 1 if ec > o[0] else (-1 if ec < o[0] else 0)
    if od == 0: return []
    pos, done = [], []
    def new(kind, e, eb, slp, tpp, mgmt):
        m = mgmt
        return dict(kind=kind, e=e, eb=eb, slp=slp, stop=e - od * slp, tp=(e + od * tpp) if tpp > 0 else None,
                    tpp=tpp, size=1.0, real=0.0, best=e, mae=0.0,
                    be=P["be_at"] * slp if m else 0.0, pl=(e + od * P["partial"] * slp) if (m and P["partial"] > 0) else None,
                    tr=P["trail"] * slp if m else 0.0)
    ok = True
    if P["vwap"]:
        vw = (eh + el + ec) / 3.0; ok = (ec > vw) == (od > 0)
    if ok:
        thr = P["big_thr"] * u * (2.2 if tf > 1 else 1.0)
        big = (eh - el) > thr
        slp, tpp = ((P["big_sl"], P["big_tp"]) if big else (P["std_sl"], P["std_tp"]))
        pos.append(new("open", ec, tf - 1, slp * u, tpp * u, True))
    legs_alive = P["legs"]; nlegs = 0; lw = min(P["leg_window"], last)
    lmg = P["legs_mgmt"] == "same"
    for j in range(tf, last + 1):
        hj, lj, cj = h[j], l[j], c[j]
        keep = []
        for p in pos:
            adv = (p["e"] - lj) if od > 0 else (hj - p["e"])
            if adv > p["mae"]: p["mae"] = adv
            if (lj <= p["stop"]) if od > 0 else (hj >= p["stop"]):
                x = p["stop"]
            elif p["tp"] is not None and ((hj >= p["tp"]) if od > 0 else (lj <= p["tp"])):
                x = p["tp"]
            elif j == last:
                x = cj
            else:
                if p["pl"] is not None and p["size"] == 1.0 and ((hj >= p["pl"]) if od > 0 else (lj <= p["pl"])):
                    p["real"] += 0.5 * (p["pl"] - p["e"]) * od; p["size"] = 0.5
                fav = hj if od > 0 else lj
                if (fav - p["best"]) * od > 0: p["best"] = fav
                gain = (p["best"] - p["e"]) * od
                if p["be"] > 0 and gain >= p["be"]:
                    p["stop"] = max(p["stop"], p["e"]) if od > 0 else min(p["stop"], p["e"])
                if p["tr"] > 0:
                    t = p["best"] - od * p["tr"]
                    p["stop"] = max(p["stop"], t) if od > 0 else min(p["stop"], t)
                keep.append(p); continue
            pts = p["real"] + p["size"] * (x - p["e"]) * od
            done.append(dict(kind=p["kind"], eb=p["eb"], xb=j, R=(pts - cost) / p["slp"], mae_R=p["mae"] / p["slp"],
                             tp_R=((p["tpp"] - cost) / p["slp"]) if p["tp"] is not None else None))
        pos = keep
        if not pos and (not legs_alive or j >= lw or nlegs >= P["leg_max"]): break
        if legs_alive and j >= 2 and j >= tf and j < lw and j < last:
            gap = (lj - h[j - 2]) if od > 0 else (l[j - 2] - hj)
            if gap <= 0 or gap < P["leg_gap"] * u:
                legs_alive = False
            elif nlegs < P["leg_max"]:
                nlegs += 1; pos.append(new("leg", cj, j, P["leg_sl"] * u, P["leg_tp"] * u, lmg))
    return done
