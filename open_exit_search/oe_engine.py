"""Opening/expansion trade + FVG continuation legs with entry variants, filters and exit management; one day -> trades.

Opening candle: the first `entry_tf` minutes (open of bar 0, close of bar tf-1, high/low over them); direction = its
colour (doji -> nothing). Directional filters (gate the opening trade AND the legs; the day is skipped if they fail):
  body_min  candle body / candle range >= body_min
  gap       'with' / 'against': the overnight gap (open - previous session close) must point with / against the
            candle direction and be >= gap_min points; 'off' = no gap condition
VWAP filter (live behaviour, opening trade only): candle close on the right side of its typical price.
Opening entry: 'close' = at the candle close (live); 'pb' = limit order pb_frac x candle range back into the candle,
  valid pb_wait minutes, no trade if not filled; stop/target distances are measured from the fill.
Big candle (range > big_thr, x2.2 for 5-min candles) -> big_sl/big_tp else std_sl/std_tp; TP 0 = no take-profit.
FVG legs (live rule): from bar max(2, tf), every bar that prints a fresh 3-candle FVG in the opening direction with
  gap >= leg_gap adds a leg at its close; the streak ends on the first bar without one; max leg_max; before leg_window.
Exit management (opening trade; legs too when legs_mgmt == 'same', in multiples of each trade's own stop):
  be_at (stop -> entry after be_at x SL in favour), partial (half off at partial x SL), trail (stop trails the best
  price by trail x SL), time_stop (close all at the close of bar time_stop-1; 999 = session close).
Per bar: stop, then TP, then partial (conservative); stop moves use the bar's extreme and apply from the next bar.
A pullback fill that also touches the stop on its fill bar is stopped out on that bar.
Trade: kind, eb, xb, R (net of cost, size-weighted, per initial stop), mae_R, tp_R (net R at full TP or None), slp."""
import numpy as np

def sim(b, P, u, cost, gap=None):
    c = b[:, 3]
    lv = np.flatnonzero(~np.isnan(c))
    if len(lv) < 30 or lv[0] != 0: return []
    lastb = int(lv[-1]); T = lastb + 1 if P["time_stop"] == 999 else min(P["time_stop"], lastb + 1)
    last = T - 1; tf = P["entry_tf"]
    if last <= tf: return []
    o, h, l = b[:, 0].astype(float), b[:, 1].astype(float), b[:, 2].astype(float); c = c.astype(float)
    ec = c[tf - 1]; eh = h[:tf].max(); el = l[:tf].min(); rngc = eh - el
    od = 1 if ec > o[0] else (-1 if ec < o[0] else 0)
    if od == 0: return []
    if P["body_min"] > 0 and (rngc <= 0 or abs(ec - o[0]) / rngc < P["body_min"]): return []
    if P["gap"] != "off":
        if gap is None or not np.isfinite(gap) or abs(gap) < P["gap_min"] * u: return []
        gs = 1 if gap > 0 else -1
        if (P["gap"] == "with" and gs != od) or (P["gap"] == "against" and gs == od): return []
    pos, done = [], []
    def new(kind, e, eb, slp, tpp, m):
        return dict(kind=kind, e=e, eb=eb, slp=slp, stop=e - od * slp, tp=(e + od * tpp) if tpp > 0 else None,
                    tpp=tpp, size=1.0, real=0.0, best=e, mae=0.0,
                    be=P["be_at"] * slp if m else 0.0, pl=(e + od * P["partial"] * slp) if (m and P["partial"] > 0) else None,
                    tr=P["trail"] * slp if m else 0.0)
    def fin(p, j, x):
        pts = p["real"] + p["size"] * (x - p["e"]) * od
        done.append(dict(kind=p["kind"], eb=p["eb"], xb=j, R=(pts - cost) / p["slp"], mae_R=p["mae"] / p["slp"],
                         tp_R=((p["tpp"] - cost) / p["slp"]) if p["tp"] is not None else None, slp=p["slp"]))
    pend = None
    ok = True
    if P["vwap"]:
        vw = (eh + el + ec) / 3.0; ok = (ec > vw) == (od > 0)
    if ok:
        big = rngc > P["big_thr"] * u * (2.2 if tf > 1 else 1.0)
        slp, tpp = ((P["big_sl"], P["big_tp"]) if big else (P["std_sl"], P["std_tp"]))
        if P["entry_mode"] == "close":
            pos.append(new("open", ec, tf - 1, slp * u, tpp * u, True))
        else:
            pend = dict(level=ec - od * P["pb_frac"] * rngc, until=tf - 1 + P["pb_wait"], slp=slp * u, tpp=tpp * u)
    legs_alive = P["legs"]; nlegs = 0; lw = min(P["leg_window"], last)
    lmg = P["legs_mgmt"] == "same"
    for j in range(tf, last + 1):
        hj, lj, cj = h[j], l[j], c[j]
        keep = []
        for p in pos:
            adv = (p["e"] - lj) if od > 0 else (hj - p["e"])
            if adv > p["mae"]: p["mae"] = adv
            if (lj <= p["stop"]) if od > 0 else (hj >= p["stop"]):
                fin(p, j, p["stop"]); continue
            if p["tp"] is not None and ((hj >= p["tp"]) if od > 0 else (lj <= p["tp"])):
                fin(p, j, p["tp"]); continue
            if j == last:
                fin(p, j, cj); continue
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
            keep.append(p)
        pos = keep
        if pend is not None:
            if j > pend["until"] or j >= last:
                pend = None
            elif (lj <= pend["level"]) if od > 0 else (hj >= pend["level"]):
                p = new("open", pend["level"], j, pend["slp"], pend["tpp"], True); pend = None
                p["mae"] = max(0.0, (p["e"] - lj) if od > 0 else (hj - p["e"]))
                if (lj <= p["stop"]) if od > 0 else (hj >= p["stop"]): fin(p, j, p["stop"])
                else: pos.append(p)
        if not pos and pend is None and (not legs_alive or j >= lw or nlegs >= P["leg_max"]): break
        if legs_alive and j >= 2 and j < lw and j < last:
            gp = (lj - h[j - 2]) if od > 0 else (l[j - 2] - hj)
            if gp <= 0 or gp < P["leg_gap"] * u:
                legs_alive = False
            elif nlegs < P["leg_max"]:
                nlegs += 1; pos.append(new("leg", cj, j, P["leg_sl"] * u, P["leg_tp"] * u, lmg))
    return done
