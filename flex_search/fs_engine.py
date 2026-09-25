"""flex_search engine: one session (NY / LDN / ASIA) of one day -> list of trades.

Mechanics (all generalised from discretionary_backtest.py + the 2026-09 experiments):
  fair value (FV) = session anchor open (bar 0 open), held flat.
  opening trade    : direction of bar 0 (doji = none); entry bar-0 close; big-candle (range > big_thr) -> big_sl/big_tp
                     else std_sl/std_tp; optional VWAP-alignment filter.
  FVG legs         : continuation legs in the opening direction on every fresh 3-candle FVG with gap >= leg_gap,
                     streak ends on first bar without one; leg_sl/leg_tp; max leg_max legs.
  reversals        : break-of-structure (close beyond previous run's last candle wick +/- min_bos) toward FV,
                     at least min_fv from FV. rev_mode: off | std (TP std_tp) | disp_fv (only BOS candles with
                     body >= disp_pts, TP = FV) | mixed (displacement -> FV, others std_tp). SL = std_sl.
  stacking         : up to maxpos reversal positions. The first needs a BOS; additional (stacked) positions can be
                     triggered by 'bos', 'fvg' (fresh FVG toward FV with gap >= stack_gap) or 'both'; stacked
                     size multiplier stack_size.
  wait_open        : reversals only after the opening trade has resolved (live behaviour) or from bar 1.
  exit             : 'hard' closes everything at the last entry-window bar; 'run' lets trades run to the end of
                     the session block (next session anchor / 17:00 ET, max 6h).
Each trade: kind, sgn, eb, xb, entry, exit, pnl (points), sl (points), size (multiplier), mae (points, >=0),
tpd (take-profit distance, points).
Same-bar SL/TP collision -> SL (conservative). No entry on the last bar.
"""
import numpy as np

def _runs_levels(o, c, h, l, n, min_bos):
    d = np.sign(c[:n] - o[:n])
    nz = np.nonzero(d)[0]
    rsh = np.full(n, np.nan); rsl = np.full(n, np.nan)
    if len(nz) < 2: return d, rsh, rsl
    dz = d[nz]; brk = np.nonzero(np.diff(dz))[0]           # last index (in nz) of each run except final
    starts = np.r_[0, brk + 1]; ends = np.r_[brk, len(nz) - 1]
    for k in range(1, len(starts)):
        ref = nz[ends[k - 1]]; cs, ce = nz[starts[k]], nz[ends[k]]
        rsh[cs:ce + 1] = h[ref] + min_bos; rsl[cs:ce + 1] = l[ref] - min_bos
    return d, rsh, rsl

def sim(block, P):
    o, h, l, c, v = (block[:, k].astype("float64") for k in range(5))
    valid = np.isfinite(c)
    nb = int(valid.sum()) if valid.all() is False else len(c)
    # last valid bar
    lv = np.nonzero(valid)[0]
    if len(lv) < 10 or not valid[0]: return []
    lastb = lv[-1]
    n = min(P["window"], lastb + 1)
    last = n - 1 if P["exit"] == "hard" else lastb
    fv = o[0]
    trades, openp = [], []   # open: dict
    def close(p, j, xp):
        trades.append(dict(kind=p["kind"], sgn=p["sgn"], eb=p["eb"], xb=j, entry=p["e"], exit=xp,
                           pnl=(xp - p["e"]) * p["sgn"], sl=p["slp"], size=p["size"], mae=p["mae"], tpd=abs(p["tp"] - p["e"])))
    # opening trade
    od = int(np.sign(c[0] - o[0]))
    open_active = False
    if P["open_trade"] and od != 0:
        ok = True
        if P["vwap"]:
            vw = (h[0] + l[0] + c[0]) / 3
            ok = (c[0] > vw) == (od > 0)
        if ok:
            big = (h[0] - l[0]) > P["big_thr"]
            slp, tpp = (P["big_sl"], P["big_tp"]) if big else (P["std_sl"], P["std_tp"])
            openp.append(dict(kind="open", sgn=od, e=c[0], sl=c[0] - od * slp, tp=c[0] + od * tpp, slp=slp, eb=0, size=1.0, mae=0.0))
            open_active = True
    # reversal structure
    d, rsh, rsl = _runs_levels(o, c, h, l, n, P["min_bos"])
    body = np.abs(c - o)
    legs_alive = P["legs"] and od != 0; nlegs = 0
    rev_on = P["rev_mode"] != "off"
    rev_start = 1 if not P["wait_open"] else None
    if rev_start is None and not open_active: rev_start = 1
    for j in range(1, last + 1):
        if not valid[j]: continue
        # 1) resolve
        still = []; freed_rev = 0
        for p in openp:
            s = p["sgn"]
            adv = (p["e"] - l[j]) if s > 0 else (h[j] - p["e"])
            hs = l[j] <= p["sl"] if s > 0 else h[j] >= p["sl"]
            ht = h[j] >= p["tp"] if s > 0 else l[j] <= p["tp"]
            if hs:
                p["mae"] = max(p["mae"], adv); close(p, j, p["sl"])
            elif ht:
                p["mae"] = max(p["mae"], adv); close(p, j, p["tp"])
            elif j == last:
                p["mae"] = max(p["mae"], adv); close(p, j, c[j])
            else:
                p["mae"] = max(p["mae"], adv); still.append(p); continue
            if p["kind"] == "open" and rev_start is None: rev_start = j + 1
            if p["kind"] in ("rev", "stack"): freed_rev += 1
        openp = still
        if j >= n - 1 or j == last: continue   # no entries on / after the last entry-window bar
        # 2) FVG continuation legs
        if legs_alive and j >= 2:
            gap = (l[j] - h[j - 2]) if od > 0 else (l[j - 2] - h[j])
            if gap <= 0 or gap < P["leg_gap"]:
                legs_alive = False
            elif nlegs < P["leg_max"]:
                nlegs += 1
                openp.append(dict(kind="leg", sgn=od, e=c[j], sl=c[j] - od * P["leg_sl"], tp=c[j] + od * P["leg_tp"], slp=P["leg_sl"], eb=j, size=1.0, mae=0.0))
        # 3) reversals
        if not rev_on or rev_start is None or j < rev_start or d[j] == 0: continue
        nrev = sum(1 for p in openp if p["kind"] in ("rev", "stack"))
        if nrev + freed_rev >= P["maxpos"]: continue
        dj = int(d[j])
        bos = (dj < 0 and c[j] < rsl[j]) or (dj > 0 and c[j] > rsh[j])   # nan compares False
        toward = (dj > 0 and c[j] < fv) or (dj < 0 and c[j] > fv)
        far = abs(c[j] - fv) >= P["min_fv"]
        trig = False; stacked = nrev > 0
        if not stacked:
            trig = bos
        else:
            st = P["stack_trig"]
            if st in ("bos", "both") and bos: trig = True
            if not trig and st in ("fvg", "both") and j >= 2:
                g = (l[j] - h[j - 2]) if dj > 0 else (l[j - 2] - h[j])
                trig = g >= P["stack_gap"]
        if not (trig and toward and far): continue
        isdisp = body[j] >= P["disp_pts"]
        m = P["rev_mode"]
        if m == "disp_fv" and not isdisp: continue
        to_fv = (m == "disp_fv") or (m == "mixed" and isdisp)
        tp = fv if to_fv else c[j] + dj * P["std_tp"]
        openp.append(dict(kind="stack" if stacked else "rev", sgn=dj, e=c[j], sl=c[j] - dj * P["std_sl"], tp=tp,
                          slp=P["std_sl"], eb=j, size=P["stack_size"] if stacked else 1.0, mae=0.0))
    return trades
