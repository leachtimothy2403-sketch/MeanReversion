"""
Copied verbatim (2026-08-24) from the sister project RCTBE's
`_ftmo_challenge_rules.py` — that file remains the source of truth for
this logic; mirror any future correction there back into this copy.
Duplicated rather than imported so this project's repo/VPS deployment
never depends on RCTBE being checked out alongside it.

Correct FTMO challenge-type simulation, replacing the single-phase
8%-max-loss/5%-daily-loss assumption used everywhere so far this
session with the two REAL challenge types:

1-STEP: target 10% ($110k), daily loss 3% ($3k) reset each midnight
        (not trailing), max loss 10% ($10k buffer) - **TRAILING**, per
        the user directly (2026-08-13): at midnight CEST, the account's
        max-ever balance is snapshotted; if it's a new high, the max
        -loss floor rises to (that high - $10k) and never comes back
        down even if balance later drops. This is a real, materially
        different mechanic from a static floor - an account that grew
        to $105k then pulled back to $96k would FAIL here (floor is
        $95k) even though $96k is still above the original $90k static
        floor. Corrected 2026-08-13 (was a fixed fail_equity=$90k the
        whole session before this) - see run_phase's trailing_max_loss
        param.
        NO minimum trading days, single phase.

2-STEP: Phase 1 (Challenge): target 10% ($110k), daily loss 5% ($5k),
        max loss 10% ($90k) - STATIC, not trailing (the user scoped the
        trailing correction to the 1-STEP specifically; 2-step's real
        -world max-loss is commonly the static kind, unlike some 1-step
        /instant-funding-style challenges - left as static here,
        revisit if that assumption turns out wrong for the actual firm
        in use). MINIMUM 4 trading days (days with >=1 real trade)
        before the phase can be marked complete, even if target is
        reached earlier - a real trade stream keeps trading after the
        target is hit (no "stop once profitable" logic exists anywhere
        in this codebase), so extra required days are genuine
        additional risk exposure, not free.
        Phase 2 (Verification): equity resets to $100k, target 5%
        ($105k), SAME daily loss ($5k) and max loss ($90k, static)
        limits, SAME 4-trading-day minimum (assumption - FTMO's exact
        Verification-phase minimum-days rule not independently
        confirmed, flagging this). Starts the very next trading day
        after Phase 1 completes, continuing chronologically in the
        same real historical sequence (deterministic walk-forward,
        not a fresh random draw).

PASS only triggers when (equity >= target) AND (trading days used >=
min_days) are BOTH true at the same check point - if equity rises above
target then dips back below before min_days is satisfied, that's not a
completed pass, matching how a real end-of-evaluation balance check
would work.

--------------------------------------------------------------------
`get_mondays_full` / `rolling_worst_window_pass_rate` added 2026-08-24,
ported from RCTBE's `_ftmo_2step_simulator.py` (verified against that
file's current content — as of this port, this logic was NOT yet also
present in RCTBE's own `_ftmo_challenge_rules.py`, only in the 2-step
simulator; if RCTBE later moves or changes it, mirror the change here
too, same "duplicated, not imported, kept in sync" convention this
whole file already follows).

Ported specifically in response to review feedback on `candidate_report.
py`'s original historical-replay check: an AVERAGE pass rate across all
weekly cohorts can look strong while hiding a cluster of failures
concentrated in one bad stretch (a COVID-2020-style regime) — exactly
the failure mode RCTBE's own real risk-selection standard uses a
rolling WORST 24-month window to catch instead of trusting an average
(see RCTBE_SYSTEM_BRIEFING.md Section 9's "month-by-month breakdown
swung wildly... no consistent safe stretch" finding, which an average
alone would have hidden). `candidate_report.py` now reports both,
clearly labeled — never mistake the average for the worst-window number.
"""
import pandas as pd

START_EQUITY = 100_000.0
ROLLING_WINDOW_MONTHS = 24   # matches RCTBE's own default


def walk_days(by_date, all_days, start):
    return [d for d in all_days if d >= start]


def get_mondays_full(all_days):
    """Every calendar Monday from the first to the last real trading day
    in `all_days`, spaced exactly 7 days apart — NOT filtered to Mondays
    that happen to have a trade. A cohort can legitimately start on a
    Monday with zero trades that specific day (the simulation just picks
    up the next real trading day); using only trade-bearing Mondays as
    anchors would silently skip real, valid cohort start points (e.g.
    a Monday a `skip_weekday`-filtered candidate never trades on)."""
    d = all_days[0]
    while d.weekday() != 0:
        d += pd.Timedelta(days=1)
    mondays = []
    last = all_days[-1]
    while d <= last:
        mondays.append(d)
        d += pd.Timedelta(days=7)
    return mondays


def rolling_worst_window_pass_rate(mondays, outcomes, window_months=ROLLING_WINDOW_MONTHS):
    """`mondays`/`outcomes` are parallel lists — one weekly cohort's
    start date and its simulate_1step/simulate_2step outcome string
    ("PASS"/"FAIL"/"STILL_GOING"). Scans every `window_months`-wide
    rolling window anchored at each Monday and returns the WORST
    observed pass rate across all of them (and that window's start
    date) — not the average. A window with fewer than 8 cohorts, or
    where >15% of cohorts are still-running (too recent to have a real
    outcome yet), is skipped as not yet a reliable read, same as RCTBE's
    own implementation."""
    worst_rate, worst_start = 1.1, None
    for anchor in mondays:
        window_end = anchor + pd.Timedelta(days=int(window_months * 30.44))
        in_window = [o for m, o in zip(mondays, outcomes) if anchor <= m < window_end]
        if len(in_window) < 8:
            continue
        n_still_going = sum(1 for o in in_window if o == "STILL_GOING")
        if n_still_going / len(in_window) > 0.15:
            continue
        n_pass = sum(1 for o in in_window if o == "PASS")
        rate = n_pass / len(in_window)
        if rate < worst_rate:
            worst_rate, worst_start = rate, anchor
    return worst_rate, worst_start


def run_phase(by_date, days, risk_amt, target_equity, daily_loss_limit, min_days,
              max_loss_buffer=None, fail_equity=None, trailing_max_loss=False,
              start_equity=START_EQUITY, max_concurrent=None):
    """Runs day-by-day through `days` (real historical trading days, each
    guaranteed >=1 trade by construction of by_date). Returns
    (outcome, reason, n_days_used, n_trades, end_equity, last_day_index)
    - last_day_index lets a 2-step caller know where to resume Phase 2.

    Two mutually exclusive max-loss modes:
    - trailing_max_loss=True: pass max_loss_buffer (a dollar amount,
      e.g. $10k = 10% of a $100k account). The floor is recomputed once
      per day AT THE START of that day (the "midnight snapshot") as
      max(high_water_mark so far) - max_loss_buffer, then held fixed
      for the rest of that day - matches "moves up when balance makes a
      new high as of midnight, never moves down" exactly.
    - trailing_max_loss=False (default): pass fail_equity (a fixed
      dollar floor, e.g. $90k) - the original static-floor behavior.

    max_concurrent (2026-08-16 fix - was silently missing here, unlike
    simulate_1step_capped in _max_concurrent_sweep.py which always had
    it): caps NEW entries taken per calendar day, matching
    --max-concurrent's live semantics exactly (DEPLOYED=3, ISONLY_PF=
    None/uncapped) - `None` (default) means uncapped, so any EXISTING
    caller that never passed this stays byte-for-byte identical.
    Remaining same-day signals beyond the cap are skipped entirely
    (equity untouched by them), same semantics as simulate_1step_capped."""
    equity = start_equity
    high_water_mark = start_equity
    n_trades = 0
    for i, day in enumerate(days):
        day_start_equity = equity
        if trailing_max_loss:
            high_water_mark = max(high_water_mark, day_start_equity)
            floor = high_water_mark - max_loss_buffer
        else:
            floor = fail_equity
        n_taken_today = 0
        for r in by_date[day]:
            if max_concurrent is not None and n_taken_today >= max_concurrent:
                break
            equity += risk_amt * r
            n_taken_today += 1
            n_trades += 1
            days_used = i + 1
            if equity <= floor:
                return "FAIL", "max_loss", days_used, n_trades, equity, i
            if equity <= day_start_equity - daily_loss_limit:
                return "FAIL", "daily_loss", days_used, n_trades, equity, i
            if equity >= target_equity and days_used >= min_days:
                return "PASS", None, days_used, n_trades, equity, i
    return "STILL_GOING", None, len(days), n_trades, equity, len(days) - 1


def simulate_1step(by_date, all_days, start, risk_pct):
    days = walk_days(by_date, all_days, start)
    risk_amt = START_EQUITY * risk_pct
    outcome, reason, n_days, n_trades, end_equity, _ = run_phase(
        by_date, days, risk_amt, target_equity=110_000.0,
        max_loss_buffer=10_000.0, trailing_max_loss=True,
        daily_loss_limit=3_000.0, min_days=1)
    return {"outcome": outcome, "reason": reason, "days": n_days, "trades": n_trades, "end_equity": end_equity}


def simulate_2step(by_date, all_days, start, risk_pct, min_days=4):
    days = walk_days(by_date, all_days, start)
    risk_amt_p1 = START_EQUITY * risk_pct
    o1, r1, d1, t1, eq1, last_idx1 = run_phase(
        by_date, days, risk_amt_p1, target_equity=110_000.0, fail_equity=90_000.0,
        daily_loss_limit=5_000.0, min_days=min_days)
    if o1 != "PASS":
        return {"outcome": o1, "phase": 1, "reason": r1, "days": d1, "trades": t1, "end_equity": eq1}

    # Phase 2 starts the NEXT trading day after Phase 1 completed, equity reset
    phase2_days = days[last_idx1 + 1:]
    risk_amt_p2 = START_EQUITY * risk_pct  # same % risk, of the RESET 100k balance
    o2, r2, d2, t2, eq2, _ = run_phase(
        by_date, phase2_days, risk_amt_p2, target_equity=105_000.0, fail_equity=90_000.0,
        daily_loss_limit=5_000.0, min_days=min_days)
    return {"outcome": o2, "phase": 2, "reason": r2, "days": d1 + d2, "days_p1": d1, "days_p2": d2,
            "trades": t1 + t2, "end_equity": eq2}
