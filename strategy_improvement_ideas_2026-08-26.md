# MeanReversion — strategy improvement ideas (overnight research, 2026-08-25/26)

While the VPS search runs overnight, I dug into three questions: could an
HMM regime filter help, is there a better entry mechanism than break-of-
structure, and what does the evidence actually say before we commit to
either. Short version up front, details below.

**Bottom line:** the current fair-value/BOS design has real but weak
edge (confirmed empirically, not just asserted). A crude volatility-regime
filter bolted onto the existing BOS signal did NOT help — it made results
slightly worse. But that's not evidence against regime-awareness in
general; it's evidence that filtering the wrong signal type doesn't work.
RCTBE (the sister project) hit this exact problem before and solved it by
changing the ENTRY MECHANISM to be regime-native, not by filtering an
unrelated one. That's the strongest lead here, and it's a proven design,
not a guess.

## 1. What I checked, and why

Before proposing anything, I wanted a real answer to "does regime-gating
even move the needle," not just intuition. Using the 2022 and 2024 real
NDX100 data already staged from last night's diagnostic, I built a cheap
regime proxy — trailing realized volatility (240-bar log-return std)
below its own trailing 2000-bar median, fully causal, no lookahead — and
re-ran the same 400-draw cross-year test, this time filtering the
existing BOS signals to only fire on "compression-like" bars.

Result: it didn't help.

| | 2022 mean avg_pf | 2024 mean avg_pf | cross-year corr |
|---|---|---|---|
| Ungated (baseline) | 0.752 | 0.716 | 0.427 |
| Gated to low-vol bars | 0.732 | 0.666 | 0.383 |

Gating made things *slightly worse* on both years and weakened the
cross-year correlation. Only 36% (2022) / 25% (2024) of individual draws
even improved from gating — most got worse or unchanged.

This is a genuinely useful negative result, not a wasted afternoon: it
tells us a generic "only trade when it's quiet" filter isn't the fix,
at least not applied on top of the current signal. The likely reason is
structural, not a weak proxy — see below.

## 2. Why a bolted-on regime filter probably can't work here

The current entry (`bos_up`/`bos_down` — price closing beyond a recent
swing high/low or wick) is, definitionally, a local **breakout** event.
Breakouts tend to happen as volatility is *expanding*, not while it's
still low. Gating that signal to "only fire during low volatility" is
asking for an event that's mechanically more likely to occur at the
boundary of — or just after — a compression period ends, restricted to
exactly the window where it's least likely to be a clean signal. That's
probably why the crude proxy filter didn't help: it's not that the
regime idea is wrong, it's that BOS isn't the right signal to gate with
it.

This matches something RCTBE already learned the hard way. I pulled
three of its sibling strategies (all read from `C:\Users\leach\RCTBE`,
with your folder access already granted, so I could check the actual
code rather than guess):

- **`layer2_optimizer.py`** — trades the MOMENT a Compression regime ends
  (a breakout at the transition). RCTBE's own notes say this produced
  only ~0.19 trades/day combined across assets — confirmed transitions
  are rare. Sound familiar? This is almost exactly the frequency problem
  we started with.
- **`layer2_range_fade.py`** — RCTBE's actual fix for that frequency
  problem. Instead of waiting for Compression to end, it trades *during*
  an ongoing Compression run: price wicks into a band near either edge
  of the CURRENT (still-open) low-vol range, and it fades back toward
  the middle. Direction, entry, stop, and target are all regime-native —
  built specifically for "price is quietly ranging," not retrofitted
  onto a breakout signal. It also has an emergency exit: if the HMM
  regime flips to Expansion mid-trade, it exits immediately rather than
  riding out the normal horizon, because the whole premise (quiet range)
  just broke.
- **`layer2_trend_exhaustion.py`** — the mirror image: fades a
  just-ENDED Expansion run (a trend running out of steam), with the same
  regime-flip emergency exit.

All three share one HMM: a 2-state (Compression/Expansion) Gaussian HMM
(`hmm_gaussian.py` — a small, in-repo Baum-Welch implementation, no
external dependency) fit on realized volatility, volume z-score, and
short-window momentum, refit on a rolling walk-forward schedule
(`layer1_regime.py`) so there's no lookahead: fitting uses only a
trailing window, standardization uses that window's own mean/std, and
labeling uses the causal forward-pass posterior (`filter()`), never the
smoothed/Viterbi versions which would leak future bars into past labels.

The pattern that actually worked for RCTBE, in other words, wasn't "add
an HMM filter to an existing signal" — it was "design the entry mechanism
around what the regime means." That's the more promising direction here
too.

## 3. Concrete proposal: a range-fade-style entry for MeanReversion

Rather than trying to rescue BOS with a regime filter, the more promising
move is to add — as a genuinely new, swept entry mode alongside the
existing BOS one, not a replacement — something shaped like
`layer2_range_fade.py`, adapted to this project's own fair-value framing
instead of a fresh HMM-only design:

- **Trigger**: while price is within a "consolidating" fair-value window
  (this project already has that concept — `consolidation_bars`/
  `consolidation_atr_mult` define exactly this), price wicks into a swept
  `fade_zone_pct` band near either edge of the current consolidation
  range, without closing beyond it (a close beyond the edge looks like a
  real breakout in progress, not a rejection worth fading — same
  distinction RCTBE's version makes).
- **Direction**: fade it — sell near the top of the range, buy near the
  bottom.
- **Stop-loss**: structural, just past the opposite edge of the range
  (same ATR-buffered-floor-and-cap philosophy already used here).
- **Target**: a swept fraction across the range (0.5 = midpoint, matching
  the existing `target_fraction` convention).
- **Exit**: existing wall-clock horizon, PLUS an emergency exit if the
  range breaks (price closes beyond either edge) mid-trade — a cheap,
  regime-flip-like mechanic that doesn't need the full HMM to start
  testing, since "did the range that defined this trade just break" is
  derivable from the same consolidation-tracking logic already in
  `_fair_value_series`.

This is testable WITHOUT building the full HMM first — it reuses this
project's existing consolidation-range concept as the "regime," which is
a cheaper (if less validated) proxy than the Gaussian HMM. If it shows
real promise, the natural follow-up is swapping the consolidation-range
proxy for RCTBE's actual walk-forward HMM Compression label, which should
be a materially cleaner regime signal (it's validated on
forward-persistent volatility, not just "is the range tight right now").

This also directly answers the frequency question differently than the
current approach: RCTBE's range-fade wasn't built to hit a frequency
target for its own sake, it was built because trading *within* a common
regime instead of *only at its rare transitions* is naturally far more
frequent — the same logic should apply here without needing to keep
loosening BOS parameters (which is what's been diluting edge quality, per
last night's finding that only `fractal`-mode draws survived the 2022-vs-
2024 cross-year check, not `raw_wick`).

## 4. If we do want the full HMM: what it would take

Not proposing this as step one, but concretely, since you asked directly:

- Port `hmm_gaussian.py` (self-contained, no external dependency — it was
  written in-repo specifically because hmmlearn wasn't installable on
  RCTBE's own Python version at the time; worth checking here since I
  found `hmmlearn` installs fine in this sandbox's Python 3.11 — could
  use either).
- Port `layer1_regime.py`'s walk-forward-safe fitting pattern
  (trailing-window fit, causal `filter()` labeling, periodic refit) —
  this is the part most likely to go wrong if rebuilt from scratch, so
  reusing the actual RCTBE code is worth it.
- Retune `window_bars`/`refit_every_bars` for 1-minute bars — RCTBE's
  values (2000/2000) were validated on 5-minute bars (~7 trading days);
  the 1-minute equivalent is a much larger raw bar count for the same
  wall-clock window, so this needs its own tuning, not a blind port of
  the numbers.
- Budget real precompute time: RCTBE's own walk-forward refit takes
  ~10-15 min/asset on 5-min bars; 1-min bars (5x the data) will take
  longer. This is a one-time-per-asset precompute step, not a per-search-
  iteration cost, so it's affordable, just not instant.
- Recompute NDX100's regime label this way and re-run the same cross-year
  robustness check (2022 vs 2024) I ran tonight, this time gating either
  the existing BOS signal or (more likely to matter, per Section 2) a new
  range-fade-style entry.

## 5. Other ideas worth a mention (lower priority, not tested tonight)

- **Multi-timeframe structure confirmation** — require a 5-min or 15-min
  BOS/structure alignment before trusting a 1-min BOS, to filter out pure
  1-min noise. Cheap to add, easy to test, doesn't need the HMM.
- **Adaptive z-score bands instead of fixed ATR multiples** — the current
  `deviation_threshold_atr` is a fixed multiple; a rolling z-score of
  price vs. fair value (mean/std over a trailing window) would adapt to
  changing volatility regimes automatically, which is a lighter-weight
  partial substitute for full regime-awareness.
- **Momentum-exhaustion confirmation** — pair the BOS/extension condition
  with a momentum-deceleration check (e.g., rate-of-change turning back
  toward zero) rather than relying on price structure alone — closer in
  spirit to `layer2_trend_exhaustion.py`'s "the move ran out of steam"
  logic than to BOS's "it broke a level."

## 6. Recommendation for tomorrow's discussion

1. Let the overnight search finish — no reason to stop it, and `fractal`
   mode alone still has thin-but-real edge worth letting it search
   through.
2. Prioritize building the range-fade-style entry mode (Section 3) as a
   new swept option alongside BOS — it's the most direct fix for the
   frequency-vs-edge tension, it's proven in spirit by RCTBE's own
   experience, and it doesn't require the HMM as a prerequisite to start
   testing.
3. Treat the full walk-forward HMM (Section 4) as a strong follow-up once
   the range-fade entry is in and its consolidation-range proxy for
   "regime" has been checked against real data the same way tonight's
   BOS test was — if the cheaper proxy shows promise, the HMM is likely
   to sharpen it further, not replace a null result with a working one.
4. I'd hold off on the multi-timeframe/z-score/exhaustion ideas (Section
   5) until we see how far the range-fade entry gets — they're
   complements, not competitors, to it.
