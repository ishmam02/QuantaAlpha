"""Alpha-decay tiering: healthy / soft (50% weight) / hard (retired, UUID kept).

D1  oracle: a constructed series decays at a date the rule must find EXACTLY
D2  "for 30+ days" is a SUSTAINED breach, not a touch (a 1-day dip is healthy)
D3  hard dominates soft, and hard sets retire + quarterly re-test dates
D4  portfolio action: 1.0 / 0.5 / 0.0 weight multipliers; soft is flagged
D5  hard decay NEVER deletes -- the UUID survives and comes back for re-test
D6  a non-positive baseline IC is not tierable (thresholds would be negative)
D7  first_hard_decay picks the earliest trigger across the book (the remine date)
"""
import numpy as np
import pandas as pd

from quantaalpha.eval.decay import (
    HEALTHY, SOFT, HARD, DecayRule, classify, classify_book, apply_weights,
    due_for_retest, first_hard_decay, rolling_ic,
)

RULE = DecayRule()
DATES = pd.bdate_range("2022-01-03", periods=500)


def _series(values):
    return pd.Series(values, index=DATES[:len(values)], dtype=float)


# ---------------------------------------------------------------------------
# D1 -- ORACLE. Baseline IC 0.040 => soft bar 0.028, hard bar 0.020.
# Construct: 200 days at 0.040 (healthy), then 0.010 forever (below BOTH bars).
# The 63d rolling mean needs 63 days to fill, then decays as the window slides.
# The rule must fire soft before hard, and both at reproducible dates.
# ---------------------------------------------------------------------------
base = 0.040
vals = [0.040] * 200 + [0.010] * 300
st = classify("F1", _series(vals), base, RULE)

assert abs(st.soft_threshold - 0.028) < 1e-12, f"D1: soft bar {st.soft_threshold}"
assert abs(st.hard_threshold - 0.020) < 1e-12, f"D1: hard bar {st.hard_threshold}"
assert st.tier == HARD, f"D1: a factor that collapses to 25% of baseline must be HARD; got {st.tier}"
assert st.soft_trigger and st.hard_trigger, "D1: both triggers must be dated"
assert st.soft_start < st.soft_trigger, "D1: start must precede trigger"
assert st.hard_start < st.hard_trigger, "D1: start must precede trigger"
assert st.soft_trigger < st.hard_trigger, (
    f"D1: soft must fire before hard; {st.soft_trigger} vs {st.hard_trigger}")
# The trigger is exactly soft_days/hard_days business days after the start.
_sd = len(pd.bdate_range(st.soft_start, st.soft_trigger))
_hd = len(pd.bdate_range(st.hard_start, st.hard_trigger))
assert _sd == RULE.soft_days, f"D1: soft run is {_sd} days, rule says {RULE.soft_days}"
assert _hd == RULE.hard_days, f"D1: hard run is {_hd} days, rule says {RULE.hard_days}"
print(f"D1 PASS  oracle: soft {st.soft_start}->{st.soft_trigger} ({_sd}d), "
      f"hard {st.hard_start}->{st.hard_trigger} ({_hd}d)")

# ---------------------------------------------------------------------------
# D2 -- a TOUCH is not decay. A rolling IC that dips under the bar for a few
# days and recovers must stay HEALTHY; counting the touch date would overstate
# decay speed by weeks (measured: 44 days on the prior 19-factor set).
# ---------------------------------------------------------------------------
# A moderate multi-day shock. It drags the 63d mean under BOTH bars for a
# stretch shorter than the 60 days hard decay requires, so the factor must NOT
# be retired -- a tradable factor that has one bad month is not a dead factor.
#
# Sizing this is not arbitrary, and the guard below is the point: a 63-day mean
# carries a bad day for 63 days, so a SINGLE catastrophic day breaches the bar
# for 63 consecutive days and legitimately trips hard decay. Depth and duration
# are not separable under a rolling mean. The fixture therefore uses a shock
# shallow enough that the breach ends before day 60.
dip = [0.040] * 120 + [-0.25] * 5 + [0.040] * 250
st_dip = classify("F2", _series(dip), base, RULE)
_roll = rolling_ic(_series(dip), RULE.window).dropna()
_breach_days = int((_roll < st_dip.hard_threshold).sum())
assert _breach_days > 0, (
    "D2: the constructed dip must actually breach the hard bar, else D2 proves nothing")
assert _breach_days < RULE.hard_days, (
    f"D2: fixture breaches for {_breach_days}d, which is >= the {RULE.hard_days}d "
    "rule -- it would legitimately be hard decay and tests nothing")
assert st_dip.tier != HARD, (
    f"D2: a {_breach_days}-day breach must NOT retire the factor "
    f"({RULE.hard_days}d required); got {st_dip.tier}")
assert st_dip.hard_trigger is None, "D2: no hard trigger for a short breach"
print(f"D2 PASS  breached the hard bar for {_breach_days}d (<{RULE.hard_days}d) "
      f"-> not retired (tier={st_dip.tier})")

# D2b -- the corollary, stated as a test so the behaviour is documented rather
# than discovered in production: ONE catastrophic day DOES retire a factor,
# because the 63d window carries it for 63 days. If that is not wanted, the
# rolling statistic needs a median (or the shock needs winsorizing) -- it is
# not something the duration threshold can fix.
_one_bad_day = [0.040] * 120 + [-2.0] + [0.040] * 250
_st1 = classify("F2b", _series(_one_bad_day), base, RULE)
assert _st1.tier == HARD, (
    "D2b: documented behaviour -- a single -2.0 IC day keeps the 63d mean under "
    f"the hard bar for a full window and retires the factor; got {_st1.tier}")
print("D2b PASS  documented: ONE catastrophic day trips hard decay (63d window carries it)")

# ---------------------------------------------------------------------------
# D3 -- hard dominates soft; retirement + quarterly re-test dates are set.
# ---------------------------------------------------------------------------
assert st.retired_on == st.hard_trigger, "D3: retired_on must be the hard trigger"
_gap = (pd.Timestamp(st.retest_on) - pd.Timestamp(st.retired_on)).days
assert _gap == RULE.retest_days, f"D3: re-test must be quarterly; got {_gap}d"
print(f"D3 PASS  hard dominates; retired {st.retired_on}, re-test {st.retest_on} (+{_gap}d)")

# ---------------------------------------------------------------------------
# D4 -- portfolio action. Healthy full, soft HALF and flagged, hard removed.
# ---------------------------------------------------------------------------
soft_vals = [0.040] * 120 + [0.024] * 200          # under 0.028, above 0.020
st_soft = classify("F3", _series(soft_vals), base, RULE)
assert st_soft.tier == SOFT, f"D4: expected soft; got {st_soft.tier}"
assert st_soft.hard_trigger is None, "D4: must not also trip hard"

healthy = classify("F4", _series([0.040] * 300), base, RULE)
assert healthy.tier == HEALTHY, f"D4: expected healthy; got {healthy.tier}"

states = {"F4": healthy, "F3": st_soft, "F1": st}
w = apply_weights({"F4": 1.0, "F3": 1.0, "F1": 1.0}, states)
assert w == {"F4": 1.0, "F3": 0.5, "F1": 0.0}, f"D4: weights {w}"
assert st_soft.flagged_for_review and not healthy.flagged_for_review, (
    "D4: soft decay is the research-review flag")
# An unmonitored factor keeps full weight (absence of data != decay).
assert apply_weights({"ZZ": 1.0}, states) == {"ZZ": 1.0}, (
    "D4: an unmonitored factor must not be silently zeroed")
print("D4 PASS  weights 1.0 / 0.5 / 0.0; soft flagged; unmonitored untouched")

# ---------------------------------------------------------------------------
# D5 -- hard decay RETAINS the UUID and comes back for re-test. This is the
# half of the rule that makes it "removed from scoring", not "deleted".
# ---------------------------------------------------------------------------
assert st.factor_id == "F1", "D5: the UUID must survive retirement"
before = due_for_retest(states.values(), asof=st.retired_on)
after = due_for_retest(states.values(), asof=st.retest_on)
assert before == [], f"D5: not due before the quarter elapses; got {before}"
assert after == ["F1"], f"D5: the retired UUID must come back for re-test; got {after}"
print("D5 PASS  hard decay retains the UUID and re-tests it quarterly")

# ---------------------------------------------------------------------------
# D6 -- a non-positive baseline is not tierable (70% of a negative number is a
# LOWER bar, so "decay" could never fire and the factor would look healthy).
# ---------------------------------------------------------------------------
for bad in (0.0, -0.02, float("nan")):
    st_bad = classify("F5", _series([0.01] * 200), bad, RULE)
    assert st_bad.tier == HARD, f"D6: baseline {bad} must not be scored; got {st_bad.tier}"
    assert st_bad.weight_multiplier == 0.0
print("D6 PASS  a non-positive/NaN baseline IC is not scored")

# ---------------------------------------------------------------------------
# D7 -- the remine date: earliest hard trigger across the book.
# ---------------------------------------------------------------------------
early = [0.040] * 80 + [0.001] * 300
st_early = classify("F6", _series(early), base, RULE)
assert st_early.tier == HARD
book = [healthy, st_soft, st, st_early]
assert first_hard_decay(book) == min(st.hard_trigger, st_early.hard_trigger), (
    "D7: must return the EARLIEST hard trigger")
assert first_hard_decay([healthy, st_soft]) is None, (
    "D7: no hard decay -> None (no remine date)")
print(f"D7 PASS  first hard decay across the book = {first_hard_decay(book)}")

# ---------------------------------------------------------------------------
# classify_book wiring
# ---------------------------------------------------------------------------
frame = pd.DataFrame({"A": _series([0.040] * 300), "B": _series(vals[:300])})
bk = classify_book(frame, {"A": base, "B": base}, RULE)
assert bk["A"].tier == HEALTHY and bk["B"].tier in (SOFT, HARD)
print("D8 PASS  classify_book tiers every column against its own baseline")

print("\nALL PASS")
