"""Alpha-decay tiering: healthy / soft decay / hard decay.

The rule (as specified, 2026-08-24):

    tier         trigger                                    portfolio action
    ---------------------------------------------------------------------------
    healthy      rolling 63d IC >= 70% of validation IC     full weight eligible
    soft decay   IC < 70% of baseline for 30+ days          weight capped at 50%
                                                            of model weight;
                                                            flagged for research
    hard decay   IC < 50% of baseline for 60+ days          removed from scoring;
                                                            UUID RETAINED (never
                                                            deleted), re-tested
                                                            quarterly

Two design points that are easy to get wrong and are load-bearing here:

* **"for 30+ days" means a SUSTAINED breach, not a touch.** A single day under
  the threshold is noise -- a 63-day rolling IC crosses its own threshold
  constantly near the boundary. The tier fires only after the breach has held
  for the full duration, and the reported ``*_start`` is the first day of that
  run while ``*_trigger`` is the day the duration was met. Using the touch date
  as the decay date overstates decay speed by weeks (measured on the prior
  19-factor set: soft touch 2022-04-11 vs trigger 2022-05-25 -- 44 days apart).

* **Hard decay never deletes.** The factor is removed from SCORING and its UUID
  is retained, because a regime that killed a factor can return and the only
  way to know is to keep re-testing it. ``quarantine`` records the retirement
  date and the next re-test date; ``due_for_retest`` reads it back.

Baseline IC is the factor's validation-window IC -- the number the factor was
admitted on. Comparing live IC to a live-window mean instead would move the
goalposts with the decay itself and could never fire.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

HEALTHY = "healthy"
SOFT = "soft_decay"
HARD = "hard_decay"

# Portfolio action per tier: the multiplier applied to the model weight.
WEIGHT_MULTIPLIER = {HEALTHY: 1.0, SOFT: 0.5, HARD: 0.0}


@dataclass(frozen=True)
class DecayRule:
    window: int = 63          # rolling IC window, trading days
    soft_frac: float = 0.70   # of baseline (validation) IC
    soft_days: int = 30       # sustained days below soft_frac
    hard_frac: float = 0.50
    hard_days: int = 60
    retest_days: int = 91     # quarterly re-test of hard-decayed factors

    def soft_threshold(self, baseline_ic: float) -> float:
        return self.soft_frac * baseline_ic

    def hard_threshold(self, baseline_ic: float) -> float:
        return self.hard_frac * baseline_ic


@dataclass
class DecayState:
    """One factor's tier and the dates that produced it."""
    factor_id: str
    baseline_ic: float
    tier: str = HEALTHY
    soft_threshold: float | None = None
    hard_threshold: float | None = None
    soft_start: str | None = None     # first day of the sustained breach
    soft_trigger: str | None = None   # day the 30-day duration was met
    hard_start: str | None = None
    hard_trigger: str | None = None
    retired_on: str | None = None     # = hard_trigger; UUID retained, not deleted
    retest_on: str | None = None
    roll_first: float | None = None
    roll_last: float | None = None
    roll_min: float | None = None
    n_days: int = 0

    @property
    def weight_multiplier(self) -> float:
        return WEIGHT_MULTIPLIER[self.tier]

    @property
    def flagged_for_review(self) -> bool:
        """Soft decay is the research-review flag; hard decay is past review."""
        return self.tier == SOFT

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weight_multiplier"] = self.weight_multiplier
        d["flagged_for_review"] = self.flagged_for_review
        return d


def rolling_ic(ic_series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling mean IC. Requires a full window before emitting a value, so the
    first ``window-1`` days are NaN rather than a mean over 3 observations."""
    s = pd.Series(ic_series).astype(float).sort_index()
    return s.rolling(window, min_periods=window).mean()


def _sustained_breach(roll: pd.Series, threshold: float, days: int
                      ) -> tuple[str | None, str | None]:
    """First (start, trigger) of a run of ``days`` consecutive days strictly
    below ``threshold``.

    ``start`` is the first day of the run; ``trigger`` is the ``days``-th day,
    i.e. the day the rule is actually satisfied. Returns (None, None) when no
    run is long enough. NaN days (before the window fills) break a run -- an
    unmeasurable day is not evidence of decay.
    """
    below = roll < threshold
    run = 0
    start_idx = None
    for ts, flag in below.items():
        if pd.isna(roll.loc[ts]):
            run, start_idx = 0, None
            continue
        if flag:
            if run == 0:
                start_idx = ts
            run += 1
            if run >= days:
                return (str(pd.Timestamp(start_idx).date()),
                        str(pd.Timestamp(ts).date()))
        else:
            run, start_idx = 0, None
    return (None, None)


def classify(factor_id: str, ic_series: pd.Series, baseline_ic: float,
             rule: DecayRule | None = None) -> DecayState:
    """Tier one factor from its live daily IC series and its validation IC.

    ``baseline_ic`` <= 0 is not tierable: the thresholds would be negative and
    "below 70% of a negative number" is not decay. Such a factor is returned as
    HARD with no dates -- it never worked, so it should not be scored.
    """
    rule = rule or DecayRule()
    st = DecayState(factor_id=factor_id, baseline_ic=float(baseline_ic))

    if not np.isfinite(baseline_ic) or baseline_ic <= 0:
        st.tier = HARD
        return st

    st.soft_threshold = rule.soft_threshold(baseline_ic)
    st.hard_threshold = rule.hard_threshold(baseline_ic)

    roll = rolling_ic(ic_series, rule.window).dropna()
    st.n_days = int(len(roll))
    if st.n_days == 0:
        return st  # nothing measurable yet -> stays healthy
    st.roll_first = float(roll.iloc[0])
    st.roll_last = float(roll.iloc[-1])
    st.roll_min = float(roll.min())

    st.soft_start, st.soft_trigger = _sustained_breach(
        roll, st.soft_threshold, rule.soft_days)
    st.hard_start, st.hard_trigger = _sustained_breach(
        roll, st.hard_threshold, rule.hard_days)

    # Hard dominates: a factor deep enough to breach 50% for 60 days is retired
    # even though it also satisfies the softer rule.
    if st.hard_trigger:
        st.tier = HARD
        st.retired_on = st.hard_trigger
        st.retest_on = str((pd.Timestamp(st.hard_trigger)
                            + pd.Timedelta(days=rule.retest_days)).date())
    elif st.soft_trigger:
        st.tier = SOFT
    return st


def classify_book(ic_frame: pd.DataFrame, baselines: dict[str, float],
                  rule: DecayRule | None = None) -> dict[str, DecayState]:
    """Tier every column of ``ic_frame`` (dates x factor_id) against its baseline."""
    rule = rule or DecayRule()
    out: dict[str, DecayState] = {}
    for fid in ic_frame.columns:
        out[str(fid)] = classify(str(fid), ic_frame[fid],
                                 baselines.get(str(fid), float("nan")), rule)
    return out


def apply_weights(model_weights: dict[str, float],
                  states: dict[str, DecayState]) -> dict[str, float]:
    """Portfolio-construction action: full / 50% / removed from scoring.

    A factor with no decay state is left at full weight (it has not been
    monitored, which is not the same as having decayed).
    """
    return {fid: w * (states[fid].weight_multiplier if fid in states else 1.0)
            for fid, w in model_weights.items()}


def due_for_retest(states: Iterable[DecayState], asof: str) -> list[str]:
    """Hard-decayed UUIDs whose quarterly re-test is due on ``asof``.

    This is the half of the rule that keeps hard decay from being a deletion:
    regimes change, and a retired factor is re-tested rather than forgotten.
    """
    t = pd.Timestamp(asof)
    return [s.factor_id for s in states
            if s.tier == HARD and s.retest_on and pd.Timestamp(s.retest_on) <= t]


def first_hard_decay(states: Iterable[DecayState]) -> str | None:
    """Earliest hard-decay trigger across the book -- the date the remine
    experiment treats as the start of its new test range."""
    ds = [s.hard_trigger for s in states if s.hard_trigger]
    return min(ds) if ds else None


__all__ = [
    "HEALTHY", "SOFT", "HARD", "WEIGHT_MULTIPLIER", "DecayRule", "DecayState",
    "rolling_ic", "classify", "classify_book", "apply_weights",
    "due_for_retest", "first_hard_decay",
]
