"""Admission by marginal contribution, tested against its own noise.

The previous rule admitted a batch when its utility ``U`` cleared a fixed
percentile of the repository. That failed in a specific and recorded way: the
first two batches are admitted unconditionally to bootstrap the ranking, so they
become the yardstick for everything after them, and a *lucky* pair sets a bar
nothing can clear. Measured on a top-k re-mine -- bootstrap net_ir
-0.1746 and -0.0213, against -0.1531 and -0.2867 on the run that worked -- the
repository froze at 18 factors and rejected 33 consecutive batches while factor
quality was statistically identical to the successful run. Rejections do not
change the repository, and eviction only removes the weakest member, so the bar
could only ever rise. A one-way ratchet driven by a two-batch draw.

The fix separates two jobs that were being done by one number.

**Admission asks whether the book gets better.** Not whether the batch ranks
well against a winners-only sample of earlier batches -- that sample drifts with
luck and has no fixed meaning -- but whether adding these factors improves the
portfolio that already exists. That question is self-normalising: a batch that
helps, helps, whatever earlier batches scored. It cannot ratchet, and it cannot
admit junk, because junk does not improve the book.

**Feedback asks how the generator is doing.** That is where a population
percentile belongs, and ranking against *every* batch evaluated -- admitted or
not -- is the honest reference for it. It gates nothing, so it cannot admit bad
factors; it only tells the generator where it stands.

``delta_net_ir`` was tried as an admission rule before and abandoned because it
was not reproducible: 3 of 4 sampled factors flipped their verdict across
combiner seeds. That is a fact about the *point estimate*, not about the
quantity. Here the same delta is measured once per seed and admitted only when

    mean(Δ) > k · se(Δ)

so the bar is set by how precisely the improvement can be measured. Noisy
deltas fail not because they are small but because they are unresolved, which
is the honest reading of a measurement that flips with the seed. It also makes
the requirement genuinely dynamic without anyone choosing a percentile.

**Saturation is handled by capacity, not by the bar.** Marginal contribution
necessarily shrinks as the repository grows -- that is what diminishing returns
means, and it is information rather than a defect. Once the repository is full
the decision becomes a *replacement*: a candidate enters only by displacing the
incumbent it beats, so the criterion stops decaying and the library improves
monotonically instead of merely growing.

**The floor catches pathologies, never weakness.** The gates that were removed
failed because they were absolute thresholds on noisy estimates of the wrong
quantity -- stand-alone IC, measured to be anti-correlated with contribution.
Nothing here rejects a factor for being weak. It rejects signals that are
broken: too sparse to price, constant, or a near-duplicate of something already
held. Those cannot starve the repository because they are not a quality bar.
"""

from __future__ import annotations

import logging
import math
import statistics as st
from dataclasses import dataclass, field

from quantaalpha.core.verdict import Verdict

logger = logging.getLogger(__name__)

# Sigma bar for CALLING a rejected batch resolvably harmful, as opposed to
# unresolved. Diagnostic only -- it never changes an admission (see the verdict
# branch in ``decide``, where both paths already return Decision(False, ...)),
# it only decides which story the ledger and the diagnosis router are told.
#
# Kept separate from ``Θ.admission.k_sigma`` on purpose. Selection wants a
# conservative bar because a false positive costs the book; feedback wants a
# sensitive one because withholding "this made the book worse" teaches nothing.
# Sharing one number meant tightening selection silently relabelled every batch
# between -k_sigma and -1σ as MARGINAL. Not a Θ field: the verdict is not part
# of the scored objective, and Θ is frozen.
_VERDICT_SIGMA = 1.0


@dataclass(frozen=True)
class Decision:
    """The verdict, and everything needed to explain it in the ledger."""

    admit: bool
    reason: str
    deltas: tuple[float, ...] = ()
    mean: float = float("nan")
    se: float = float("nan")
    t_stat: float = float("nan")
    displaced: str | None = None
    pathology: str | None = None
    # Structured verdict (T1): the branch in ``decide`` already knows which case
    # it is, so it sets this directly instead of leaving the verdict to be
    # re-parsed from ``reason`` by substring matching downstream. Serialized to
    # ``.value`` so the persisted form is a plain string ("net_harmful", ...)
    # and the downstream ``classify_verdict`` can ``Verdict(record["verdict"])``
    # it back without the str-Enum repr gotcha. Defaults to NO_DATA so a
    # Decision constructed without it (none in production, but defensive) reads
    # as "nothing to diagnose" rather than crashing.
    verdict: Verdict = Verdict.NO_DATA

    def as_record(self) -> dict:
        return {
            "admitted": self.admit,
            "reason": self.reason,
            "delta_mean": self.mean,
            "delta_se": self.se,
            "delta_t": self.t_stat,
            "delta_per_seed": list(self.deltas),
            "displaced": self.displaced,
            "pathology": self.pathology,
            "verdict": self.verdict.value,
        }


@dataclass
class PopulationStats:
    """Every batch's utility, admitted or not -- the feedback reference.

    Kept separate from the repository on purpose. The repository is a
    winners-only sample and makes a poor yardstick for "how is generation
    going"; this one is the actual distribution the generator produces.
    """

    utilities: list[float] = field(default_factory=list)

    def observe(self, u: float) -> None:
        if u is not None and u == u:
            self.utilities.append(float(u))

    def percentile(self, u: float) -> float:
        """Fraction of all evaluated batches this one is at least as good as."""
        if not self.utilities or u is None or u != u:
            return float("nan")
        return sum(1 for v in self.utilities if v <= u) / len(self.utilities)

    def summary(self, u: float) -> str:
        pct = self.percentile(u)
        if pct != pct:
            return "no population yet"
        return (f"{100 * pct:.0f}th percentile of {len(self.utilities)} evaluated "
                f"batch(es)")


def check_pathology(metrics: dict, theta) -> tuple[str | None, Verdict | None]:
    """A reason to reject that is not about quality, or ``(None, None)``.

    Deliberately narrow. Each condition describes a signal that cannot be
    priced or is not a distinct bet -- never one that is merely weak. Returns
    the prose ``why`` AND the structured ``Verdict`` so the caller does not have
    to re-parse the prose to recover the verdict (the brittle substring path T1
    retires to a fallback).
    """
    adm = theta.admission
    coverage = metrics.get("coverage")
    if coverage is not None and coverage == coverage and coverage < adm.min_coverage:
        return (f"coverage {coverage:.1%} < min_coverage {adm.min_coverage:.0%} "
                "(too sparse to price)", Verdict.TOO_SPARSE)

    for key in ("signal_std", "std"):
        s = metrics.get(key)
        if s is not None and s == s and float(s) <= 0.0:
            return "signal is constant (zero variance)", Verdict.CONSTANT

    rho = metrics.get("rho_max")
    bar = theta.gates.rho_bar
    if bar is not None and rho is not None and rho == rho and float(rho) > float(bar):
        return (f"rho_max {float(rho):.3f} > rho_bar {float(bar):.2f} (duplicate)",
                Verdict.REDUNDANT)
    return None, None


def _mean_se(deltas) -> tuple[float, float]:
    vals = [float(d) for d in deltas if d is not None and d == d]
    if not vals:
        return float("nan"), float("nan")
    mean = st.mean(vals)
    if len(vals) < 2:
        # One measurement carries no information about its own spread. Report
        # an infinite standard error so the test cannot pass on it by accident;
        # the caller decides whether a single-seed run is acceptable.
        return mean, float("inf")
    return mean, st.stdev(vals) / math.sqrt(len(vals))


def decide(
    deltas,
    zoo_size: int,
    theta,
    metrics: dict | None = None,
    weakest: tuple[str, float] | None = None,
) -> Decision:
    """Admit, reject, or replace.

    ``deltas`` is one marginal contribution per combiner seed -- the paired
    difference between the book with these factors and the book without them.
    ``weakest`` is the incumbent with the smallest contribution and its value,
    consulted only once the repository is at capacity.
    """
    adm = theta.admission

    if not adm.enabled:
        return Decision(True, "gating disabled", verdict=Verdict.NO_DATA)

    if metrics:
        why, pverdict = check_pathology(metrics, theta)
        if why:
            return Decision(False, f"pathology: {why}", pathology=why, verdict=pverdict)

    if zoo_size < adm.min_size:
        # Still bootstrapping. Note what this no longer implies: because the bar
        # below is measured against the CURRENT book rather than against these
        # batches, an unusually good bootstrap draw no longer sets a standard
        # that later batches must beat. Bootstrapping fills the repository; it
        # does not calibrate the gate.
        return Decision(True, f"bootstrapping (|zoo|={zoo_size} < {adm.min_size})",
                        verdict=Verdict.BOOTSTRAP)

    mean, se = _mean_se(deltas)
    if mean != mean:
        return Decision(False, "no usable marginal contribution measured",
                        verdict=Verdict.NO_DATA)

    # se == 0 means every seed returned the SAME delta -- the seeds agreed
    # exactly, which is unanimity, not absence of evidence. It happens whenever
    # the combiner's per-seed bootstrap cannot move the weights: with
    # ``shrinkage: "auto"`` on weak factors the Ledoit-Wolf intensity saturates
    # at delta=1, every seed collapses to the same equal-weight vector, and the
    # five evaluations become one repeated five times.
    #
    # The sign still has to carry through. Mapping se==0 to t=0.0 for a negative
    # mean called a unanimously HARMFUL batch (mean=-0.73 on the first real
    # decision of the smoke run) "not resolved", which reads as a measurement
    # problem and, worse, routes the refinement through the MARGINAL directive
    # table instead of NET_HARMFUL -- telling the generator to push harder on a
    # direction that is actively hurting the book. Rejection was already
    # correct; the *diagnosis* was not. -inf mirrors the +inf the positive
    # branch already used.
    t = mean / se if se not in (0.0, float("inf")) and se == se else (
        (float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0)
        if se == 0.0 else 0.0)
    bar = float(adm.k_sigma)

    if t <= bar:
        # One branch, two different failures, and the ledger should not conflate
        # them now that ``reason`` is persisted and read as analysis input. A
        # batch with mean < 0 and a large |t| is not unresolved -- it is
        # resolvably HARMFUL, measured precisely enough to be sure the book gets
        # worse. Calling that "not resolved" reads as a measurement problem when
        # it is a clean negative result, and the two say opposite things about
        # how the generator is doing. The verdict is identical either way --
        # and now it is carried structurally too, so a prose rewording can no
        # longer flip NET_HARMFUL into MARGINAL downstream.
        # The verdict bar is DECOUPLED from the admission bar (2026-08-15).
        #
        # Both branches below already return Decision(False, ...) -- admission is
        # settled by `t <= bar` above, and the verdict only LABELS the rejection
        # for the diagnosis router and the ledger. So the two thresholds answer
        # different questions and should not share a number:
        #
        #   admission  -- a decision with a cost (a false positive pollutes the
        #                 book), so it is deliberately conservative. Raising
        #                 k_sigma to 3.0 (Harvey-Liu-Zhu) is right for that.
        #   verdict    -- feedback. Being conservative here does not prevent a
        #                 bad outcome, it just withholds information: a batch at
        #                 t=-2.5 IS resolvably harmful, and telling the generator
        #                 "not resolved" says a measurement problem occurred when
        #                 a clean negative result did.
        #
        # Tying them meant raising k_sigma silently relabelled every batch
        # between -k_sigma and -1 sigma from NET_HARMFUL to MARGINAL, degrading
        # the feedback channel as a side effect of tightening selection.
        # Deliberately NOT a Θ field: the verdict is diagnostic, not part of the
        # scored objective, and Θ is frozen.
        if mean < 0 and t < -_VERDICT_SIGMA:
            why = (f"contribution is resolvably NEGATIVE: mean {mean:+.5f} "
                   f"(t={t:.2f}, se {se:.5f}) -- the book gets worse with this batch")
            v = Verdict.NET_HARMFUL
        else:
            why = (f"contribution not resolved: mean {mean:+.5f} "
                   f"<= {bar:g}x se {se:.5f} (t={t:.2f})")
            v = Verdict.MARGINAL
        # Non-blocking mode: the measurement and the verdict are unchanged --
        # diagnosis still learns that this batch was harmful or unresolved --
        # but the batch enters the pool anyway. Selection for the BOOK is a
        # separate, later decision; this one only decides what the search is
        # allowed to remember.
        if not getattr(adm, "blocking", True):
            return Decision(True, f"observed (non-blocking): {why}",
                            tuple(deltas), mean, se, t, verdict=v)
        return Decision(False, why, tuple(deltas), mean, se, t, verdict=v)

    capacity = adm.capacity
    if capacity and zoo_size >= capacity:
        if weakest is None:
            return Decision(False, f"repository full ({zoo_size}) and no incumbent to displace",
                            tuple(deltas), mean, se, t, verdict=Verdict.FULL)
        name, worst = weakest
        if mean <= worst:
            return Decision(False,
                            f"repository full: contribution {mean:+.5f} does not beat "
                            f"the weakest incumbent {worst:+.5f}",
                            tuple(deltas), mean, se, t, verdict=Verdict.FULL)
        return Decision(True,
                        f"replaces {name} ({mean:+.5f} > {worst:+.5f}), t={t:.2f}",
                        tuple(deltas), mean, se, t, displaced=name, verdict=Verdict.REPLACED)

    return Decision(True, f"contribution {mean:+.5f} > {bar:g}x se {se:.5f} (t={t:.2f})",
                    tuple(deltas), mean, se, t, verdict=Verdict.ADMITTED)


def should_evict(contribution: float, theta) -> bool:
    """Whether an incumbent has stopped earning its place.

    Re-measured against the repository *as it now stands*, which is the whole
    point: a factor that was additive when the zoo held twelve members can be
    redundant at a hundred and fifty, and nothing about its own metrics changes
    to say so. Eviction uses a slacker bar than admission -- a member has to be
    actively unhelpful, not merely unproven -- so a factor does not oscillate in
    and out on measurement noise.
    """
    if contribution != contribution:
        return False
    return contribution < float(theta.admission.evict_below)


__all__ = ["Decision", "PopulationStats", "check_pathology", "decide", "should_evict"]
