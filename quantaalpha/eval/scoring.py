"""Repository-relative scoring and the utility ``U`` (Eq. 10-12).

Each dimension is scored by **rank against the current repository** rather
than by absolute value::

    R(f, m, zoo) = (1/|zoo|) · Σ 1[m(f) < m(f')]        (Eq. 11)
    e_j          = 1 − R                                 (Eq. 10)
    U            = Σ_j ω_j · e_j                         (Eq. 12)

Two things are easy to get wrong here and both are load-bearing:

1. **Effectiveness ranks on net IR, not on RankIC.** RankIC is reported, and
   RankIC is what the *gate* uses, but the *score* is the realized economic
   edge after costs. That distinction is the whole difference between this
   objective and a relabelled IC.
2. **Lower-is-better dimensions are negated before ranking** — turnover and
   ρ_max. Miss a sign flip and the search will actively pursue redundant,
   high-turnover factors while the numbers look fine.

There is deliberately **no ``λR(f)`` penalty term**. Novelty is handled by the
ρ_max gate plus the diversity dimension; complexity by the cx gate plus the
over-fitting dimension. Adding a penalty would double-count both.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from quantaalpha.eval.protocol import Protocol

# dimension -> (metric key, higher_is_better)
RANKED_DIMENSIONS: dict[str, tuple[str, bool]] = {
    # Ranks on the book's absolute risk-adjusted return, NOT on marginal
    # contribution. Marginal contribution was the original choice, on the
    # reasoning that candidates sharing a zoo produce near-identical absolute
    # net_ir. Two independent measurements say otherwise:
    #
    #   * delta_net_ir gating flipped 3 of 4 verdicts across combiner seeds --
    #     the deltas are smaller than the noise on them;
    #   * delta necessarily decays toward zero as the repository grows, so
    #     every batch ties at rank 0 and the heaviest dimension (omega 0.30)
    #     turns into noise exactly when the repository starts working. Observed
    #     live on run 20260731_144810: batches 5-10 all scored e_effectiveness
    #     = 0.00, freezing the zoo at 12, and the gate rejected the best batch
    #     of the entire run (net ARR -1.3%) while having admitted one at
    #     -2.80%. A bar that shuts as the repository improves is a ratchet, not
    #     a quality criterion.
    #
    # Replaying that ledger: ranking on delta_net_ir admits 4/10 and rejects
    # the best batch; on net_ir it admits 7/10, rejects the worst (-10.2%) and
    # admits the best. net_ir rather than net_arr because `arr` already ranks
    # net_arr -- reusing it here would put 0.50 of the weight on one quantity
    # under two names, where net_ir is risk-adjusted and stays distinct.
    "effectiveness": ("net_ir", True),
    "arr": ("net_arr", True),
    "stability": ("rank_icir", True),
    "turnover": ("turnover_book", False),
    "diversity": ("rho_max", False),
    "decay": ("decay_slope", True),
}


def _usable(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def rank(value: float, incumbent_values: Iterable[float]) -> float:
    """R (Eq. 11) — fraction of the repository that beats ``value``.

    Eq. 11 carries a ``1/|zoo|`` factor and is therefore undefined at
    ``|zoo| = 0``; this is where that case is pinned down.

    **Neutral (``0.5``), not ``0.0``.** Returning ``0.0`` makes ``e_j = 1`` on
    every dimension, so a batch ranked against nothing scores near the maximum
    ``U`` -- and since evolutionary fitness is ``U`` under the net-cost objective,
    the very first batch becomes a fitness leader no legitimately-scored batch
    can overtake. Measured on run 20260728_151227: batch 1 scored ``U=0.9500``
    against an empty repository while batches 2-6, scored against real
    incumbents, reached at most ``0.3750`` -- and those five ranked in exactly
    their net-ARR order, so the ranking was working and only the empty case was
    broken. With nothing to compare against, a candidate is neither above nor
    below the repository, which is what ``0.5`` says.
    """
    values = [v for v in incumbent_values if _usable(v)]
    if not values:
        return 0.5
    beaten_by = sum(1 for v in values if value < v)
    return float(beaten_by) / float(len(values))


def _overfit_score(m: dict, theta: Protocol) -> float:
    """The documented exception to repository ranking (§3.4).

    Over-fitting risk is "assessed from the factor's structure and refinement
    history rather than by repository rank", so it is computed directly in
    [0, 1]: half from structural parsimony, half from how much the IS edge
    survived into the OOS proxy.

    Each half is included only when it can actually be computed, and the score
    is the mean of whatever is available. The previous version defaulted a
    missing term to ``1.0`` and averaged it in regardless, so with
    ``gamma_cx: null`` -- the shipped configuration -- ``cx_term`` was pinned at
    1.0 on every candidate and half of this dimension was a constant. At
    ``omega["overfit"] = 0.10`` that made 0.05 of the total utility weight
    inert while still appearing in the objective.
    """
    complexity = m.get("cx")
    gap = m.get("is_oos_gap")

    terms: list[float] = []
    if _usable(complexity) and theta.gates.gamma_cx:
        terms.append(1.0 - min(1.0, float(complexity) / float(theta.gates.gamma_cx)))
    if _usable(gap) and theta.overfit.delta_ref:
        terms.append(1.0 - min(1.0, max(0.0, float(gap)) / float(theta.overfit.delta_ref)))

    if not terms:
        # Nothing measurable: score neutrally rather than award a free 1.0.
        return 0.5
    return float(sum(terms) / len(terms))


def dimension_scores(m: dict, zoo_metrics: Sequence[dict], theta: Protocol) -> dict[str, float]:
    """One ``e_j`` per dimension named in ``Θ.utility.omega``."""
    scores: dict[str, float] = {}

    for dimension in theta.utility.omega:
        if dimension == "overfit":
            scores["overfit"] = _overfit_score(m, theta)
            continue

        spec = RANKED_DIMENSIONS.get(dimension)
        if spec is None:
            continue
        key, higher_is_better = spec

        value = m.get(key)
        if not _usable(value):
            # Nothing to rank on: score neutrally rather than penalising a
            # factor for a metric the engine could not compute.
            scores[dimension] = 0.5
            continue

        sign = 1.0 if higher_is_better else -1.0
        incumbents = [sign * float(z[key]) for z in zoo_metrics if _usable(z.get(key))]
        scores[dimension] = 1.0 - rank(sign * float(value), incumbents)

    return scores


def utility(m: dict, zoo_metrics: Sequence[dict], theta: Protocol) -> float:
    """U (Eq. 12) — the scalar the search maximizes."""
    mode = theta.utility.mode
    if mode == "net_ir":
        value = m.get("net_ir")
        return float(value) if _usable(value) else float("nan")
    if mode == "net_arr":
        value = m.get("net_arr")
        return float(value) if _usable(value) else float("nan")

    scores = dimension_scores(m, zoo_metrics, theta)
    return float(sum(theta.utility.omega.get(k, 0.0) * v for k, v in scores.items()))


__all__ = ["RANKED_DIMENSIONS", "dimension_scores", "rank", "utility"]
