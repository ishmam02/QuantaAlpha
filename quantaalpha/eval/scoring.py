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
    # Rank on MARGINAL contribution, not the book's absolute performance:
    # candidates sharing a zoo produce near-identical absolute net_ir, so
    # ranking on it barely discriminates. Must match what the gate admits on.
    "effectiveness": ("delta_net_ir", True),
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

    ``0.0`` for an empty zoo, so ``e_j = 1``: the first factor is trivially the
    best thing in the book, which is the only coherent convention when there is
    nothing to compare against.
    """
    values = [v for v in incumbent_values if _usable(v)]
    if not values:
        return 0.0
    beaten_by = sum(1 for v in values if value < v)
    return float(beaten_by) / float(len(values))


def _overfit_score(m: dict, theta: Protocol) -> float:
    """The documented exception to repository ranking (§3.4).

    Over-fitting risk is "assessed from the factor's structure and refinement
    history rather than by repository rank", so it is computed directly in
    [0, 1]: half from structural parsimony, half from how much the IS edge
    survived into the OOS proxy.
    """
    complexity = m.get("cx")
    gap = m.get("is_oos_gap")

    cx_term = 1.0
    if _usable(complexity) and theta.gates.gamma_cx:
        cx_term = 1.0 - min(1.0, float(complexity) / float(theta.gates.gamma_cx))

    gap_term = 1.0
    if _usable(gap) and theta.overfit.delta_ref:
        gap_term = 1.0 - min(1.0, max(0.0, float(gap)) / float(theta.overfit.delta_ref))

    return float(0.5 * cx_term + 0.5 * gap_term)


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
