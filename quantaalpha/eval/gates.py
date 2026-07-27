"""``F_Θ`` — hard feasibility gates (Eq. 9).

Admissibility and scoring are **separate steps**, and this is the admissibility
one. Gates are *fixed absolute floors* — a property of the market and the
existing book, not a bar that drifts with search progress. Scoring
(``scoring.py``) is repository-relative. Do not let a threshold leak from one
into the other: a gate that moved with the population would let the search
congratulate itself for getting worse.

Thresholds come from Θ, **not** from ``configs/experiment.yaml``. That file's
``quality_gate`` block is largely decorative in the current tree — only
``consistency_enabled`` is actually threaded, ``complexity_enabled`` and
``redundancy_enabled`` are hardcoded ``True`` at ``proposal.py:360-361``, and
the thresholds are never read at all.

Evaluated on train / valid only (Property 3). The operator passes IS and
OOS-proxy metrics; the test window never reaches here.
"""

from __future__ import annotations

import math

from quantaalpha.eval.protocol import Protocol


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def turnover_value(m: dict, theta: Protocol):
    """The turnover the gate reads, per ``Θ.gates.turnover_basis``.

    ``book`` is faithful to Eq. 6 but is structurally capped at
    ``n_drop/topk`` (0.10), so ``τ_max = 0.30`` cannot bind under top-k
    dropout — capacity pressure comes from κ₂ instead. ``solo`` is the variant
    that does discriminate between signals, offered for the sensitivity pass.
    """
    key = "turnover_book" if theta.gates.turnover_basis == "book" else "turnover_solo"
    return m.get(key)


def feasible(m: dict, theta: Protocol) -> tuple[bool, list[str]]:
    """Return ``(is_feasible, failed_gate_names)``.

    The list matters as much as the boolean: the summarizer turns it into a
    rejection *reason* for the generator, and "RankICIR 0.11 < γ_ir 0.20" is
    far more actionable than a score.

    A metric that is missing or NaN does not fail its gate — it cannot be
    judged, and failing it would silently reject every factor whenever an
    upstream metric went unavailable.
    """
    failed: list[str] = []

    checks = (
        # PRIMARY admissibility: does f improve the book it joins? Stand-alone
        # RankIC is an average over the whole cross-section, but the top-k book
        # is a bet on its extreme tail, so the two come apart -- the pilot's 18
        # factors all failed the RankIC gate yet were worth +7pp of annual
        # return over the base-features-only null model.
        ("delta_net_ir", m.get("delta_net_ir"), theta.gates.gamma_delta, "min"),
        # Liveness floors: is the signal predictive at all? Deliberately low --
        # these catch dead or inverted signals, they do not judge quality.
        ("rank_ic", m.get("rank_ic"), theta.gates.gamma_ic, "min"),
        ("rank_icir", m.get("rank_icir"), theta.gates.gamma_ir, "min"),
        ("turnover", turnover_value(m, theta), theta.gates.tau_max, "max"),
        ("rho_max", m.get("rho_max"), theta.gates.rho_bar, "max"),
        ("cx", m.get("cx"), theta.gates.gamma_cx, "max"),
    )

    for name, value, threshold, direction in checks:
        # A null threshold disables the gate. Used for the stand-alone IC floors,
        # which measurement showed to be anti-correlated with what they were
        # meant to proxy: on the pilot's 18 factors the best contributor
        # (+0.1893 net IR) had rank_ic +0.0004, several strong contributors had
        # NEGATIVE rank_ic, and the best-IC factor (+0.0237) was the worst
        # contributor (-0.0833). Keeping them as a "liveness" floor rejected all
        # nine factors that actually improved the book.
        if threshold is None or _missing(value):
            continue
        if direction == "min" and float(value) < float(threshold):
            failed.append(name)
        elif direction == "max" and float(value) > float(threshold):
            failed.append(name)

    return (not failed), failed


def describe_failures(m: dict, theta: Protocol, failed: list[str]) -> list[str]:
    """Human-readable "value vs threshold" lines, for the LLM feedback."""
    thresholds = {
        "delta_net_ir": ("Marginal net IR vs the book without it",
                         m.get("delta_net_ir"), theta.gates.gamma_delta, "<", "gamma_delta"),
        "rank_ic": ("RankIC", m.get("rank_ic"), theta.gates.gamma_ic, "<", "gamma_ic"),
        "rank_icir": ("RankICIR", m.get("rank_icir"), theta.gates.gamma_ir, "<", "gamma_ir"),
        "turnover": ("Turnover", turnover_value(m, theta), theta.gates.tau_max, ">", "tau_max"),
        "rho_max": ("rho_max", m.get("rho_max"), theta.gates.rho_bar, ">", "rho_bar"),
        "cx": ("Complexity", m.get("cx"), theta.gates.gamma_cx, ">", "gamma_cx"),
    }
    lines = []
    for name in failed:
        if name not in thresholds:
            continue
        label, value, threshold, op, sym = thresholds[name]
        try:
            shown = f"{float(value):.4f}"
        except (TypeError, ValueError):
            shown = str(value)
        lines.append(f"{label} {shown} {op} {sym} {threshold}")
    return lines


__all__ = ["describe_failures", "feasible", "turnover_value"]
