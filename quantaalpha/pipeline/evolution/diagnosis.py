"""Diagnose-and-refine: turn a parent's rejection into a directional refinement.

The paper's mutation (Eq. 6) is "self-reflect to diagnose the faulty decision
node k, freeze the prefix, refine only a_k." The paper leaves the
self-reflection step -- the prompt and the output schema for k -- **unspecified**,
so this module designs it. The raw signal the diagnosis needs already lives on
the trajectory (``admitted``, ``reason``/``pathology``, ``weakest_dimensions``,
``e_*``); the only missing piece was the verdict string, which ``_to_series``
dropped until T0.1 threaded it through.

Design choice (user-confirmed Q1, 2026-08-13): **hybrid**. The structured fields
(verdict / weakness_dimension / refine_target / frozen_layers / mechanism_hint)
are produced **deterministically** by a rule table -- the directional search
signal is guaranteed, not left to an LLM's mood. An optional LLM call (in the
RefinementOperator) may rewrite ``directive_text`` for fluency; the structured
fields never depend on it.

Design choice (Q2): k ∈ {hypothesis, expression}. Cost / turnover / overfit /
diversity / decay faults freeze the hypothesis and refine the **expression**
(the cost-stall case -- the signal is sound, the construction is too expensive).
Signal-quality faults (effectiveness / arr / stability) rewrite the
**hypothesis** (the premise itself is the problem) and re-derive the expression.

This is the paper's "directional search signal, allowing even imperfect
localization to move the child toward a different region," made concrete and
reproducible.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from quantaalpha.core.verdict import Verdict
from .segment_profiling import build_segments, SegmentProfile

logger = logging.getLogger(__name__)


# ``Verdict`` is imported from ``quantaalpha.core.verdict`` (see top of file) so
# ``eval.admission`` can emit the structured verdict without importing from this
# pipeline module (layering: eval must not depend on pipeline). Re-exported here
# via the import for backward-compatible ``from ...diagnosis import Verdict``.


class RefineTarget(str, Enum):
    """Which layer of the trajectory Eq. 6's k points at (Q2: no code layer)."""

    HYPOTHESIS = "hypothesis"  # rewrite the premise; expression re-derived
    EXPRESSION = "expression"  # keep the hypothesis; refine the construction
    SIGN = "sign"  # freeze the construction; correct the direction, re-test
    # SIGN is the first-class deterministic refine for a parent whose
    # pre-registered IC direction was OPPOSITE the measured one while the edge
    # cleared the bar (|t| >= k_sigma): the construction has a real edge, the
    # LABEL was backwards. No LLM call -- the expression is frozen verbatim, the
    # direction is corrected to the measured sign, the hypothesis records the
    # correction, and the factor is re-tested. See ``sign_flip_directive``.
    RECOMBINE = "recombine"  # crossover: author a NEW child from two parents'
    # validated strengths (Eq. 7 idea-recombination). Nothing is frozen -- the
    # child is a fresh hypothesis + fresh expression combining the two best
    # parents' complementary validated ideas, not a literal AST fusion. The
    # strength diagnosis (strength_diagnosis.py) locates what worked in each
    # parent; the crossover LLM authors the child. See ``strength_diagnosis``.


# Which property of the factor a dimension measures. The directive table is
# keyed by (verdict, category) so a turnover weakness and a diversity weakness
# get different instructions even under the same verdict.
_DIMENSION_CATEGORY: dict[str, str] = {
    "turnover": "cost",          # lower is better -- the cost-stall lever
    "diversity": "redundancy",   # lower rho_max is better
    "overfit": "overfit",       # structural parsimony / OOS survival
    "effectiveness": "signal",  # net_ir of the book (higher is better)
    "arr": "signal",            # net annualised return
    "stability": "signal",      # RankICIR
    "decay": "decay",           # edge fading across OOS
}


def _dimension_category(dim: str | None) -> str:
    if not dim:
        return "unknown"
    return _DIMENSION_CATEGORY.get(dim, "unknown")


@dataclass
class RefinementDirective:
    """The structured output of the self-reflection step.

    Everything an operator needs to breed a *refinement* of the parent rather
    than an orthogonal restart. The structured fields are deterministic; only
    ``directive_text`` may be rewritten by an LLM for fluency (Q1 hybrid).
    """

    verdict: Verdict
    weakness_dimension: str | None
    refine_target: RefineTarget
    # Layers kept verbatim from the parent. ``["hypothesis"]`` for an
    # expression-refine (the premise is sound); ``[]`` for a hypothesis-refine
    # (the premise itself changes, so nothing is carried verbatim).
    frozen_layers: list[str] = field(default_factory=list)
    mechanism_hint: str = ""
    directive_text: str = ""
    # T3 multi-segment: one entry per severe weakness dimension, each naming the
    # actual operator + parameter + factor located in the parent's AST (fix #1:
    # more than one weakness; fix #2: expression-aware; fix #3: the param is the
    # magnitude). The primary (most-severe) target's mechanism_hint mirrors the
    # scalar ``mechanism_hint`` above for backward compatibility. Empty for the
    # structural / winner fallbacks, which are single-purpose.
    targets: list[dict] = field(default_factory=list)
    # The frozen prefix itself -- the parent's hypothesis and factor expressions
    # the child builds on. Carried as plain data so it survives task
    # serialization into a parallel child process (where ``parent_trajectories``
    # is stripped; see ``factor_mining._serialize_task_for_parallel``).
    parent_hypothesis: str = ""
    # The parent's PRE-REGISTERED IC sign. An expression-refine freezes the
    # premise, so the premise's directional prediction still stands and must
    # travel with it. Without this the frozen hypothesis arrives with no
    # direction and the falsifiability gate rejects every refine child --
    # which silenced the one operator measured to actually improve factors.
    parent_expected_ic_sign: str = ""
    parent_factors: list[dict[str, Any]] = field(default_factory=list)
    # The raw metric vector, for the optional LLM rationale step and for audit.
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    # T5: set when BOTH the expression and hypothesis levers have been pulled
    # >=K times along the lineage without admission -- the refinement direction
    # is exhausted, so the caller should breed an orthogonal restart instead.
    # ``is_refinement()`` returns False when this is set, routing the parent to
    # ORTHOGONAL in the T6 selection rewire.
    exhausted_lever: bool = False
    # Per-segment solo measurement of the parent's expression (the ablation
    # evaluator's measurement-only summary -- which sub-tree carries the rank
    # edge, which temporal window is IC-neutral, whether the core's sign is
    # stable across sub-samples). Empty when the ablation is off
    # (QA_ABLATION_DIAGNOSIS) or the parent has no factor / the eval failed.
    # Measurement only -- never a remedy; the summary closes with "how to fix
    # that is yours to determine". See ``segment_ablation``. The table path also
    # uses the ablation OBJECT (not just this string) in ``_build_target`` to
    # route on the broken part + apply the IC-neutral-window backstop.
    ablation_summary: str = ""

    def is_refinement(self) -> bool:
        """False when the verdict gives no basis to refine, or the lever is
        exhausted (both expression and hypothesis tried >=K times without
        admission -> fall back to orthogonal)."""
        return self.verdict not in (Verdict.NO_DATA,) and not self.exhausted_lever

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for task serialization (must survive a Process pickle)."""
        return {
            "verdict": self.verdict.value,
            "weakness_dimension": self.weakness_dimension,
            "refine_target": self.refine_target.value,
            "frozen_layers": list(self.frozen_layers),
            "mechanism_hint": self.mechanism_hint,
            "directive_text": self.directive_text,
            "parent_hypothesis": self.parent_hypothesis,
            "parent_expected_ic_sign": self.parent_expected_ic_sign,
            "parent_factors": [dict(f) for f in self.parent_factors],
            "targets": [dict(t) for t in self.targets],
            "exhausted_lever": self.exhausted_lever,
            "ablation_summary": self.ablation_summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RefinementDirective":
        return cls(
            verdict=Verdict(d.get("verdict", "no_data")),
            weakness_dimension=d.get("weakness_dimension"),
            refine_target=RefineTarget(d.get("refine_target", "expression")),
            frozen_layers=list(d.get("frozen_layers", [])),
            mechanism_hint=d.get("mechanism_hint", ""),
            directive_text=d.get("directive_text", ""),
            parent_hypothesis=d.get("parent_hypothesis", ""),
            parent_expected_ic_sign=d.get("parent_expected_ic_sign", ""),
            parent_factors=list(d.get("parent_factors", [])),
            targets=[dict(t) for t in d.get("targets", [])],
            raw_metrics={},  # not serialized; reconstruct from the trajectory if needed
            exhausted_lever=bool(d.get("exhausted_lever", False)),
            ablation_summary=d.get("ablation_summary", ""),
        )


# --------------------------------------------------------------------------
# Verdict parsing
# --------------------------------------------------------------------------

def classify_verdict(metrics: dict[str, Any]) -> Verdict:
    """Parse ``admission.decide``'s record into a Verdict.

    Reads the structured ``verdict`` field first (T1): ``admission.decide`` sets
    it at each branch and ``Decision.as_record`` serializes it to ``.value``, so
    the authoritative verdict is carried as a plain string -- no prose to
    re-parse. Falls back to the legacy substring classification of
    ``reason`` / ``pathology`` **only** when the field is absent (records written
    before T1), and logs a warning when it does -- so a prose rewording can no
    longer silently degrade every reject to MARGINAL.

    Order matters in the fallback: pathology rejections are structural (a
    duplicate / unpriceable signal is a different failure from a weak
    contribution) and ``check_pathology`` runs before the marginal-contribution
    test in ``decide``, so they appear as a ``pathology: ...`` reason. The
    admission flag then disambiguates the contribution verdicts.
    """
    raw = metrics.get("verdict")
    if raw is not None and raw != "":
        try:
            return raw if isinstance(raw, Verdict) else Verdict(str(raw))
        except ValueError:
            # An unrecognized verdict string (a future value, or a corrupt
            # record) is better served by the legacy classification than a crash.
            logger.warning(
                "verdict field present but unrecognized (%r); falling back to "
                "substring classification for reason=%r",
                raw, metrics.get("reason"))

    # Legacy / fallback path: re-derive the verdict from the prose. Logged so a
    # prose rewording that would silently degrade classification is loud instead.
    logger.warning(
        "verdict field absent; falling back to substring classification for "
        "reason=%r", metrics.get("reason"))

    admitted = bool(metrics.get("admitted", True))
    reason = str(metrics.get("reason") or "")
    pathology = str(metrics.get("pathology") or "")

    # Pathology rejections (check_pathology) -- structural, not contribution.
    if pathology or reason.startswith("pathology:"):
        why = pathology or reason
        if "rho_max" in why or "duplicate" in why:
            return Verdict.REDUNDANT
        if "coverage" in why or "sparse" in why:
            return Verdict.TOO_SPARSE
        if "constant" in why or "zero variance" in why:
            return Verdict.CONSTANT
        return Verdict.TOO_SPARSE  # unknown pathology -> treat as unpriceable

    if "no usable marginal contribution" in reason or "gating disabled" in reason:
        return Verdict.NO_DATA

    if admitted:
        if reason.startswith("replaces"):
            return Verdict.REPLACED
        if "bootstrapping" in reason:
            return Verdict.BOOTSTRAP
        return Verdict.ADMITTED

    # Rejected on contribution. Split the two failures the ledger must not
    # conflate (see admission.decide): resolvably NEGATIVE vs not resolved.
    if "resolvably NEGATIVE" in reason or "book gets worse" in reason:
        return Verdict.NET_HARMFUL
    if "repository full" in reason:
        return Verdict.FULL
    if "not resolved" in reason or "contribution" in reason:
        return Verdict.MARGINAL
    return Verdict.MARGINAL


# A dimension is a refinement target when its e_* score is below this. The e_*
# scores are [0,1] "how good is this property", so a low score is a weakness;
# 0.25 lets the clearly-weak dims (0.05-0.20 in the run) count while excluding
# the merely-middling (0.30-0.40). If nothing qualifies, the single weakest is
# still returned so a directive always carries at least one target.
_SEVERITY_THRESHOLD = 0.25
_MAX_TARGETS = 3


def _parse_weakest_dimensions_str(weak: str) -> list[tuple[str, float]]:
    """Parse "turnover (e=0.10), arr (e=0.18)" -> [(turnover, 0.10), (arr, 0.18)]."""
    out: list[tuple[str, float]] = []
    for part in weak.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split("(")[0].strip()
        e = float("nan")
        m = re.search(r"e=([-\d.]+)", part)
        if m:
            try:
                e = float(m.group(1))
            except ValueError:
                pass
        if name:
            out.append((name, e))
    return out


def weakest_dimensions(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    """All severe weakness dimensions, most-severe first (fix #1: not just one).

    Sources the ``weakest_dimensions`` string ("turnover (e=0.10), arr (e=0.18)")
    and the ``e_*`` scalar vector, keeps dims whose e_* < ``_SEVERITY_THRESHOLD``
    (or, if none qualify, the single weakest), and returns up to
    ``_MAX_TARGETS``. Each entry drives one targeted sub-tree refine in the
    directive (T3 multi-segment).
    """
    scored: dict[str, float] = {}
    weak = metrics.get("weakest_dimensions")
    if weak and isinstance(weak, str):
        for name, e in _parse_weakest_dimensions_str(weak):
            scored[name] = e
    # Fold in any e_* keys the string did not list; prefer the scalar when both
    # exist (it is the authoritative score).
    for k, v in metrics.items():
        if k.startswith("e_") and isinstance(v, (int, float)) and v == v:
            scored[k[2:]] = float(v)

    if not scored:
        return []
    # Sort ascending by score (lowest = weakest); NaN scores sort last via inf.
    items = sorted(scored.items(),
                   key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) and kv[1] == kv[1]
                                   else float("inf")))
    severe = [(d, e) for d, e in items
              if isinstance(e, (int, float)) and e == e and e < _SEVERITY_THRESHOLD]
    if not severe:
        severe = [items[0]]  # always at least the weakest
    return severe[:_MAX_TARGETS]


def _weakest_dimension(metrics: dict[str, Any]) -> str | None:
    """The single weakest dimension (backward-compat wrapper over weakest_dimensions)."""
    dims = weakest_dimensions(metrics)
    return dims[0][0] if dims else None


# --------------------------------------------------------------------------
# The directive table: (verdict, dimension_category) -> refinement instruction
# --------------------------------------------------------------------------

# Each entry: (refine_target, frozen_layers, base_text). The base_text is a
# template rendered in diagnose() with the parent's metrics; it gives the
# high-level guidance for the (verdict, dimension_category) pair. The
# expression-aware PER-TARGET mechanism (the actual operator + parameter +
# factor located in the parent's AST, fix #2/#3) is built by ``_build_target``
# below from the T2 ``SegmentProfile``s and appended to the directive text.
# frozen_layers == ["hypothesis"] means "keep the parent hypothesis verbatim";
# [] means "the hypothesis itself is rewritten".
_DIRECTIVES: dict[tuple[Verdict, str], tuple[RefineTarget, list[str], str]] = {
    # ---- Cost / turnover: the cost-stall case. The signal is sound; the
    # construction is too expensive. Freeze the hypothesis, refine the
    # expression to trade less for the same alpha. -------------------------
    (Verdict.NET_HARMFUL, "cost"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's contribution to the book was resolvably "
        "NEGATIVE. Its signal has predictive content (RankIC {rank_ic:+.4f}) but "
        "the book it produces turns over {turnover_book:.4f} per day, and the "
        "trading cost of that turnover exceeds the edge. "
        "The premise is not what failed -- the CONSTRUCTION is. Keep the "
        "hypothesis and change the expression so the same signal is expressed at "
        "a cost the edge can pay. How to do that is yours to determine."),
    (Verdict.MARGINAL, "cost"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's contribution could not be resolved from noise "
        "({reason}). Its signal has predictive content (RankIC {rank_ic:+.4f}) but "
        "turnover is {turnover_book:.4f} per day and the net contribution after "
        "cost is indistinguishable from zero. "
        "Keep the hypothesis and change the expression so the net contribution "
        "clears the noise. How to do that is yours to determine."),
    # ---- Redundancy: the factor duplicates the zoo. ----------------------
    (Verdict.NET_HARMFUL, "redundancy"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's maximum correlation with a factor already in "
        "the repository is rho_max {rho_max:.3f}. At that correlation it carries "
        "almost no information the book does not already hold, and its "
        "contribution was resolvably NEGATIVE. "
        "Keep the hypothesis and change the expression so the factor is less "
        "correlated with what the repository already captures. How to achieve "
        "that decorrelation is yours to determine."),
    (Verdict.MARGINAL, "redundancy"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's maximum correlation with an existing "
        "repository factor is rho_max {rho_max:.3f}, and its contribution could "
        "not be resolved from noise. It largely restates information the book "
        "already has. "
        "Keep the hypothesis and change the expression so it is less correlated "
        "with the repository. How to achieve that is yours to determine."),
    # ---- Overfit: the edge did not survive out of sample. ----------------
    (Verdict.NET_HARMFUL, "overfit"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's in-sample edge did not survive out of sample, "
        "and its contribution to the book was resolvably NEGATIVE. The pattern is "
        "consistent with the expression fitting sample-specific noise rather than "
        "a persistent effect. "
        "Keep the hypothesis and change the expression so the edge it captures "
        "survives out of sample. How to do that is yours to determine."),
    (Verdict.MARGINAL, "overfit"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's out-of-sample edge is materially weaker than "
        "its in-sample edge, and its contribution could not be resolved from "
        "noise -- the fit looks sample-specific rather than structural. "
        "Keep the hypothesis and change the expression so the edge persists out "
        "of sample. How to do that is yours to determine."),
    # ---- Signal quality: the premise itself is the fault. Rewrite the
    # hypothesis and re-derive the expression. ----------------------------
    (Verdict.NET_HARMFUL, "signal"): (
        RefineTarget.HYPOTHESIS, [],
        "MEASUREMENT: this parent's contribution was resolvably NEGATIVE, and the "
        "weakest measured dimension is the signal itself ({weakness}: e={e_weak:.2f}) "
        "-- not its cost and not its overlap with the repository. The premise did "
        "not produce a real edge. "
        "Revise the HYPOTHESIS; the expression will be re-derived from it. What "
        "the revised premise should be is yours to determine."),
    (Verdict.MARGINAL, "signal"): (
        RefineTarget.HYPOTHESIS, [],
        "MEASUREMENT: this parent's contribution could not be resolved from noise "
        "and the weakest measured dimension is the signal itself ({weakness}: "
        "e={e_weak:.2f}). The premise is not producing enough edge to measure. "
        "Revise the HYPOTHESIS; the expression will be re-derived from it. What "
        "the revised premise should be is yours to determine."),
    # ---- Decay: the edge is fading across the OOS window. ---------------
    (Verdict.NET_HARMFUL, "decay"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's edge FADED across the out-of-sample window -- "
        "it was present early and had weakened by the end -- and its contribution "
        "to the book was resolvably NEGATIVE. "
        "Keep the hypothesis and change the expression so the edge it captures "
        "persists across the window instead of decaying. How to do that is yours "
        "to determine."),
    (Verdict.MARGINAL, "decay"): (
        RefineTarget.EXPRESSION, ["hypothesis"],
        "MEASUREMENT: this parent's edge decays across the out-of-sample window "
        "and its contribution could not be resolved from noise. "
        "Keep the hypothesis and change the expression so the edge persists. How "
        "to do that is yours to determine."),
}


# --------------------------------------------------------------------------
# Expression-aware target construction (T3, fixes #2/#3). ``_DIRECTIVES`` fixes
# the LAYER (expression vs hypothesis) and the high-level guidance; these
# helpers read the parent's parsed AST (T2 ``SegmentProfile``s) to name the
# ACTUAL operator + parameter + factor behind each weakness -- the #3 magnitude
# -- and to emit one target per severe dimension (fix #1: multi-segment).
# --------------------------------------------------------------------------

def _find_extreme_window_temporal(segments, want_min):
    """The (factor, op, window, signature) of the smallest/largest temporal window."""
    best = None
    for seg in segments or []:
        for t in seg.temporal_ops:
            for w in t["windows"]:
                take = best is None or (w < best[2] if want_min else w > best[2])
                if take:
                    best = (seg.factor_name, t["op"], w, t["signature"])
    return best


def _find_heaviest(segments):
    """The (factor, depth, n_free_params) most likely to be overfit."""
    best = None
    for seg in segments or []:
        key = (seg.n_free_params, seg.depth)
        if best is None or key > best[1]:
            best = (seg.factor_name, key)
    if best is None:
        return None
    fname, (nfree, depth) = best
    return (fname, depth, nfree)


def _shallowest_signal_factor(segments):
    """The first factor carrying a raw-signal sub-pattern (the signal source)."""
    for seg in segments or []:
        if seg.sub_patterns.get("signal"):
            return seg.factor_name
    return None


def _strongest_subpattern(segments):
    """The (factor, op) to amplify on a winner (T3/T6 ADMITTED push-further).

    Prefer per-factor combiner credit (T4) -- the highest-weight factor is the
    edge worth strengthening; without credit, the deepest / most-temporal factor
    is a structural proxy. ``op`` is the factor's first temporal op (the horizon
    to extend), or ``None`` for a pure signal/xs factor.
    """
    best = None
    for seg in segments or []:
        w = seg.combiner_weight
        if w is not None and w == w:
            key = (1, w)  # credit-bearing ranks above structural proxies
        else:
            key = (0, seg.depth + len(seg.temporal_ops))
        if best is None or key > best[1]:
            best = (seg, key)
    if best is None:
        return None
    seg = best[0]
    op = seg.temporal_ops[0]["op"] if seg.temporal_ops else None
    return (seg.factor_name, op)


# --------------------------------------------------------------------------
# Per-segment ablation augmentation (B5). ``ablation`` is a ``SegmentAblation``
# (duck-typed -- no import, to keep this module free of a segment_ablation
# dependency) or None. These read its ``per_part`` / ``window_sensitivity`` /
# ``core_sign_stability`` to (a) add measured solo significance to a located
# sub-tree's ``mechanism_hint`` and (b) apply the IC-neutral-window backstop --
# the Q2 defense that refuses a window-targeted hint when the window is
# IC-neutral and the weakness is IC-related (decay/signal), pointing at the core
# instead. Measurement only; never a remedy.
# --------------------------------------------------------------------------

def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _ablation_part_note(ablation, sig) -> str:
    """Measurement-only solo significance for the sub-tree with this signature."""
    if ablation is None or not hasattr(ablation, "per_part") or not sig:
        return ""
    pm = ablation.per_part.get(sig)
    if pm is None:
        return ""
    parts = []
    if _is_finite(getattr(pm, "rank_ic", None)):
        parts.append(f"solo rank_ic {pm.rank_ic:+.4f}")
    if _is_finite(getattr(pm, "t_nw", None)):
        parts.append(f"t_nw {pm.t_nw:+.2f}")
    if _is_finite(getattr(pm, "ic_pos_frac", None)):
        parts.append(f"ic_pos_frac {pm.ic_pos_frac:.2f}")
    if _is_finite(getattr(pm, "turnover_solo", None)):
        parts.append(f"solo turnover {pm.turnover_solo:.3f}")
    return "; ".join(parts)


def _window_ic_neutral(ablation, sig) -> bool:
    """True if the temporal op at ``sig`` is IC-neutral (the window-trap signal)."""
    if ablation is None or not hasattr(ablation, "window_sensitivity") or not sig:
        return False
    ws = ablation.window_sensitivity.get(sig)
    return bool(ws and ws.get("ic_neutral"))


def _core_health(ablation) -> tuple[str, str]:
    """('healthy'|'weak'|'unknown', note) from the core's solo t_nw + sign stability.

    |t_nw| >= 3 (the admission bar) with a stable sign => the edge is real and the
    construction is the problem (the temporal lever is appropriate for a COST
    weakness). A weak / sign-unstable core => the edge itself is the problem
    (point at the core, not the window). Measurement only.
    """
    if ablation is None or not hasattr(ablation, "core_sign_stability"):
        return ("unknown", "")
    css = ablation.core_sign_stability or {}
    csig = css.get("core_signature")
    pm = ablation.per_part.get(csig) if (csig and hasattr(ablation, "per_part")) else None
    stable = css.get("stable")
    if pm is None or not _is_finite(getattr(pm, "t_nw", None)):
        return ("unknown", "")
    healthy = abs(pm.t_nw) >= 3.0 and bool(stable)
    return ("healthy" if healthy else "weak",
            f"solo t_nw {pm.t_nw:+.2f}, sign {'stable' if stable else 'unstable'}")


def _build_target(dim, e_score, verdict, segments, metrics, ablation=None):
    """One targeted sub-tree refine (fix #1/#2/#3): dimension -> located op+param+factor.

    Returns a plain dict (serializable for T5's ``refine_actions`` lineage walk).
    ``e_score`` is the severity (low = weak); NaN is normalized to None for JSON.
    """
    if isinstance(e_score, float) and e_score != e_score:
        e_score = None
    category = _dimension_category(dim)
    tgt = {"dimension": dim, "category": category, "e_score": e_score,
           "factor": None, "op": None, "param": None,
           "mechanism_hint": "", "subtree_signature": None}

    if category == "cost":
        # LOCATE the turnover driver; do NOT prescribe the remedy. The shortest
        # temporal window is what makes the signal jump day to day and therefore
        # what the book pays to chase. Naming it is diagnosis; saying "lengthen
        # it" was a prescription, and the model then treated every weakness as a
        # window problem (measured: 2 of 3 refine children only moved a window,
        # both went from positive to net_harmful).
        t = _find_extreme_window_temporal(segments, want_min=True)
        if t is not None:
            fname, op, win, sig = t
            hint = (f"the fastest-moving component is {op} "
                    f"{int(round(win))} on {fname}; it is the "
                    "largest contributor to day-to-day turnover")
            note = _ablation_part_note(ablation, sig)
            if note:
                hint += f" ({note})"
            _cstate, _cnote = _core_health(ablation)
            if _cnote:
                # A cost weakness with a healthy core is the case the temporal
                # lever is MADE for: the edge is real, the construction is what
                # costs. State the core's measured health (no remedy).
                hint += f"; the core's edge is {_cstate} ({_cnote})"
            tgt.update(factor=fname, op=op, param=win, subtree_signature=sig,
                       mechanism_hint=hint)
        else:
            tgt.update(mechanism_hint=(
                "no temporal window in this factor (pure cross-sectional): the "
                "turnover comes from the signal itself changing rank day to day"))

    elif category == "overfit":
        t = _find_heaviest(segments)
        if t is not None and (t[1] > 3 or t[2] > 2):
            fname, depth, nfree = t
            tgt.update(factor=fname, op=None, param=None,
                       mechanism_hint=(f"the most complex component is {fname}: "
                                       f"expression depth {depth}, {nfree} free "
                                       "parameters"))
        else:
            tgt.update(mechanism_hint="no single component dominates the complexity")

    elif category == "redundancy":
        rho = metrics.get("rho_max")
        try:
            rho_f = float(rho) if rho is not None else None
        except (TypeError, ValueError):
            rho_f = None
        # Two DIFFERENT overlaps, and they fail for different reasons: rho_max is
        # overlap with the repository, rho_within is how much this batch's own
        # factors duplicate EACH OTHER. Both are measurements; neither carries a
        # remedy (see the system rule that prompts diagnose but never prescribe).
        within = metrics.get("rho_within")
        try:
            within_f = float(within) if within is not None else None
        except (TypeError, ValueError):
            within_f = None
        parts = []
        if rho_f is not None and rho_f == rho_f:
            tgt["param"] = rho_f
            parts.append(f"measured overlap with the repository: rho_max {rho_f:.2f}")
        else:
            parts.append("measured overlap with the repository is high")
        if within_f is not None and within_f == within_f and within_f > 0.8:
            parts.append(
                f"and the factors of this hypothesis have rank correlation "
                f"{within_f:.2f} with EACH OTHER -- they measure one direction, "
                f"not {int(metrics.get('n_factors') or 3)}"
            )
        tgt["mechanism_hint"] = "; ".join(parts)

    elif category == "signal":
        # The premise itself is weak -- name the signal source to strengthen.
        fname = _shallowest_signal_factor(segments)
        _cstate, _cnote = _core_health(ablation)
        if fname:
            hint = (f"the signal source carrying the premise is "
                    f"{fname}; it did not produce a measurable edge")
        else:
            hint = "the premise did not produce a measurable edge"
        if _cnote:
            hint += f" ({_cnote})"
        tgt.update(factor=fname, mechanism_hint=hint)

    elif category == "decay":
        # LOCATE the slowest component; do NOT prescribe "shift the horizon".
        t = _find_extreme_window_temporal(segments, want_min=False)
        if t is not None:
            fname, op, win, sig = t
            if _window_ic_neutral(ablation, sig):
                # Q2 backstop: the window is IC-neutral (solo rank_ic flat across
                # the sweep), so the decay is NOT a window problem -- moving the
                # window does not move the edge. Point at the CORE, where the edge
                # lives and where the fading originates, instead of handing the
                # model a window to move (the measured failure mode). The structured
                # target follows the hint: subtree_signature is the CORE's (so T5
                # lineage tracks the core lever, not the window we refused to move),
                # and op/param are cleared -- there is no window to target here.
                _cstate, _cnote = _core_health(ablation)
                _csig = (ablation.core_sign_stability or {}).get("core_signature") \
                    if ablation is not None else None
                hint = (f"the {op} window is IC-neutral (solo rank_ic flat across "
                        f"windows), so the decay is not in the window; the edge "
                        f"lives in the core on {fname}")
                if _cnote:
                    hint += f" ({_cnote})"
                tgt.update(factor=fname, op=None, param=None,
                           subtree_signature=_csig, mechanism_hint=hint)
            else:
                note = _ablation_part_note(ablation, sig)
                hint = (f"the slowest-moving component is {op} "
                        f"{int(round(win))} on {fname}")
                if note:
                    hint += f" ({note})"
                tgt.update(factor=fname, op=op, param=win, subtree_signature=sig,
                           mechanism_hint=hint)
        else:
            tgt.update(mechanism_hint="no temporal window dominates this factor")

    else:  # "unknown"
        tgt.update(mechanism_hint="address this weakness while keeping the signal")

    return tgt


def _render_directive_text(base_text, targets, metrics, primary_dim):
    """High-level guidance (from ``_DIRECTIVES``) + the per-target refinement list."""
    ctx = {
        "weakness": primary_dim or "this dimension",
        "e_weak": _e_weak(metrics, primary_dim),
        "reason": str(metrics.get("reason") or ""),
        "rank_ic": metrics.get("rank_ic", metrics.get("RankIC", float("nan"))),
        "turnover_book": metrics.get("turnover_book", float("nan")),
        "rho_max": metrics.get("rho_max", float("nan")),
        "rho_within": metrics.get("rho_within", float("nan")),
    }
    try:
        intro = base_text.format(**ctx)
    except (KeyError, IndexError):
        intro = base_text
    if not targets:
        return intro
    lines = [intro, "Targeted refinements (most severe first):"]
    for i, t in enumerate(targets, 1):
        e = t.get("e_score")
        e_str = f" (e={e:.2f})" if isinstance(e, (int, float)) and e == e else ""
        lines.append(f"{i}. {t['dimension']}{e_str}: {t['mechanism_hint']}")
    return "\n".join(lines)


def _fallback_directive(verdict: Verdict, metrics: dict[str, Any],
                        segments: list[SegmentProfile] | None = None) -> RefinementDirective:
    """A directive for verdicts with no dimension-specific row, or unknown category.

    These are the verdicts whose instruction does not depend on a dimension:
    ADMITTED / REPLACED / BOOTSTRAP (winners -- refine to push further), the
    structural rejections REDUNDANT / TOO_SPARSE / CONSTANT, FULL, and the
    unknown-category fallback for the contribution verdicts. The winner branch
    is expression-aware (T3/T6): it reads the parent's AST (T2 ``segments``) to
    name the strong sub-pattern to amplify, rather than a generic "push further".
    """
    rank_ic = metrics.get("rank_ic", metrics.get("RankIC"))
    reason = str(metrics.get("reason") or "")

    if verdict in (Verdict.ADMITTED, Verdict.REPLACED, Verdict.BOOTSTRAP):
        # A winner: LOCATE where the credit sits. T3/T6 reads the parent's AST
        # to name the sub-pattern the combiner actually paid for.
        #
        # 2026-08-23: this branch PRESCRIBED -- "strengthen {op} / extend its
        # horizon on {fname}". Two prescriptions in six words: it named the
        # parent's own top operator as the thing to keep, and "extend its
        # horizon" as the edit to make. Measured in the 10-dir smoke: both
        # admitted parents produced this directive 5x each ("strengthen TS_MEAN
        # / extend its horizon", "strengthen TS_ZSCORE / extend its horizon"),
        # so whichever operator happened to be admitted first was fed back as
        # the operator to reuse -- a self-reinforcing loop that is a plausible
        # driver of TS_MEAN holding 64-83% of the window-summarizer slot in
        # every mine on record. It also pushes toward LONGER horizons, which
        # [[qa-cost-aware-longer-horizon-bias]] already measured as biasing away
        # from the short-horizon edge. Rewritten to the same standard as the
        # REDUNDANT branch below: state where the credit is measured, state what
        # must be achieved, leave the how open.
        sp = _strongest_subpattern(segments)
        if sp is not None:
            fname, op = sp
            where = (f"the combiner's credit is concentrated in {op} on {fname}"
                     if op else f"the combiner's credit is concentrated in {fname}")
            hint = where
            targets = [{
                "dimension": None, "category": "winner", "e_score": None,
                "factor": fname, "op": op, "param": None,
                "mechanism_hint": hint, "subtree_signature": None,
            }]
        else:
            where = ("the measurement does not localize the credit to one "
                     "sub-pattern")
            hint = where
            targets = []
        return RefinementDirective(
            verdict=verdict,
            weakness_dimension=None,
            refine_target=RefineTarget.EXPRESSION,
            frozen_layers=["hypothesis"],
            mechanism_hint=hint,
            directive_text=(
                f"MEASUREMENT: this parent was ADMITTED -- it contributed "
                f"measurably to the book (RankIC {_fmt(rank_ic, '+.4f')}), and "
                f"{where}. Keep the hypothesis and change the expression so that "
                "the contribution the measurement located is larger, without "
                "discarding what earned the admission. Which part to change, and "
                "how, is yours to determine."
            ),
            targets=targets,
            parent_hypothesis="",
            parent_factors=[],
            raw_metrics=metrics,
        )

    if verdict is Verdict.REDUNDANT:
        rho = metrics.get("rho_max")
        return RefinementDirective(
            verdict=verdict,
            weakness_dimension="diversity",
            refine_target=RefineTarget.EXPRESSION,
            frozen_layers=["hypothesis"],
            # 2026-08-15: this path still PRESCRIBED ("orthogonalize the inputs",
            # "use different input fields or a different construction"). It is a
            # second redundancy directive that the earlier prescription sweep
            # missed. Rewritten to the same standard as the rest: state the
            # measurement, state what must be achieved, leave the how open.
            mechanism_hint="measured overlap with the repository is at rho_max",
            directive_text=(
                f"MEASUREMENT: this parent's signal has rank correlation "
                f"{_fmt(rho, '.3f')} with a factor already in the repository. At "
                "that overlap the combiner cannot separate them, so the parent's "
                "contribution beyond what the book already holds is not "
                "measurable. Keep the hypothesis and change the expression so "
                "that the signal it produces is distinguishable from the "
                "incumbent's. How to do that is yours to determine."
            ),
            parent_hypothesis="",
            parent_factors=[],
            raw_metrics=metrics,
        )

    if verdict in (Verdict.TOO_SPARSE, Verdict.CONSTANT):
        return RefinementDirective(
            verdict=verdict,
            weakness_dimension="coverage",
            refine_target=RefineTarget.EXPRESSION,
            frozen_layers=["hypothesis"],
            mechanism_hint="broaden coverage",
            directive_text=(
                "This parent's signal could not be priced across the universe "
                "(too sparse / zero variance). Refine the EXPRESSION so the factor "
                "is defined and non-constant for more names -- broaden the input "
                "fields or soften hard thresholds. Keep the hypothesis."
            ),
            parent_hypothesis="",
            parent_factors=[],
            raw_metrics=metrics,
        )

    if verdict is Verdict.FULL:
        return RefinementDirective(
            verdict=verdict,
            weakness_dimension=None,
            refine_target=RefineTarget.EXPRESSION,
            frozen_layers=["hypothesis"],
            mechanism_hint="marginal contribution is below the weakest incumbent",
            # 2026-08-23: this also PRESCRIBED ("strengthen the signal or cut
            # turnover") -- the same sweep that fixed the ADMITTED branch above.
            # Naming the two levers is naming the remedy; the measurement is that
            # the bar exists and where it sits.
            directive_text=(
                "MEASUREMENT: the repository is full: " + reason +
                ". Marginal contribution is scored net of cost against the "
                "weakest incumbent, so a factor enters only by displacing it. "
                "Keep the hypothesis and change the expression so the measured "
                "marginal contribution clears that bar. How to do that is yours "
                "to determine."
            ),
            parent_hypothesis="",
            parent_factors=[],
            raw_metrics=metrics,
        )

    if verdict is Verdict.NO_DATA:
        # No basis to refine. The controller falls back to an orthogonal
        # mutation for this parent.
        return RefinementDirective(
            verdict=verdict,
            weakness_dimension=None,
            refine_target=RefineTarget.EXPRESSION,
            frozen_layers=[],
            mechanism_hint="",
            directive_text="",
            parent_hypothesis="",
            parent_factors=[],
            raw_metrics=metrics,
        )

    # Unknown dimension category under a contribution verdict: fall back to a
    # generic expression-refine that preserves the signal.
    return RefinementDirective(
        verdict=verdict,
        weakness_dimension=_weakest_dimension(metrics),
        refine_target=RefineTarget.EXPRESSION,
        frozen_layers=["hypothesis"],
        mechanism_hint="preserve the signal, fix the weakness",
        directive_text=(
            "This parent did not improve the book. Refine the EXPRESSION to "
            "address its weakness ({weakness}) while keeping the signal "
            f"(RankIC {_fmt(rank_ic, '+.4f')}). Keep the hypothesis."
        ),
        parent_hypothesis="",
        parent_factors=[],
        raw_metrics=metrics,
    )


def _fmt(value, spec: str) -> str:
    """Format a measured value for a directive.

    ``spec`` is a BARE format spec (``".3f"``, ``"+.4f"``) as every call site
    passes it -- so this must use the ``format(value, spec)`` builtin.

    It previously did ``spec.format(float(value))``, i.e. it called ``str.format``
    ON THE SPEC. A string with no ``{}`` placeholders returns itself, so
    ``".3f".format(0.97)`` produced the literal ``".3f"`` and every directive
    built with ``_fmt`` shipped the format spec to the model where the measured
    number belonged: *"this parent's contribution ... (RankIC +.4f)"*. The whole
    point of these directives is to state the measurement, and the measurement
    was the one thing that never arrived. Silent because the output is a
    plausible-looking short token and directive text is not logged.
    """
    try:
        if value is None or (isinstance(value, float) and value != value):
            return "n/a"
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# Lineage-aware diagnosis (T5, fix #5): detect an exhausted refinement lever.
# --------------------------------------------------------------------------
# A refinement lever is "exhausted" after it has been pulled this many times
# along a lineage WITHOUT the resulting child being admitted. The K-th attempt
# on the same sub-tree switches the layer (EXPRESSION -> HYPOTHESIS); once BOTH
# layers are spent, the diagnosis routes the parent to an orthogonal restart.
_EXHAUSTION_K = 2


def _signature_hit_count(ancestors, signature: str | None) -> int:
    """How many non-admitted ancestors already refined this exact sub-tree.

    Each ancestor's ``refine_actions`` records the levers pulled to produce it;
    its ``backtest_metrics["admitted"]`` records whether that lever paid off. A
    lever pulled but NOT admitted is a failed attempt -- count those. ``None``
    signatures (no located sub-tree, e.g. a pure-xs cost factor, or a
    HYPOTHESIS-refine whose signal-category target has no sub-tree) never match,
    so they cannot be exhausted via this counter -- the HYPOTHESIS lever uses
    ``_hypothesis_hit_count`` instead.
    """
    if not signature:
        return 0
    count = 0
    for anc in ancestors or []:
        actions = getattr(anc, "refine_actions", None) or []
        admitted = bool((getattr(anc, "backtest_metrics", None) or {}).get("admitted", True))
        if admitted:
            continue  # the lever worked -> not a failed attempt
        for act in actions:
            sigs = act.get("target_subtree_signatures") or []
            if signature in sigs:
                count += 1
                break  # one failed attempt per ancestor is enough
    return count


def _hypothesis_hit_count(ancestors, dimension: str | None) -> int:
    """Non-admitted ancestors that already rewrote this dimension's HYPOTHESIS.

    The HYPOTHESIS lever (signal-quality faults: rewrite the premise) has no
    located sub-tree, so it has no ``subtree_signature`` to compare -- it is
    keyed by ``(refine_target=="hypothesis", weakness_dimension)`` instead. A
    premise rewritten >=K times for the same dimension without admission is
    spent, and there is no lower layer to switch to -> orthogonal restart.
    """
    if not dimension:
        return 0
    count = 0
    hyp = RefineTarget.HYPOTHESIS.value
    for anc in ancestors or []:
        admitted = bool((getattr(anc, "backtest_metrics", None) or {}).get("admitted", True))
        if admitted:
            continue
        for act in getattr(anc, "refine_actions", None) or []:
            if act.get("refine_target") == hyp and act.get("weakness_dimension") == dimension:
                count += 1
                break
    return count


def _apply_lineage(directive: RefinementDirective,
                   ancestors: list | None) -> RefinementDirective:
    """Detect an exhausted refinement lever and switch the directive (fix #5).

    Two levers, two counters, switched in order so "both spent" is reachable:

    * EXPRESSION-refine (cost / turnover / overfit / redundancy / decay -- a
      located sub-tree with a ``subtree_signature``): if that signature has
      been refined >=K times without admission, the expression lever is spent.
      If the HYPOTHESIS lever for the SAME dimension is ALSO spent -> orthogonal
      restart (``exhausted_lever``). Otherwise switch to HYPOTHESIS (rewrite the
      premise for a fresh construction) -- still a refinement.
    * HYPOTHESIS-refine (signal-quality -- the premise is the fault, no
      sub-tree): if the premise for this dimension has been rewritten >=K times
      without admission, the hypothesis lever is spent -> orthogonal restart.

    Only the primary (most-severe) target is checked. ``None`` / missing
    ancestors are a no-op, so callers that do not pass a lineage (the unit
    tests, legacy paths) get the pre-T5 directive unchanged.
    """
    if not ancestors or not directive.targets:
        return directive
    primary = directive.targets[0]
    dim = primary.get("dimension") or "this"
    mech = primary.get("mechanism_hint") or "the same sub-tree"

    def _note(hits: int) -> str:
        return (f"\n\nLineage note: the {dim} lever ({mech}) has been refined "
                f"{hits} time(s) along this lineage without admission.")

    if directive.refine_target is RefineTarget.EXPRESSION:
        ehits = _signature_hit_count(ancestors, primary.get("subtree_signature"))
        if ehits < _EXHAUSTION_K:
            return directive
        hhits = _hypothesis_hit_count(ancestors, dim)
        if hhits >= _EXHAUSTION_K:
            # Both the expression sub-tree and the hypothesis for this dimension
            # are spent -> orthogonal restart.
            directive.exhausted_lever = True
            directive.mechanism_hint = (f"exhausted: expression ({mech}, {ehits}x) and "
                                        f"hypothesis ({dim}, {hhits}x) levers both tried "
                                        "without admission -> orthogonal restart")
            directive.directive_text = (directive.directive_text or "") + _note(ehits) + (
                " Both the EXPRESSION and HYPOTHESIS levers are exhausted. Breed "
                "an ORTHOGONAL restart (a fresh direction) rather than refining further.")
            logger.info("lineage: both levers exhausted (expr %s %d, hyp %s %d) -> orthogonal",
                        primary.get("subtree_signature"), ehits, dim, hhits)
            return directive
        # Expression spent, hypothesis still fresh -> rewrite the premise.
        directive.refine_target = RefineTarget.HYPOTHESIS
        directive.frozen_layers = []
        directive.mechanism_hint = (f"exhausted expression lever ({mech} tried "
                                    f"{ehits}x) -> rewrite the HYPOTHESIS for a "
                                    "fresh premise and re-derive the expression")
        directive.directive_text = (directive.directive_text or "") + _note(ehits) + (
            " The EXPRESSION lever is exhausted. Rewrite the HYPOTHESIS for a "
            "fresh premise and re-derive the expression; do not keep tuning the "
            "same operator/window.")
        directive.targets = []  # the located sub-tree targets no longer apply
        logger.info("lineage: exhausted EXPRESSION lever (%s, %d hits) -> HYPOTHESIS",
                    primary.get("subtree_signature"), ehits)
        return directive

    # HYPOTHESIS-refine: the premise itself is the fault. If the lineage already
    # rewrote this dimension's premise >=K times without admission, the lever is
    # spent and there is no lower layer to switch to -> orthogonal restart.
    hhits = _hypothesis_hit_count(ancestors, dim)
    if hhits >= _EXHAUSTION_K:
        directive.exhausted_lever = True
        directive.mechanism_hint = (f"exhausted hypothesis lever ({dim}, {hhits}x) "
                                    "tried without admission -> orthogonal restart")
        directive.directive_text = (directive.directive_text or "") + _note(hhits) + (
            " The HYPOTHESIS lever is exhausted -- the premise for this weakness "
            "has been rewritten without admission. Breed an ORTHOGONAL restart "
            "(a fresh direction) rather than refining further.")
        logger.info("lineage: exhausted HYPOTHESIS lever (%s, %d hits) -> orthogonal",
                    dim, hhits)
    return directive


# --------------------------------------------------------------------------
# First-class deterministic refine: the SIGN-FLIP (the measured-direction fix).
# --------------------------------------------------------------------------
# The significance bar a sign-flip candidate must have cleared. Mirrors
# ``admission.k_sigma`` (Harvey-Liu-Zhu 3.0): a parent rejected for a sign
# MISMATCH has ALREADY cleared this bar -- the sign/mechanism gate only runs on
# |t| >= k_sigma, so a sign-mismatch reject implies |t| >= k_sigma. This is a
# safety guard, not the primary condition. Kept here rather than read from Theta
# because the diagnosis does not hold Theta; the constant is stable (the
# multiple-testing bar that RISES is ``fdr_t_required``, a separate gate).
_SIGN_FLIP_T_BAR = 3.0


def sign_flip_directive(parent: Any, metrics: dict[str, Any]) -> RefinementDirective | None:
    """The first-class deterministic sign-flip refine (the measured-direction fix).

    Fires when the parent's pre-registered IC direction (``sign_predicted``) is
    OPPOSITE its measured direction (``sign_realized``) AND the edge clears the
    significance bar (``|t_nw| >= _SIGN_FLIP_T_BAR``). The construction has a real
    edge -- it was rejected ONLY because the hypothesis labeled the direction
    backwards. The fix is deterministic: FREEZE the expression verbatim, CORRECT
    the direction to the measured one, RE-TEST. No LLM call (both the diagnosis
    and the child are deterministic), so this OVERRIDES any LLM ``exhausted=true``
    -- a mislabeled edge is never an exhausted lever, it is a candidate the search
    measured wrongly and can correct for free.

    This is the most literal "learn from the measured mistake": the search
    measured that its prediction was backwards and corrects it, admitting a real
    edge that the sign gate wrongly rejected. Measurement only -- the
    ``directive_text`` states the measured directions and the correction, never a
    remedy to invent.

    Returns ``None`` when the condition does not hold (no sign mismatch, no
    measured sign, or the edge did not clear the bar) so the caller falls through
    to the LLM/table diagnosis unchanged.
    """
    sign_pred = str(metrics.get("sign_predicted", "") or "").strip().lower()
    sign_real = str(metrics.get("sign_realized", "") or "").strip().lower()
    if sign_pred not in ("positive", "negative"):
        return None
    if sign_real not in ("positive", "negative"):
        return None
    if sign_pred == sign_real:
        return None
    t_nw = metrics.get("t_nw")
    if not _is_finite(t_nw) or abs(float(t_nw)) < _SIGN_FLIP_T_BAR:
        return None

    verdict = classify_verdict(metrics)
    pred_word = "positive" if sign_pred == "positive" else "negative"
    real_word = "positive" if sign_real == "positive" else "negative"
    t_val = float(t_nw)
    note = (
        f"MEASUREMENT: the pre-registered direction ({pred_word}) is opposite the "
        f"measured direction ({real_word}, |t| {t_val:.2f}). The construction "
        f"carries a real edge -- it was rejected only because the hypothesis "
        f"labeled the direction backwards. The construction is RETAINED unchanged; "
        f"the pre-registered direction is CORRECTED to the measured one "
        f"({real_word}). How to act on that is yours to determine."
    )
    return RefinementDirective(
        verdict=verdict,
        weakness_dimension="sign",
        refine_target=RefineTarget.SIGN,
        # The expression is frozen verbatim (the construction is sound); the
        # hypothesis is rewritten to record the measured correction.
        frozen_layers=["expression"],
        mechanism_hint=(
            f"predicted {pred_word}, realized {real_word} (opposite), |t| {t_val:.2f}; "
            f"construction retained, direction corrected to {real_word}"
        ),
        directive_text=note,
        parent_hypothesis=str(getattr(parent, "hypothesis", "") or ""),
        parent_expected_ic_sign=real_word,  # the corrected direction the child carries
        parent_factors=[dict(f) for f in (getattr(parent, "factors", None) or [])],
        raw_metrics=dict(metrics),
        exhausted_lever=False,  # a mislabeled edge is never an exhausted lever
    )


def diagnose(parent: Any, ancestors: list | None = None, *,
             ablation: Any = None, ablation_summary: str = "") -> RefinementDirective | None:
    """Map a parent trajectory's backtest result to a refinement directive.

    Returns ``None`` when the parent carries no objective vector (no ``U``) --
    nothing to diagnose, and the caller breeds orthogonally as today. Returns a
    directive with ``verdict=NO_DATA`` (``is_refinement() is False``) when there
    is a ``U`` but no usable verdict, so the caller falls back to orthogonal for
    that parent without losing the rest.

    T3: for contribution verdicts (NET_HARMFUL / MARGINAL) the directive is
    expression-aware and multi-segment -- the parent's factor expressions are
    parsed (T2 ``SegmentProfile``s) and every severe weakness dimension drives
    one targeted sub-tree refine whose ``mechanism_hint`` names the ACTUAL
    operator + parameter + factor behind the weakness (fix #1 multi-segment,
    #2 expression-aware, #3 the param is the magnitude). Structural / winner
    verdicts stay single-purpose (``_fallback_directive``).

    T5: ``ancestors`` (the parent's ancestor trajectories, from
    ``TrajectoryPool.get_ancestors``) let the diagnosis detect a refinement lever
    pulled >=K times along the lineage without admission (fix #5). When the
    primary lever is exhausted the directive switches EXPRESSION -> HYPOTHESIS,
    and when both are spent it sets ``exhausted_lever`` so the caller breeds an
    orthogonal restart. Omit ``ancestors`` (or pass ``None``) for the pre-T5
    behaviour -- the directive is then lineage-unaware, as in the unit tests.

    ``parent`` is a ``StrategyTrajectory`` (duck-typed on ``backtest_metrics``,
    ``hypothesis``, ``factors``).
    """
    metrics: dict[str, Any] = dict(getattr(parent, "backtest_metrics", None) or {})
    if "U" not in metrics:
        return None  # no objective vector -- nothing to diagnose

    verdict = classify_verdict(metrics)
    segments = build_segments(parent)

    # Structural / winner verdicts: the weakness is implied by the verdict (or
    # there is none to locate). Single-purpose -- no multi-segment list.
    if verdict in (Verdict.REDUNDANT, Verdict.TOO_SPARSE, Verdict.CONSTANT,
                   Verdict.ADMITTED, Verdict.REPLACED, Verdict.BOOTSTRAP,
                   Verdict.FULL, Verdict.NO_DATA):
        directive = _fallback_directive(verdict, metrics, segments)
    else:
        # Contribution verdicts (NET_HARMFUL / MARGINAL): locate every severe
        # weakness in the parent's AST (T2) and emit one targeted sub-tree refine
        # per dimension (fix #1/#2/#3).
        weak_dims = weakest_dimensions(metrics)
        if not weak_dims:
            # No scored dimension at all -- nothing to locate.
            directive = _fallback_directive(verdict, metrics, segments)
        else:
            primary_dim, _ = weak_dims[0]
            primary_cat = _dimension_category(primary_dim)
            row = _DIRECTIVES.get((verdict, primary_cat))
            if row is None:
                refine_target, frozen_layers, base_text = (
                    RefineTarget.EXPRESSION, ["hypothesis"],
                    "This parent did not improve the book (RankIC {rank_ic:+.4f}). Refine "
                    "the EXPRESSION to address its weakness while keeping the signal. Keep "
                    "the hypothesis.")
            else:
                refine_target, frozen_layers, base_text = row

            targets = [_build_target(dim, e, verdict, segments, metrics, ablation)
                       for dim, e in weak_dims]
            directive_text = _render_directive_text(base_text, targets, metrics, primary_dim)
            directive = RefinementDirective(
                verdict=verdict,
                weakness_dimension=primary_dim,
                refine_target=refine_target,
                frozen_layers=list(frozen_layers),
                mechanism_hint=targets[0]["mechanism_hint"] if targets else "",
                directive_text=directive_text,
                targets=targets,
                parent_hypothesis="",
                parent_factors=[],
                raw_metrics=metrics,
            )

    # Common tail: the frozen prefix (carried for the constructor) + T5 lineage.
    directive.parent_hypothesis = str(getattr(parent, "hypothesis", "") or "")
    directive.parent_factors = [dict(f) for f in (getattr(parent, "factors", None) or [])]
    # Mirror llm_diagnosis._to_directive: carry the parent's pre-registered
    # direction so a frozen expression-refine child is born with it (else the
    # falsifiability gate rejects it no_mechanism). getattr(parent,
    # "expected_ic_sign") resolves once the trajectory persists it
    # (controller.create_trajectory_from_loop_result); sign_predicted is a
    # belt-and-braces fallback for trajectories persisted before that.
    directive.parent_expected_ic_sign = str(
        getattr(parent, "expected_ic_sign", "")
        or metrics.get("sign_predicted", "") or "").strip().lower()
    # Per-segment ablation summary (B3/B5): measurement-only prose carried onto
    # the directive so a parallel refine child inherits it via to_dict/from_dict.
    # Empty when the ablation is off / failed (QA_ABLATION_DIAGNOSIS) -- the
    # directive is then byte-identical to the pre-ablation diagnosis.
    directive.ablation_summary = ablation_summary or ""
    return _apply_lineage(directive, ancestors)


def _e_weak(metrics: dict[str, Any], dim: str | None) -> float:
    """The e_j score of the weakest named dimension, or nan."""
    if not dim:
        return float("nan")
    val = metrics.get(f"e_{dim}")
    if isinstance(val, (int, float)):
        return float(val)
    return float("nan")


__all__ = [
    "Verdict",
    "RefineTarget",
    "RefinementDirective",
    "classify_verdict",
    "diagnose",
    "sign_flip_directive",
]