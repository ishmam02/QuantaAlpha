"""Strength diagnosis — the crossover mirror of diagnose-and-refine.

The paper's Eq. 7 crossover *"identifies trajectory segments that consistently
contribute to high cumulative rewards — hypothesis templates, factor
construction patterns, or strategic repair actions — and merges them into a
highly coherent sequence."* The refine mutation (``diagnosis.py``) locates the
SHORTFALL of one parent; this module locates the STRENGTH, inverted, so the
crossover LLM can author a NEW child from the two best parents' complementary
VALIDATED ideas rather than a literal equation splice.

Three segment types are located, mirroring the paper verbatim:

1. **Hypothesis templates** — extracted from each parent's ``hypothesis_details``
   (reason / observation / justification / knowledge), surfaced as the template
   the LLM reasons over.
2. **Factor construction patterns** — ``build_segments`` +
   ``_strongest_subpattern`` locate the highest-``combiner_weight`` factor (the
   construction that drove the book), its operator, horizon, and combiner credit.
3. **Strategic repair actions** — ``repair_actions_summary`` walks the lineage
   and surfaces the repairs that were made AND validated (the resulting child was
   admitted) — the credible-lineage material.

And the *"consistently contribute"* step: ``lineage_validated_segments`` walks
the admitted ancestors and aggregates all three segment types by recurrence +
admission, so the identification is a trajectory-level roll-up ("this
construction pattern recurred in N admitted ancestors"), not a point-in-time
per-parent estimate. This is the piece the old crossover never had -- it spliced
one factor's AST and never saw the lineage at all.

Same discipline as the refine diagnosis: **diagnose-never-prescribe**. State
what the measurements show WORKED and where they locate it; never state what the
child should be or how to combine; close with *"how to combine that is yours to
determine."* No market-specific priors -- every number is measured on the market
the run is on (the ``_METRIC_GLOSS`` is measurement-only).

Hybrid (mirrors ``llm_diagnosis``): the structured fields (strongest dimension,
per-dimension located strengths, hypothesis template, lineage-validated
segments, validated repairs) are produced **deterministically**; an optional LLM
(QA_LLM_STRENGTH_DIAGNOSIS, default on) authors the prose ``directive_text`` for
fluency. Falls back to the deterministic table on any LLM failure, so a strength
diagnosis never blocks a round.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from quantaalpha.core.verdict import Verdict
from .diagnosis import (
    RefineTarget,
    _dimension_category,
    _find_extreme_window_temporal,
    _strongest_subpattern,
    classify_verdict,
    weakest_dimensions,
)
from .segment_profiling import SegmentProfile, build_segments

logger = logging.getLogger(__name__)

# On by default (mirrors QA_LLM_DIAGNOSIS). Set QA_LLM_STRENGTH_DIAGNOSIS=0 to use
# the deterministic strength table only.
def _enabled() -> bool:
    return os.environ.get("QA_LLM_STRENGTH_DIAGNOSIS", "1").strip().lower() not in (
        "0", "false", "no")


# Reuse the refine diagnosis's measurement gloss + rendering verbatim -- it is
# measurement-only (no priors) and already covers the objective vector, the e_*
# dimension scores, factor_attribution, the verdict, and the lineage. The
# strength prompt reasons over the SAME numbers, inverted.
from .llm_diagnosis import (
    _GLOSS_ALIASES,
    _METRIC_GLOSS,
    _fmt,
    _get,
    _measurement_block,
    _population_block,
)


# A dimension is a strength when its e_* score is above this. The e_* scores are
# [0,1] repository percentiles (1 = best), so a HIGH score is a strength --
# the inverse of diagnosis._SEVERITY_THRESHOLD (0.25). 0.75 lets the clearly-
# strong dims count while excluding the merely-middling. If nothing qualifies,
# the single strongest is still returned so a directive always carries at least
# one located strength (the inverse of "always at least the weakest").
_STRENGTH_THRESHOLD = 0.75
_MAX_TARGETS = 3


def strongest_dimensions(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    """All strong dimensions, strongest first (the inverse of weakest_dimensions).

    Sources the ``weakest_dimensions`` string (which names the k=2 lowest) AND
    the full ``e_*`` scalar vector, keeps dims whose e_* >
    ``_STRENGTH_THRESHOLD`` (or, if none qualify, the single strongest), and
    returns up to ``_MAX_TARGETS`` sorted DESCENDING by score. Each entry drives
    one located-strength target in the directive.
    """
    scored: dict[str, float] = {}
    # The weakest_dimensions string is still useful context (it names what the
    # gate flagged), but for STRENGTH we re-score every e_* and keep the highs.
    for k, v in metrics.items():
        if k.startswith("e_") and isinstance(v, (int, float)) and v == v:
            scored[k[2:]] = float(v)

    if not scored:
        return []
    items = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    strong = [(d, e) for d, e in items
              if isinstance(e, (int, float)) and e == e and e > _STRENGTH_THRESHOLD]
    if not strong:
        strong = [items[0]]  # always at least the strongest
    return strong[:_MAX_TARGETS]


# --------------------------------------------------------------------------
# Per-dimension strength location (the inverse of diagnosis._build_target).
# Each helper reads the parent's parsed AST (SegmentProfiles) to name the
# ACTUAL operator + parameter + factor behind the strength -- measurement only,
# never a remedy.
# --------------------------------------------------------------------------

def _seg_for(segments: list[SegmentProfile], fname: str | None) -> SegmentProfile | None:
    for seg in segments or []:
        if seg.factor_name == fname:
            return seg
    return None


def _cheapest_factor(segments: list[SegmentProfile]) -> tuple[str, float] | None:
    """The (factor, turnover_share) that costs the book the least.

    Per-factor ``turnover_share`` from ``factor_attribution`` -- the real cost
    axis, not a trajectory-level proxy. None when no factor carries the field.
    """
    best = None
    for seg in segments or []:
        ts = seg.turnover_share
        if ts is None or ts != ts:
            continue
        if best is None or ts < best[1]:
            best = (seg.factor_name, ts)
    return best


def _find_shallowest(segments: list[SegmentProfile]) -> tuple[str, int, int] | None:
    """The (factor, depth, n_free_params) least likely to be overfit -- the
    inverse of diagnosis._find_heaviest."""
    best = None
    for seg in segments or []:
        key = (seg.n_free_params, seg.depth)
        if best is None or key < best[1]:
            best = (seg.factor_name, key)
    if best is None:
        return None
    fname, (nfree, depth) = best
    return (fname, depth, nfree)


def _credit_note(seg: SegmentProfile | None) -> str:
    """Measurement-only combiner credit for a located factor (the strength
    analogue of ``_ablation_part_note``)."""
    if seg is None:
        return ""
    parts = []
    if seg.combiner_weight is not None and seg.combiner_weight == seg.combiner_weight:
        parts.append(f"combiner weight {seg.combiner_weight:+.4f}")
    if seg.weight_stability is not None and seg.weight_stability == seg.weight_stability:
        parts.append(f"weight stability {seg.weight_stability:.2f}")
    if seg.rank_ic is not None and seg.rank_ic == seg.rank_ic:
        parts.append(f"rank_ic {seg.rank_ic:+.4f}")
    if seg.turnover_share is not None and seg.turnover_share == seg.turnover_share:
        parts.append(f"turnover share {seg.turnover_share:.3f}")
    return "; ".join(parts)


def _build_strength_target(dim: str, e_score: float, segments: list[SegmentProfile],
                           metrics: dict[str, Any]) -> dict[str, Any]:
    """One located-strength target (the inverse of diagnosis._build_target).

    Returns a plain serializable dict. ``e_score`` is the strength (high =
    strong); NaN normalized to None. Measurement-only ``mechanism_hint`` --
    names what worked and where, never a remedy.
    """
    if isinstance(e_score, float) and e_score != e_score:
        e_score = None
    category = _dimension_category(dim)
    tgt: dict[str, Any] = {"dimension": dim, "category": category, "e_score": e_score,
                          "factor": None, "op": None, "param": None,
                          "mechanism_hint": "", "subtree_signature": None}

    if category == "signal":
        # The premise paid off -- name the construction that drove the book.
        sp = _strongest_subpattern(segments)
        if sp is not None:
            fname, op = sp
            seg = _seg_for(segments, fname)
            note = _credit_note(seg)
            hint = (f"the construction that drove the book is {op} on {fname}"
                    if op else f"the construction that drove the book is {fname}")
            if note:
                hint += f" ({note})"
            sig = None
            if seg is not None and seg.temporal_ops:
                sig = seg.temporal_ops[0].get("signature")
            tgt.update(factor=fname, op=op, subtree_signature=sig, mechanism_hint=hint)
        else:
            tgt.update(mechanism_hint="the premise produced a measurable edge")

    elif category == "cost":
        # Cheap to trade -- the strength the cost-stall cases lack. Name the
        # cheapest construction (lowest turnover share) and the most stable
        # (largest window) temporal component.
        cheap = _cheapest_factor(segments)
        t = _find_extreme_window_temporal(segments, want_min=False)
        parts = []
        if cheap is not None:
            fname, ts = cheap
            tgt["factor"] = fname
            parts.append(f"the cheapest component is {fname} (turnover share {ts:.3f})")
        if t is not None:
            fname, op, win, sig = t
            tgt.update(op=op, param=win, subtree_signature=sig)
            parts.append(f"the most stable component is {op} {int(round(win))} on {fname}")
        tgt["mechanism_hint"] = "; ".join(parts) or "the factor trades cheaply"

    elif category == "overfit":
        # Parsimonious -- the inverse of overfit risk. Name the shallowest /
        # lowest-parameter component (the part most likely to survive OOS).
        sh = _find_shallowest(segments)
        if sh is not None:
            fname, depth, nfree = sh
            tgt.update(factor=fname, mechanism_hint=(
                f"the most parsimonious component is {fname}: expression depth "
                f"{depth}, {nfree} free parameters -- the part most likely to "
                "survive out of sample"))
        else:
            tgt.update(mechanism_hint="the construction is parsimonious")

    elif category == "redundancy":
        # Distinctive -- low overlap with the repository. rho_max is the measured
        # overlap; a low value is the strength.
        rho = metrics.get("rho_max")
        try:
            rho_f = float(rho) if rho is not None else None
        except (TypeError, ValueError):
            rho_f = None
        if rho_f is not None and rho_f == rho_f:
            tgt["param"] = rho_f
            tgt["mechanism_hint"] = (
                f"the factor is distinctive: max correlation with the repository "
                f"is rho_max {rho_f:.3f} (low) -- it carries information the book "
                "did not already hold")
        else:
            tgt["mechanism_hint"] = "the factor is distinctive from the repository"

    elif category == "decay":
        # Persistent -- the edge did not fade. Name the slowest-moving (largest-
        # window) temporal component as the persistence source.
        t = _find_extreme_window_temporal(segments, want_min=False)
        if t is not None:
            fname, op, win, sig = t
            tgt.update(factor=fname, op=op, param=win, subtree_signature=sig,
                       mechanism_hint=(f"the edge persists: the slowest-moving "
                                       f"component is {op} {int(round(win))} on "
                                       f"{fname}"))
        else:
            tgt.update(mechanism_hint="the edge persists across the out-of-sample window")

    else:  # "unknown"
        tgt.update(mechanism_hint="this dimension is a measured strength of the factor")

    return tgt


# --------------------------------------------------------------------------
# The lineage-validated aggregation (the "consistently contribute" step).
# --------------------------------------------------------------------------

def _admitted(traj: Any) -> bool:
    m = getattr(traj, "backtest_metrics", None) or {}
    return bool(m.get("admitted", m.get("feasible", False)))


def _repair_record(act: dict, admitted_on: str) -> dict[str, Any]:
    """One validated repair action (the trimmed refine_actions fields + where it
    was admitted). Measurement only."""
    return {
        "verdict": act.get("verdict"),
        "weakness_dimension": act.get("weakness_dimension"),
        "refine_target": act.get("refine_target"),
        "mechanism_hint": act.get("mechanism_hint"),
        "target_subtree_signatures": list(act.get("target_subtree_signatures") or []),
        "admitted_on": admitted_on,
    }


def repair_actions_summary(parent: Any, ancestors: list | None = None) -> list[dict]:
    """Strategic repair actions VALIDATED on the lineage.

    Walks the parent + its ancestors; for each ADMITTED trajectory, its
    ``refine_actions`` are the repairs that PRODUCED an admitted factor -- i.e.
    a strategic repair action validated in a previously successful trajectory
    (the paper's credible-lineage material). Non-admitted trajectories' repairs
    were not validated and are not surfaced here. Empty when the lineage has no
    admitted refine parent (e.g. an original-direction parent, or a crossover
    child with empty refine_actions).

    Data caveat: ``refine_actions`` is a TRIMMED 6-field summary -- the full
    ``RefinementDirective`` (ablation reasoning, detailed targets) is not
    persisted. So this names the validated repair and the sub-tree it touched,
    not the full before/after reasoning. Enough to inherit the DECISION; not
    enough to replay the rationale.
    """
    out: list[dict] = []
    chain = [parent] + list(ancestors or [])
    for traj in chain:
        if not _admitted(traj):
            continue
        actions = getattr(traj, "refine_actions", None) or []
        tid = getattr(traj, "trajectory_id", "")
        for act in actions:
            if not isinstance(act, dict):
                continue
            # A crossover child also writes a refine_actions entry (its task
            # carries a refine_directive with refine_target="recombine"), but a
            # RECOMBINATION IS NOT A REPAIR: it addressed no diagnosed weakness
            # (weakness_dimension is None), so reporting it as "a strategic
            # repair action validated in a previously successful trajectory"
            # would credit the lineage with a fix that never happened.
            if act.get("refine_target") == RefineTarget.RECOMBINE.value:
                continue
            out.append(_repair_record(act, tid))
    return out


def _hypothesis_template(parent: Any) -> dict[str, str]:
    """The parent's hypothesis template components (the populated string fields
    of ``hypothesis_details``). These are the template the crossover LLM reasons
    over -- reason / observation / justification / knowledge -- not the raw
    hypothesis prose alone."""
    hd = getattr(parent, "hypothesis_details", None) or {}
    if not isinstance(hd, dict):
        return {}
    return {k: v for k, v in hd.items() if isinstance(v, str) and v.strip()}


def lineage_validated_segments(parent: Any,
                                ancestors: list | None) -> dict[str, Any]:
    """Aggregate the three segment types across the ADMITTED lineage by
    recurrence -- the paper's *"consistently contribute to high cumulative
    rewards"* made concrete.

    * **construction_patterns:** ``build_segments`` on each admitted ancestor;
      temporal-op signatures aggregated by recurrence across admitted factors.
      The parent's own strongest sub-pattern is its immediate instance; a
      signature recurring in N admitted ancestors is a construction that
      *consistently* contributed.
    * **hypothesis_templates:** the ``hypothesis_details`` components populated
      across admitted ancestors, with a recurrence count (a component populated
      in >=2 admitted ancestors is a template that *consistently* paid off).
    * **repair_actions:** ``repair_actions_summary`` -- the validated repairs.

    Each record carries its recurrence count and (for construction patterns) the
    ancestor it was admitted on. The parent is included in the chain when it was
    itself admitted. Empty/absent ancestors => a per-parent-only roll-up (still
    useful, just not cross-trajectory).
    """
    chain = [parent] + list(ancestors or [])
    admitted_chain = [t for t in chain if _admitted(t)]

    # --- construction patterns: aggregate temporal-op signatures by recurrence
    # Recurrence counts DISTINCT ADMITTED TRAJECTORIES the signature appears in,
    # NOT occurrences. `DELAY($close,1)` twice inside one expression is one
    # factor's internal repetition, not a pattern that "consistently contributed
    # across the lineage" -- counting occurrences reported "recurred in 2
    # admitted ancestors" for a single parent with no ancestors at all, which is
    # exactly the false lineage claim this block exists to avoid.
    sig_trajs: dict[str, set] = {}
    sig_first: dict[str, dict] = {}
    for traj in admitted_chain:
        tid = getattr(traj, "trajectory_id", None) or id(traj)
        for seg in build_segments(traj):
            for top in seg.temporal_ops:
                sig = top.get("signature")
                if not sig:
                    continue
                if sig not in sig_trajs:
                    sig_first[sig] = {
                        "signature": sig,
                        "op": top.get("op"),
                        "windows": list(top.get("windows") or []),
                        "factor_name": seg.factor_name,
                        "combiner_weight": seg.combiner_weight,
                        "turnover_share": seg.turnover_share,
                    }
                    sig_trajs[sig] = set()
                sig_trajs[sig].add(tid)
    construction = [{**sig_first[k], "recurrence": len(sig_trajs[k])}
                    for k in sig_trajs]
    construction.sort(key=lambda r: r["recurrence"], reverse=True)

    # --- hypothesis templates: count populated hypothesis_details keys across admitted
    key_counts: dict[str, int] = {}
    key_instance: dict[str, str] = {}
    for traj in admitted_chain:
        hd = getattr(traj, "hypothesis_details", None) or {}
        if not isinstance(hd, dict):
            continue
        for k, v in hd.items():
            if isinstance(v, str) and v.strip():
                key_counts[k] = key_counts.get(k, 0) + 1
                key_instance.setdefault(k, v)
    hypothesis_templates = [
        {"component": k, "recurrence": key_counts[k], "instance": key_instance[k]}
        for k in sorted(key_counts, key=lambda kk: key_counts[kk], reverse=True)
    ]

    repair = repair_actions_summary(parent, ancestors)
    return {
        "construction_patterns": construction,
        "hypothesis_templates": hypothesis_templates,
        "repair_actions": repair,
    }


# --------------------------------------------------------------------------
# The StrengthDirective (strength/weakness-agnostic reuse of the
# RefinementDirective shape). Carries the located strengths + lineage-validated
# segments as plain data so it survives task serialization into a parallel child.
# --------------------------------------------------------------------------

@dataclass
class StrengthDirective:
    """The structured output of the strength-diagnosis step.

    Everything the crossover LLM needs to author a child that inherits the
    VALIDATED ideas of two best parents. The structured fields are deterministic;
    only ``directive_text`` may be authored by an LLM for fluency (hybrid, mirrors
    ``RefinementDirective``). Carries the paper's three segment types:
    ``targets`` (per-dimension located construction strengths), ``hypothesis_template``
    (the hypothesis template), and ``lineage_segments`` (the recurrence-aggregated
    construction patterns + hypothesis templates + validated repair actions).
    """

    verdict: Verdict
    strongest_dimension: str | None
    refine_target: RefineTarget = RefineTarget.RECOMBINE
    # Nothing is frozen -- a crossover child is a fresh combination, not a
    # refinement of one parent. Kept for shape-parity with RefinementDirective.
    frozen_layers: list[str] = field(default_factory=list)
    mechanism_hint: str = ""
    directive_text: str = ""
    # Per-dimension located strengths (one per strong e_* dimension), each naming
    # the actual operator + parameter + factor behind the strength.
    targets: list[dict] = field(default_factory=list)
    # The parent's hypothesis template components (hypothesis_details strings).
    hypothesis_template: dict[str, str] = field(default_factory=dict)
    # The lineage-validated roll-up: construction_patterns (recurrence), hypothesis
    # templates (recurrence), and validated repair actions.
    lineage_segments: dict[str, Any] = field(default_factory=dict)
    # The frozen prefix (both parents' factors + the inherited signs) -- the
    # constructor's inspiration channel. Carried as plain data for parallel safety.
    parent_hypothesis: str = ""
    parent_expected_ic_sign: str = ""
    parent_factors: list[dict[str, Any]] = field(default_factory=list)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for task serialization (must survive a Process pickle)."""
        return {
            "verdict": self.verdict.value,
            "strongest_dimension": self.strongest_dimension,
            "refine_target": self.refine_target.value,
            "frozen_layers": list(self.frozen_layers),
            "mechanism_hint": self.mechanism_hint,
            "directive_text": self.directive_text,
            "targets": [dict(t) for t in self.targets],
            "hypothesis_template": dict(self.hypothesis_template),
            "lineage_segments": _serialize_lineage(self.lineage_segments),
            "parent_hypothesis": self.parent_hypothesis,
            "parent_expected_ic_sign": self.parent_expected_ic_sign,
            "parent_factors": [dict(f) for f in self.parent_factors],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrengthDirective":
        return cls(
            verdict=Verdict(d.get("verdict", "no_data")),
            strongest_dimension=d.get("strongest_dimension"),
            refine_target=RefineTarget(d.get("refine_target", "recombine")),
            frozen_layers=list(d.get("frozen_layers", [])),
            mechanism_hint=d.get("mechanism_hint", ""),
            directive_text=d.get("directive_text", ""),
            targets=[dict(t) for t in d.get("targets", [])],
            hypothesis_template=dict(d.get("hypothesis_template", {})),
            lineage_segments=dict(d.get("lineage_segments", {})),
            parent_hypothesis=d.get("parent_hypothesis", ""),
            parent_expected_ic_sign=d.get("parent_expected_ic_sign", ""),
            parent_factors=[dict(f) for f in d.get("parent_factors", [])],
        )


def _serialize_lineage(seg: dict[str, Any]) -> dict[str, Any]:
    """Coerce a lineage_segments dict to plain JSON-serializable data."""
    def _flt(v):
        if isinstance(v, dict):
            return {k: _flt(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_flt(x) for x in v]
        if isinstance(v, float) and v != v:
            return None
        return v
    return _flt(seg)


# --------------------------------------------------------------------------
# Deterministic strength-table fallback (the inverse of diagnosis._DIRECTIVES).
# --------------------------------------------------------------------------

def _render_strength_text(targets: list[dict], lineage_segments: dict,
                          metrics: dict[str, Any]) -> str:
    """The deterministic strength directive text -- measurement only, closes
    'yours to determine'."""
    lines = ["MEASUREMENT: the strengths this parent's measurements validate:"]
    for i, t in enumerate(targets, 1):
        e = t.get("e_score")
        e_str = f" (e={e:.2f})" if isinstance(e, (int, float)) and e == e else ""
        lines.append(f"{i}. {t['dimension']}{e_str}: {t['mechanism_hint']}")

    # A lineage claim requires the pattern to appear in MORE THAN ONE admitted
    # trajectory. The parent itself is in the admitted chain, so recurrence == 1
    # means "this parent only" -- calling that "consistently contributed along
    # this trajectory" would assert a validation the measurement does not have.
    cpat = [p for p in (lineage_segments.get("construction_patterns") or [])
            if int(p.get("recurrence", 0) or 0) >= 2]
    if cpat:
        top = cpat[0]
        lines.append(
            f"Lineage: the construction pattern {top.get('signature')} recurs in "
            f"{top.get('recurrence', 0)} admitted trajectories on this lineage -- a "
            "construction that consistently contributed.")
    repair = lineage_segments.get("repair_actions") or []
    if repair:
        r = repair[0]
        lines.append(
            f"Lineage: a validated repair ({r.get('refine_target')} on "
            f"{r.get('weakness_dimension')}) produced an admitted factor -- a "
            "strategic repair action validated in a previously successful trajectory.")

    lines.append(
        "These are the validated decisions a crossover child may inherit. Which "
        "to combine, and how, is yours to determine.")
    return "\n".join(lines)


def _strength_fallback_directive(parent: Any, ancestors: list | None,
                                 metrics: dict[str, Any],
                                 segments: list[SegmentProfile]) -> StrengthDirective:
    """The deterministic strength directive (the table path)."""
    strong_dims = strongest_dimensions(metrics)
    targets = [_build_strength_target(dim, e, segments, metrics)
               for dim, e in strong_dims] or []
    primary = strong_dims[0][0] if strong_dims else None
    lineage = lineage_validated_segments(parent, ancestors)
    text = _render_strength_text(targets, lineage, metrics)
    return StrengthDirective(
        verdict=classify_verdict(metrics),
        strongest_dimension=primary,
        refine_target=RefineTarget.RECOMBINE,
        frozen_layers=[],
        mechanism_hint=targets[0]["mechanism_hint"] if targets else "",
        directive_text=text,
        targets=targets,
        hypothesis_template=_hypothesis_template(parent),
        lineage_segments=lineage,
        parent_hypothesis=str(getattr(parent, "hypothesis", "") or ""),
        parent_expected_ic_sign=str(
            getattr(parent, "expected_ic_sign", "")
            or metrics.get("sign_predicted", "") or "").strip().lower(),
        parent_factors=[dict(f) for f in (getattr(parent, "factors", None) or [])],
        raw_metrics=dict(metrics),
    )


# --------------------------------------------------------------------------
# The LLM-authored strength diagnosis (hybrid: deterministic structure + LLM prose).
# --------------------------------------------------------------------------

_SYSTEM_STRENGTH = (
    "You are diagnosing the STRENGTHS of a quantitative equity alpha factor that "
    "has just been evaluated, so a second parent can be combined with it. You are "
    "given the factor, every measurement taken on it, what its lineage validated, "
    "and the attempts that came before it.\n\n"
    "Your job is to LOCATE WHAT WORKED, not to design a child. Say what the "
    "measurements show this factor did WELL and WHERE in it the measurements "
    "locate that strength (which construction sub-pattern, which scored dimension, "
    "which horizon, which validated repair on its lineage). Reason from the "
    "measurements given and the structure of the expression. If the measurements "
    "do not support a specific strength, say so -- 'nothing is resolvably strong "
    "beyond noise' is a valid diagnosis. Do not manufacture a strength to have "
    "something to say.\n\n"
    "You state the measured strength and the part of the factor the measurements "
    "locate it in. You do NOT state what the child should be, how to combine this "
    "parent with another, or what the next factor should do -- how to combine the "
    "strengths of two parents is not your call. Reason from the numbers you are "
    "shown and the structure of the expression. Do not assume facts about this "
    "market that is not in the data given to you -- you are not told which market "
    "this is, and any assumption about which effects should work here would be a "
    "guess.\n\n"
    "Use the lineage. If a construction pattern or a repair action recurred across "
    "admitted ancestors, that is a decision validated in previously successful "
    "trajectories -- name it as the credible-lineage material a child may inherit. "
    "Do not propose a combination -- naming the validated strength is the "
    "diagnosis; what to combine it with is not your call.\n\n"
    "Return JSON with exactly these keys:\n"
    '  "strength": what worked and how the measurements show it (2-4 sentences)\n'
    '  "strongest": one short label for the primary strength, or "none"\n'
    '  "directive": the measured strength stated for whoever combines this parent '
    "with another (3-6 sentences). Name the strength, where the measurements "
    "locate it, and the observed outcomes. Do NOT state what to combine or how; "
    "close with 'how to combine that is yours to determine.'"
)


def _lineage_block(ancestors: list | None) -> str:
    """How this factor's lineage scored (the credibility context)."""
    if not ancestors:
        return ""
    steps = []
    for a in ancestors[-4:]:
        am = dict(getattr(a, "backtest_metrics", None) or {})
        steps.append(
            f"  - {getattr(getattr(a, 'phase', None), 'value', '?')}: "
            f"RankIC {_fmt(_get(am, 'rank_ic'))}, t_nw {_fmt(_get(am, 't_nw'))}, "
            f"contribution {_fmt(am.get('delta_mean'))}, admitted {am.get('admitted')}"
        )
    return "\n\n## How this factor's lineage has scored\n" + "\n".join(steps)


def _lineage_segments_block(lineage: dict[str, Any]) -> str:
    """The validated segments (construction patterns + hypothesis templates +
    repairs), rendered measurement-only."""
    parts: list[str] = []
    cpat = lineage.get("construction_patterns") or []
    if cpat:
        rows = [f"  - {p.get('signature')} (op {p.get('op')}, in "
                f"{p.get('recurrence', 0)} admitted trajector"
                f"{'y' if int(p.get('recurrence', 0) or 0) == 1 else 'ies'} on this "
                f"lineage)" for p in cpat[:4]]
        parts.append("### Construction patterns validated on the lineage\n" + "\n".join(rows))
    htemp = lineage.get("hypothesis_templates") or []
    if htemp:
        rows = [f"  - {h['component']} (populated in {h['recurrence']} admitted "
                f"trajector{'y' if h['recurrence'] == 1 else 'ies'} on this "
                f"lineage): {h['instance'][:160]}" for h in htemp[:4]]
        parts.append("### Hypothesis templates validated on the lineage\n" + "\n".join(rows))
    repair = lineage.get("repair_actions") or []
    if repair:
        rows = [f"  - {r.get('refine_target')} on {r.get('weakness_dimension')} "
                f"(admitted on {r.get('admitted_on')}): {r.get('mechanism_hint', '')[:160]}"
                for r in repair[:4]]
        parts.append("### Strategic repair actions validated on the lineage\n" + "\n".join(rows))
    if not parts:
        return ""
    return "\n\n## Trajectory segments that consistently contributed\n" + "\n\n".join(parts)


def _build_strength_prompt(parent: Any, metrics: dict[str, Any],
                            population: list | None, ancestors: list | None,
                            lineage: dict[str, Any]) -> str:
    hypothesis = getattr(parent, "hypothesis", "") or "(none recorded)"
    factors = getattr(parent, "factors", None) or []
    expr_lines = [f"  - {f.get('name', 'unnamed')}: {f.get('expression', '')}"
                  for f in factors] or ["  (no expressions recorded)"]

    hd = _hypothesis_template(parent)
    template_block = ""
    if hd:
        template_block = ("\n\n## Hypothesis template (the hypothesis_details)\n"
                          + "\n".join(f"  - {k}: {v[:200]}" for k, v in hd.items()))

    return (
        "## The factor being strength-diagnosed\n"
        f"Hypothesis: {hypothesis}\n"
        "Expressions:\n" + "\n".join(expr_lines) +
        template_block +
        "\n\n## Everything measured on it\n" + _measurement_block(metrics) +
        _lineage_block(ancestors) +
        _lineage_segments_block(lineage) +
        "\n\n## Attempts that came before this one\n" + _population_block(population) +
        "\n\nLocate this factor's measured strengths and where the measurements "
        "locate them. Do not state what to combine or how; how to combine that "
        "is yours to determine."
    )


def _apply_llm_prose(base: StrengthDirective, payload: dict) -> StrengthDirective:
    """Hybrid: keep the deterministic structure; let the LLM author the prose."""
    strength = str(payload.get("strength", "") or "").strip()
    directive = str(payload.get("directive", "") or "").strip()
    if directive:
        base.directive_text = directive
    if strength:
        base.mechanism_hint = strength
    strongest = str(payload.get("strongest", "") or "").strip()
    # If the LLM's label names a real scored dimension, prefer it; else keep the
    # deterministic primary. Never invent a dimension the metrics do not carry.
    if strongest and strongest.lower() not in ("none", "n/a", "nothing"):
        scored = {k[2:] for k in base.raw_metrics if k.startswith("e_")}
        if strongest in scored:
            base.strongest_dimension = strongest
    return base


def diagnose_strength(parent: Any, ancestors: list | None = None,
                      population: list | None = None, *,
                      ablation_eval: Any = None) -> StrengthDirective | None:
    """Locate a parent's validated strengths for crossover recombination.

    Returns ``None`` only when there is nothing to diagnose (no objective
    vector), matching ``llm_diagnose``'s contract. Otherwise returns a
    ``StrengthDirective`` whose structured fields (strongest dimension, located
    targets, hypothesis template, lineage-validated segments, validated repairs)
    are deterministic and whose prose may be LLM-authored.

    ``ablation_eval`` is accepted for signature parity with ``diagnose_parent``
    (so the controller can pass the same callable) but is not used here -- the
    strength diagnosis reads the already-computed ``factor_attribution`` /
    ``segment`` credit rather than re-running a solo ablation.
    """
    del ablation_eval  # signature parity; not used (see docstring)
    metrics: dict[str, Any] = dict(getattr(parent, "backtest_metrics", None) or {})
    if "U" not in metrics:
        return None  # no objective vector -- nothing to diagnose

    segments = build_segments(parent)
    base = _strength_fallback_directive(parent, ancestors, metrics, segments)
    if not _enabled():
        return base

    try:
        from quantaalpha.llm.client import APIBackend

        raw = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=_build_strength_prompt(parent, metrics, population,
                                                ancestors, base.lineage_segments),
            system_prompt=_SYSTEM_STRENGTH,
            json_mode=True,
        )
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(payload, dict) or not payload.get("directive"):
            raise ValueError(f"strength diagnosis returned no directive: {str(payload)[:200]}")
        base = _apply_llm_prose(base, payload)
        logger.info(
            "llm_strength_diagnosis: verdict=%s strongest=%s",
            base.verdict.value, base.strongest_dimension,
        )
        return base
    except Exception as exc:
        logger.warning("llm_strength_diagnosis failed (%s); falling back to the table", exc)
        return base


__all__ = [
    "StrengthDirective",
    "strongest_dimensions",
    "repair_actions_summary",
    "lineage_validated_segments",
    "diagnose_strength",
]