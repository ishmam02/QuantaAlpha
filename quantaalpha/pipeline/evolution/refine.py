"""The diagnose-and-refine mutation operator (the paper's Eq. 6, implemented).

Parallel to ``MutationOperator``, but where mutation generates an ORTHOGONAL
new hypothesis (a diversification/random-restart), refinement BUILDS ON the
parent: diagnose why the parent fell short, freeze the sound layers, and refine
only the diagnosed weakness. This is the operator that makes the search *learn*
rather than drift -- the measured signature of the old operator was a trendless
marginal contribution (t=+0.42 over 79 batches, ``controller.py:623-625``).

Architecture (user-confirmed decisions 2026-08-13):

* **Q1 hybrid.** The structured directive (verdict / weakness_dimension /
  refine_target / frozen_layers / mechanism_hint) is produced **deterministically**
  by ``diagnosis.diagnose`` -- a rule table over the verdict string + the weakest
  scored dimension. The directional signal is guaranteed, not LLM-iffy. An
  *optional* LLM may rewrite ``directive_text`` for fluency, gated on
  ``QA_REFINE_LLM_RATIONALE=true`` and OFF by default so the tests stay hermetic
  and the default path is fully reproducible.
* **Q2 hypothesis + expression.** Cost / turnover / overfit / diversity / decay
  faults freeze the hypothesis and refine the expression (the cost-stall case).
  Signal-quality faults rewrite the hypothesis. No code layer -- the coder
  renders code from the expression, so refining the expression refines the code.
* **No LLM round-trip in the operator itself.** Unlike mutation (which calls
  the LLM to invent an orthogonal hypothesis), the refinement DIRECTION comes
  from ``diagnose``. The only LLM calls happen later, inside the loop's
  hypothesis/construct steps, guided by the directive.

The operator does not touch the frozen protocol ``Θ``. The controller dispatches
to it for every non-tail parent (the explore tail and any non-diagnosable parent
fall back to orthogonal mutation).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from quantaalpha.log import logger
from .diagnosis import (
    RefinementDirective, RefineTarget, diagnose, sign_flip_directive,
)
from .trajectory import StrategyTrajectory, format_metric, format_objective_note

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evolution_prompts.yaml"

# Off by default: the rule-based directive_text is already fluent and the
# structured fields are the load-bearing part. When ON, an LLM rewrites the
# text for style only (the structured fields are never touched).
_LLM_RATIONALE = os.environ.get("QA_REFINE_LLM_RATIONALE", "false").lower() in (
    "1", "true", "yes", "on")


class RefinementOperator:
    """Diagnose a parent and produce a refinement directive + prompt suffix.

    The counterpart of ``MutationOperator``. Where
    mutation asks the LLM for an orthogonal direction, refinement asks the
    metrics: ``diagnose(parent)`` reads the verdict and the weakest dimension
    off the trajectory and returns a structured directive. The suffix rendered
    here is the AlphaEvolve "artifact side-channel" -- it tells the LLM *why*
    the parent failed, not merely that it did.
    """

    def __init__(self, prompt_path: Optional[Path] = None):
        self.prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, Any]:
        if self.prompt_path and self.prompt_path.exists():
            try:
                all_prompts = yaml.safe_load(self.prompt_path.read_text(encoding="utf-8")) or {}
                refine_prompts = all_prompts.get("refine", {})
                if refine_prompts:
                    return refine_prompts
            except Exception as e:
                logger.warning(f"Failed to load refine prompts from {self.prompt_path}: {e}")
        # Minimal fallback (English) if the YAML section is missing.
        logger.warning("Using minimal fallback prompts for refinement operator")
        return {
            "suffix_template": (
                "---\n\n## Refinement Round Guidance\n\n{parent_summary}\n\n"
                "### Diagnosis\n{directive_text}\n\n{keep_change_note}\n"
            ),
            "frozen_prefix_factors": (
                "### Factors to REFINE (do not discard)\n{parent_factors}\n\n"
                "### Directive\n{directive_text}\n"
            ),
            "revise_hypothesis_block": (
                "### Parent hypothesis to revise\n{parent_hypothesis}\n\n"
                "### Diagnosis\n{directive_text}\n"
            ),
            "keep_change": {
                "expression": "KEEP the hypothesis; REFINE the expressions.",
                "hypothesis": "REVISE the hypothesis; re-derive the expressions.",
            },
        }

    # ------------------------------------------------------------------
    def diagnose_parent(self, parent: StrategyTrajectory,
                        ancestors: list | None = None,
                        population: list | None = None, *,
                        ablation_eval=None) -> RefinementDirective | None:
        """The deterministic self-reflection step (Eq. 6's missing k-selection).

        Returns ``None`` for a parent with no ``U`` (nothing to diagnose), or a
        directive with ``is_refinement() is False`` when there is a ``U`` but no usable
        verdict -- both cases tell the caller to breed orthogonally instead.

        ``ancestors`` (T5) is the parent's lineage (from
        ``TrajectoryPool.get_ancestors``); it lets the diagnosis detect a
        refinement lever pulled >=K times without admission and switch the
        layer / route to an orthogonal restart (fix #5). Omit for the pre-T5
        lineage-unaware behaviour.

        ``ablation_eval`` (optional callable ``parent -> SegmentAblation | None``)
        runs the per-segment solo measurement of the parent's expression -- the
        AlphaEvolve "which sub-tree is broken" signal. It is computed HERE, once,
        and threaded to BOTH the LLM path (shown as a measurement block + carried
        on ``directive.ablation_summary``) and its table fallback (which routes
        ``_build_target`` on the broken part and applies the IC-neutral-window
        backstop). It is run only when the parent has a ``U`` (otherwise
        ``llm_diagnose`` returns ``None`` and the heavy eval would be wasted),
        and any failure is caught so an ablation error never blocks a diagnosis.
        """
        # The LLM authors the diagnosis (AlphaEvolve); the hardcoded
        # (verdict, dimension) table is the fallback, not the default. See
        # llm_diagnosis for why the table cannot compound.
        from quantaalpha.pipeline.evolution.llm_diagnosis import llm_diagnose

        _m = dict(getattr(parent, "backtest_metrics", None) or {})
        if "U" not in _m:
            # No U -> llm_diagnose returns None; nothing to diagnose, and the
            # sign/mechanism gate (which is what produces sign_realized) never
            # ran, so a sign-flip cannot apply either.
            return None

        # First-class deterministic refine: the SIGN-FLIP (the measured-direction
        # fix). A parent rejected for a sign MISMATCH whose edge cleared the bar
        # (|t_nw| >= k_sigma) is a mislabeled edge, NOT an exhausted lever -- the
        # construction has a real edge, the hypothesis just labeled its direction
        # backwards. Correct the direction and re-test the FROZEN construction.
        # No LLM call (both the diagnosis and the child are deterministic), so
        # this is checked BEFORE the ablation/LLM path: the construction is frozen
        # verbatim so the heavy per-segment eval would be wasted, and the
        # short-circuit must not be suppressed by an LLM ``exhausted=true``
        # verdict -- a backwards label is never an exhausted lever. This is the
        # most literal "learn from the measured mistake": the search measured that
        # its prediction was backwards and corrects it, admitting a real edge that
        # the sign gate wrongly rejected. See ``diagnosis.sign_flip_directive``.
        sd = sign_flip_directive(parent, _m)
        if sd is not None:
            logger.info(
                f"sign-flip refine: predicted={_m.get('sign_predicted')} "
                f"realized={_m.get('sign_realized')} |t|={abs(float(_m.get('t_nw') or 0)):.2f} "
                f"-> freeze construction, correct direction to {sd.parent_expected_ic_sign}"
            )
            return sd

        abl, summary = None, ""
        if callable(ablation_eval):
            try:
                abl = ablation_eval(parent)
                summary = getattr(abl, "summary", "") or "" if abl is not None else ""
            except Exception as e:
                logger.warning(f"ablation_eval failed ({e}); diagnosing without it")
                abl, summary = None, ""
        return llm_diagnose(parent, ancestors, population,
                            ablation=abl, ablation_summary=summary)

    def _maybe_llm_rationale(self, directive: RefinementDirective) -> str:
        """Optional fluency rewrite of ``directive_text`` (Q1 hybrid), off by default.

        The structured fields are never touched; only the prose is rewritten,
        and any failure falls back to the deterministic rule-based text so the
        operator never blocks on the LLM.
        """
        if not _LLM_RATIONALE or not directive.directive_text:
            return directive.directive_text
        try:
            from quantaalpha.llm.client import APIBackend

            system = (
                "You rewrite a factor-refinement instruction for clarity and style. "
                "Preserve the directive's meaning, the named weakness, and every "
                "instruction to keep or change a specific layer. Do not add new "
                "requirements. Output the rewritten instruction text only."
            )
            user = (
                f"Verdict: {directive.verdict.value}\n"
                f"Weakness dimension: {directive.weakness_dimension}\n"
                f"Mechanism: {directive.mechanism_hint}\n"
                f"Refine target: {directive.refine_target.value}\n\n"
                f"Instruction to rewrite:\n{directive.directive_text}"
            )
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user, system_prompt=system, json_mode=False)
            text = (resp or "").strip()
            return text or directive.directive_text
        except Exception as e:
            logger.warning(f"refine LLM rationale failed ({e}); using rule-based text")
            return directive.directive_text

    # ------------------------------------------------------------------
    def _parent_summary(self, parent: StrategyTrajectory) -> str:
        """A compact summary of the parent for the prompt (the frozen prefix)."""
        parts = [f"Hypothesis: {parent.hypothesis or 'N/A'}"]
        if parent.factors:
            factor_strs = []
            for f in parent.factors[:5]:
                name = f.get("name", "unknown")
                expr = f.get("expression", "")
                factor_strs.append(f"  - {name}: {expr}")
            parts.append("Factor expressions:\n" + "\n".join(factor_strs))
        metrics = parent.backtest_metrics or {}
        if metrics:
            # Render the objective note (verdict + weakest dims) first, then
            # the raw metrics -- this IS the artifact side-channel.
            note = format_objective_note(parent)
            if note:
                parts.append(note)
            metric_str = ", ".join(
                f"{k}={format_metric(v)}" for k, v in metrics.items()
                if v is not None and not k.startswith("e_")
            )
            if metric_str:
                parts.append(f"Metrics: {metric_str}")
            # The full per-dimension score vector (e_effectiveness, e_arr,
            # e_stability, e_turnover, e_diversity, e_overfit, e_decay). The
            # diagnosis directive names the WEAKEST dimension, but a refiner that
            # sees only that one score has no view of the rest of the profile --
            # it cannot tell that fixing turnover (e=0.10) at the cost of
            # stability (e=0.62) is a bad trade. These were filtered out here
            # before; surface them as a labeled block so the whole profile is
            # visible and tradeoffs are informed, not blind.
            dim_str = ", ".join(
                f"{k[2:]}={format_metric(v)}" for k, v in sorted(metrics.items())
                if k.startswith("e_") and v is not None
            )
            if dim_str:
                parts.append(f"Dimension scores (e_j, 0-1): {dim_str}")
        return "\n".join(parts)

    def _keep_change_note(self, directive: RefinementDirective) -> str:
        table = self.prompts.get("keep_change", {})
        key = directive.refine_target.value
        return table.get(key, table.get(
            "expression", "KEEP the hypothesis; REFINE the expressions."))

    def generate_refine_prompt_suffix(
        self, parent: StrategyTrajectory, directive: RefinementDirective,
    ) -> str:
        """Render the deterministic suffix that carries the directive to the LLM.

        No LLM call: the suffix is built straight from the directive + parent
        summary. (The optional rationale rewrite happens on ``directive_text``
        before this, if ``QA_REFINE_LLM_RATIONALE`` is on.)
        """
        directive_text = self._maybe_llm_rationale(directive)
        template = self.prompts.get("suffix_template")
        if template:
            try:
                return template.format(
                    parent_summary=self._parent_summary(parent),
                    directive_text=directive_text,
                    keep_change_note=self._keep_change_note(directive),
                )
            except (KeyError, IndexError):
                pass
        # Fallback (no template / render error)
        return (
            f"\n---\n## Refinement Round Guidance\n\n"
            f"{self._parent_summary(parent)}\n\n"
            f"### Diagnosis\n{directive_text}\n\n"
            f"{self._keep_change_note(directive)}\n"
        )

    # ------------------------------------------------------------------
    def frozen_prefix_factors_block(self, directive: RefinementDirective) -> str:
        """The block the factor constructor injects when refining expressions.

        Empty when the directive does not freeze the hypothesis (i.e. a
        hypothesis-refine re-derives expressions from a new premise and has no
        frozen expression prefix to show).
        """
        if directive.refine_target is not RefineTarget.EXPRESSION:
            return ""
        factors = directive.parent_factors
        if not factors:
            return ""
        factor_strs = []
        for f in factors[:5]:
            name = f.get("name", "unknown")
            expr = f.get("expression", "")
            factor_strs.append(f"  - {name}: {expr}")
        parent_factors = "\n".join(factor_strs)
        template = self.prompts.get("frozen_prefix_factors")
        if template:
            try:
                return template.format(
                    parent_factors=parent_factors,
                    directive_text=directive.directive_text,
                )
            except (KeyError, IndexError):
                pass
        return (f"### Factors to REFINE (do not discard)\n{parent_factors}\n\n"
                f"### Directive\n{directive.directive_text}\n")

    def revise_hypothesis_block(self, directive: RefinementDirective) -> str:
        """The block the hypothesis generator injects when revising the premise."""
        template = self.prompts.get("revise_hypothesis_block")
        if template:
            try:
                return template.format(
                    parent_hypothesis=directive.parent_hypothesis or "N/A",
                    directive_text=directive.directive_text,
                )
            except (KeyError, IndexError):
                pass
        return (f"### Parent hypothesis to revise\n{directive.parent_hypothesis}\n\n"
                f"### Diagnosis\n{directive.directive_text}\n")

    # ------------------------------------------------------------------
    def build_task_extras(self, parent: StrategyTrajectory,
                          ancestors: list | None = None,
                          population: list | None = None, *,
                          ablation_eval=None,
                          directive=None) -> dict[str, Any] | None:
        """Produce the structured task fields for a refine child of ``parent``.

        Returns ``None`` when the parent cannot be diagnosed (no ``U``, no
        verdict) or the verdict gives no basis to refine (``is_refinement`` False);
        in both cases the caller falls back to an orthogonal mutation for this
        parent. Otherwise returns the fields the controller merges into the
        task dict and the loop reads to enter refine mode:

            ``strategy_suffix``  -- the prompt text (the side-channel)
            ``refine_mode``      -- True
            ``refine_directive`` -- the structured directive, plain-dict
            ``parent_prefix``    -- {hypothesis, factors} the child builds on

        Everything is plain JSON-serializable data so it survives
        ``_serialize_task_for_parallel`` (which strips ``parent_trajectories``
        in parallel mode -- the directive/prefix must therefore be data, not a
        trajectory object).

        ``ancestors`` (T5) is passed through to ``diagnose_parent`` so the
        lineage-aware exhausted-lever switch (fix #5) can fire; the lineage note
        is baked into ``directive_text`` here, so the parallel child needs no
        pool access.
        """
        # Reuse a directive computed once by the caller (EvolutionController.
        # _cached_diagnosis) when supplied, so the bucket classification and this
        # build cannot diverge on the refine-vs-orthogonal call. Falls back to
        # diagnosing here (with ablation_eval) on a miss -- e.g. an ADMITTED-PUSH
        # parent the classification does not pre-diagnose.
        if directive is None:
            directive = self.diagnose_parent(parent, ancestors, population,
                                             ablation_eval=ablation_eval)
        if directive is None or not directive.is_refinement():
            return None

        suffix = self.generate_refine_prompt_suffix(parent, directive)
        # Pre-render the step-specific prompt blocks here (the controller owns
        # the operator and its templates) so the loop steps only inject a
        # string and never need to import the operator or load YAML. Empty
        # string for the block that does not apply to this refine_target.
        refine_factors_block = self.frozen_prefix_factors_block(directive)
        revise_hypothesis_block = self.revise_hypothesis_block(directive)
        return {
            "strategy_suffix": suffix,
            "refine_mode": True,
            "refine_directive": directive.to_dict(),
            "parent_prefix": {
                "hypothesis": directive.parent_hypothesis,
                "factors": [dict(f) for f in directive.parent_factors],
                # Belt-and-braces: carry the parent's pre-registered direction so
                # the proposal.py freeze path's first disjunct
                # (pp.get("expected_ic_sign")) resolves. The load-bearing source is
                # directive.parent_expected_ic_sign (set in both diagnosis paths),
                # which itself reads the trajectory's expected_ic_sign.
                "expected_ic_sign": directive.parent_expected_ic_sign,
            },
            # Pre-rendered prompt blocks the loop hands to its steps.
            "refine_factors_block": refine_factors_block,
            "revise_hypothesis_block": revise_hypothesis_block,
        }


__all__ = ["RefinementOperator"]