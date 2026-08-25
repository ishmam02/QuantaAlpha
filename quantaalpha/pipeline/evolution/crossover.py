"""
Crossover operator for combining multiple parent strategies.

The crossover operator takes multiple parent trajectories and generates a hybrid
strategy that combines their strengths while avoiding their weaknesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from quantaalpha.log import logger
from .trajectory import StrategyTrajectory, RoundPhase


# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evolution_prompts.yaml"


# -- Two-best parent selection (Eq. 7) --------------------------------------
# The paper's crossover selects the TWO BEST-performing parents and recombines
# their VALIDATED ideas -- not a high-signal x low-turnover niche pair, not a
# diversity heuristic. Selection ranks on the shrunk marginal-contribution
# estimate (``fitness_of`` from the controller, falling back to the primary
# metric) and forms disjoint top-2 pairs: group 1 = the two best, group 2 = the
# next two, etc. The only tie-break is to AVOID pairing two children of the SAME
# direction when they are fitness-near-equal -- two same-direction near-clones
# are exactly the pair the ``check_crossover`` distinctive-vocabulary gate then
# struggles to separate, and they carry little complementary information anyway.

# The fitness band within which the direction tie-break may swap the second
# pick, as a fraction of the population's (max-min) fitness spread. 5%: a nudge
# that only fires when the second and third candidates are genuinely
# indistinguishable on fitness, never overriding a real quality gap.
_TIE_BREAK_BAND = 0.05


def _fitness(t: StrategyTrajectory, fitness_of: dict[str, float] | None) -> float:
    """The shrunk marginal-contribution estimate when available, else the
    single-seed primary metric. Zero when neither is present (sorts last)."""
    if fitness_of and t.trajectory_id in fitness_of:
        v = fitness_of[t.trajectory_id]
    else:
        v = t.get_primary_metric()
    return 0.0 if v is None else float(v)


def _tie_break_band(values: list[float]) -> float:
    """The fitness gap within which two candidates count as near-equal."""
    if len(values) < 2:
        return 0.0
    spread = max(values) - min(values)
    if spread != spread or spread <= 0:
        return 0.0
    return _TIE_BREAK_BAND * spread


class CrossoverOperator:
    """Eq. 7 crossover: recombine the VALIDATED IDEAS of the two best parents.

    The crossover process:
    1. Select the two best-performing parents (``select_crossover_pairs``).
    2. For each parent, locate the construction decisions its measurements
       validated -- strongest construction sub-pattern, strongest scored
       dimension, the lineage-validated repair (``diagnose_strength``).
    3. Hand both parents' strength directives to the hypothesis generator as
       inspiration, and let it AUTHOR A NEW hypothesis that combines those
       validated ideas -- not a literal splice of the two expressions, not a
       plain average. The child states which validated decision it inherits from
       which parent (credible lineage) and pre-registers an IC sign.
    4. The ``check_crossover`` gate verifies the child carries vocabulary
       distinctive to each parent (genuine dual-parent inheritance).
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """
        Initialize crossover operator.
        
        Args:
            prompt_path: Path to YAML file containing prompts.
                        If None, uses default prompt path.
        """
        self.prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> dict[str, str]:
        """Load prompts from YAML file."""
        if self.prompt_path and self.prompt_path.exists():
            try:
                all_prompts = yaml.safe_load(self.prompt_path.read_text(encoding="utf-8")) or {}
                crossover_prompts = all_prompts.get("crossover", {})
                if crossover_prompts:
                    return crossover_prompts
            except Exception as e:
                logger.warning(f"Failed to load crossover prompts from {self.prompt_path}: {e}")
        
        # Minimal fallback prompts (English). The crossover round authors its
        # child via the hypothesis generator + the suffix, so only
        # parent_template / phase_names / suffix_template are live (no separate
        # "draft a fusion" LLM step).
        logger.warning("Using minimal fallback prompts for crossover operator")
        return {
            "parent_template": "Parent {idx}: {hypothesis}",
            "phase_names": {
                "original": "Original Round",
                "mutation": "Mutation Round",
                "crossover": "Crossover Round"
            },
        }
    
    def _format_parent_summary(self, parent: StrategyTrajectory, idx: int) -> str:
        """Format a single parent trajectory for the prompt."""
        phase_names = self.prompts.get("phase_names", {
            "original": "Original Round",
            "mutation": "Mutation Round",
            "crossover": "Crossover Round"
        })
        phase_name = phase_names.get(parent.phase.value, "Unknown")
        
        factors_str = ""
        if parent.factors:
            for f in parent.factors[:3]:
                name = f.get("name", "unknown")
                # Full expression, not an 80-char truncation: the RECOMBINE
                # instruction asks the LLM to inherit construction sub-patterns
                # across parents -- it cannot recombine a construction it can
                # only see the first 80 chars of. The constructor-facing
                # inspiration block already carries full expressions; this keeps
                # the two channels consistent.
                expr = f.get("expression", "")
                factors_str += f"  - {name}: {expr}\n"
        else:
            factors_str = "  N/A\n"
        
        from quantaalpha.pipeline.evolution.trajectory import (
            format_metric,
            format_objective_note,
        )

        metrics_str = ""
        if parent.backtest_metrics:
            for k, v in parent.backtest_metrics.items():
                if v is not None:
                    metrics_str += f"  - {k}: {format_metric(v)}\n"
        if not metrics_str:
            metrics_str = "  N/A\n"

        # SIGN-ALIGNMENT BANNER. The crossover guidance warns that parents whose
        # realized RankIC have OPPOSITE signs have produced children weaker than
        # either parent. rank_ic is already in the metric dump above, but buried
        # among ~20 entries; surface each parent's measured sign at the top so
        # the generator sees opposite-sign risk per parent. The sign is read from
        # the parent's own rank_ic -- data-driven, no market prior assumed.
        _ric = (parent.backtest_metrics or {}).get("rank_ic")
        if isinstance(_ric, (int, float)) and _ric == _ric:
            _dir = "NEGATIVE" if _ric < 0 else "positive"
            metrics_str = (
                f"  >> MEASURED SIGN: this parent's realized RankIC is "
                f"{_ric:+.4f} ({_dir}) -- i.e. a high factor value was followed "
                f"by {'LOWER' if _ric < 0 else 'HIGHER'} forward returns.\n"
                + metrics_str)

        # Admissibility verdict + the gate it missed. Empty when no verdict is present.
        objective_note = format_objective_note(parent)
        if objective_note:
            metrics_str = f"  {objective_note}\n{metrics_str}"
        
        template = self.prompts.get("parent_template", "")
        if template:
            return template.format(
                idx=idx,
                phase_name=phase_name,
                direction_id=parent.direction_id,
                hypothesis=parent.hypothesis[:300] if parent.hypothesis else "N/A",
                factors=factors_str,
                metrics=metrics_str,
                feedback=parent.feedback[:200] if parent.feedback else "N/A"
            )
        
        # Default format
        return f"""### Parent {idx}: {phase_name}
**Direction ID**: {parent.direction_id}
**Hypothesis**: {parent.hypothesis[:300] if parent.hypothesis else 'N/A'}
**Factors**:
{factors_str}
**Metrics**:
{metrics_str}
**Feedback**:
{parent.feedback[:200] if parent.feedback else 'N/A'}
---
"""
    
    def _parent_sign(self, parent: StrategyTrajectory, strength) -> str:
        """The parent's pre-registered IC sign (data-driven)."""
        if strength is not None and getattr(strength, "parent_expected_ic_sign", ""):
            return str(strength.parent_expected_ic_sign).strip().lower()
        return str(getattr(parent, "expected_ic_sign", "") or "").strip().lower()

    def _render_strength_block(
        self,
        parents: list[StrategyTrajectory],
        strengths: list,
    ) -> str:
        """The suffix's {strength_block}: what each parent's measurement
        validated (the located strengths + the lineage-validated segments),
        measurement-only. Each ``StrengthDirective.directive_text`` already
        states the located strength and closes with "how to combine that is
        yours to determine", so this only labels it per parent -- it adds no
        prescription of its own."""
        parts: list[str] = []
        for i, (p, s) in enumerate(zip(parents, strengths)):
            phase = getattr(getattr(p, "phase", None), "value", "?")
            header = f"#### Parent {i + 1} (Direction {p.direction_id}, {phase})"
            if s is not None and getattr(s, "directive_text", ""):
                parts.append(f"{header}\n{s.directive_text}")
            else:
                # No objective vector / no strength located -- say so rather
                # than manufacture one. The parent's measured outcomes are still
                # in the parent summary above.
                parts.append(
                    f"{header}\nNo validated strength was located beyond what the "
                    "parent summary reports (the strength diagnosis had no "
                    "objective vector to read)."
                )
        return "\n\n".join(parts)

    def _strength_inspiration_block(
        self,
        parents: list[StrategyTrajectory],
        strengths: list,
    ) -> str:
        """The CONSTRUCTOR-facing inspiration block (injected into the factor
        constructor's feedback channel by ``proposal.py`` on the recombine path).

        This is NOT the child. It surfaces the two parents' validated strengths
        and their factor expressions as the ideas a fresh child may inherit, and
        tells the constructor to realize the new hypothesis rather than copy an
        expression. Measurement-only: the strength prose is the diagnosis's own
        (diagnose-never-prescribe); this block adds no remedy, no market prior."""
        lines: list[str] = [
            "### Crossover: validated ideas to draw from (inspiration, not the child)",
            "",
            "Two parents were selected for crossover. Below are the construction "
            "decisions each one's measurements validated, and each parent's factor "
            "expressions. These are the ideas the new factor may inherit -- realize "
            "the new hypothesis from them; do not copy an expression from either "
            "parent.",
            "",
        ]
        for i, (p, s) in enumerate(zip(parents, strengths)):
            phase = getattr(getattr(p, "phase", None), "value", "?")
            lines.append(f"#### Parent {i + 1} (Direction {p.direction_id}, {phase})")
            if s is not None and getattr(s, "directive_text", ""):
                lines.append("Validated strengths:")
                lines.append(s.directive_text)
            else:
                lines.append("Validated strengths: (none located beyond the metrics)")
            exprs = [f.get("expression", "") for f in (p.factors or []) if f.get("expression")]
            if exprs:
                lines.append("Expressions:")
                for e in exprs[:3]:
                    lines.append(f"  - {e}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _combined_strength_text(
        self,
        parents: list[StrategyTrajectory],
        strengths: list,
    ) -> str:
        """A short routing directive text (the ``refine_directive.directive_text``
        for the recombine path). Names that two parents were recombined and
        their strongest dimensions; carries no prescription."""
        bits: list[str] = []
        for i, (p, s) in enumerate(zip(parents, strengths)):
            dim = getattr(s, "strongest_dimension", None) if s else None
            dim = f" (strongest dimension: {dim})" if dim else ""
            bits.append(f"parent {i + 1} (direction {p.direction_id}){dim}")
        return ("Crossover recombination of " + " and ".join(bits)
                + ". Combine the validated ideas; the child is a new factor, not a "
                  "splice or an average.")

    def build_task_extras(
        self,
        parents: list[StrategyTrajectory],
        strengths: list | None = None,
    ) -> dict[str, Any] | None:
        """Eq. 7: ship the two parents' validated strengths as CONSTRUCTOR
        inspiration (not a frozen splice), and a routing directive so the loop
        enters refine mode and the gate fires.

        The old crossover built a mechanical AST splice and shipped it as the
        literal child ("these expressions ARE this round's children -- do not
        replace them"). The measured result was that the far parent contributed
        essentially nothing (child-vs-far-parent expression similarity 0.543
        against a null of 0.417 whose own p90 is 0.592 -- below chance; children
        inherited a mean 21% of the far parent's distinctive tokens). Crossover
        had degenerated into editing one lookback constant on the nearer parent.

        Now the two parents' strength directives (what each parent's measurement
        VALIDATED -- strongest construction sub-pattern, strongest scored
        dimension, the lineage-validated repair) travel to the hypothesis
        generator (the suffix) and the constructor (the inspiration block) as
        IDEAS, and the LLM authors a NEW hypothesis + a FRESH expression. Nothing
        is frozen (``frozen_layers=[]``), so there is no ``refine_factors_block``;
        ``refine_target="recombine"`` routes through proposal.py's default
        generate+construct path with the inspiration block injected.

        ``strengths`` is the list of ``StrengthDirective`` (one per parent, order
        matching ``parents``), computed once by the controller (T6) so the suffix
        and this build cannot diverge. ``None`` entries are tolerated (the parent
        had no objective vector) -- the block says so rather than invent a
        strength.

        Returns ``None`` only when there are fewer than 2 parents; the caller
        falls back to an orthogonal mutation for that slot.
        """
        if len(parents) < 2:
            return None
        strengths = list(strengths or [None] * len(parents))
        # Pad/truncate to parents length defensively.
        if len(strengths) < len(parents):
            strengths += [None] * (len(parents) - len(strengths))
        a, b = parents[0], parents[1]
        sa, sb = strengths[0], strengths[1]

        crossover_parents = {
            "a": [f.get("expression", "") for f in (a.factors or []) if f.get("expression")],
            "b": [f.get("expression", "") for f in (b.factors or []) if f.get("expression")],
        }
        crossover_strengths = [s.to_dict() if s is not None else None for s in strengths]

        refine_directive = {
            "refine_target": "recombine",
            "verdict": "crossover",
            "weakness_dimension": None,
            "directive_text": self._combined_strength_text(parents, strengths),
            "frozen_layers": [],
        }

        # ``parent_prefix`` keeps the single-dict shape the loop/proposal read,
        # with the second parent under ``parent_b`` (the constructor does not
        # freeze on the recombine path, so this is audit/sign context only).
        parent_prefix = {
            "hypothesis": (sa.parent_hypothesis if sa is not None else (a.hypothesis or "")),
            "factors": [dict(f) for f in (a.factors or [])],
            "expected_ic_sign": self._parent_sign(a, sa),
            "parent_b": {
                "hypothesis": (sb.parent_hypothesis if sb is not None else (b.hypothesis or "")),
                "factors": [dict(f) for f in (b.factors or [])],
                "expected_ic_sign": self._parent_sign(b, sb),
            },
        }

        suffix = self.generate_crossover_prompt_suffix(parents, strengths)

        return {
            "strategy_suffix": suffix,
            "refine_mode": True,
            "refine_directive": refine_directive,
            "parent_prefix": parent_prefix,
            # The two parents' expressions -- the ``check_crossover`` gate reads
            # these to verify the child carries vocabulary distinctive to EACH
            # parent (genuine dual-parent inheritance). Without them the loop
            # cannot tell a crossover task from a refine task.
            "crossover_parents": crossover_parents,
            # Constructor-facing inspiration (proposal.py injects it on the
            # recombine path); audit record of the two directives.
            "crossover_strength_block": self._strength_inspiration_block(parents, strengths),
            "crossover_strengths": crossover_strengths,
            # No frozen-prefix block (crossover builds a fresh expression) and no
            # hypothesis-revise block (the generator authors a new premise).
            "refine_factors_block": "",
            "revise_hypothesis_block": "",
        }

    def generate_crossover_prompt_suffix(
        self,
        parents: list[StrategyTrajectory],
        strengths: list | None = None,
    ) -> str:
        """The prompt suffix appended to the hypothesis generator on a crossover
        round (Eq. 7).

        Renders the two parents' summaries (hypotheses, expressions, measured
        outcomes, measured sign) and a "what each parent's measurement
        validated" block built from their strength directives, then formats the
        ``suffix_template``. No mechanical splice, no separate "draft a fusion"
        LLM step -- the hypothesis generator authors the new hypothesis fresh
        from this suffix (the ``refine_target="recombine"`` value falls through
        proposal.py's freeze/sign/hypothesis branches to the default generate
        path, with this suffix on the direction).

        ``strengths`` is the list of ``StrengthDirective`` (one per parent),
        computed once by the controller; ``None`` entries are tolerated.
        """
        strengths = list(strengths or [None] * len(parents))
        if len(strengths) < len(parents):
            strengths += [None] * (len(parents) - len(strengths))

        parent_summaries = "\n".join(
            self._format_parent_summary(p, i + 1) for i, p in enumerate(parents)
        )
        strength_block = self._render_strength_block(parents, strengths)

        suffix_template = self.prompts.get("suffix_template")
        if suffix_template:
            return suffix_template.format(
                parent_summaries=parent_summaries,
                strength_block=strength_block,
            )

        # Fallback when no template loaded.
        return (
            "\n---\n\n## Crossover Round Guidance\n\n"
            "### Parent Strategy Summaries\n" + parent_summaries + "\n\n"
            "### What each parent's measurement validated\n" + strength_block + "\n\n"
            "Author a new hypothesis that combines the validated ideas above.\n"
        )
    
    def select_crossover_pairs(
        self,
        candidates: list[StrategyTrajectory],
        crossover_size: int = 2,
        crossover_n: int = 3,
        prefer_diverse: bool = True,
        selection_strategy: str = "best",
        top_percent_threshold: float = 0.3,
        fitness_of: dict[str, float] | None = None,
    ) -> list[list[StrategyTrajectory]]:
        """Select the TWO BEST-performing parents per crossover group (Eq. 7).

        Ranks candidates by the shrunk marginal-contribution estimate
        (``fitness_of`` from the controller, falling back to the primary metric)
        and forms ``crossover_n`` disjoint top groups: group 1 = the
        ``crossover_size`` best, group 2 = the next ``crossover_size``, and so
        on. Each group is the best-REMAINING parents -- the paper's "select two
        best performing", not a high-signal x low-turnover niche pair and not a
        diversity heuristic.

        The one tie-break: when filling a group, among candidates whose fitness
        is within ``_TIE_BREAK_BAND`` of the best remaining, prefer a
        ``direction_id`` not already in the group, so a pair is not two
        same-direction near-clones (which carry little complementary information
        and which the ``check_crossover`` distinctive-vocabulary gate then
        struggles to separate). The tie-break only fires within the near-equal
        band -- it never overrides a real quality gap.

        ``prefer_diverse`` / ``selection_strategy`` / ``top_percent_threshold``
        are accepted for signature compatibility; crossover is now pure
        two-best-by-fitness (the diversity / weighted-sampling paths paired
        one-high x one-low, which Eq. 7 does not ask for).
        """
        del prefer_diverse, selection_strategy, top_percent_threshold
        if len(candidates) < crossover_size:
            return []

        # Rank best-first on shrunk fitness (or the primary-metric fallback).
        ranked = sorted(
            candidates, key=lambda t: _fitness(t, fitness_of), reverse=True
        )
        band = _tie_break_band([_fitness(t, fitness_of) for t in ranked])

        groups: list[list[StrategyTrajectory]] = []
        pool = list(ranked)
        while len(groups) < crossover_n and len(pool) >= crossover_size:
            group: list[StrategyTrajectory] = []
            used_dirs: set = set()
            while len(group) < crossover_size and pool:
                if not group:
                    pick = pool.pop(0)
                else:
                    # Among the near-equal front (fitness within `band` of the
                    # best remaining), prefer a direction not already in the
                    # group. The front is contiguous (pool is sorted); stop at
                    # the band edge. Fall back to the plain best remaining.
                    top_val = _fitness(pool[0], fitness_of)
                    pick = None
                    for j, cand in enumerate(pool):
                        if _fitness(cand, fitness_of) < top_val - band:
                            break  # left the near-equal band
                        if cand.direction_id not in used_dirs:
                            pick = pool.pop(j)
                            break
                    if pick is None:
                        pick = pool.pop(0)
                group.append(pick)
                used_dirs.add(pick.direction_id)
            groups.append(group)

        return groups
