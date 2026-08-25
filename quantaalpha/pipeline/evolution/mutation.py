"""
Mutation operator for generating orthogonal strategies.

The mutation operator takes a parent trajectory and generates a new hypothesis
that explores an orthogonal/independent direction from the parent. This ensures
diversity in the exploration space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend
from .trajectory import StrategyTrajectory, RoundPhase


# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evolution_prompts.yaml"


class MutationOperator:
    """
    Generates orthogonal (mutated) strategies from parent trajectories.
    
    The mutation process:
    1. Takes a parent trajectory's hypothesis, factors, and feedback
    2. Generates a new hypothesis that explores an orthogonal direction
    3. The new hypothesis should be fundamentally different to ensure diversity
    
    Key principles:
    - Orthogonality: New strategy should be nearly independent from parent
    - Diversity: Avoid repeating exploration paths
    - Learning: Use feedback from parent to avoid known pitfalls
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """
        Initialize mutation operator.
        
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
                mutation_prompts = all_prompts.get("mutation", {})
                if mutation_prompts:
                    return mutation_prompts
            except Exception as e:
                logger.warning(f"Failed to load mutation prompts from {self.prompt_path}: {e}")
        
        # Minimal fallback prompts (English)
        logger.warning("Using minimal fallback prompts for mutation operator")
        return {
            "system": "You are a quantitative finance strategy expert. Generate orthogonal strategies.",
            "user": "Generate an orthogonal strategy based on parent: {parent_hypothesis}",
            "simple_user": "Generate orthogonal hypothesis: {parent_hypothesis}",
            "fallback_templates": [
                "Explore mean reversion characteristics",
                "Study volume-price nonlinear relationships",
                "Analyze cross-cycle trend signals",
                "Mine market microstructure liquidity features",
            ]
        }
    
    def generate_mutation(
        self,
        parent: StrategyTrajectory,
        use_detailed_prompt: bool = True
    ) -> dict[str, str]:
        """
        Generate a mutated (orthogonal) strategy from parent.
        
        Args:
            parent: The parent trajectory to mutate from
            use_detailed_prompt: Whether to use detailed prompt (returns structured output)
                               or simple prompt (returns just hypothesis text)
        
        Returns:
            Dictionary containing mutation results:
            - "new_hypothesis": The new hypothesis text
            - "exploration_direction": Direction description (if detailed)
            - "orthogonality_reason": Why this is orthogonal (if detailed)
            - "expected_characteristics": Expected characteristics (if detailed)
        """
        # Format parent information
        parent_hypothesis = parent.hypothesis or "N/A"
        
        parent_factors = ""
        if parent.factors:
            for f in parent.factors[:5]:
                name = f.get("name", "unknown")
                expr = f.get("expression", "")
                desc = f.get("description", "")
                parent_factors += f"- {name}: {expr}\n  Description: {desc}\n"
        else:
            parent_factors = "N/A"
        
        from quantaalpha.pipeline.evolution.trajectory import (
            format_metric,
            format_objective_note,
        )

        parent_metrics = ""
        if parent.backtest_metrics:
            for k, v in parent.backtest_metrics.items():
                if v is not None:
                    parent_metrics += f"- {k}: {format_metric(v)}\n"
        if not parent_metrics:
            parent_metrics = "N/A"

        # Whether the parent was admitted, and if not, which gate it missed.
        # Empty string when there is no objective vector to refine on.
        objective_note = format_objective_note(parent)
        if objective_note:
            parent_metrics = f"{objective_note}\n{parent_metrics}"

        parent_feedback = parent.feedback or "N/A"
        
        # Build prompt
        system_prompt = self.prompts.get("system", "")
        
        if use_detailed_prompt:
            user_prompt = self.prompts.get("user", "").format(
                parent_hypothesis=parent_hypothesis,
                parent_factors=parent_factors,
                parent_metrics=parent_metrics,
                parent_feedback=parent_feedback
            )
        else:
            user_prompt = self.prompts.get("simple_user", "").format(
                parent_hypothesis=parent_hypothesis,
                parent_factors=parent_factors
            )
        
        # Call LLM
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=use_detailed_prompt
            )
            
            if use_detailed_prompt:
                result = self._parse_detailed_response(response)
            else:
                result = {"new_hypothesis": response.strip()}
            
            logger.info(f"Generated mutation from parent {parent.trajectory_id}")
            return result
            
        except Exception as e:
            # A failed call yields NOTHING, by design. This used to substitute a
            # canned hypothesis -- one of six hardcoded strategy ideas, or a
            # rule that answered "momentum" with "mean reversion" -- which then
            # entered the search indistinguishable from a reasoned proposal.
            # Those are exactly the stale, market-specific priors the system is
            # forbidden to inject: a mined result built on one would be crediting
            # the search with an answer it was handed. An empty slot is honest;
            # a canned one is a silent prior. The caller skips on {}.
            logger.error(
                f"Mutation generation failed for parent {parent.trajectory_id} "
                f"({e}) -- skipping this slot rather than substituting a canned "
                f"hypothesis"
            )
            return {}
    
    def _parse_detailed_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from LLM."""
        import json
        import re
        
        # Extract JSON from response
        text = response.strip()
        
        # Try to find JSON block
        fence_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        
        try:
            data = json.loads(text)
            return {
                "new_hypothesis": data.get("new_hypothesis", ""),
                "exploration_direction": data.get("exploration_direction", ""),
                # 2026-08-15: the prompt no longer asks the model to justify
                # ORTHOGONALITY (chasing difference for its own sake is not how
                # the search is scored -- redundancy is measured and enforced at
                # admission). It now asks what the parent's MEASUREMENT
                # established and how that led to the next hypothesis. Both new
                # keys fall back to the old one so trajectories written by the
                # previous schema still read cleanly.
                "parent_reading": data.get(
                    "parent_reading", data.get("orthogonality_reason", "")),
                "evidence_link": data.get(
                    "evidence_link", data.get("orthogonality_reason", "")),
                "expected_characteristics": data.get("expected_characteristics", "")
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, treat entire response as hypothesis
            return {"new_hypothesis": response.strip()}
    
    # ``_generate_fallback_hypothesis`` was REMOVED 2026-08-15.
    #
    # It answered a failed LLM call with a canned strategy idea: a keyword rule
    # ("momentum" -> "explore mean reversion", "volume" -> "explore price
    # patterns") over a list of six hardcoded directions, falling through to
    # ``random.choice``. Every one of those is a market prior chosen by the
    # author, and none of them is measured. Injected into the search they are
    # indistinguishable from a hypothesis the system reasoned its way to, so any
    # factor descended from one would credit the search with an answer it was
    # given. Failures now skip the slot instead -- see ``generate_mutation``.


    def generate_mutation_prompt_suffix(self, parent: StrategyTrajectory) -> str:
        """
        Generate a prompt suffix to be appended to the hypothesis generator.
        
        This suffix instructs the hypothesis generator to explore orthogonal directions.
        
        Args:
            parent: The parent trajectory
            
        Returns:
            Prompt suffix string
        """
        mutation_result = self.generate_mutation(parent, use_detailed_prompt=True)

        # An empty result means the LLM call failed and NOTHING was substituted
        # (the canned-hypothesis fallback was removed). Return an empty suffix so
        # the caller emits the base prompt rather than a guidance block whose
        # every field is blank -- an empty "Proposed next hypothesis:" heading
        # reads as an instruction to invent one, which is the trapdoor again.
        if not mutation_result.get("new_hypothesis"):
            logger.warning(
                f"no mutation guidance for parent {parent.trajectory_id}; "
                f"emitting the base prompt with no mutation block"
            )
            return ""

        # Use template from prompts if available
        suffix_template = self.prompts.get("suffix_template")
        if suffix_template:
            return suffix_template.format(
                parent_summary=parent.to_summary_text(),
                parent_reading=mutation_result.get('parent_reading', ''),
                new_hypothesis=mutation_result.get('new_hypothesis', 'Explore new direction'),
                exploration_direction=mutation_result.get('exploration_direction', ''),
                evidence_link=mutation_result.get('evidence_link', ''),
            )
        
        # Default suffix (English)
        suffix = f"""

---

## Mutation Round Guidance

An earlier hypothesis has been measured. This round proposes what to test next.

### Parent Strategy Summary
{parent.to_summary_text()}

### What the parent's measurement establishes
{mutation_result.get('parent_reading', '')}

### Proposed next hypothesis
- **Hypothesis**: {mutation_result.get('new_hypothesis', 'Explore new direction')}
- **Exploration Dimension**: {mutation_result.get('exploration_direction', '')}
- **How the parent's evidence led here**: {mutation_result.get('evidence_link', '')}

### What this round is scored on
1. The marginal contribution of the resulting factors to the existing book, net
   of trading cost. A premise the book already expresses contributes nothing
   measurable, however sound it is.
2. Redundancy against the book is measured separately and enforced at admission,
   so being different from the parent is not itself worth anything.

Resemblance to the parent is an outcome of your reasoning, not a target.
"""
        return suffix
