"""
Evolution controller for managing the original→mutation→crossover cycle.

The controller orchestrates the evolutionary process:
1. Original round: Initial exploration in each direction
2. Mutation round: Orthogonal exploration from each original trajectory
3. Crossover round: Combine trajectories across directions
4. Repeat mutation→crossover cycle
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import threading

from quantaalpha.log import logger
from .trajectory import (
    StrategyTrajectory,
    TrajectoryPool,
    RoundPhase,
    _PRIMARY_METRIC,
    _REQUIRE_FEASIBLE,
    is_admissible,
    format_metric,
)
from .mutation import MutationOperator
from .crossover import CrossoverOperator



def expected_factor_count(
    num_directions: int,
    crossover_n: int,
    max_rounds: int,
    factors_per_hypothesis: int,
) -> int:
    """How many factors a run of this shape is expected to mine.

    Mirrors the task generation in ``get_all_tasks_for_current_phase``:

    * round 0 -- ORIGINAL, one task per direction                  -> D
    * round 1 -- MUTATION of each original trajectory              -> D
    * rounds 2+ -- CROSSOVER and MUTATION alternating, each
      operating on the previous phase's crossover results          -> C each

    so ``batches = D + D + C*(max_rounds - 2)`` for ``max_rounds >= 2``.

    This is an *upper* estimate: a factor whose implementation fails to produce
    a usable signal is dropped, so a run can finish just short of it (17 rather
    than 18 was observed). Callers sizing a budget should treat it as a target
    and bound the search separately rather than assuming it is always reached.

    It lives beside the loop it describes on purpose -- a copy of this formula
    in a driver script would silently drift the first time the phase order
    changes.
    """
    d, c, r = max(int(num_directions), 0), max(int(crossover_n), 0), int(max_rounds)
    if r <= 0:
        batches = 0
    elif r == 1:
        batches = d
    else:
        batches = d + d + c * (r - 2)
    return batches * max(int(factors_per_hypothesis), 0)


@dataclass
class EvolutionConfig:
    """Configuration for evolution process."""
    # Number of planning directions (parallel original rounds)
    num_directions: int = 2
    
    # Steps per loop (5 for: propose/construct/calculate/backtest/feedback)
    steps_per_loop: int = 5
    
    # Maximum total rounds (original + mutation + crossover rounds)
    max_rounds: int = 10
    
    # Enable/disable mutation phase; when false, skip mutation rounds entirely
    mutation_enabled: bool = True

    # Enable/disable crossover phase; when false, skip crossover rounds entirely
    crossover_enabled: bool = True
    
    # Crossover parameters
    crossover_size: int = 2  # Number of parents per crossover
    crossover_n: int = 3     # Number of crossover combinations per round
    
    # Whether to prefer diverse crossover combinations
    prefer_diverse_crossover: bool = True
    
    # Parent selection for crossover: best | random | weighted | weighted_inverse | top_percent_plus_random
    parent_selection_strategy: str = "best"

    # Fraction of the previous phase kept as mutation parents, best first. 1.0
    # reproduces the old behaviour of mutating everything.
    mutation_top_fraction: float = 1.0
    # Top percent threshold when parent_selection_strategy = "top_percent_plus_random"
    top_percent_threshold: float = 0.3

    # Enable parallel execution within each round
    parallel_enabled: bool = False
    
    # Path to save trajectory pool
    pool_save_path: Optional[str] = None
    
    # Path to evolution prompts
    mutation_prompt_path: Optional[str] = None
    crossover_prompt_path: Optional[str] = None
    
    # Start with empty trajectory pool (ignore existing data)
    fresh_start: bool = True

    # Learning-aware reseed: rounds of NO repository growth before the search
    # regenerates NEW, outcome-informed directions (treatment arm only; the
    # control arm has no ledger and the gate no-ops). NOT a Theta field -- the
    # frozen protocol hash must not move.
    reseed_after_stale_rounds: int = 2
    # The controller owns the live direction list so it can GROW it on a reseed.
    # Seeded from the initial planning output; ``num_directions`` is the frozen
    # initial count (also the reseed batch size).
    directions: list[str] = field(default_factory=list)
    initial_direction: str = ""
    informed_prompt_path: Optional[str] = None


class EvolutionController:
    """
    Controls the evolutionary exploration process.
    
    The evolution cycle:
    1. Original rounds: Run initial exploration for each planning direction
    2. Mutation rounds: Generate orthogonal strategies from each original
    3. Crossover rounds: Combine top trajectories across all directions
    4. Repeat: mutation → crossover → mutation → crossover → ...
    
    The controller:
    - Tracks all trajectories in a pool
    - Determines which phase/round to run next
    - Generates strategy guidance for each round
    - Manages parent selection for mutation/crossover
    
    After crossover, the number of parallel branches changes:
    - Initial: num_directions branches
    - After first crossover: crossover_n branches (and so on)
    """
    
    def __init__(self, config: EvolutionConfig):
        """
        Initialize evolution controller.
        
        Args:
            config: Evolution configuration
        """
        self.config = config
        
        # Initialize trajectory pool with fresh_start option
        pool_path = Path(config.pool_save_path) if config.pool_save_path else None
        self.pool = TrajectoryPool(save_path=pool_path, fresh_start=config.fresh_start)
        
        # Initialize operators
        mutation_path = Path(config.mutation_prompt_path) if config.mutation_prompt_path else None
        crossover_path = Path(config.crossover_prompt_path) if config.crossover_prompt_path else None
        self.mutation_op = MutationOperator(prompt_path=mutation_path)
        self.crossover_op = CrossoverOperator(prompt_path=crossover_path)
        
        # State tracking
        self._current_round = 0
        self._current_phase = RoundPhase.ORIGINAL
        self._directions_completed = set()  # Track which directions completed original
        self._crossover_groups: list[list[StrategyTrajectory]] = []  # Current crossover groups
        self._crossover_idx = 0  # Which crossover group is next
        
        # Track active branch count (changes after crossover)
        self._active_branch_count = config.num_directions
        # Track trajectories to mutate in current mutation round
        self._mutation_targets: list[StrategyTrajectory] = []
        self._mutation_idx = 0  # Current index in mutation targets

        # Learning-aware reseed state. The controller owns the live direction
        # list (grown on reseed) and a per-direction outcome tally used to mark
        # saturated directions and to build the digest fed to the LLM.
        self._directions: list[str] = list(getattr(config, "directions", []) or [])
        self._direction_status: list[dict] = [
            self._blank_direction_status(i, "initial")
            for i in range(len(self._directions))
        ]
        self._best_zoo_size = -1
        self._stale_rounds = 0
        self._reseed_count = 0

    @staticmethod
    def _blank_direction_status(direction_id: int, source: str) -> dict:
        return {
            "direction_id": direction_id,
            "admitted_count": 0,
            "rejected_count": 0,
            "last_admit_round": -1,
            "attempts": 0,
            "saturated": False,
            "source": source,
        }

    def get_current_state(self) -> dict[str, Any]:
        """Get current evolution state."""
        return {
            "round": self._current_round,
            "phase": self._current_phase.value,
            "directions_completed": list(self._directions_completed),
            "active_branch_count": self._active_branch_count,
            "mutation_targets_remaining": len(self._mutation_targets) - self._mutation_idx if self._mutation_targets else 0,
            "crossover_groups_remaining": len(self._crossover_groups) - self._crossover_idx,
            "pool_stats": self.pool.get_statistics(),
        }
    
    def get_next_task(self) -> Optional[dict[str, Any]]:
        """
        Determine the next task to run.
        
        Returns:
            Dictionary describing the next task:
            - "phase": RoundPhase (original/mutation/crossover)
            - "direction_id": Which direction (for original/mutation)
            - "parent_trajectories": Parent trajectories (for mutation/crossover)
            - "strategy_suffix": Prompt suffix for hypothesis generator
            - "round_idx": Current round index
            
            Returns None if evolution is complete.
        """
        # Round budget: max_rounds, or QA_TARGET_ZOO-driven (see _rounds_exhausted)
        if self._rounds_exhausted():
            return None
        
        # Phase: ORIGINAL
        if self._current_phase == RoundPhase.ORIGINAL:
            return self._get_original_task()
        
        # Phase: MUTATION
        elif self._current_phase == RoundPhase.MUTATION:
            return self._get_mutation_task()
        
        # Phase: CROSSOVER
        elif self._current_phase == RoundPhase.CROSSOVER:
            return self._get_crossover_task()
        
        return None
    
    def get_all_tasks_for_current_phase(self) -> list[dict[str, Any]]:
        """
        Get all remaining tasks for the current phase.
        
        This is used for parallel execution - returns all tasks that can
        be executed in parallel within the current round/phase.
        
        Returns:
            List of task dictionaries, or empty list if phase is complete
        """
        if self._rounds_exhausted():
            return []
        
        tasks = []
        
        # Phase: ORIGINAL - collect all remaining original tasks
        if self._current_phase == RoundPhase.ORIGINAL:
            for d in range(len(self._directions)):
                if d not in self._directions_completed:
                    tasks.append({
                        "phase": RoundPhase.ORIGINAL,
                        "direction_id": d,
                        "direction": self._directions[d],
                        "parent_trajectories": [],
                        "strategy_suffix": "",
                        "round_idx": self._current_round,
                    })
            
            # If no tasks, transition phase for next call
            if not tasks:
                self._current_round += 1
                # Transition based on enabled phases
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                elif self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                else:
                    return []  # No evolution, just original
                return self.get_all_tasks_for_current_phase()
        
        # Phase: MUTATION - collect all remaining mutation tasks
        elif self._current_phase == RoundPhase.MUTATION:
            # Skip if mutation is disabled
            if not self.config.mutation_enabled:
                if self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                    self._current_round += 1
                    return self.get_all_tasks_for_current_phase()
                return []
            
            # Prepare mutation targets if needed
            if not self._mutation_targets:
                self._prepare_mutation_targets()
            
            for idx, parent in enumerate(self._mutation_targets):
                if idx < self._mutation_idx:
                    continue  # Skip already processed
                
                # Check if this mutation already exists
                existing = [t for t in self.pool.get_all()
                           if t.round_idx == self._current_round 
                           and t.phase == RoundPhase.MUTATION
                           and parent.trajectory_id in t.parent_ids]
                if existing:
                    continue
                
                suffix = self.mutation_op.generate_mutation_prompt_suffix(parent)
                zc = self._build_zoo_context()
                if zc:
                    suffix = suffix + "\n" + zc
                tasks.append({
                    "phase": RoundPhase.MUTATION,
                    "direction_id": idx,
                    "parent_trajectories": [parent],
                    "strategy_suffix": suffix,
                    "round_idx": self._current_round,
                })
            
            # If no tasks, transition phase for next call
            if not tasks:
                self._mutation_targets = []
                self._mutation_idx = 0
                self._current_round += 1
                
                if self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                else:
                    # Stay in mutation mode
                    self._current_phase = RoundPhase.MUTATION
                return self.get_all_tasks_for_current_phase()
        
        # Phase: CROSSOVER - collect all remaining crossover tasks
        elif self._current_phase == RoundPhase.CROSSOVER:
            # Skip if crossover is disabled
            if not self.config.crossover_enabled:
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                    self._current_round += 1
                    return self.get_all_tasks_for_current_phase()
                return []
            
            for idx in range(self._crossover_idx, len(self._crossover_groups)):
                parents = self._crossover_groups[idx]
                suffix = self.crossover_op.generate_crossover_prompt_suffix(parents)
                zc = self._build_zoo_context()
                if zc:
                    suffix = suffix + "\n" + zc
                tasks.append({
                    "phase": RoundPhase.CROSSOVER,
                    "direction_id": idx,
                    "parent_trajectories": parents,
                    "strategy_suffix": suffix,
                    "round_idx": self._current_round,
                })
            
            # If no tasks, transition phase for next call
            if not tasks:
                self._current_round += 1
                
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                else:
                    # Stay in crossover mode, prepare new groups
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                return self.get_all_tasks_for_current_phase()
        
        return tasks
    
    def advance_phase_after_parallel_completion(self, completed_tasks: list[dict[str, Any]]):
        """
        Update controller state after parallel tasks complete.
        
        Called after all parallel tasks in a phase complete to 
        advance the controller to the next phase.
        
        Args:
            completed_tasks: List of completed task dictionaries
        """
        if not completed_tasks:
            return
        
        phase = completed_tasks[0]["phase"]
        
        if phase == RoundPhase.ORIGINAL:
            # Mark all directions as completed
            for task in completed_tasks:
                self._directions_completed.add(task["direction_id"])
            
            # Transition based on enabled phases
            if len(self._directions_completed) >= len(self._directions):
                self._current_round += 1
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                    logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
                elif self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                    logger.info(f"All original rounds complete, transitioning to crossover (round {self._current_round})")
                else:
                    logger.info("Neither mutation nor crossover enabled, evolution complete")
        
        elif phase == RoundPhase.MUTATION:
            # Update mutation index to skip completed
            self._mutation_idx = len(self._mutation_targets)
            self._mutation_targets = []
            self._mutation_idx = 0
            self._current_round += 1
            
            # Transition based on enabled phases
            if self.config.crossover_enabled:
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                logger.info(f"All mutation rounds complete, transitioning to crossover (round {self._current_round})")
            else:
                # Stay in mutation mode
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All mutation rounds complete, continuing with mutation (round {self._current_round})")
        
        elif phase == RoundPhase.CROSSOVER:
            # Update crossover index
            self._crossover_idx = len(self._crossover_groups)
            self._current_round += 1
            
            # Transition based on enabled phases
            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All crossover rounds complete, transitioning to mutation (round {self._current_round})")
            else:
                # Stay in crossover mode, prepare new groups
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                logger.info(f"All crossover rounds complete, continuing with crossover (round {self._current_round})")

        # Learning-aware reseed (parallel path): after a breeding round that did
        # not grow the repository, regenerate informed directions. The caller
        # re-asks get_all_tasks_for_current_phase, which now emits ORIGINAL tasks
        # for the new (and re-opened) directions. Dormant under
        # QA_SEQUENTIAL_EVOLUTION=true (the default) but covers the parallel path.
        if phase in (RoundPhase.MUTATION, RoundPhase.CROSSOVER):
            self._reseed_if_stale()
    
    def _get_original_task(self) -> Optional[dict[str, Any]]:
        """Get next original round task."""
        # Find a direction that hasn't completed original
        for d in range(len(self._directions)):
            if d not in self._directions_completed:
                return {
                    "phase": RoundPhase.ORIGINAL,
                    "direction_id": d,
                    "direction": self._directions[d],
                    "parent_trajectories": [],
                    "strategy_suffix": "",  # No guidance for original
                    "round_idx": self._current_round,
                }
        
        # All directions completed original, transition to next phase
        self._current_round += 1
        return self._transition_to_next_phase_after_original()
    
    def _transition_to_next_phase_after_original(self) -> Optional[dict[str, Any]]:
        """
        Determine and transition to the next phase after original round completes.
        
        Returns the first task of the next phase, or None if evolution is complete.
        """
        # Case 1: Both mutation and crossover enabled - follow standard flow
        if self.config.mutation_enabled and self.config.crossover_enabled:
            self._current_phase = RoundPhase.MUTATION
            logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
            return self._get_mutation_task()
        
        # Case 2: Only mutation enabled - go to mutation
        elif self.config.mutation_enabled:
            self._current_phase = RoundPhase.MUTATION
            logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
            return self._get_mutation_task()
        
        # Case 3: Only crossover enabled - go to crossover
        elif self.config.crossover_enabled:
            self._prepare_crossover_groups()
            self._current_phase = RoundPhase.CROSSOVER
            logger.info(f"All original rounds complete, transitioning to crossover (round {self._current_round})")
            return self._get_crossover_task()
        
        # Case 4: Neither enabled - evolution is complete after original
        else:
            logger.info("Neither mutation nor crossover enabled, evolution complete after original")
            return None
    
    def _get_mutation_task(self) -> Optional[dict[str, Any]]:
        """Get next mutation round task."""
        # If mutation is disabled, skip to crossover or stay in mutation loop
        if not self.config.mutation_enabled:
            if self.config.crossover_enabled:
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                self._current_round += 1
                return self._get_crossover_task()
            return None
        
        # If mutation targets not prepared, prepare them
        if not self._mutation_targets:
            self._prepare_mutation_targets()
        
        # Process next mutation target
        while self._mutation_idx < len(self._mutation_targets):
            parent = self._mutation_targets[self._mutation_idx]
            direction_id = self._mutation_idx  # Use index as new direction ID
            
            # Check if this mutation already exists
            existing = [t for t in self.pool.get_all()
                       if t.round_idx == self._current_round 
                       and t.phase == RoundPhase.MUTATION
                       and parent.trajectory_id in t.parent_ids]
            if existing:
                self._mutation_idx += 1
                continue
            
            # Generate mutation guidance
            suffix = self.mutation_op.generate_mutation_prompt_suffix(parent)
            zc = self._build_zoo_context()
            if zc:
                suffix = suffix + "\n" + zc
            
            task = {
                "phase": RoundPhase.MUTATION,
                "direction_id": direction_id,
                "parent_trajectories": [parent],
                "strategy_suffix": suffix,
                "round_idx": self._current_round,
            }
            
            self._mutation_idx += 1
            return task
        
        # All mutation tasks complete, transition to next phase
        self._mutation_targets = []  # Reset for next mutation round
        self._mutation_idx = 0
        self._current_round += 1

        # Learning-aware reseed: if breeding has stalled, regenerate informed
        # directions and return to ORIGINAL with new material instead of
        # transitioning into yet more breeding of the same saturated space.
        if self._reseed_if_stale():
            return self._get_original_task()
        
        # Determine next phase based on config
        if self.config.crossover_enabled:
            self._prepare_crossover_groups()
            self._current_phase = RoundPhase.CROSSOVER
            logger.info(f"All mutation rounds complete, transitioning to crossover (round {self._current_round})")
            return self._get_crossover_task()
        else:
            # Stay in mutation mode (mutation-only loop)
            logger.info(f"All mutation rounds complete, continuing with mutation (round {self._current_round})")
            return self._get_mutation_task()
    
    def _prepare_mutation_targets(self):
        """
        Prepare targets for current mutation round.
        
        For the first mutation round (after original), mutate each original trajectory.
        For subsequent mutation rounds (after crossover), mutate each crossover result.
        """
        self._mutation_targets = []
        self._mutation_idx = 0
        
        # Get the previous round's outputs
        prev_round = self._current_round - 1
        
        if prev_round < 0:
            # This shouldn't happen - mutation should come after original
            logger.warning("Mutation round before any original rounds")
            return
        
        # Get trajectories from the previous phase
        prev_phase_trajs = []
        
        # After original round (round 0), we mutate original trajectories
        if self._current_round == 1:
            prev_phase_trajs = self.pool.get_by_phase(RoundPhase.ORIGINAL)
        else:
            # After crossover round, we mutate the crossover outputs
            # Find crossover trajectories from the most recent crossover round
            all_crossover = self.pool.get_by_phase(RoundPhase.CROSSOVER)
            # Get the most recent crossover round index
            if all_crossover:
                max_crossover_round = max(t.round_idx for t in all_crossover)
                prev_phase_trajs = [t for t in all_crossover if t.round_idx == max_crossover_round]
        
        if not prev_phase_trajs:
            # Fallback: get all trajectories from previous round
            prev_phase_trajs = [t for t in self.pool.get_all() if t.round_idx == prev_round]
        
        # Do not mutate a factor that F_Theta rejected (treatment arm only).
        prev_phase_trajs = self._admissible_parents(
            prev_phase_trajs, minimum=1, what="mutation targets"
        )

        # Select mutation parents on FITNESS, not on arrival order.
        #
        # This sorted by direction_id and mutated everything, which is not a
        # selection operator at all: the below-median half of the population was
        # bred exactly as often as the best of it. Crossover has always ranked
        # its parents (parent_selection_strategy); mutation never did, so half
        # the evolutionary budget was spent on directions the evidence already
        # said were not working. Measured consequence: marginal contribution had
        # no resolvable trend across a run (t=+0.42 over 79 batches), i.e. the
        # search was drifting rather than improving.
        #
        # mutation_top_fraction keeps a tail of weaker parents rather than
        # breeding only the leader, because the fitness is noisy and a strict
        # top-1 collapses diversity in a population this small.
        prev_phase_trajs = self._rank_by_fitness(prev_phase_trajs)
        frac = float(getattr(self.config, "mutation_top_fraction", 1.0) or 1.0)
        if 0.0 < frac < 1.0 and len(prev_phase_trajs) > 1:
            keep = max(1, int(round(frac * len(prev_phase_trajs))))
            dropped = len(prev_phase_trajs) - keep
            prev_phase_trajs = prev_phase_trajs[:keep]
            logger.info(f"mutation: kept the top {keep} of {keep + dropped} "
                        f"parent(s) by {_PRIMARY_METRIC}, dropped {dropped}")
        self._mutation_targets = prev_phase_trajs
        
        # Update active branch count
        self._active_branch_count = len(self._mutation_targets)
        
        logger.info(f"Prepared {len(self._mutation_targets)} mutation targets for round {self._current_round}")
    
    def _prepare_crossover_groups(self):
        """
        Prepare crossover groups for the next crossover round.
        
        Crossover candidates are selected from the two most recent rounds:
        - First crossover (after round 1): original (round 0) + mutation (round 1)
        - Subsequent crossovers: previous mutation + previous crossover
        
        This ensures that crossover combines the latest evolutionary results,
        not arbitrarily old trajectories.
        """
        # Find the two most recent rounds to use as crossover candidates
        candidates = self._admissible_parents(
            self._get_crossover_candidates(),
            minimum=self.config.crossover_size,
            what="crossover candidates",
        )

        if len(candidates) < self.config.crossover_size:
            logger.warning(f"Not enough candidates for crossover: {len(candidates)} < {self.config.crossover_size}")
            self._crossover_groups = []
            self._crossover_idx = 0
            return
        
        # Rank crossover parents by the shrunk marginal-contribution estimate
        # rather than the single-seed point estimate, so crossover pairing
        # matches the mutation-parent ranking and both track the seed-averaged
        # truth instead of a noise-dominated point estimate.
        self._crossover_groups = self.crossover_op.select_crossover_pairs(
            candidates=candidates,
            crossover_size=self.config.crossover_size,
            crossover_n=self.config.crossover_n,
            prefer_diverse=self.config.prefer_diverse_crossover,
            selection_strategy=self.config.parent_selection_strategy,
            top_percent_threshold=self.config.top_percent_threshold,
            fitness_of=self._shrunk_fitness(candidates),
        )
        self._crossover_idx = 0
        logger.info(f"Prepared {len(self._crossover_groups)} crossover groups from {len(candidates)} candidates")
    
    def _shrunk_fitness(
        self, trajectories: list[StrategyTrajectory]
    ) -> dict[str, float]:
        """Empirical-Bayes shrinkage of each parent's seed-averaged marginal
        contribution toward the group mean, weighted by its own standard error.

        Selection used to rank on ``m_delta_net_ir`` -- a single-seed point
        estimate that flipped 3 of 4 verdicts across combiner seeds. With
        ``delta_mean`` (averaged over ``test_seeds``) and ``delta_se`` now on
        the trajectory, this shrinks each parent's estimate:

            shrunk_i = mu + (tau^2 / (tau^2 + se_i^2)) * (delta_mean_i - mu)

        where ``mu`` is the group mean and ``tau^2`` is the between-parent
        signal variance, estimated as ``Var(delta_mean) - mean(se^2)`` (the
        within-batch noise removed from the observed spread). When measurement
        noise dominates (``se^2`` >> ``tau^2``) the shrinkage factor -> 0 and
        parents pull to ``mu`` -- selection honestly admits it cannot tell them
        apart, instead of breeding from the noise winner. When a parent is
        well-measured and parents genuinely differ, the factor -> 1 and the raw
        ``delta_mean`` stands.

        Falls back to ``get_primary_metric()`` (the point estimate, e.g. U for
        the control arm) for any trajectory without ``delta_mean``, so arms that
        do not run marginal_contribution mode are untouched.
        """
        picks = []
        for t in trajectories:
            m = (t.backtest_metrics or {}).get("delta_mean")
            s = (t.backtest_metrics or {}).get("delta_se")
            picks.append((t, m, s))

        have = [(t, float(m), s) for t, m, s in picks
                if m is not None and m == m]
        scores: dict[str, float] = {}

        if len(have) >= 2:
            means = [m for _, m, _ in have]
            mu = statistics.fmean(means)
            between = statistics.pvariance(means)
            # A missing/non-finite se means we have no spread info for that
            # parent. Exclude it from the within-batch variance (so it does not
            # leak inf into `within`), and at the shrink step trust its
            # seed-averaged delta_mean rather than pretend to shrink -- the
            # average is already the improvement over the single-seed point.
            def _se2(s):
                if s is None or s != s:
                    return None
                try:
                    sf = float(s)
                except (TypeError, ValueError):
                    return None
                if math.isinf(sf):
                    return None
                return sf * sf
            se2s = [_se2(s) for _, _, s in have]
            known = [v for v in se2s if v is not None]
            within = statistics.fmean(known) if known else 0.0
            tau2 = max(0.0, between - within)
            for (t, m, _), se2 in zip(have, se2s):
                # No usable spread info (missing / inf se): trust the
                # seed-averaged delta_mean rather than pretend to shrink.
                if se2 is None or (tau2 + se2) <= 0.0:
                    shrink = 1.0
                else:
                    shrink = tau2 / (tau2 + se2)
                scores[t.trajectory_id] = mu + shrink * (m - mu)

        # Anything without a delta_mean ranks on its primary metric (point
        # estimate) so the population is not silently shrunk for a reason
        # unrelated to quality.
        for t, m, _ in picks:
            scores.setdefault(t.trajectory_id, t.get_primary_metric() or 0.0)
        return scores

    def _rank_by_fitness(
        self, trajectories: list[StrategyTrajectory]
    ) -> list[StrategyTrajectory]:
        """Best first, on a shrunk estimate of marginal contribution, ties and
        misses last.

        See ``_shrunk_fitness``. Trajectories whose fitness is missing sort to
        the back rather than being dropped: a metric that failed to extract is
        not evidence that the trajectory was bad, and discarding it silently
        would shrink the population for a reason unrelated to quality.
        """
        scores = self._shrunk_fitness(trajectories)

        def key(t):
            v = scores.get(t.trajectory_id)
            if v is None:
                v = t.get_primary_metric()
            return (v is not None, v if v is not None else 0.0)

        ranked = sorted(trajectories, key=key, reverse=True)

        raw_means = [float(t.backtest_metrics["delta_mean"])
                     for t in trajectories
                     if (t.backtest_metrics or {}).get("delta_mean") is not None
                     and t.backtest_metrics["delta_mean"]
                     == t.backtest_metrics["delta_mean"]]
        if raw_means:
            shrunk_vals = [scores[t.trajectory_id] for t in trajectories
                           if t.trajectory_id in scores
                           and (t.backtest_metrics or {}).get("delta_mean") is not None]
            shrunk_vals = [v for v in shrunk_vals if v is not None]
            mu = statistics.fmean(raw_means)
            best = max(shrunk_vals) if shrunk_vals else float("nan")
            worst = min(shrunk_vals) if shrunk_vals else float("nan")
            logger.info(
                f"parents ranked on shrunk delta_mean: best {best:.5f} "
                f"worst {worst:.5f} (group mean {mu:.5f}, "
                f"{len(raw_means)}/{len(ranked)} seed-averaged)")
        else:
            scored = [t for t in ranked if t.get_primary_metric() is not None]
            if scored:
                logger.info(
                    f"parents ranked on {_PRIMARY_METRIC}: best "
                    f"{scored[0].get_primary_metric():.5f} worst "
                    f"{scored[-1].get_primary_metric():.5f} "
                    f"({len(scored)}/{len(ranked)} scored)")
            else:
                logger.warning(
                    f"no trajectory carries {_PRIMARY_METRIC} or delta_mean -- "
                    "mutation parents fall back to arrival order, which is not "
                    "selection. Check the runner emits it.")
        return ranked

    def _admissible_parents(
        self,
        trajectories: list[StrategyTrajectory],
        minimum: int,
        what: str,
    ) -> list[StrategyTrajectory]:
        """Restrict breeding stock to factors that passed F_Theta.

        The objective is `argmax U(m(f))` over `f in F_Theta`, so an inadmissible
        factor should not seed the next generation. Without this the gates are
        purely decorative: a rejected factor remains a fully eligible crossover
        and mutation parent, and the search happily breeds from it.

        Only active under the treatment arm (`QA_REQUIRE_FEASIBLE=true`); the
        control arm keeps today's unfiltered behaviour exactly.

        Falls back to the unfiltered set when filtering would leave too few
        parents to continue. Feasibility can legitimately be rare early in a run,
        and stalling the search is a worse failure than breeding from a rejected
        factor -- but the fallback is logged loudly, because a run that never
        recovers is telling you the gates are mis-calibrated.
        """
        if not _REQUIRE_FEASIBLE or not trajectories:
            return trajectories

        admissible = [t for t in trajectories if is_admissible(t)]
        if len(admissible) < minimum:
            logger.warning(
                f"F_theta: only {len(admissible)}/{len(trajectories)} {what} are admissible "
                f"(need {minimum}); falling back to the unfiltered set. If this persists, "
                f"the gates are too tight for this run."
            )
            return trajectories

        if len(admissible) < len(trajectories):
            logger.info(
                f"F_theta: {len(admissible)}/{len(trajectories)} {what} admissible; "
                f"{len(trajectories) - len(admissible)} rejected factor(s) excluded from breeding"
            )
        return admissible

    def _get_crossover_candidates(self) -> list[StrategyTrajectory]:
        """
        Get candidates for crossover from the two most recent relevant rounds.
        
        Logic depends on enabled phases:
        - Both enabled:
          - First crossover: original (round 0) + mutation (round 1)
          - Subsequent crossovers: latest mutation + latest crossover
        - Crossover-only (mutation disabled):
          - First crossover: original trajectories only
          - Subsequent crossovers: two most recent crossover rounds
        
        Returns:
            List of trajectories to use as crossover candidates
        """
        all_trajs = self.pool.get_all()
        if not all_trajs:
            return []
        
        # Get trajectories by phase
        original_trajs = self.pool.get_by_phase(RoundPhase.ORIGINAL)
        mutation_trajs = self.pool.get_by_phase(RoundPhase.MUTATION)
        crossover_trajs = self.pool.get_by_phase(RoundPhase.CROSSOVER)
        
        # Find the most recent mutation round
        latest_mutation_round = -1
        if mutation_trajs:
            latest_mutation_round = max(t.round_idx for t in mutation_trajs)
        
        # Find the most recent crossover round
        latest_crossover_round = -1
        if crossover_trajs:
            latest_crossover_round = max(t.round_idx for t in crossover_trajs)
        
        candidates = []
        
        # ========================================================
        # CROSSOVER-ONLY MODE (mutation disabled)
        # ========================================================
        if not self.config.mutation_enabled:
            # Case 1: First crossover - use original trajectories
            if latest_crossover_round < 0:
                candidates.extend(original_trajs)
                logger.info(f"First crossover (crossover-only mode): using {len(original_trajs)} original trajectories")
            
            # Case 2: Subsequent crossovers - use two most recent crossover rounds
            else:
                # Get unique crossover round indices, sorted descending
                crossover_rounds = sorted(set(t.round_idx for t in crossover_trajs), reverse=True)
                
                if len(crossover_rounds) >= 2:
                    # Use two most recent crossover rounds
                    round1, round2 = crossover_rounds[0], crossover_rounds[1]
                    trajs_round1 = [t for t in crossover_trajs if t.round_idx == round1]
                    trajs_round2 = [t for t in crossover_trajs if t.round_idx == round2]
                    candidates.extend(trajs_round1)
                    candidates.extend(trajs_round2)
                    logger.info(f"Crossover-only mode: using {len(trajs_round1)} from round {round1} + "
                               f"{len(trajs_round2)} from round {round2}")
                else:
                    # Only one crossover round exists, use that + original
                    latest_crossovers = [t for t in crossover_trajs if t.round_idx == latest_crossover_round]
                    candidates.extend(latest_crossovers)
                    candidates.extend(original_trajs)
                    logger.info(f"Crossover-only mode (fallback): using {len(latest_crossovers)} crossover + "
                               f"{len(original_trajs)} original")
            
            return candidates
        
        # ========================================================
        # STANDARD MODE (mutation enabled)
        # ========================================================
        # Case 1: First crossover (no previous crossover exists)
        # Use: original + latest mutation
        if latest_crossover_round < 0:
            candidates.extend(original_trajs)
            if latest_mutation_round >= 0:
                candidates.extend([t for t in mutation_trajs if t.round_idx == latest_mutation_round])
            logger.info(f"First crossover: using {len(original_trajs)} original + "
                       f"{len(candidates) - len(original_trajs)} mutation (round {latest_mutation_round})")
        
        # Case 2: Subsequent crossover
        # Use: latest mutation + latest crossover
        else:
            # Add latest mutation trajectories
            if latest_mutation_round >= 0:
                latest_mutations = [t for t in mutation_trajs if t.round_idx == latest_mutation_round]
                candidates.extend(latest_mutations)
                logger.info(f"Adding {len(latest_mutations)} mutation trajectories from round {latest_mutation_round}")
            
            # Add latest crossover trajectories
            latest_crossovers = [t for t in crossover_trajs if t.round_idx == latest_crossover_round]
            candidates.extend(latest_crossovers)
            logger.info(f"Adding {len(latest_crossovers)} crossover trajectories from round {latest_crossover_round}")
        
        return candidates
    
    def _get_crossover_task(self) -> Optional[dict[str, Any]]:
        """Get next crossover round task."""
        # If crossover is disabled, skip to mutation or stay in crossover loop
        if not self.config.crossover_enabled:
            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                self._current_round += 1
                return self._get_mutation_task()
            return None
        
        # Check if there are remaining crossover groups
        if self._crossover_idx >= len(self._crossover_groups):
            # All crossover tasks complete, transition to next phase
            self._current_round += 1

            # Learning-aware reseed: if breeding has stalled, regenerate informed
            # directions and return to ORIGINAL with new material.
            if self._reseed_if_stale():
                return self._get_original_task()
            
            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All crossover rounds complete, transitioning to mutation (round {self._current_round})")
                return self._get_mutation_task()
            else:
                # Stay in crossover mode (crossover-only loop)
                # Prepare next crossover groups from the two most recent rounds
                self._prepare_crossover_groups()
                logger.info(f"All crossover rounds complete, continuing with crossover (round {self._current_round})")
                return self._get_crossover_task()
        
        # Get next crossover group
        parents = self._crossover_groups[self._crossover_idx]
        
        # Generate crossover guidance
        suffix = self.crossover_op.generate_crossover_prompt_suffix(parents)
        zc = self._build_zoo_context()
        if zc:
            suffix = suffix + "\n" + zc
        
        task = {
            "phase": RoundPhase.CROSSOVER,
            "direction_id": self._crossover_idx,  # Use crossover index as direction
            "parent_trajectories": parents,
            "strategy_suffix": suffix,
            "round_idx": self._current_round,
        }
        
        self._crossover_idx += 1
        return task
    
    def report_task_complete(
        self,
        task: dict[str, Any],
        trajectory: StrategyTrajectory
    ):
        """
        Report that a task has been completed.
        
        Args:
            task: The task that was completed
            trajectory: The resulting trajectory
        """
        # Add trajectory to pool
        self.pool.add(trajectory)
        
        # Update state based on phase
        phase = task["phase"]
        direction_id = task["direction_id"]
        
        if phase == RoundPhase.ORIGINAL:
            self._directions_completed.add(direction_id)
            self._update_direction_status(direction_id, trajectory)
            logger.info(f"Original round complete for direction {direction_id}")

        elif phase == RoundPhase.MUTATION:
            logger.info(f"Mutation round complete for direction {direction_id}")

        elif phase == RoundPhase.CROSSOVER:
            logger.info(f"Crossover round complete (group {direction_id})")

    def _update_direction_status(self, direction_id: int, trajectory: StrategyTrajectory) -> None:
        """Tally one ORIGINAL outcome onto the per-direction status.

        Only ORIGINAL tasks carry a real direction index (mutation/crossover
        use a task index), so this is called from the ORIGINAL branch only.
        ``admitted`` is surfaced by ``_extract_net_cost_metrics`` (Part E); the
        ``feasible`` fallback keeps control-arm trajectories readable, though
        the reseed gate (_has_treatment_data) means this tally only matters for
        the treatment arm.
        """
        if direction_id < 0 or direction_id >= len(self._direction_status):
            return
        st = self._direction_status[direction_id]
        st["attempts"] += 1
        metrics = trajectory.backtest_metrics or {}
        if bool(metrics.get("admitted", metrics.get("feasible", True))):
            st["admitted_count"] += 1
            try:
                st["last_admit_round"] = max(st["last_admit_round"], int(trajectory.round_idx))
            except (TypeError, ValueError):
                pass
        else:
            st["rejected_count"] += 1

    def create_trajectory_from_loop_result(
        self,
        task: dict[str, Any],
        hypothesis: Any,
        experiment: Any,
        feedback: Any
    ) -> StrategyTrajectory:
        """
        Create a trajectory from loop execution results.
        
        Args:
            task: The task that was executed
            hypothesis: The hypothesis object
            experiment: The experiment object (with factors and results)
            feedback: The feedback object
            
        Returns:
            A new StrategyTrajectory
        """
        phase = task["phase"]
        direction_id = task["direction_id"]
        round_idx = task["round_idx"]
        
        # Generate trajectory ID
        traj_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
        
        # Extract hypothesis info
        hypothesis_text = str(hypothesis) if hypothesis else ""
        hypothesis_details = {}
        if hypothesis:
            for attr in ["hypothesis", "reason", "concise_reason", "concise_observation",
                        "concise_justification", "concise_knowledge"]:
                if hasattr(hypothesis, attr):
                    hypothesis_details[attr] = getattr(hypothesis, attr, "")
        
        # Extract factor info
        factors = []
        if experiment and hasattr(experiment, "sub_tasks"):
            for idx, task_obj in enumerate(experiment.sub_tasks):
                factor_info = {
                    "name": getattr(task_obj, "factor_name", f"factor_{idx}"),
                    "expression": getattr(task_obj, "factor_expression", ""),
                    "description": getattr(task_obj, "factor_description", ""),
                }
                # Try to get code
                if (hasattr(experiment, "sub_workspace_list") and 
                    idx < len(experiment.sub_workspace_list)):
                    ws = experiment.sub_workspace_list[idx]
                    if ws and hasattr(ws, "code_dict") and ws.code_dict:
                        factor_info["code"] = ws.code_dict.get("factor.py", "")
                factors.append(factor_info)
        
        # Extract backtest metrics
        backtest_metrics = {}
        backtest_result = getattr(experiment, "result", None) if experiment else None
        if backtest_result is not None:
            backtest_metrics = self._extract_metrics(backtest_result)
        
        # Extract feedback info
        feedback_text = str(feedback) if feedback else ""
        feedback_details = {}
        if feedback:
            for attr in ["observations", "hypothesis_evaluation", "new_hypothesis", 
                        "reason", "decision"]:
                if hasattr(feedback, attr):
                    feedback_details[attr] = getattr(feedback, attr, "")
        
        # Get parent IDs
        parent_ids = [p.trajectory_id for p in task.get("parent_trajectories", [])]
        
        return StrategyTrajectory(
            trajectory_id=traj_id,
            direction_id=direction_id,
            round_idx=round_idx,
            phase=phase,
            hypothesis=hypothesis_text,
            hypothesis_details=hypothesis_details,
            factors=factors,
            backtest_result=backtest_result,
            backtest_metrics=backtest_metrics,
            feedback=feedback_text,
            feedback_details=feedback_details,
            parent_ids=parent_ids,
        )
    
    def _extract_metrics(self, result: Any) -> dict[str, Optional[float]]:
        """Extract metrics from backtest result."""
        import pandas as pd
        
        metrics = {
            "IC": None,
            "ICIR": None,
            "RankIC": None,
            "RankICIR": None,
            "annualized_return": None,
            "information_ratio": None,
            "max_drawdown": None
        }
        
        if result is None:
            return metrics
        
        try:
            index_mapping = {
                'IC': ['IC', 'ic'],
                'ICIR': ['ICIR', 'icir'],
                'RankIC': ['RankIC', 'Rank IC', 'rank_ic'],
                'RankICIR': ['RankICIR', 'Rank ICIR', 'rank_icir'],
                'annualized_return': [
                    '1day.excess_return_with_cost.annualized_return',
                    '1day.excess_return_without_cost.annualized_return',
                    'annualized_return',
                    'Annualized Return'
                ],
                'information_ratio': [
                    '1day.excess_return_with_cost.information_ratio',
                    '1day.excess_return_without_cost.information_ratio',
                    'information_ratio',
                    'Information Ratio'
                ],
                'max_drawdown': [
                    '1day.excess_return_with_cost.max_drawdown',
                    '1day.excess_return_without_cost.max_drawdown',
                    'max_drawdown',
                    'Max Drawdown'
                ],
            }
            
            if isinstance(result, pd.DataFrame):
                col = result.columns[0] if len(result.columns) > 0 else 0
                for target, names in index_mapping.items():
                    for name in names:
                        if name in result.index:
                            val = result.loc[name, col] if col in result.columns else result.loc[name]
                            if pd.notna(val):
                                metrics[target] = float(val)
                                break
            
            elif isinstance(result, pd.Series):
                for target, names in index_mapping.items():
                    for name in names:
                        if name in result.index:
                            val = result[name]
                            if pd.notna(val):
                                metrics[target] = float(val)
                                break

            # Net-of-cost engine extras (E_theta / NetCostFactorRunner). Purely
            # additive: the control arm's Qlib result carries none of these
            # keys, so its behaviour is unchanged.
            metrics.update(self._extract_net_cost_metrics(result))
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")

        return metrics

    # Keys emitted by NetCostFactorRunner._to_series, by coercion type.
    _NET_COST_FLOAT_KEYS = (
        "U", "rho_max", "turnover_book", "turnover_solo", "cx", "cost_bps", "zoo_size",
        # Marginal contribution -- the fitness that anchors selection to the
        # book rather than to a percentile of a winners-only sample. delta_mean
        # / delta_se are the seed-averaged estimate and its standard error that
        # _rank_by_fitness shrinks; absent outside marginal_contribution mode.
        "delta_net_ir", "delta_net_arr", "base_net_ir",
        "delta_mean", "delta_se",
        # Realized per-factor outcome + the admission bar that judged it.
        # Surfaced so the evolution prompts (format_objective_note) and the
        # reseed digest can read the verdict off the trajectory instead of
        # falling back to ``feasible`` (which the net-cost runner never sets,
        # so rejected batches used to render as ADMITTED).
        "net_ir", "net_arr", "tau_admit",
        "e_effectiveness", "e_arr", "e_stability", "e_turnover", "e_diversity",
        "e_overfit", "e_decay",
    )
    _NET_COST_STR_KEYS = ("theta_hash", "zoo_hash", "failed_gates", "weakest_dimensions")
    # Bool flags from the net-cost runner. ``admitted`` is the live verdict the
    # digest and format_objective_note key on; ``in_zoo`` is its persistence
    # counterpart. The control arm's Qlib Series carries neither, so these stay
    # absent and its trajectories read exactly as before.
    _NET_COST_BOOL_KEYS = ("admitted", "in_zoo")

    def _extract_net_cost_metrics(self, result: Any) -> dict[str, Any]:
        """Pull the E_theta metric vector off a result, when present."""
        import pandas as pd

        if isinstance(result, pd.DataFrame):
            if result.empty or len(result.columns) == 0:
                return {}
            series = result.iloc[:, 0]
        elif isinstance(result, pd.Series):
            series = result
        else:
            return {}

        extras: dict[str, Any] = {}
        for key in self._NET_COST_FLOAT_KEYS:
            if key in series.index:
                val = series[key]
                if pd.notna(val):
                    try:
                        extras[key] = float(val)
                    except (TypeError, ValueError):
                        pass
        for key in self._NET_COST_STR_KEYS:
            if key in series.index and pd.notna(series[key]):
                extras[key] = str(series[key])
        if "feasible" in series.index and pd.notna(series["feasible"]):
            extras["feasible"] = bool(series["feasible"])
        for key in self._NET_COST_BOOL_KEYS:
            if key in series.index and pd.notna(series[key]):
                extras[key] = bool(series[key])
        return extras
    
    # ------------------------------------------------------------------
    # Learning-aware reseed: when breeding stops growing the repository,
    # regenerate NEW directions informed by the run's trial history.
    # ------------------------------------------------------------------

    def _has_treatment_data(self) -> bool:
        """True when any trajectory carries the net-of-cost objective vector.

        The A/B gate for every reseed behaviour. The control arm (RankIC, no
        ledger) never produces a ``U`` metric, so this returns False and the
        whole reseed path is a no-op, keeping the arms comparable. Gating on
        data presence rather than an env var is what makes the control arm
        byte-for-byte unchanged.
        """
        for t in self.pool.get_all():
            if "U" in (t.backtest_metrics or {}):
                return True
        return False

    def _zoo_size(self) -> int | None:
        """How many factors the repository holds, or None if unknowable here.

        The same source ``_rounds_exhausted`` sizes the target from, so the two
        cannot disagree about whether progress is being made. A missing ledger
        yields 0 (``replay_repository`` returns {} rather than raising), which is
        why ``_has_treatment_data`` is checked first: a control arm with an empty
        default ledger path must not read as "stuck at 0" and trigger a reseed.
        """
        try:
            from quantaalpha.eval.ledger import replay_repository

            return len(replay_repository(os.environ.get("QA_LEDGER")))
        except Exception:
            return None

    def _reseed_if_stale(self) -> bool:
        """Regenerate informed directions when the repository stops growing.

        Returns True when the phase was reset to ORIGINAL, so the caller returns
        an original task instead of its own transition. Keyed on repository
        GROWTH rather than on rejections: a round can legitimately reject
        everything while the repository is still climbing, and reseeding then
        would discard parents that are working. Silent no-op for the control arm
        (no ``U``) and when ``reseed_after_stale_rounds`` <= 0.
        """
        if not self._has_treatment_data():
            return False
        n = int(getattr(self.config, "reseed_after_stale_rounds", 0) or 0)
        if n <= 0:
            return False
        size = self._zoo_size()
        if size is None:
            return False
        if size > self._best_zoo_size:
            self._best_zoo_size = size
            self._stale_rounds = 0
            return False
        self._stale_rounds += 1
        if self._stale_rounds < n:
            return False
        # Stalled for n rounds: build the digest and ask for new directions.
        self._stale_rounds = 0
        digest = self._build_reseed_digest()
        if not digest:
            logger.warning(
                "Repository stuck but no digestible trial history; skipping reseed"
            )
            return False
        new_dirs = self._generate_informed_directions(digest)
        if not new_dirs:
            logger.warning(
                f"Repository stuck at {size} factor(s) for {n} round(s): "
                "informed-direction generation returned nothing; retrying next stale window"
            )
            return False
        # Mark saturated directions: explored and no admission within the last n
        # rounds. A direction that admitted recently keeps its headroom and
        # stays eligible; a never-admitted or long-dormant one is skipped so the
        # ORIGINAL phase spends its budget on the new directions instead.
        for st in self._direction_status:
            recent = (
                st["last_admit_round"] >= 0
                and (self._current_round - st["last_admit_round"]) < n
            )
            st["saturated"] = bool(st["attempts"] >= 1 and not recent)
        # Grow the direction list (never replace -- working parents stay).
        self._reseed_count += 1
        src = f"reseed_{self._reseed_count}"
        for d in new_dirs:
            self._directions.append(d)
            self._direction_status.append(
                self._blank_direction_status(len(self._directions) - 1, src)
            )
        # Re-open eligible directions: keep saturated ids completed (skip them),
        # drop non-saturated ones so they get another ORIGINAL pass; the new ids
        # were never completed.
        self._directions_completed = {
            d for d in self._directions_completed
            if d < len(self._direction_status) and self._direction_status[d]["saturated"]
        }
        self._current_phase = RoundPhase.ORIGINAL
        logger.warning(
            f"Repository stuck at {size} factor(s) for {n} round(s): generated "
            f"{len(new_dirs)} informed direction(s) (reseed #{self._reseed_count}); "
            "returning to ORIGINAL with new material. Mutation and crossover can "
            "only recombine what already exists, so a stalled search needs new "
            "directions, not more breeding."
        )
        return True

    def _build_reseed_digest(self) -> str:
        """Per-direction outcome summary fed to the informed-direction LLM.

        Groups ORIGINAL trajectories by ``direction_id`` (a real index only
        there), mapped to their direction string via ``self._directions``. For
        each direction: the verdict tally (admitted vs rejected, binned
        redundant / net-harmful / marginal from ``failed_gates`` + ``delta_mean``
        -- the reason string lives only in the ledger, not on the trajectory),
        the top admitted factors' signatures, last-admit round, and a SATURATED
        flag. Returns "" when there is no digestible history.
        """
        if not self._directions:
            return ""
        by_dir: dict[int, list[StrategyTrajectory]] = {}
        for t in self.pool.get_by_phase(RoundPhase.ORIGINAL):
            did = getattr(t, "direction_id", None)
            if isinstance(did, int) and 0 <= did < len(self._directions):
                by_dir.setdefault(did, []).append(t)
        if not by_dir:
            return ""

        lines: list[str] = []
        for did in sorted(by_dir):
            trajs = by_dir[did]
            st = self._direction_status[did]
            direction_text = (self._directions[did] or "")[:200]
            admitted = [
                t for t in trajs
                if bool((t.backtest_metrics or {}).get(
                    "admitted", (t.backtest_metrics or {}).get("feasible", True)))
            ]
            redundant = net_harmful = marginal = 0
            for t in trajs:
                if t in admitted:
                    continue
                m = t.backtest_metrics or {}
                fg = str(m.get("failed_gates") or "")
                dm = m.get("delta_mean")
                if "rho_max" in fg:
                    redundant += 1
                elif isinstance(dm, (int, float)) and dm < 0:
                    net_harmful += 1
                else:
                    marginal += 1
            parts = [
                f'- Direction {did} ("{direction_text}"):',
                f"  attempts={st['attempts']} admitted={st['admitted_count']} "
                f"rejected={st['rejected_count']} "
                f"(redundant={redundant}, net_harmful={net_harmful}, marginal={marginal})",
                f"  last_admit_round={st['last_admit_round']}",
            ]
            for t in admitted[:3]:
                m = t.backtest_metrics or {}
                exprs = [f.get("expression", "")[:80] for f in (t.factors or [])[:2]]
                parts.append(
                    f"  admitted: [{' | '.join(exprs)}] U={format_metric(m.get('U'))} "
                    f"delta_mean={format_metric(m.get('delta_mean'))} "
                    f"rho_max={format_metric(m.get('rho_max'))}"
                )
            parts.append(f"  SATURATED={'yes' if st['saturated'] else 'no'}")
            lines.append("\n".join(parts))
        return "\n".join(lines)

    def _generate_informed_directions(self, digest: str) -> list[str]:
        """Ask the LLM for new orthogonal directions from the digest."""
        prompt_path = getattr(self.config, "informed_prompt_path", None)
        if not prompt_path or not Path(prompt_path).exists():
            logger.warning("No informed-planning prompt path configured; cannot reseed")
            return []
        from quantaalpha.pipeline.planning import generate_informed_directions

        n = max(1, int(getattr(self.config, "num_directions", 2) or 2))
        try:
            return generate_informed_directions(
                initial_direction=getattr(self.config, "initial_direction", "") or "",
                n=n,
                prompt_file=Path(prompt_path),
                history_summary=digest,
                use_llm=True,
                allow_fallback=False,
            )
        except Exception as exc:
            logger.warning(f"Informed direction generation failed: {exc}")
            return []

    def _build_zoo_context(self) -> str:
        """Cumulative repository summary appended to mutation/crossover guidance.

        Tells the breeder what the book already captures so mutations and
        crossovers aim at ORTHOGONAL signal rather than re-saturating the same
        space. Empty for the control arm (no ``U``), so its ``strategy_suffix``
        and the resulting ``effective_direction`` (loop.py) are byte-identical to
        today. Capped at 8 admitted trajectories to bound the token cost.
        """
        if not self._has_treatment_data():
            return ""
        admitted = [
            t for t in self.pool.get_all()
            if bool((t.backtest_metrics or {}).get(
                "admitted", (t.backtest_metrics or {}).get("feasible", True)))
        ][:8]
        if not admitted:
            return ""
        lines = ["## Repository Context (what the book already captures)"]
        for t in admitted:
            m = t.backtest_metrics or {}
            exprs = [f.get("expression", "")[:60] for f in (t.factors or [])[:2]]
            lines.append(
                f"- [{' | '.join(exprs)}] U={format_metric(m.get('U'))} "
                f"rho_max={format_metric(m.get('rho_max'))}"
            )
        lines.append(
            "Aim for ORTHOGONAL signal not already in the book; do not duplicate "
            "the admitted factor logic above."
        )
        return "\n".join(lines)

    def _rounds_exhausted(self) -> bool:
        """Should evolution stop?

        Normally ``max_rounds`` decides. But when the treatment arm gates
        admission on ``U``, a fixed round count produces a repository whose size
        depends on how many batches happened to clear the bar -- and factor
        count moves the combiner independently of factor quality, so an arm that
        mines the same amount but admits less is handicapped for a reason that
        has nothing to do with its objective.

        ``QA_TARGET_ZOO`` makes the budget follow the outcome instead of an
        estimate: keep mining until the repository actually holds that many
        admitted factors. A guessed multiplier cannot do this -- the admission
        rate is not known before the run and drifts as the repository improves,
        which is the whole point of an adaptive bar.

        ``QA_MAX_ROUNDS_CAP`` bounds the cost. Hitting it means the arm could
        not reach the target, which is itself a result and is logged as such
        rather than being retried forever.
        """
        if self._current_round < self.config.max_rounds:
            return False

        target = os.environ.get("QA_TARGET_ZOO")
        if not target:
            return True
        try:
            target_n = int(target)
        except ValueError:
            logger.warning(f"QA_TARGET_ZOO={target!r} is not an integer; ignoring")
            return True

        cap = int(os.environ.get("QA_MAX_ROUNDS_CAP", self.config.max_rounds * 3))
        try:
            from quantaalpha.eval.ledger import replay_repository

            admitted = len(replay_repository(os.environ.get("QA_LEDGER")))
        except Exception as exc:
            logger.warning(f"cannot size the repository ({exc}); stopping at max_rounds")
            return True

        if admitted >= target_n:
            logger.info(
                f"Evolution complete: repository holds {admitted} admitted factor(s) "
                f">= QA_TARGET_ZOO={target_n} after round {self._current_round}"
            )
            return True
        if self._current_round >= cap:
            logger.warning(
                f"Evolution stopping at the cap ({cap} rounds) with {admitted}/{target_n} "
                f"admitted factors. The arm could not reach the target: its admission "
                f"rate is too low for this budget, which is a result, not a retry condition."
            )
            return True
        logger.info(
            f"Extending evolution past max_rounds: {admitted}/{target_n} admitted "
            f"factor(s) after round {self._current_round} (cap {cap})"
        )
        return False

    def is_complete(self) -> bool:
        """Check if evolution is complete."""
        return self._rounds_exhausted()
    
    def get_best_trajectories(self, top_n: int = 5) -> list[StrategyTrajectory]:
        """Get the best performing trajectories."""
        all_trajs = self.pool.get_all()
        
        # Filter to successful trajectories
        valid = [t for t in all_trajs if t.is_successful()]
        
        # Sort by primary metric
        valid.sort(key=lambda t: t.get_primary_metric() or 0, reverse=True)
        
        return valid[:top_n]
    
    def save_state(self, path: Path):
        """Save controller state to disk."""
        import json
        
        state = {
            "current_round": self._current_round,
            "current_phase": self._current_phase.value,
            "directions_completed": list(self._directions_completed),
            "crossover_idx": self._crossover_idx,
            "active_branch_count": self._active_branch_count,
            "mutation_idx": self._mutation_idx,
            "mutation_target_ids": [t.trajectory_id for t in self._mutation_targets],
            # Learning-aware reseed state: the grown direction list and the
            # per-direction outcome tally must survive a restart or a resumed
            # run would forget it had already saturated those directions.
            "directions": list(self._directions),
            "direction_status": list(self._direction_status),
            "best_zoo_size": self._best_zoo_size,
            "stale_rounds": self._stale_rounds,
            "config": {
                "num_directions": self.config.num_directions,
                "max_rounds": self.config.max_rounds,
                "mutation_enabled": self.config.mutation_enabled,
                "crossover_enabled": self.config.crossover_enabled,
                "crossover_size": self.config.crossover_size,
                "crossover_n": self.config.crossover_n,
            }
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved evolution state to {path}")
    
    def load_state(self, path: Path):
        """Load controller state from disk."""
        import json
        
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        self._current_round = state.get("current_round", 0)
        self._current_phase = RoundPhase(state.get("current_phase", "original"))
        self._directions_completed = set(state.get("directions_completed", []))
        self._crossover_idx = state.get("crossover_idx", 0)
        self._active_branch_count = state.get("active_branch_count", self.config.num_directions)
        self._mutation_idx = state.get("mutation_idx", 0)

        # Restore learning-aware reseed state. Fall back to the config-seeded
        # values for state files written before this feature existed.
        saved_dirs = state.get("directions")
        if isinstance(saved_dirs, list) and saved_dirs:
            self._directions = list(saved_dirs)
        saved_status = state.get("direction_status")
        if isinstance(saved_status, list) and len(saved_status) == len(self._directions):
            self._direction_status = list(saved_status)
        else:
            self._direction_status = [
                self._blank_direction_status(i, "initial")
                for i in range(len(self._directions))
            ]
        self._best_zoo_size = int(state.get("best_zoo_size", -1))
        self._stale_rounds = int(state.get("stale_rounds", 0))

        # Restore mutation targets from IDs
        mutation_target_ids = state.get("mutation_target_ids", [])
        self._mutation_targets = []
        for tid in mutation_target_ids:
            traj = self.pool.get(tid)
            if traj:
                self._mutation_targets.append(traj)
        
        # Re-prepare crossover groups if in crossover phase
        if self._current_phase == RoundPhase.CROSSOVER:
            self._prepare_crossover_groups()
        
        logger.info(f"Loaded evolution state from {path}")

