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
from .refine import RefinementOperator


# T6 diagnosability-based mutation routing. Each prev-phase parent is bucketed
# by what its evaluation verdict SAYS to do with it, not by where it ranks on
# fitness (the old rank+tail-cut routing bred the best parents and restarted the
# weakest -- backwards: the failures need refining, the winners go to crossover).
# See ``_prepare_mutation_targets`` / ``_build_mutation_task``.
_BUCKET_REFINE = "refine"            # rejected + a usable verdict -> fix why it failed
_BUCKET_ORTHOGONAL = "orthogonal"    # NO_DATA / no usable verdict / exhausted lever
_BUCKET_ADMITTED_PUSH = "admitted_push"  # a gated admitted winner -> push the edge


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


# How many prior attempts the diagnosis is shown. Enough to reveal a repeated
# failure, few enough to leave room for the parent's own detail.
_POPULATION_SAMPLE = 8


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

    # VESTIGIAL under the T6 diagnosability-based bucket routing (see
    # ``_prepare_mutation_targets``): selection now routes each parent to
    # REFINE / ORTHOGONAL / ADMITTED-PUSH by its verdict + diagnosability, not by
    # its fitness-rank position, so there is no "weakest tail" to cut. The field
    # is kept for config-file compatibility and no longer drives routing.
    orthogonal_tail_fraction: float = 0.25

    # T6 ADMITTED-PUSH: a config-gated fraction of ADMITTED winners that ALSO get
    # a "push the edge further" refine task (local-search exploitation of a
    # working factor), instead of going to crossover only. 0.0 (default) = a
    # clean split -- admitted winners are crossover-only, never mutated; the
    # mutation budget goes entirely to refining rejected diagnosable parents.
    # >0 fills the lowest-priority mutation slots with push-further refines
    # (REFINE > ORTHOGONAL > ADMITTED-PUSH), so it never crowds out a
    # failure-refine. NOT a Theta field.
    admitted_push_fraction: float = 0.0
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

    # Learning-aware reseed: regenerate NEW, outcome-informed directions when the
    # repository grows too slowly. NOT a Theta field -- the frozen protocol hash
    # must not move.
    # Rounds of INSUFFICIENT growth (below ``growth_floor`` admissions) before a
    # stale reseed fires. 1 fires on the first slow round; the previous 2 let a
    # creeping zoo (an occasional admission) defer reseed indefinitely while the
    # search inbred the same directions and the round cap gave up short of 150.
    reseed_after_stale_rounds: int = 1
    # A round whose admissions < growth_floor counts as stale (does not reset the
    # counter). 2 means a 1-admission creep is still "stale"; only a burst of
    # >= 2 resets. Set >= 1; a creep slower than this triggers reseed instead of
    # inbreeding.
    growth_floor: int = 2
    # Scheduled immigration: every ``reseed_interval`` rounds, inject fresh
    # informed directions REGARDLESS of growth -- steady anti-inbreed that does
    # not wait for stagnation. 0 disables.
    reseed_interval: int = 3
    # The controller owns the live direction list so it can GROW it on a reseed.
    # Seeded from the initial planning output; ``num_directions`` is the frozen
    # initial count (also the reseed batch size).
    directions: list[str] = field(default_factory=list)
    initial_direction: str = ""
    informed_prompt_path: Optional[str] = None
    # Whether the Alpha158(20) seed library is injected into direction-planning.
    # Mirrors ``planning_cfg.seed_in_generation`` so the RESEED path
    # (``_generate_informed_directions`` -> ``generate_informed_directions``) honors
    # the same flag the round-0 path (``generate_parallel_directions``) already
    # reads from factor_mining. Without this the reseed would default to
    # seed_in_generation=True and keep injecting the OHLCV seeds after the
    # round-0 directions were de-primed -- half a de-prime. NOT a Theta field.
    seed_in_generation: bool = True


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
        # The refine prompts live in the same evolution_prompts.yaml (``refine:``
        # section), so the operator reuses the mutation prompt path.
        self.refine_op = RefinementOperator(prompt_path=mutation_path)
        
        # State tracking
        self._current_round = 0
        self._current_phase = RoundPhase.ORIGINAL
        self._directions_completed = set()  # Track which directions completed original
        self._crossover_groups: list[list[StrategyTrajectory]] = []  # Current crossover groups
        self._crossover_idx = 0  # Which crossover group is next
        
        # Track active branch count (changes after crossover)
        self._active_branch_count = config.num_directions
        # Track trajectories to mutate in current mutation round. ``_mutation_buckets``
        # is the parallel per-target routing decision (T6): REFINE / ORTHOGONAL /
        # ADMITTED-PUSH, set by ``_prepare_mutation_targets`` from each parent's
        # verdict + diagnosability (NOT its fitness rank) and read by
        # ``_build_mutation_task`` so the serial and parallel paths route identically.
        self._mutation_targets: list[StrategyTrajectory] = []
        self._mutation_buckets: list[str] = []
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
        # Zoo size at the previous reseed check, so growth is measured per round
        # (admissions since the last check) rather than vs an all-time high -- a
        # creeping zoo must still register as stale.
        self._zoo_size_at_last_check = -1

        # RESUME. The pool survives a restart (fresh_start=False) but this
        # state block does not: without the restore below a resumed run rebuilds
        # the zoo from the ledger and then re-runs ORIGINAL round 0 for every
        # direction, re-mining work that is already in the pool.
        if not config.fresh_start and self.pool._trajectories:
            self._restore_progress()

    def _restore_progress(self) -> None:
        """Rebuild round / phase / completed-direction state from the pool.

        Only three things have to be recovered for the loop to continue where it
        stopped:

        * ``_current_round`` -- the highest round any trajectory reached. The
          loop increments it on each phase transition, so resuming a round lower
          than the pool's maximum would re-do rounds already recorded.
        * ``_directions_completed`` -- every direction that already has an
          ORIGINAL trajectory. ``_get_original_task`` hands out the first
          direction NOT in this set, so an empty set means all 10 originals are
          re-mined before anything else runs.
        * ``_current_phase`` -- the phase of the most recent trajectory. The
          transition helpers advance from here, so starting at ORIGINAL after a
          crossover round would walk the whole cycle again.

        Everything else (crossover groups, mutation targets) is derived per
        round from the pool, so it rebuilds itself on the next call.
        """
        trajs = list(self.pool._trajectories.values())
        if not trajs:
            return

        max_round = max((getattr(t, "round_idx", 0) or 0) for t in trajs)

        completed = set()
        for t in trajs:
            ph = getattr(t, "phase", None)
            if ph is not None and getattr(ph, "value", ph) == RoundPhase.ORIGINAL.value:
                completed.add(getattr(t, "direction_id", None))
        completed.discard(None)

        # Phase of the LATEST trajectory: created_at is an ISO string, so a
        # plain max() orders correctly; fall back to round_idx when absent.
        def _key(t):
            return (getattr(t, "round_idx", 0) or 0, str(getattr(t, "created_at", "")))
        latest = max(trajs, key=_key)
        ph = getattr(latest, "phase", None)
        phase = RoundPhase.ORIGINAL
        if ph is not None:
            try:
                phase = ph if isinstance(ph, RoundPhase) else RoundPhase(getattr(ph, "value", ph))
            except ValueError:
                phase = RoundPhase.ORIGINAL

        self._current_round = int(max_round)
        self._directions_completed = completed
        self._current_phase = phase
        # RDAgentLog takes ONE message argument -- printf-style args raise.
        n_dirs = len(self._directions) or self.config.num_directions
        logger.info(
            f"RESUME: restored from {len(trajs)} pooled trajectories -- "
            f"round {self._current_round}, phase {self._current_phase.value}, "
            f"{len(completed)}/{n_dirs} directions already have an ORIGINAL trajectory"
        )

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
                        # ORIGINAL rounds are where reseed re-enters
                        # exploration, and they were the ONLY phase that got no
                        # memory at all ("No guidance for original"). Measured
                        # consequence: the round-6 reseed produced 0 back-
                        # references to any prior round -- it explored genuinely
                        # new ground (48/51 novel hypotheses, six operators never
                        # used before) and re-derived it from nothing, then
                        # admitted nothing. Exploration does not require amnesia;
                        # what has already been measured and failed is exactly
                        # what a fresh direction should not re-tread.
                        "strategy_suffix": self._build_zoo_context(),
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

                # T6: route by the bucket decided in _prepare_mutation_targets
                # (shared with _get_mutation_task so the serial and parallel paths
                # cannot diverge on the refine/orthogonal/push decision).
                bucket = (self._mutation_buckets[idx]
                          if idx < len(self._mutation_buckets)
                          else _BUCKET_REFINE)
                tasks.append(self._build_mutation_task(parent, idx, bucket))

            # If no tasks, transition phase for next call
            if not tasks:
                self._mutation_targets = []
                self._mutation_buckets = []
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
                # T6: compute each parent's strength directive ONCE (cached) so
                # the suffix and build_task_extras share one diagnosis.
                strengths = [self._cached_strength(p) for p in parents]
                suffix = self.crossover_op.generate_crossover_prompt_suffix(parents, strengths)
                zc = self._build_zoo_context()
                if zc:
                    suffix = suffix + "\n" + zc
                # Eq. 7: ship the two parents' validated strengths as constructor
                # inspiration (not a literal splice). None when fewer than 2
                # parents -> unchanged LLM-only crossover.
                _cx_extras = self.crossover_op.build_task_extras(parents, strengths) or {}
                tasks.append({
                    "phase": RoundPhase.CROSSOVER,
                    "direction_id": idx,
                    "parent_trajectories": parents,
                    "strategy_suffix": suffix,
                    "round_idx": self._current_round,
                    **_cx_extras,
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
            self._mutation_buckets = []
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
        # Budget guard. The three phase getters call each other on every
        # transition (original->mutation->crossover->mutation->...), and
        # ``get_next_task``'s ``_rounds_exhausted`` check is only on the OUTER
        # entry -- so once a transition chain starts, nothing re-checks it.
        # Measured: a resumed controller whose pool has no minable parents
        # recursed to round 1116+ with max_rounds=15 and QA_MAX_ROUNDS_CAP=60
        # both set, i.e. the budget was bypassed entirely. Each getter now
        # re-checks before doing any work.
        if self._rounds_exhausted():
            return None
        # Find a direction that hasn't completed original
        for d in range(len(self._directions)):
            if d not in self._directions_completed:
                return {
                    "phase": RoundPhase.ORIGINAL,
                    "direction_id": d,
                    "direction": self._directions[d],
                    "parent_trajectories": [],
                    # See the note in get_all_tasks_for_current_phase: ORIGINAL
                    # was the only phase generated with no memory of what had
                    # already been measured and failed.
                    "strategy_suffix": self._build_zoo_context(),
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
    
    def _get_ablation_eval(self):
        """Lazy per-segment ablation closure, env-gated (``QA_ABLATION_DIAGNOSIS=1``).

        Returns a callable ``parent -> SegmentAblation | None`` that runs the
        per-segment solo measurement of the parent's expression (the AlphaEvolve
        "which sub-tree is broken" signal), or ``None`` when the flag is OFF (the
        default). With the flag off, ``build_task_extras`` gets ``ablation_eval=None``
        and the diagnosis is byte-identical to the pre-ablation path -- this is the
        no-regression guarantee. Turning the ablation on for a relaunch is one flag.

        The heavy ``CustomFactorCalculator`` (the ~163 MB/col qlib load) and the
        ``EvaluationOperator`` panel are built ONCE and cached on
        ``self._ablation_eval_cache``; the closure's ``eval_signal``/``score`` are
        the metrics.py primitives wired exactly as the controller's own
        coverage-probe precedent (``EvaluationOperator(theta)`` + ``_windows`` +
        ``_panel`` + ``label_frame`` + ``_cross_sectional_corr(_slice(s, win), ...)``).
        An eval error on any parent returns ``None`` so an ablation failure never
        blocks a diagnosis (the refine.py seam catches it too, defence in depth).

        Wired into the single per-parent diagnosis (``_cached_diagnosis``) that
        BOTH the bucket classification (``_prepare_mutation_targets``) and the task
        build (``_build_mutation_task`` -> ``build_task_extras``) reuse, so the
        ablation runs ONCE per parent per round and the bucket decision and the
        built task see the same directive (and the same ablation_summary). The
        prior design ran the ablation only in the build and a second, ablation-less
        diagnosis in the classification; the two LLM calls disagreed (the
        classification saw a different prompt and the LLM path is not deterministic
        across calls), so a REFINE bucket silently fell back to an orthogonal
        restart. Caching one directive fixes both the disagreement and the cost.
        """
        if os.environ.get("QA_ABLATION_DIAGNOSIS", "0").strip().lower() not in (
                "1", "true", "yes", "on"):
            return None
        if getattr(self, "_ablation_eval_cache", None) is not None:
            return self._ablation_eval_cache

        import numpy as np
        import pandas as pd
        from quantaalpha.eval.data import align_signal
        from quantaalpha.eval.metrics import (
            _cross_sectional_corr, _slice, label_frame, newey_west_t,
        )
        from quantaalpha.eval.operator import EvaluationOperator
        from quantaalpha.eval.protocol import default_protocol_path, load_protocol
        from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator
        from quantaalpha.pipeline.evolution.segment_ablation import ablate

        theta = load_protocol(os.environ.get("QA_PROTOCOL") or default_protocol_path())
        op = EvaluationOperator(theta)
        start, end, win = op._windows(False)
        panel = op._panel(start, end)
        label = label_frame(panel, theta)
        # The factor-expression renderer (CustomFactorCalculator) needs the LONG
        # qlib format -- a MultiIndex (datetime, instrument) frame with $close /
        # $volume / ... columns -- but ``op._panel`` returns a PanelBundle of WIDE
        # (dates x instruments) per-field frames. A bare ``CustomFactorCalculator()``
        # has neither data_df nor config, so ``calculate_factor`` raised "No stock
        # data provided and no config for loading" on every sub-expression and the
        # ablation silently produced all-NaN (the 30x "Factor computation failed
        # [abl]" in the 0823 smoke). Build the long frame once from the bundle's
        # already-universe-masked fields and hand it to the calculator; the panel
        # is the eval window's, so the solo metrics match the gate's window.
        _fields = (("$open", panel.open), ("$high", panel.high),
                   ("$low", panel.low), ("$close", panel.close),
                   ("$volume", panel.volume), ("$amount", panel.amount),
                   ("$vwap", panel.vwap))
        long_df = pd.concat({n: f.stack() for n, f in _fields}, axis=1)
        long_df.index.names = ["datetime", "instrument"]
        calc = CustomFactorCalculator(data_df=long_df, auto_extract_cache=False)
        _nan = float("nan")
        _empty = pd.Series(dtype=float)

        def eval_signal(sub_expr: str):
            """Compute + align a sub-expression to the panel (opaque handle)."""
            sig = calc.calculate_factor("abl", sub_expr)
            if sig is None or (hasattr(sig, "empty") and sig.empty):
                return None
            return align_signal(sig, panel)

        def _rank_turnover(handle) -> float:
            """Construction-agnostic solo turnover (no book, no combiner).

            The mean absolute day-to-day change in the signal's cross-sectional
            rank, counted only across adjacent days BOTH in-universe (so
            name entry/exit does not inject spurious jumps). This is the turnover
            a long-short book on the sub-tree's signal would incur, independent of
            the portfolio construction -- which is what the ablation needs to
            attribute cost to a sub-tree. The prior ``solo_turnover()`` built a
            ``topk_dropout`` book and raised ``NotImplementedError`` for the
            ``mean_variance`` construction (Defect C, unmasked once Fix A made the
            calculator actually render). Scale is a per-name per-day rank fraction
            (0..1); only the RELATIVE turnover across sub-trees / windows is
            load-bearing for routing, so the absolute scale is irrelevant here.
            """
            if handle is None:
                return _nan
            s = _slice(handle, win)
            if s.empty:
                return _nan
            uni = _slice(panel.universe, win).reindex(
                index=s.index, columns=s.columns).fillna(False)
            ranks = s.where(uni).rank(axis=1, pct=True)
            d = np.abs(np.diff(ranks.values, axis=0))
            both = uni.values[1:] & uni.values[:-1]
            if not both.any():
                return _nan
            return float(np.nanmean(np.where(both, d, np.nan)))

        def score(handle) -> dict:
            """SOLO metrics for one handle: IC and cost SEPARATELY (no net_ir)."""
            if handle is None:
                return {"rank_ic": _nan, "t_nw": _nan, "ic_pos_frac": _nan,
                        "monotonicity": _nan, "turnover_solo": _nan, "ric_series": _empty}
            ric = _cross_sectional_corr(_slice(handle, win), label, "spearman").dropna()
            if ric.empty:
                return {"rank_ic": _nan, "t_nw": _nan, "ic_pos_frac": _nan,
                        "monotonicity": _nan, "turnover_solo": _nan, "ric_series": _empty}
            return {
                "rank_ic": float(ric.mean()),
                "t_nw": float(newey_west_t(ric)),
                "ic_pos_frac": float((ric > 0).mean()),
                # monotonicity is not load-bearing for routing (the IC sign +
                # t_nw + turnover carry the diagnosis); left NaN to avoid the
                # extra quantile_metrics cost on every sub-tree + window variant.
                "monotonicity": _nan,
                "turnover_solo": _rank_turnover(handle),
                "ric_series": ric,
            }

        def ablation_eval(parent):
            factors = getattr(parent, "factors", None) or []
            if not factors:
                return None
            expr = (factors[0] or {}).get("expression", "") or ""
            if not expr:
                return None
            sign = str(getattr(parent, "expected_ic_sign", "") or "").strip().lower()
            try:
                return ablate(expr, sign, eval_signal=eval_signal, score=score)
            except Exception as e:
                logger.warning(f"ablation_eval failed for parent "
                               f"{getattr(parent, 'trajectory_id', '?')} ({e})")
                return None

        self._ablation_eval_cache = ablation_eval
        return ablation_eval

    def _cached_diagnosis(self, parent: StrategyTrajectory):
        """One ``diagnose_parent`` per parent per round, cached + reused.

        The bucket classification (``_prepare_mutation_targets``) and the task
        build (``_build_mutation_task`` -> ``build_task_extras``) used to call
        ``diagnose_parent`` separately: the classification WITHOUT
        ``ablation_eval``, the build WITH it. The LLM diagnosis path runs at a
        non-zero temperature with no chat cache, so two separate calls on the same
        parent can return different structured verdicts, AND the two calls saw
        different prompts (the per-part ablation block was threaded only into the
        build). They disagreed: classification -> ``is_refinement`` True (REFINE
        bucket), build -> not-refinement (``build_task_extras`` returns None) ->
        the refine silently fell back to an orthogonal restart. The 0823 smoke
        showed this on BOTH parents ("2 REFINE" classified, then both
        "mutation[..] ORTHOGONAL" built).

        Computing the directive ONCE here -- WITH the ablation, so the cached
        directive carries ``ablation_summary`` -- and reusing it in the build
        makes the bucket and the task agree by construction and halves the
        per-round diagnosis LLM calls (the build no longer re-diagnoses). The
        cache is cleared at the start of each ``_prepare_mutation_targets``;
        ``_classify_mutation_bucket`` (state-load re-derivation) populates it on a
        miss so a subsequent build finds the directive. With the ablation flag OFF,
        ``_get_ablation_eval()`` is ``None`` and the only behavioural change vs the
        prior path is the removal of the redundant second diagnosis call -- the
        no-regression guarantee.
        """
        cache = getattr(self, "_diagnosis_cache", None)
        if cache is None:
            cache = {}
            self._diagnosis_cache = cache
        tid = parent.trajectory_id
        if tid in cache:
            return cache[tid]
        d = self.refine_op.diagnose_parent(
            parent, self.pool.get_ancestors(tid),
            self._diagnosis_population(parent),
            ablation_eval=self._get_ablation_eval())
        cache[tid] = d
        return d

    def _cached_strength(self, parent: StrategyTrajectory):
        """One ``diagnose_strength`` per parent per crossover batch, cached +
        reused (sibling of ``_cached_diagnosis``).

        Eq. 7 crossover locates, for each of the two best parents, the
        construction decisions its measurements VALIDATED. The suffix
        (``generate_crossover_prompt_suffix``) and the task build
        (``build_task_extras``) both need the SAME directive per parent: a
        separate diagnosis call in each would run the LLM strength diagnosis
        twice per parent (non-zero temperature, no chat cache) and the two
        could disagree, exactly the divergence ``_cached_diagnosis`` fixed for
        mutation. Computing it once here and passing it to both makes the
        suffix's "what each parent's measurement validated" and the constructor
        inspiration block the same directive by construction.

        ``diagnose_strength`` reads the parent's already-computed
        ``factor_attribution`` / segments (no ablation re-run), so -- unlike
        ``_cached_diagnosis`` -- there is no ablation flag to thread. The
        cache is cleared at the start of each ``_prepare_crossover_groups``;
        ``None`` is stored (and returned) when the parent has no objective
        vector, so ``build_task_extras`` tolerates it.
        """
        cache = getattr(self, "_strength_cache", None)
        if cache is None:
            cache = {}
            self._strength_cache = cache
        tid = parent.trajectory_id
        if tid in cache:
            return cache[tid]
        from quantaalpha.pipeline.evolution.strength_diagnosis import diagnose_strength
        d = diagnose_strength(
            parent,
            self.pool.get_ancestors(tid),
            self._diagnosis_population(parent),
        )
        cache[tid] = d
        return d

    def _build_mutation_task(self, parent: StrategyTrajectory, direction_id: int,
                             bucket: str = _BUCKET_REFINE) -> dict[str, Any]:
        """Build one mutation task for ``parent`` routed by ``bucket`` (T6).

        The refine-vs-orthogonal decision is made once in
        ``_prepare_mutation_targets`` from the parent's verdict + diagnosability
        and carried here as ``bucket`` so the serial ``_get_mutation_task`` and
        the parallel ``get_all_tasks_for_current_phase`` cannot diverge on it:

          * REFINE / ADMITTED-PUSH -- ``build_task_extras`` (Eq. 6 diagnose +
            freeze + refine, or the ADMITTED push-further directive) yields a
            refinement task. The directive is the one cached in
            ``_prepare_mutation_targets`` (via ``_cached_diagnosis``), so the
            bucket and the task are the same diagnosis by construction; if the
            bucket was REFINE the cached directive is a refinement and the task is
            built. A cache miss (an ADMITTED-PUSH parent the classification does
            not pre-diagnose, or a state load) falls back to diagnosing here, and
            if THAT yields no refinement the parent falls through to the orthogonal
            path rather than being dropped.
          * ORTHOGONAL -- the explore / restart path; the diagnosis is skipped.

        T5: the parent's lineage is passed to ``build_task_extras`` here (parent
        process) so the exhausted-lever note is baked into the serialized
        directive and the parallel child needs no pool access.
        """
        extras = None
        if bucket in (_BUCKET_REFINE, _BUCKET_ADMITTED_PUSH):
            # T5: pass the parent's lineage so the diagnosis can detect an
            # exhausted refinement lever (fix #5).
            ancestors = self.pool.get_ancestors(parent.trajectory_id)
            # Reuse the directive cached in _prepare_mutation_targets (via
            # _cached_diagnosis) so the REFINE bucket decided there and the task
            # built here cannot diverge on the refine-vs-orthogonal call. A cache
            # miss (an ADMITTED-PUSH parent the classification does not
            # pre-diagnose, or a state load) falls back to diagnosing here.
            extras = self.refine_op.build_task_extras(
                parent, ancestors, self._diagnosis_population(parent),
                ablation_eval=self._get_ablation_eval(),
                directive=getattr(self, "_diagnosis_cache", {}).get(
                    parent.trajectory_id))

        if extras is not None:
            # REFINE / ADMITTED-PUSH child (Eq. 6): build on the parent's
            # construction. The directive is the load-bearing guidance, so the
            # zoo-context digest (an "avoid the explored space" signal that
            # would fight "build on THIS parent") is NOT appended -- only the
            # orthogonal path gets it.
            task = {
                "phase": RoundPhase.MUTATION,
                "direction_id": direction_id,
                "parent_trajectories": [parent],
                "round_idx": self._current_round,
                **extras,
            }
            _rd = extras.get("refine_directive", {}) or {}
            _tag = "ADMITTED-PUSH" if bucket == _BUCKET_ADMITTED_PUSH else "REFINE"
            logger.info(
                f"mutation[{direction_id}] {_tag} parent {parent.trajectory_id} "
                f"(verdict={_rd.get('verdict')}, target={_rd.get('refine_target')}, "
                f"weakness={_rd.get('weakness_dimension')})"
            )
            return task

        # ORTHOGONAL mutation -- the explore / restart path. Also the defensive
        # fallback when a REFINE/ADMITTED-PUSH bucket's diagnosis yields no task.
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
        logger.info(
            f"mutation[{direction_id}] ORTHOGONAL parent {parent.trajectory_id}"
        )
        return task

    def _classify_mutation_bucket(self, parent: StrategyTrajectory) -> str:
        """Re-derive a parent's T6 mutation bucket from its current metrics.

        Mirrors the classification in ``_prepare_mutation_targets`` so a state
        load (which restores targets from IDs but may carry no buckets in a
        pre-T6 state file) re-derives the same routing. An admitted parent that
        IS a mutation target was selected for ADMITTED-PUSH -- the unselected
        admitted winners are crossover-only and absent from the target list, so
        seeing an admitted parent here means it was pushed.
        """
        metrics = parent.backtest_metrics or {}
        if "U" not in metrics:
            return _BUCKET_ORTHOGONAL
        if bool(metrics.get("admitted", False)):
            return _BUCKET_ADMITTED_PUSH
        d = self._cached_diagnosis(parent)
        if d is not None and d.is_refinement():
            return _BUCKET_REFINE
        return _BUCKET_ORTHOGONAL

    def _get_mutation_task(self) -> Optional[dict[str, Any]]:
        """Get next mutation round task."""
        # Budget guard. The three phase getters call each other on every
        # transition (original->mutation->crossover->mutation->...), and
        # ``get_next_task``'s ``_rounds_exhausted`` check is only on the OUTER
        # entry -- so once a transition chain starts, nothing re-checks it.
        # Measured: a resumed controller whose pool has no minable parents
        # recursed to round 1116+ with max_rounds=15 and QA_MAX_ROUNDS_CAP=60
        # both set, i.e. the budget was bypassed entirely. Each getter now
        # re-checks before doing any work.
        if self._rounds_exhausted():
            return None
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
            bucket = (self._mutation_buckets[self._mutation_idx]
                      if self._mutation_idx < len(self._mutation_buckets)
                      else _BUCKET_REFINE)

            # Check if this mutation already exists
            existing = [t for t in self.pool.get_all()
                       if t.round_idx == self._current_round
                       and t.phase == RoundPhase.MUTATION
                       and parent.trajectory_id in t.parent_ids]
            if existing:
                self._mutation_idx += 1
                continue

            task = self._build_mutation_task(parent, direction_id, bucket)
            self._mutation_idx += 1
            return task

        # All mutation tasks complete, transition to next phase
        self._mutation_targets = []  # Reset for next mutation round
        self._mutation_buckets = []
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

        T6: each prev-phase parent is routed to a mutation BUCKET by its
        evaluation verdict + diagnosability, not by its fitness rank (the old
        rank+tail-cut routing bred the best parents and restarted the weakest --
        backwards: the failures need refining, the winners go to crossover):

          * REFINE        -- rejected (``admitted`` False) with a usable verdict
                            (``diagnose().is_refinement()``): fix why it failed.
          * ORTHOGONAL    -- ``NO_DATA`` / no usable verdict / an exhausted lever:
                            nothing to diagnose, restart orthogonally.
          * ADMITTED-PUSH -- a config-gated fraction (``admitted_push_fraction``)
                            of admitted winners ALSO get a "push the edge further"
                            refine; the rest are crossover-only.

        Bucket priority REFINE > ORTHOGONAL > ADMITTED-PUSH so a push-further
        never crowds out a failure-refine. Bucketing is per-parent and
        order-independent, so the serial ``_get_mutation_task`` and the parallel
        ``get_all_tasks_for_current_phase`` route identically.
        """
        self._mutation_targets = []
        self._mutation_buckets = []
        self._mutation_idx = 0
        # One diagnose_parent per parent per round, reused by _build_mutation_task
        # (see _cached_diagnosis). Cleared each round so a parent re-evaluated in a
        # later round (e.g. a crossover child) gets a fresh directive.
        self._diagnosis_cache = {}

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

        # Do not mutate a factor that F_Theta rejected.
        prev_phase_trajs = self._admissible_parents(
            prev_phase_trajs, minimum=1, what="mutation targets"
        )

        # T6 bucket classification (see docstring). The directive computed here
        # via ``_cached_diagnosis`` is the SAME one ``_build_mutation_task`` reuses,
        # so the bucket decided here matches the task routed there by construction
        # (no reliance on cross-call determinism).
        refine_parents: list[StrategyTrajectory] = []
        orthogonal_parents: list[StrategyTrajectory] = []
        admitted_parents: list[StrategyTrajectory] = []
        for t in prev_phase_trajs:
            metrics = t.backtest_metrics or {}
            if "U" not in metrics:
                orthogonal_parents.append(t)  # no objective -> nothing to diagnose
                continue
            if bool(metrics.get("admitted", False)):
                admitted_parents.append(t)  # winner -> crossover stock / push
                continue
            d = self._cached_diagnosis(t)
            if d is not None and d.is_refinement():
                refine_parents.append(t)
            else:
                # NO_DATA, an exhausted lever, or a non-refinement verdict.
                orthogonal_parents.append(t)

        # ADMITTED-PUSH: a config-gated fraction of admitted winners ALSO get a
        # push-further refine (local-search exploitation). Default 0.0 = admitted
        # are crossover-only (NOT mutated). The pushed fraction is the BEST
        # winners by fitness (rank once, reuse); the rest -> DROP-TO-CROSSOVER.
        push_frac = float(getattr(self.config, "admitted_push_fraction", 0.0) or 0.0)
        admitted_ranked = self._rank_by_fitness(admitted_parents)
        push_n = int(round(push_frac * len(admitted_ranked))) if push_frac > 0 else 0
        push_n = max(0, min(push_n, len(admitted_ranked)))
        admitted_push = admitted_ranked[:push_n]

        # Rank REFINE best-first by fitness so ``mutation_top_fraction`` (if <1.0)
        # drops the weakest refines, not the strongest. Routing is bucket-based;
        # ranking only orders WITHIN a bucket for the optional cap.
        refine_ranked = self._rank_by_fitness(refine_parents)
        targets = (list(refine_ranked) + list(orthogonal_parents)
                   + list(admitted_push))
        buckets = ([_BUCKET_REFINE] * len(refine_ranked)
                   + [_BUCKET_ORTHOGONAL] * len(orthogonal_parents)
                   + [_BUCKET_ADMITTED_PUSH] * len(admitted_push))

        # Optional total cap (mutation_top_fraction). Cuts from the back -- the
        # lowest-priority ADMITTED-PUSH slots first, then ORTHOGONAL -- so every
        # REFINE survives. frac==1.0 (the default) cuts nothing.
        frac = float(getattr(self.config, "mutation_top_fraction", 1.0) or 1.0)
        if 0.0 < frac < 1.0 and len(targets) > 1:
            keep = max(1, int(round(frac * len(targets))))
            if keep < len(targets):
                logger.info(f"mutation: capped {len(targets)} -> {keep} targets "
                            f"(dropped {len(targets) - keep} lowest-priority)")
                targets, buckets = targets[:keep], buckets[:keep]

        self._mutation_targets = targets
        self._mutation_buckets = buckets
        self._mutation_idx = 0
        self._active_branch_count = len(targets)

        logger.info(
            f"Prepared {len(targets)} mutation targets for round "
            f"{self._current_round}: {len(refine_ranked)} REFINE, "
            f"{len(orthogonal_parents)} ORTHOGONAL, {push_n} ADMITTED-PUSH "
            f"({max(0, len(admitted_parents) - push_n)} admitted -> crossover-only)"
        )
    
    def _prepare_crossover_groups(self):
        """
        Prepare crossover groups for the next crossover round.

        Crossover candidates are selected from the two most recent rounds:
        - First crossover (after round 1): original (round 0) + mutation (round 1)
        - Subsequent crossovers: previous mutation + previous crossover

        This ensures that crossover combines the latest evolutionary results,
        not arbitrarily old trajectories.
        """
        # T6: a fresh strength-diagnosis cache per crossover batch (sibling of
        # ``_diagnosis_cache`` cleared in ``_prepare_mutation_targets``). The two
        # parents' ``diagnose_strength`` directives are computed once per parent
        # here (via ``_cached_strength``) and reused by both the suffix and
        # ``build_task_extras`` so they cannot diverge.
        self._strength_cache = {}

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
    
    def _diagnosis_population(self, exclude: StrategyTrajectory | None = None) -> list:
        """The prior attempts the diagnosis is allowed to see.

        The search has never had this channel. Every hypothesis is generated
        against an empty ``Trace`` (a fresh ``AlphaAgentLoop`` per task builds a
        fresh one), so the generator proposes as if every round were the first --
        which is measurably what happens: 0 of 7 rounds show round-over-round
        improvement, and operators produce worse children than their parents
        54 times to 15.

        Sampled the way FunSearch samples its program database: the best-scoring
        attempts so the model can see what worked, plus the most recent so it can
        see what was just tried and failed. Deduplicated, parent excluded.
        """
        try:
            everything = [t for t in self.pool.get_all()
                          if (t.backtest_metrics or {}).get("U") is not None]
        except Exception:
            return []
        if exclude is not None:
            everything = [t for t in everything
                          if t.trajectory_id != exclude.trajectory_id]
        if not everything:
            return []

        half = max(1, _POPULATION_SAMPLE // 2)
        scores = self._shrunk_fitness(everything)
        best = sorted(everything,
                      key=lambda t: scores.get(t.trajectory_id, float("-inf")),
                      reverse=True)[:half]
        recent = list(reversed(everything))[:half]

        seen, out = set(), []
        for t in best + recent:
            if t.trajectory_id not in seen:
                seen.add(t.trajectory_id)
                out.append(t)
        return out[:_POPULATION_SAMPLE]

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
        the objective) for any trajectory without ``delta_mean``, so runs that
        do not use marginal_contribution mode are untouched.
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

        Only active when `QA_REQUIRE_FEASIBLE=true`; otherwise the unfiltered
        behaviour is kept exactly.

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
        # Budget guard. The three phase getters call each other on every
        # transition (original->mutation->crossover->mutation->...), and
        # ``get_next_task``'s ``_rounds_exhausted`` check is only on the OUTER
        # entry -- so once a transition chain starts, nothing re-checks it.
        # Measured: a resumed controller whose pool has no minable parents
        # recursed to round 1116+ with max_rounds=15 and QA_MAX_ROUNDS_CAP=60
        # both set, i.e. the budget was bypassed entirely. Each getter now
        # re-checks before doing any work.
        if self._rounds_exhausted():
            return None
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

        # T6: compute each parent's strength directive ONCE (cached) so the
        # suffix and build_task_extras see the same diagnosis (no double-diagnose
        # divergence). Ancestors + the prior-attempts population are threaded by
        # ``_cached_strength``.
        strengths = [self._cached_strength(p) for p in parents]

        # Generate crossover guidance
        suffix = self.crossover_op.generate_crossover_prompt_suffix(parents, strengths)
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
        # Eq. 7: ship the two parents' validated strengths as constructor
        # inspiration (not a literal splice). ``None`` when fewer than 2 parents.
        task.update(self.crossover_op.build_task_extras(parents, strengths) or {})
        
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
        ``feasible`` fallback keeps trajectories readable when ``admitted`` is
        absent.
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
        
        # Trajectory ID: reuse the one the executed task already carries.
        #
        # This used to call generate_id() a SECOND time. generate_id hashes
        # datetime.now() to microseconds, so the id minted here could never equal
        # the one factor_mining._run_evolution_task minted at its line 190 and
        # wrote into the factor library. The pool therefore referenced one id
        # space and the library another, and every ``parent_trajectory_ids``
        # link pointed at an id the library did not contain -- measured on the
        # live run: 27 factors carried parent links, **0 of them resolvable**.
        #
        # The consequence was silent and expensive: no parent-vs-child
        # comparison is possible, so the system could not answer whether
        # mutation or crossover actually improves on what it started from --
        # which is the whole question the evolution loop exists to settle.
        traj_id = (
            task.get("trajectory_id")
            or StrategyTrajectory.generate_id(direction_id, round_idx, phase)
        )
        
        # Extract hypothesis info
        hypothesis_text = str(hypothesis) if hypothesis else ""
        # Persist the pre-registered direction onto the trajectory. The
        # AlphaAgentHypothesis object carries expected_ic_sign here; str() above
        # discards it, so capture it before the object goes out of scope. A later
        # diagnosis (this trajectory as a refine parent) reads it back via
        # getattr(parent, "expected_ic_sign", "") to refreeze a child's
        # direction; without it every expression-refine child is born directionless
        # and the falsifiability gate rejects it no_mechanism (1817 run: 3/3 frozen
        # children rejected for an empty sign).
        expected_ic_sign = (
            str(getattr(hypothesis, "expected_ic_sign", "") or "").strip().lower()
            if hypothesis else ""
        )
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
            backtest_metrics = self._extract_metrics(
                backtest_result, [f.get("expression", "") for f in factors])
        
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

        # T5: record the refine directive that PRODUCED this trajectory, so a
        # descendant's diagnosis can detect an exhausted lever (fix #5). One
        # entry per refine parent; orthogonal children carry none (no
        # ``refine_directive`` on the task). A CROSSOVER child does carry one
        # (refine_target="recombine", verdict="crossover") so its lineage records
        # that it was recombined -- but ``repair_actions_summary`` skips those,
        # since a recombination repaired no diagnosed weakness. The
        # ``target_subtree_signatures`` are the levers pulled -- the canonical
        # AST signatures from the T3 targets -- which a descendant compares to
        # its own directive's targets.
        refine_actions: list[dict] = []
        rd = task.get("refine_directive")
        if isinstance(rd, dict) and rd.get("verdict"):
            refine_actions = [{
                "round_idx": round_idx,
                "verdict": rd.get("verdict"),
                "weakness_dimension": rd.get("weakness_dimension"),
                "refine_target": rd.get("refine_target"),
                "mechanism_hint": rd.get("mechanism_hint", ""),
                "target_subtree_signatures": [t.get("subtree_signature")
                                               for t in (rd.get("targets") or [])
                                               if t.get("subtree_signature")],
            }]

        # Eq.7 crossover audit: persist the two parents' strength directives
        # that produced this child, so the recombination that was actually
        # attempted is inspectable after the run (which validated decision was
        # offered, from which parent) rather than only inferable from the prose.
        extra_info: dict[str, Any] = {}
        cx_strengths = task.get("crossover_strengths")
        if cx_strengths:
            extra_info["crossover_strengths"] = cx_strengths

        return StrategyTrajectory(
            trajectory_id=traj_id,
            direction_id=direction_id,
            round_idx=round_idx,
            phase=phase,
            hypothesis=hypothesis_text,
            expected_ic_sign=expected_ic_sign,
            hypothesis_details=hypothesis_details,
            factors=factors,
            backtest_result=backtest_result,
            backtest_metrics=backtest_metrics,
            feedback=feedback_text,
            feedback_details=feedback_details,
            parent_ids=parent_ids,
            refine_actions=refine_actions,
            extra_info=extra_info,
        )
    
    def _extract_metrics(self, result: Any,
                         factor_exprs: list[str] | None = None) -> dict[str, Optional[float]]:
        """Extract metrics from backtest result.

        ``factor_exprs`` (the trajectory's own factor expressions, in order)
        selects which per-factor tearsheet to promote to top-level when the
        net-cost engine nests several under ``factor_tearsheets``; see
        ``_extract_net_cost_metrics``.
        """
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
            # additive: a Qlib-only result carries none of these keys, so its
            # behaviour is unchanged.
            metrics.update(self._extract_net_cost_metrics(result, factor_exprs))
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")

        return metrics

    # Keys emitted by NetCostFactorRunner._to_series, by coercion type.
    _NET_COST_FLOAT_KEYS = (
        "U", "rho_max", "rho_within", "turnover_book", "turnover_solo", "cx",
        "cost_bps", "zoo_size",
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
        # The t-statistic behind the admission verdict (mean/se). Distinguishes a
        # resolvably-negative contribution from an unresolved one -- the same
        # distinction the ``reason`` string makes in prose -- in a scalar the
        # diagnosis can branch on.
        "delta_t",
        # Deflated Sharpe (book) + its trial count -- written to the _to_series
        # payload (net_cost_runner._to_series) but absent here, so the gloss's
        # `dsr` line never rendered. Carried so the diagnosis sees the
        # multiple-testing-discounted book Sharpe.
        "dsr", "dsr_n_trials",
    )
    _NET_COST_STR_KEYS = (
        "theta_hash", "zoo_hash", "failed_gates", "weakest_dimensions",
        # The verdict string and its pathology sub-reason (admission.decide /
        # check_pathology), plus the incumbent a replacement evicted. These were
        # written to the ledger but dropped at _to_series, so the most diagnostic
        # text in the pipeline never reached the trajectory -- which is what
        # kept the mutation operator from turning a rejection into a directional
        # refinement instruction. ``displaced`` is an expression string or None;
        # the pd.notna guard below skips the None case so it reads as absent.
        # ``verdict`` (T1) is the *structured* verdict (``Verdict.value``, e.g.
        # "net_harmful") set at each decide() branch -- the authoritative form
        # ``classify_verdict`` reads first, retiring the brittle substring parse
        # of ``reason`` to a logged fallback.
        "reason", "pathology", "displaced", "verdict",
    )
    # List-valued (object) keys that none of the float/str/bool coercions handle.
    # ``delta_per_seed`` is the per-combiner-seed marginal-contribution vector --
    # the reproducible evidence behind the verdict, kept for auditability. The
    # diagnosis itself uses delta_mean/delta_se/reason; this is the raw record.
    _NET_COST_LIST_KEYS = ("delta_per_seed",)
    # Dict-valued keys (T4). ``factor_attribution`` is the per-factor combiner
    # credit, keyed by expression -> {weight, weight_raw, weight_stability,
    # ic_mean, ic_std, rank_ic, turnover_share}. pd.notna on a dict is truthy-
    # ambiguous (same caveat as list keys), so it is guarded on type in the
    # extract loop; the dict is carried through as-is so segment_profiling can
    # fold the credit into each SegmentProfile.
    # ``factor_tearsheets`` (per-factor admission scalars: t_nw, monotonicity,
    # sign_predicted/realized, fdr_*, capacity, ...) is carried through as-is for
    # lineage/debuggability; the CANDIDATE entry's scalars are ALSO promoted to
    # top-level in _extract_net_cost_metrics so _METRIC_GLOSS finds them.
    _NET_COST_DICT_KEYS = ("factor_attribution", "factor_tearsheets")
    # Bool flags from the net-cost runner. ``admitted`` is the live verdict the
    # digest and format_objective_note key on; ``in_zoo`` is its persistence
    # counterpart. A Qlib-only Series carries neither, so these stay absent
    # and its trajectories read exactly as before.
    _NET_COST_BOOL_KEYS = ("admitted", "in_zoo")

    def _extract_net_cost_metrics(self, result: Any,
                                  factor_exprs: list[str] | None = None) -> dict[str, Any]:
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
        # List-valued keys (e.g. delta_per_seed). pd.notna on a list returns a
        # bool array, which is truthy-ambiguous, so guard on type instead.
        for key in self._NET_COST_LIST_KEYS:
            if key in series.index:
                val = series[key]
                if isinstance(val, (list, tuple)):
                    extras[key] = list(val)
        # Dict-valued keys (T4): per-factor combiner credit. Same notna caveat
        # as list keys, so guard on type; carry the dict through as-is.
        for key in self._NET_COST_DICT_KEYS:
            if key in series.index:
                val = series[key]
                if isinstance(val, dict):
                    extras[key] = val
        if "feasible" in series.index and pd.notna(series["feasible"]):
            extras["feasible"] = bool(series["feasible"])
        for key in self._NET_COST_BOOL_KEYS:
            if key in series.index and pd.notna(series[key]):
                extras[key] = bool(series[key])
        # Per-factor tearsheet scalars -- the admission gate's per-factor
        # measurement (t_nw, monotonicity, sign_predicted/realized, fdr_*,
        # capacity, ...). _METRIC_GLOSS (llm_diagnosis) lists these as flat
        # top-level keys, but _to_series nests them under
        # factor_tearsheets[expr] and the dict allowlist above only passes the
        # whole dict through -- so the diagnosis LLM's "everything measured"
        # block rendered ~2 of 22 gloss lines (only the batch aggregates
        # rho_max/cx/net_ir/...). Promote the CANDIDATE factor's scalars to
        # top-level so the gloss finds them. sign_predicted is also the
        # belt-and-braces source for the refreeze sign (the load-bearing source
        # is the trajectory's expected_ic_sign; see
        # create_trajectory_from_loop_result). turnover_solo is already a
        # batch-aggregate float key, so it is NOT re-promoted here.
        fts = series["factor_tearsheets"] if "factor_tearsheets" in series.index else None
        sel = None
        if isinstance(fts, dict) and fts:
            for e in (factor_exprs or []):
                if e and e in fts:
                    sel = fts[e]
                    break
            if sel is None:
                # factors_per_hypothesis=1 -> one entry; else the strongest signal.
                sel = next(iter(fts.values())) if len(fts) == 1 else max(
                    fts.values(), key=lambda v: abs(float((v or {}).get("t_nw") or 0)))
        if isinstance(sel, dict):
            for k in ("t_nw", "rank_ic_neutral", "monotonicity", "q_spread",
                      "ls_sharpe", "sign_predicted", "sign_realized",
                      "mechanism_validated", "fdr_t_required", "fdr_n_tests",
                      "capacity_cny", "best_horizon", "ic_pos_frac",
                      "exposure_size"):
                if k in sel and k not in extras:
                    v = sel[k]
                    if pd.notna(v):
                        if isinstance(v, bool):
                            extras[k] = v
                        elif isinstance(v, (int, float)):
                            extras[k] = float(v)
                        else:
                            extras[k] = v
        return extras
    
    # ------------------------------------------------------------------
    # Learning-aware reseed: when breeding stops growing the repository,
    # regenerate NEW directions informed by the run's trial history.
    # ------------------------------------------------------------------

    def _zoo_size(self) -> int | None:
        """How many factors the repository holds, or None if unknowable here.

        The same source ``_rounds_exhausted`` sizes the target from, so the two
        cannot disagree about whether progress is being made. A missing ledger
        yields 0 (``replay_repository`` returns {} rather than raising); an empty
        ledger reads 0, and 0 > -1 (the initial ``_best_zoo_size``) resets the
        stale counter, so a run that has not admitted anything does not reseed.
        """
        try:
            from quantaalpha.eval.ledger import replay_repository

            return len(replay_repository(os.environ.get("QA_LEDGER")))
        except Exception:
            return None

    def _reseed_if_stale(self) -> bool:
        """Regenerate informed directions when the repository grows too slowly.

        Returns True when the phase was reset to ORIGINAL, so the caller returns
        an original task instead of its own transition. Two triggers:

        * STALE: the round admitted fewer than ``growth_floor`` factors for
          ``reseed_after_stale_rounds`` consecutive rounds. Keyed on GROWTH, not
          on rejections -- a round can legitimately reject everything while the
          repository is still climbing. The growth-floor is the fix for the
          creep: a zoo that admits one factor every few rounds used to reset the
          stale counter every time (any growth at all reset it) and so never
          reseeded, inbreeding the same directions until the round cap gave up
          short of the target.
        * SCHEDULED: every ``reseed_interval`` rounds, inject fresh informed
          directions regardless of growth -- steady immigration that does not
          wait for stagnation.

        Silent no-op when ``reseed_after_stale_rounds`` <= 0 and
        ``reseed_interval`` <= 0.
        """
        n = int(getattr(self.config, "reseed_after_stale_rounds", 0) or 0)
        interval = int(getattr(self.config, "reseed_interval", 0) or 0)
        if n <= 0 and interval <= 0:
            return False
        size = self._zoo_size()
        if size is None:
            return False
        if size > self._best_zoo_size:
            self._best_zoo_size = size
        if self._zoo_size_at_last_check < 0:
            # First check after (re)start: establish the baseline. Without this,
            # the -1 init would read a spurious size+1 "growth" and mask a stall
            # that is already in progress on a resumed run.
            self._zoo_size_at_last_check = size
            return False

        # Per-round growth (admissions since the last check), not vs an all-time
        # high -- a creeping zoo must still register as stale.
        growth = size - self._zoo_size_at_last_check
        self._zoo_size_at_last_check = size

        growth_floor = int(getattr(self.config, "growth_floor", 1) or 0)
        if growth >= growth_floor:
            self._stale_rounds = 0
        else:
            self._stale_rounds += 1

        fire_stale = n > 0 and self._stale_rounds >= n
        fire_scheduled = (
            interval > 0
            and self._current_round > 0
            and (self._current_round % interval == 0)
        )
        if not (fire_stale or fire_scheduled):
            return False
        reason = "stuck" if fire_stale else "scheduled-immigration"
        return self._do_reseed(size, reason, n)

    def _do_reseed(self, size: int, reason: str, stale_window: int) -> bool:
        """Build the digest, generate fresh directions, return to ORIGINAL.

        Shared by the stale and scheduled triggers. ``stale_window`` is the
        ``reseed_after_stale_rounds`` value, used to decide which directions have
        been dormant long enough to mark saturated.
        """
        self._stale_rounds = 0
        digest = self._build_reseed_digest()
        if not digest:
            logger.warning(
                f"Repository {reason} but no digestible trial history; skipping reseed"
            )
            return False
        new_dirs = self._generate_informed_directions(digest)
        if not new_dirs:
            logger.warning(
                f"Repository {reason} at {size} factor(s): informed-direction "
                "generation returned nothing; retrying next window"
            )
            return False
        # Mark saturated directions: explored and no admission within the last
        # `window` rounds. A direction that admitted recently keeps its headroom
        # and stays eligible; a never-admitted or long-dormant one is skipped so
        # the ORIGINAL phase spends its budget on the new directions. The window
        # is lenient (>= 2) regardless of the tight stale trigger, so a direction
        # that admitted last round is not prematurely marked explored-out.
        window = max(2, stale_window)
        for st in self._direction_status:
            recent = (
                st["last_admit_round"] >= 0
                and (self._current_round - st["last_admit_round"]) < window
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
            f"Repository {reason} at {size} factor(s): generated "
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
        all_admitted_exprs: list[str] = []  # for operator-coverage measurement
        for did in sorted(by_dir):
            trajs = by_dir[did]
            st = self._direction_status[did]
            direction_text = (self._directions[did] or "")[:200]
            admitted = [
                t for t in trajs
                if bool((t.backtest_metrics or {}).get(
                    "admitted", (t.backtest_metrics or {}).get("feasible", True)))
            ]
            for t in admitted:
                for f in (t.factors or []):
                    expr = f.get("expression", "") if isinstance(f, dict) else ""
                    if expr:
                        all_admitted_exprs.append(expr)
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

        # Operator-coverage measurement across every admitted factor in the
        # repository. This is the "lesson summary of what was learned" enriched
        # with operator-class coverage: the reseed LLM already sees WHICH
        # directions admitted/saturated above, and now also sees WHICH operator
        # classes the population has exhausted and which it has never touched --
        # so the informed directions can broaden toward REGBETA/RSI/MACD/COUNT
        # deliberately, not just toward unused signal space. Measurement only;
        # the block states the convergence and closes "yours to determine", so
        # it prescribes no remedy and carries no market prior.
        try:
            from quantaalpha.factors.operator_coverage import coverage_block
            cov = coverage_block(all_admitted_exprs)
            if cov:
                lines.append("")
                lines.append("Operator coverage across admitted factors:")
                lines.append(cov)
        except Exception:
            logger.exception("operator-coverage block failed; omitting from reseed digest")
        return "\n".join(lines)

    def _generate_informed_directions(self, digest: str) -> list[str]:
        """Ask the LLM for new orthogonal directions from the digest.

        Falls back to canned templates if the LLM returns nothing -- a reseed
        that injects zero directions is useless, and a canned opposite-direction
        restart beats inbreeding the same exhausted space.
        """
        prompt_path = getattr(self.config, "informed_prompt_path", None)
        if not prompt_path or not Path(prompt_path).exists():
            logger.warning("No informed-planning prompt path configured; cannot reseed")
            return []
        from quantaalpha.pipeline.planning import generate_informed_directions

        n = max(1, int(getattr(self.config, "num_directions", 2) or 2))
        common = dict(
            initial_direction=getattr(self.config, "initial_direction", "") or "",
            n=n,
            prompt_file=Path(prompt_path),
            history_summary=digest,
            use_llm=True,
            seed_in_generation=getattr(self.config, "seed_in_generation", True),
        )
        # The canned-fallback retry that used to sit here is gone (2026-08-15).
        # It re-called the planner with allow_fallback=True, which returned n
        # hardcoded "{base} + volatility regime switch"-style directions -- author
        # priors entering the search as if the planner had reasoned to them. A
        # reseed that produces nothing simply does not reseed this round; the
        # search continues on the directions it already has, which is honest and
        # visible in the log.
        try:
            dirs = generate_informed_directions(**common)
        except Exception as exc:
            logger.warning(f"Informed direction generation failed: {exc}")
            dirs = []
        if not dirs:
            logger.warning(
                "Informed-direction generation returned nothing; NOT substituting "
                "canned directions -- this round does not reseed"
            )
        return dirs

    def _coverage_and_headroom(self) -> str:
        """What the repository ALREADY SPANS, and what it leaves unexplained.

        Two measurements, both diagnosis rather than instruction (the system
        rule: state what was measured, never a remedy):

        **Coverage.** The repository's admitted signals are clustered by their
        pairwise rank correlation. Listing factor expressions -- which is all the
        context did before -- tells the generator what exists but not what SPACE
        is occupied: 9 factors spanning 5 correlated clusters look like 9
        directions and behave like 5. A new factor landing inside an occupied
        cluster cannot register as a contribution no matter how sound it is,
        because the gate measures contribution BEYOND the book.

        **Headroom.** The book's own IC against forward returns, and the IC that
        remains in the residual (return minus the book's prediction). This is the
        residual-targeting signal: it says how much predictable variation the
        book has NOT captured, so "there is nothing left to find" and "you keep
        proposing what is already held" are distinguishable states rather than
        the same silence.

        Correlations are measured in RANK space -- what the ICIR combiner fits on
        -- so ``X`` and ``RANK(X)`` count as one direction, not two.

        Cached on the repository's expression set: recomputed only when the zoo
        changes, and degrades to "" on any failure so a missing signal cache can
        never break generation.
        """
        try:
            import json
            import os

            ledger = os.environ.get("QA_LEDGER")
            if not ledger or not Path(ledger).exists():
                return ""
            exprs: list[str] = []
            with open(ledger, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("admitted") in (True, "True"):
                        fe = rec.get("factor_exprs") or []
                        if isinstance(fe, str):
                            import ast as _ast
                            fe = _ast.literal_eval(fe)
                        exprs.extend(fe or [])
            exprs = list(dict.fromkeys(exprs))
            if len(exprs) < 2:
                return ""
            key = tuple(exprs)
            if getattr(self, "_coverage_key", None) == key:
                return self._coverage_text

            from quantaalpha.eval.data import align_signal, load_factor_signal
            from quantaalpha.eval.metrics import (
                _cross_sectional_corr, _slice, label_frame, rho_max,
            )
            from quantaalpha.eval.operator import EvaluationOperator
            from quantaalpha.eval.protocol import default_protocol_path, load_protocol

            theta = load_protocol(os.environ.get("QA_PROTOCOL") or default_protocol_path())
            op = EvaluationOperator(theta)
            start, end, win = op._windows(False)
            panel = op._panel(start, end)
            label = label_frame(panel, theta)

            sigs = {}
            for e in exprs:
                try:
                    sigs[e] = align_signal(load_factor_signal(e), panel)
                except Exception:
                    continue
            if len(sigs) < 2:
                return ""

            # Greedy clustering at |rho| >= 0.5 -- the same notion of "one
            # direction" the de-dup screen uses.
            names = list(sigs)
            unassigned, clusters = set(names), []
            for a in names:
                if a not in unassigned:
                    continue
                fam = [a]
                unassigned.discard(a)
                for b in list(unassigned):
                    c = _cross_sectional_corr(sigs[a], sigs[b], "spearman")
                    if not c.empty and abs(float(c.mean())) >= 0.5:
                        fam.append(b)
                        unassigned.discard(b)
                clusters.append(fam)

            # Headroom: how much of the label the book's own signals leave.
            ics = []
            for e, s in sigs.items():
                ic = _cross_sectional_corr(_slice(s, win), label, "spearman").dropna()
                if len(ic):
                    ics.append(abs(float(ic.mean())))
            best_ic = max(ics) if ics else float("nan")

            lines = [
                "## Repository coverage (measured, not a target)",
                f"- {len(sigs)} admitted signals occupy {len(clusters)} distinct "
                f"directions at |rank corr| >= 0.5.",
            ]
            for i, fam in enumerate(clusters, 1):
                head = fam[0][:64]
                extra = f" (+{len(fam)-1} correlated with it)" if len(fam) > 1 else ""
                lines.append(f"  - direction {i}: {head}{extra}")
            lines.append(
                f"- Strongest single-factor |rank IC| currently held: "
                f"{best_ic:.4f}."
            )

            # RESIDUAL HEADROOM. Fit the book on the admitted signals, then ask
            # how much of the forward return its prediction does NOT explain.
            # This is what makes "the space is exhausted" and "you keep
            # re-proposing what is already held" distinguishable: if the book's
            # own IC is small, predictable variation remains and the failure is
            # duplication; if it is large, the book already holds the signal.
            try:
                from quantaalpha.eval import combiner as _comb

                pred, _attr = _comb.fit_predict(sigs, None, panel, theta)
                book_ic = _cross_sectional_corr(
                    _slice(pred, win), label, "spearman").dropna()
                if len(book_ic):
                    b = float(book_ic.mean())
                    lines.append(
                        f"- The book's combined prediction has rank IC "
                        f"{b:+.4f} against the forward return it is fitted to. "
                        f"The remaining cross-sectional variation is what a new "
                        f"factor would have to predict; a factor that only "
                        f"re-explains what this prediction already captures "
                        f"scores zero on the gate."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"residual headroom unavailable: {exc}")
            lines.append(
                "- A candidate whose rank correlation with any listed direction "
                "is high cannot be measured as a contribution: the gate scores "
                "what a factor adds BEYOND the book, and a duplicate adds "
                "nothing measurable however sound its premise."
            )
            self._coverage_key = key
            self._coverage_text = "\n".join(lines)
            return self._coverage_text
        except Exception as exc:  # noqa: BLE001 -- context must never break generation
            logger.warning(f"coverage context unavailable: {exc}")
            return ""

    def _build_zoo_context(self) -> str:
        """Cumulative repository summary appended to mutation/crossover guidance.

        Tells the breeder what the book already captures so mutations and
        crossovers aim at ORTHOGONAL signal rather than re-saturating the same
        space. Capped at 8 admitted trajectories to bound the token cost.

        Also carries the REJECTIONS. This is the only channel that survives
        across batches: ``AlphaAgentLoop.__init__`` builds a fresh
        ``Trace(scen=scen)`` per batch, so ``trace.hist`` is always empty at
        generation time and every batch is prompted as if it were the first
        round. The 20260815 run showed the cost -- the generator DISCOVERED the
        market's reversal structure in round 0 (feedback said "directionally
        supports the reversal hypothesis") and then un-learned it: factor
        descriptions mentioning reversal/inverse fell 33% -> 0% -> 17% -> 11%
        across rounds. Passing only the 5 ADMITTED trajectories meant the 19
        REJECTIONS -- which is where the "this failed, and here is the measured
        sign" evidence lives -- never reached the next hypothesis.

        Rejections are shown with their verdict and their SIGNED RankIC, because
        on this market the sign is the lesson: essentially every mined factor
        has negative raw RankIC (short-horizon mean reversion), and a generator
        that never sees that keeps proposing momentum-framed ideas.
        """
        all_trajs = self.pool.get_all()

        def _is_admitted(t) -> bool:
            """Did this batch actually HELP -- not merely: was it let in.

            Split on the VERDICT, never on the ``admitted`` flag. Under
            ``admission.blocking: false`` the flag is True for almost
            everything (that is the whole point -- the pool accumulates), so a
            flag-based split silently empties the "what has already FAILED"
            section below. That section is the only place the signed-RankIC
            lesson survives across batches, and losing it would undo the memory
            channel while looking like it still worked.
            """
            m = t.backtest_metrics or {}
            v = str(m.get("verdict") or "").strip().lower()
            if v:
                return v in ("admitted", "replaced", "bootstrap")
            # Pre-verdict records: fall back to the old flag.
            return bool(m.get("admitted", m.get("feasible", True)))

        def _has_content(t) -> bool:
            """A row worth sending: it names a factor, or it carries a number.

            Without this, a trajectory holding neither renders as
            ``- [] U=N/A rho_max=N/A`` -- a Repository Context header followed
            by rows with an empty expression and no measurements. That is not
            merely wasted tokens: an empty ``[]`` under "what the book already
            captures" reads as a repository holding nameless factors.
            """
            m = t.backtest_metrics or {}
            if any((f.get("expression") or "").strip() for f in (t.factors or [])):
                return True
            return any(m.get(k) is not None
                       for k in ("U", "rho_max", "rank_ic", "verdict"))

        all_trajs = [t for t in all_trajs if _has_content(t)]
        admitted = [t for t in all_trajs if _is_admitted(t)][:8]
        # Most recent rejections first: the newest failures are the ones the
        # next hypothesis should avoid repeating.
        rejected = [t for t in all_trajs if not _is_admitted(t)][-6:]

        if not admitted and not rejected:
            return ""

        lines = []
        if admitted:
            lines.append("## Repository Context (what the book already captures)")
            for t in admitted:
                m = t.backtest_metrics or {}
                exprs = [f.get("expression", "")[:60] for f in (t.factors or [])[:2]]
                lines.append(
                    f"- [{' | '.join(exprs)}] U={format_metric(m.get('U'))} "
                    f"rho_max={format_metric(m.get('rho_max'))}"
                )
            # Was: "Aim for ORTHOGONAL signal not already in the book; do not
            # duplicate the admitted factor logic above." That is an instruction
            # about what to produce. The coverage block below states the same
            # situation as a MEASUREMENT -- which directions are occupied and why
            # a duplicate cannot score -- and leaves what to do about it open.
            cov = self._coverage_and_headroom()
            if cov:
                lines.append("")
                lines.append(cov)

        # --- accumulated failure PATTERNS, not just recent rejections --------
        # Storing "batch 14 scored -0.08" teaches nothing. Counting WHY factors
        # failed, across the whole run, produces a lesson that transfers: "nine
        # factors so far had their entire edge removed by size neutralization"
        # is actionable in a way no per-batch scalar is.
        pattern = self._failure_patterns(all_trajs)
        if pattern:
            if lines:
                lines.append("")
            lines.extend(pattern)

        if rejected:
            if lines:
                lines.append("")
            lines.append("## What has already FAILED (do not repeat these)")
            for t in rejected:
                m = t.backtest_metrics or {}
                exprs = [f.get("expression", "")[:60] for f in (t.factors or [])[:1]]
                verdict = m.get("verdict") or "rejected"
                # The FACTOR'S own signed IC, never the composite book's. See
                # the note on `rank_ic_own` in net_cost_runner._to_series: this
                # line used to render the book number, which was positive on
                # every prompt while the factors were negative 71% of the time.
                ric = m.get("rank_ic_own")
                if ric is None:
                    ric = m.get("rank_ic_neutral")
                ric_s = f" RankIC={format_metric(ric)}" if ric is not None else ""
                lines.append(f"- [{' | '.join(exprs)}] {verdict}{ric_s}")
            # THE AGGREGATE, stated once. Every prompt carried at most 3 signed
            # observations (median 0), and no prompt in the whole run stated the
            # overall direction. A 71/29 split is not inferable from 0-3 points,
            # so the generator was asked to learn a base rate it was never shown:
            # measured sign accuracy 54% against a 71% base rate, i.e. worse
            # than answering with the majority every time.
            signed = []
            for t in all_trajs:
                mm = t.backtest_metrics or {}
                vv = mm.get("rank_ic_own", mm.get("rank_ic_neutral"))
                try:
                    vv = float(vv)
                except (TypeError, ValueError):
                    continue
                if vv == vv:
                    signed.append(vv)
            if len(signed) >= 8:
                neg = sum(1 for v in signed if v < 0)
                lines.append("")
                lines.append(
                    f"## Direction, measured across this run\n"
                    f"- {neg} of {len(signed)} factors scored so far realized a "
                    f"NEGATIVE information coefficient "
                    f"({100.0 * neg / len(signed):.0f}%).\n"
                    f"- This is the measured outcome on this market over this "
                    f"period, not an assumption. What it implies for the next "
                    f"hypothesis is yours to determine.")
            lines.append(
                "These were measured and REJECTED. Read the RankIC SIGN as "
                "evidence, not decoration: if the realized sign is consistently "
                "OPPOSITE to what the rejected premise predicted, then that "
                "premise was directionally wrong and restating it -- however "
                "reworded -- will fail again. If the sign matched but the "
                "magnitude was too small, the premise pointed the right way and "
                "what it was measured through is where the shortfall sits. "
                "What to do with either reading is yours to determine."
            )
        return "\n".join(lines)

    def _failure_patterns(self, trajectories: list) -> list[str]:
        """Aggregate WHY factors have failed, across the whole run so far.

        Six failures that used to arrive as one scalar are now distinguishable
        measurements, so they can be counted. A running tally is the closest
        thing the search has to a lesson: it survives batches, it names a cause
        rather than a score, and it says which causes are recurring.

        Every line is a count of something measured. No line says what to do
        about it -- the counts are the evidence, and what follows from them is
        the model's to work out.
        """
        buckets = {
            "size_exposure": 0, "no_signal": 0, "tails_only": 0,
            "duplicate": 0, "unstable": 0, "fast_decay": 0,
        }
        n = 0
        for t in trajectories:
            m = t.backtest_metrics or {}
            sheets = m.get("factor_tearsheets")
            rows = list(sheets.values()) if isinstance(sheets, dict) else ([m] if m else [])
            for sheet in rows:
                if not isinstance(sheet, dict):
                    continue
                raw_ic = sheet.get("rank_ic")
                neu_ic = sheet.get("rank_ic_neutral")
                exp_sz = sheet.get("exposure_size")
                t_nw = sheet.get("t_nw")
                mono = sheet.get("monotonicity")
                rho = sheet.get("rho_max")
                pos = sheet.get("ic_pos_frac")
                if neu_ic is None and t_nw is None:
                    continue
                n += 1
                try:
                    if (exp_sz is not None and abs(float(exp_sz)) > 0.4
                            and raw_ic and neu_ic is not None
                            and abs(float(neu_ic)) < 0.4 * abs(float(raw_ic))):
                        buckets["size_exposure"] += 1
                    if t_nw is not None and abs(float(t_nw)) < 3.0:
                        buckets["no_signal"] += 1
                    if mono is not None and mono == mono and abs(float(mono)) < 0.30:
                        buckets["tails_only"] += 1
                    if rho is not None and float(rho) >= 0.60:
                        buckets["duplicate"] += 1
                    if pos is not None and float(pos) < 0.52:
                        buckets["unstable"] += 1
                except (TypeError, ValueError):
                    continue
        if not n:
            return []

        label = {
            "size_exposure": "had most of their edge removed by size neutralization "
                             "(the raw correlation was largely company size)",
            "no_signal":     "did not reach |t| = 3 on their neutralized correlation",
            "tails_only":    "moved only in the extreme deciles, with no gradient between",
            "duplicate":     "correlated 0.60 or higher with a factor already held",
            "unstable":      "kept their sign on barely half the days",
        }
        out = [f"## What the measurements have shown so far ({n} factors scored)"]
        for key, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
            if count and key in label:
                out.append(f"- {count} of {n} {label[key]}")
        return out if len(out) > 1 else []

    def _rounds_exhausted(self) -> bool:
        """Should evolution stop?

        Normally ``max_rounds`` decides. But when the objective gates
        admission on ``U``, a fixed round count produces a repository whose size
        depends on how many batches happened to clear the bar -- and factor
        count moves the combiner independently of factor quality, so a run that
        mines the same amount but admits less is handicapped for a reason that
        has nothing to do with its objective.

        ``QA_TARGET_ZOO`` makes the budget follow the outcome instead of an
        estimate: keep mining until the repository actually holds that many
        admitted factors. A guessed multiplier cannot do this -- the admission
        rate is not known before the run and drifts as the repository improves,
        which is the whole point of an adaptive bar.

        ``QA_MAX_ROUNDS_CAP`` bounds the cost. Hitting it means the search could
        not reach the target, which is itself a result and is logged as such
        rather than being retried forever.
        """
        if self._current_round < self.config.max_rounds:
            return False

        cap = int(os.environ.get("QA_MAX_ROUNDS_CAP", self.config.max_rounds * 3))

        # QA_TARGET_MINED -- stop when the LIBRARY reaches N factors.
        #
        # This exists because QA_TARGET_ZOO was being used for a job it cannot
        # do. The canonical 150 came from ``expected_factor_count(10,10,5,3)``,
        # whose own docstring says "how many factors a run of this shape is
        # expected to MINE" -- a GENERATION count. Setting it as the ADMISSION
        # target asked for 150 admitted out of 150 generated: a 100% admission
        # rate, i.e. a gate that never rejects. With crossover_n cut to 5 the
        # shape generates only 105, so the standing config asked for 143%.
        #
        # Worse, the target is unreachable in principle, not just in practice:
        # admission is MARGINAL contribution to the existing book, so every
        # admitted factor raises the bar for the next one. The rate falls toward
        # zero by construction. A library target is the honest way to ask for
        # "150 factors"; the zoo then settles wherever the objective says.
        mined_target = os.environ.get("QA_TARGET_MINED")
        if mined_target:
            try:
                mined_n = int(mined_target)
            except ValueError:
                logger.warning(f"QA_TARGET_MINED={mined_target!r} is not an integer; ignoring")
                mined_n = 0
            if mined_n > 0:
                mined = self._library_size()
                if mined >= mined_n:
                    logger.info(
                        f"Evolution complete: library holds {mined} mined factor(s) "
                        f">= QA_TARGET_MINED={mined_n} after round {self._current_round}"
                    )
                    return True
                if self._current_round >= cap:
                    logger.warning(
                        f"Evolution stopping at the cap ({cap} rounds) with "
                        f"{mined}/{mined_n} mined factors."
                    )
                    return True
                logger.info(
                    f"Extending evolution: {mined}/{mined_n} mined factor(s) "
                    f"after round {self._current_round} (cap {cap})"
                )
                return False

        target = os.environ.get("QA_TARGET_ZOO")
        if not target:
            return True
        try:
            target_n = int(target)
        except ValueError:
            logger.warning(f"QA_TARGET_ZOO={target!r} is not an integer; ignoring")
            return True
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
                f"admitted factors. The search could not reach the target: its admission "
                f"rate is too low for this budget, which is a result, not a retry condition."
            )
            return True
        logger.info(
            f"Extending evolution past max_rounds: {admitted}/{target_n} admitted "
            f"factor(s) after round {self._current_round} (cap {cap})"
        )
        return False

    @staticmethod
    def _library_size() -> int:
        """How many factors the run has MINED so far (the library, not the zoo).

        Reads the same file ``loop.py`` writes: ``data/factorlib/
        all_factors_library_<FACTOR_LIBRARY_SUFFIX>.json``. Counts every entry,
        admitted or not -- the library is the record of everything the search
        produced, which is the quantity ``QA_TARGET_MINED`` bounds.

        Returns 0 when the file does not exist yet or cannot be read, so a
        missing library reads as "keep going" rather than stopping the run.
        """
        import json
        from pathlib import Path

        suffix = os.environ.get("FACTOR_LIBRARY_SUFFIX", "")
        name = f"all_factors_library_{suffix}.json" if suffix else "all_factors_library.json"
        # controller.py -> evolution -> pipeline -> quantaalpha -> <project root>
        path = Path(__file__).resolve().parents[3] / "data" / "factorlib" / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        factors = data.get("factors")
        if isinstance(factors, dict):
            return len(factors)
        meta = data.get("metadata") or {}
        try:
            return int(meta.get("total_factors") or 0)
        except (TypeError, ValueError):
            return 0

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
            # T6 per-target routing buckets (REFINE/ORTHOGONAL/ADMITTED-PUSH),
            # parallel to mutation_target_ids. Absent in pre-T6 state files; the
            # loader re-derives them when missing.
            "mutation_buckets": list(self._mutation_buckets),
            # Learning-aware reseed state: the grown direction list and the
            # per-direction outcome tally must survive a restart or a resumed
            # run would forget it had already saturated those directions.
            "directions": list(self._directions),
            "direction_status": list(self._direction_status),
            "best_zoo_size": self._best_zoo_size,
            "stale_rounds": self._stale_rounds,
            "zoo_size_at_last_check": self._zoo_size_at_last_check,
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
        self._zoo_size_at_last_check = int(state.get("zoo_size_at_last_check", -1))

        # Restore mutation targets from IDs. Re-derive the T6 routing buckets from
        # each restored target's current metrics when the state file carries none
        # (pre-T6 files) -- an admitted parent that IS a mutation target was
        # selected for ADMITTED-PUSH (the rest are crossover-only and absent), so
        # the re-derivation matches the classification in _prepare_mutation_targets.
        mutation_target_ids = state.get("mutation_target_ids", [])
        saved_buckets = state.get("mutation_buckets")
        self._mutation_targets = []
        self._mutation_buckets = []
        for i, tid in enumerate(mutation_target_ids):
            traj = self.pool.get(tid)
            if not traj:
                continue
            self._mutation_targets.append(traj)
            if isinstance(saved_buckets, list) and i < len(saved_buckets):
                self._mutation_buckets.append(str(saved_buckets[i]))
            else:
                self._mutation_buckets.append(self._classify_mutation_bucket(traj))
        
        # Re-prepare crossover groups if in crossover phase
        if self._current_phase == RoundPhase.CROSSOVER:
            self._prepare_crossover_groups()
        
        logger.info(f"Loaded evolution state from {path}")

