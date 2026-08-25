"""
Factor workflow with session control and evolution support.

Supports three round phases:
- Original: Initial exploration in each direction
- Mutation: Orthogonal exploration from parent trajectories
- Crossover: Hybrid strategies from multiple parents

Supports parallel execution within each phase when enabled.
"""

from typing import Any
from pathlib import Path
import fire
import signal
import sys
import threading
from multiprocessing import Process, Queue
from functools import wraps
import time
import ctypes
import os
import pickle
from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
from quantaalpha.pipeline.planning import generate_parallel_directions
from quantaalpha.pipeline.planning import load_run_config
from quantaalpha.pipeline.loop import AlphaAgentLoop
from quantaalpha.pipeline.evolution import (
    EvolutionController,
    EvolutionConfig,
    StrategyTrajectory,
    RoundPhase,
)
# Name of the scalar being optimized, so the objective's logs do not claim to
# show RankIC while actually showing U.
from quantaalpha.pipeline.evolution.trajectory import _PRIMARY_METRIC
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.log import logger
from quantaalpha.log.time import measure_time
from quantaalpha.llm.config import LLM_SETTINGS


def force_timeout():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Prefer the timeout parameter
            seconds = LLM_SETTINGS.factor_mining_timeout

            if sys.platform != "win32":
                # Unix/Linux: Use SIGALRM signal
                def handle_timeout(signum, frame):
                    logger.error(
                        f"Force terminating execution, exceeded {seconds} seconds"
                    )
                    sys.exit(1)

                signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            else:
                # Windows: Use daemon thread for timeout
                result_container = [None]
                exception_container = [None]

                def target():
                    try:
                        result_container[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception_container[0] = e

                worker = threading.Thread(target=target, daemon=True)
                worker.start()
                worker.join(timeout=seconds)

                if worker.is_alive():
                    logger.error(
                        f"Force terminating execution, exceeded {seconds} seconds"
                    )
                    os._exit(1)

                if exception_container[0] is not None:
                    raise exception_container[0]

                return result_container[0]

        return wrapper

    return decorator


def _run_branch(
    direction: str | None,
    step_n: int,
    use_local: bool,
    idx: int,
    log_root: str,
    log_prefix: str,
    quality_gate_cfg: dict = None,
):
    if log_root:
        branch_name = f"{log_prefix}_{idx:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=None,
        use_local=use_local,
        quality_gate_config=quality_gate_cfg or {},
    )
    model_loop.user_initial_direction = direction
    model_loop.run(step_n=step_n, stop_event=None)


def _run_evolution_task(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    stop_event: threading.Event | None,
    quality_gate_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single evolution task (one small loop).

    Args:
        task: Evolution task descriptor
        directions: List of original directions
        step_n: Steps per round
        use_local: Use local backtest
        user_direction: User initial direction
        log_root: Log root directory
        stop_event: Stop event
        quality_gate_cfg: Quality gate config

    Returns:
        Dict containing trajectory data
    """
    phase = task["phase"]
    direction_id = task["direction_id"]
    strategy_suffix = task.get("strategy_suffix", "")
    round_idx = task["round_idx"]
    parent_trajectories = task.get("parent_trajectories", [])
    # Refine-mode fields (present when the objective vector ``U`` is available;
    # absent on orthogonal mutation tasks, so the loop stays byte-identical to
    # before).
    # See evolution/refine.py + diagnosis.py: a RefinementOperator produces these
    # from a parent's verdict; the loop uses them to freeze the parent's sound
    # layers and refine only the diagnosed weakness (Eq. 6).
    refine_mode = bool(task.get("refine_mode", False))
    refine_directive = task.get("refine_directive") or None
    parent_prefix = task.get("parent_prefix") or None
    # Crossover carries its two parents' expressions instead of a single
    # parent prefix; the loop needs them to apply the both-parent check.
    crossover_parents = task.get("crossover_parents") or None
    refine_factors_block = task.get("refine_factors_block", "")
    revise_hypothesis_block = task.get("revise_hypothesis_block", "")
    # Crossover (Eq. 7): the two parents' validated strengths as constructor
    # inspiration. Empty on refine/orthogonal tasks.
    crossover_strength_block = task.get("crossover_strength_block", "")

    # Per-round deterministic LLM seed: base (CHAT_SEED env, immutable) + round
    # index. Round 0 reproduces the old fixed-seed protocol exactly; later
    # rounds draw from a different but reproducible seed. At CHAT_TEMPERATURE=0
    # this is inert (the provider is deterministic regardless of seed); it only
    # contributes diversity once the user opts in by raising the temperature,
    # and then the run stays reproducible because the seed is a function of the
    # round, not of wall-clock randomness.
    try:
        _chat_seed_base = int(os.environ.get("CHAT_SEED", "42"))
        LLM_SETTINGS.chat_seed = _chat_seed_base + int(round_idx)
    except (TypeError, ValueError):
        pass

    # Resolve direction by phase. Prefer the controller-attached string
    # (task["direction"]) so a reseed-grown direction reaches the loop even
    # though the local `directions` list was frozen at round 0; fall back to
    # that list to keep the controller (which seeds from the same list)
    # byte-identical.
    direction = task.get("direction")
    if direction is None:
        if phase in (RoundPhase.ORIGINAL, RoundPhase.MUTATION):
            direction = directions[direction_id] if direction_id < len(directions) else None
        else:  # CROSSOVER
            direction = None

    trajectory_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
    # Publish it on the task so the ONE id is shared. The controller's
    # create_trajectory_from_loop_result receives this same dict (sequential
    # path: by reference; parallel path: round-tripped as result["task"]) and
    # now reuses this value instead of minting a second one. Without this the
    # library recorded children under one id space and parents under another,
    # and no parent link in the library ever resolved.
    task["trajectory_id"] = trajectory_id
    parent_ids = [p.trajectory_id for p in parent_trajectories] or task.get(
        "parent_trajectory_ids", []
    )

    if log_root:
        branch_name = f"{phase.value}_{round_idx:02d}_{direction_id:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)

    logger.info(
        f"Starting evolution task: phase={phase.value}, round={round_idx}, direction={direction_id}"
    )

    # Create and run loop
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=stop_event,
        use_local=use_local,
        strategy_suffix=strategy_suffix,
        evolution_phase=phase.value,
        trajectory_id=trajectory_id,
        parent_trajectory_ids=parent_ids,
        direction_id=direction_id,
        round_idx=round_idx,
        quality_gate_config=quality_gate_cfg or {},
        refine_mode=refine_mode,
        refine_directive=refine_directive,
        parent_prefix=parent_prefix,
        crossover_parents=crossover_parents,
        refine_factors_block=refine_factors_block,
        revise_hypothesis_block=revise_hypothesis_block,
        crossover_strength_block=crossover_strength_block,
    )
    model_loop.user_initial_direction = user_direction

    # Run one small loop (5 steps)
    model_loop.run(step_n=step_n, stop_event=stop_event)

    traj_data = model_loop._get_trajectory_data()
    traj_data["task"] = task

    return traj_data


def _parallel_task_worker(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    result_queue: Queue,
    task_idx: int,
    quality_gate_cfg: dict[str, Any] | None = None,
):
    """
    Worker for parallel evolution tasks. Runs one evolution task in a separate process and puts result in queue.
    Args: task, directions, step_n, use_local, user_direction, log_root, result_queue, task_idx, quality_gate_cfg.
    """
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS

        RD_AGENT_SETTINGS.use_file_lock = False
        RD_AGENT_SETTINGS.pickle_cache_folder_path_str = str(
            Path(log_root) / f"pickle_cache_{task_idx}"
        )

        traj_data = _run_evolution_task(
            task=task,
            directions=directions,
            step_n=step_n,
            use_local=use_local,
            user_direction=user_direction,
            log_root=log_root,
            stop_event=None,
            quality_gate_cfg=quality_gate_cfg,
        )
        result_queue.put(
            {
                "success": True,
                "task_idx": task_idx,
                "task": task,
                "traj_data": traj_data,
            }
        )
    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "task_idx": task_idx,
                "task": task,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def _serialize_task_for_parallel(task: dict[str, Any]) -> dict[str, Any]:
    """Serialize task for use in child process (parent_trajectories are complex objects)."""
    serialized = task.copy()

    # RoundPhase -> string
    if "phase" in serialized and isinstance(serialized["phase"], RoundPhase):
        serialized["phase"] = serialized["phase"]

    # Convert parent_trajectories to serializable info
    if "parent_trajectories" in serialized:
        serialized["parent_trajectory_ids"] = [
            p.trajectory_id for p in serialized.get("parent_trajectories", [])
        ]
        # Child process does not need full trajectory objects; strategy_suffix has required info
        serialized["parent_trajectories"] = []

    return serialized


def _run_tasks_parallel(
    tasks: list[dict[str, Any]],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    quality_gate_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run multiple evolution tasks in parallel.
    Returns list of results, each with task and traj_data.

    `quality_gate_cfg` must be forwarded here: the serial path passes it, and
    without it the parallel path silently runs with the fallback defaults at
    loop.py:95. That was harmless only while `consistency_enabled` happened to
    be false in YAML; flipping it on would have disabled the check with no
    warning now that `parallel_enabled` defaults to true.
    """
    if not tasks:
        return []

    result_queue = Queue()

    # BOUNDED. This used to start one Process per task with no cap: with
    # num_directions=10 that is 10 concurrent full backtests, each holding its
    # own panel + zoo signals. Measured on this box a single mine process is
    # ~1.5 GB resident, so 10 of them is ~15 GB on a 16 GB machine with swap
    # already near full -- an OOM/freeze, not a slowdown. The cap is the
    # constraint that actually binds (memory), not the core count.
    #
    # QA_MAX_PARALLEL_TASKS overrides. Default: leave a core and ~4 GB for the
    # OS and the parent, and never exceed the task count.
    try:
        import multiprocessing as _mp
        _cores = _mp.cpu_count()
    except Exception:
        _cores = 4
    _mem_gb = 0.0
    try:
        _mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass
    _by_mem = max(1, int((_mem_gb - 4.0) // 1.5)) if _mem_gb else 2
    _default = max(1, min(_cores - 1, _by_mem, 4))
    try:
        max_workers = int(os.environ.get("QA_MAX_PARALLEL_TASKS", _default))
    except ValueError:
        max_workers = _default
    max_workers = max(1, min(max_workers, len(tasks)))

    logger.info(
        f"Starting {len(tasks)} parallel evolution tasks, {max_workers} at a time "
        f"(cores={_cores}, ram={_mem_gb:.0f}GB, ~1.5GB/worker)"
    )

    results = []
    pending = list(enumerate(tasks))
    procs: dict[int, Process] = {}
    collected = 0

    def _launch(idx: int, task: dict) -> None:
        p = Process(
            target=_parallel_task_worker,
            args=(
                _serialize_task_for_parallel(task),
                directions,
                step_n,
                use_local,
                user_direction,
                log_root,
                result_queue,
                idx,
                quality_gate_cfg,
            ),
        )
        p.start()
        procs[idx] = p
        logger.info(
            f"Started task {idx}: phase={task['phase'].value}, "
            f"direction={task['direction_id']} ({len(procs)}/{max_workers} slots)"
        )

    # A slot is freed by the RESULT, not by is_alive(). Reaping on is_alive()
    # deadlocks whenever len(tasks) > max_workers: multiprocessing's queue
    # feeder thread can keep a process "alive" after it has put its result, so
    # the loop sat blocked in get() with a full window and never refilled.
    # Verified on (tasks, cap) = (10,3), (8,2), (5,1): all complete, peak
    # concurrency never exceeds the cap.
    while collected < len(tasks):
        while pending and len(procs) < max_workers:
            idx, task = pending.pop(0)
            _launch(idx, task)

        result = result_queue.get()
        collected += 1
        if result["success"]:
            original_task = tasks[result["task_idx"]]
            result["task"] = original_task
            result["traj_data"]["task"] = original_task
            results.append(result)
            logger.info(f"Task {result['task_idx']} completed")
        else:
            logger.error(f"Task {result['task_idx']} failed: {result['error']}")
            logger.error(result.get("traceback", ""))

        finished = procs.pop(result["task_idx"], None)
        if finished is not None:
            finished.join()

    for p in procs.values():
        p.join()

    logger.info(f"Parallel tasks done: {len(results)}/{len(tasks)} succeeded")

    return results


def run_evolution_loop(
    initial_direction: str | None,
    evolution_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
    stop_event: threading.Event | None = None,
    quality_gate_cfg: dict[str, Any] | None = None,
):
    """
    Run evolution loop: Original -> Mutation -> Crossover -> Mutation -> ...
    Supports parallel execution per phase.
    """
    quality_gate_cfg = quality_gate_cfg or {}
    from quantaalpha.core.conf import RD_AGENT_SETTINGS

    RD_AGENT_SETTINGS.use_file_lock = False
    logger.info("Evolution mode: file lock disabled to avoid deadlock")

    # Parse config
    num_directions = int(planning_cfg.get("num_directions", 2))
    max_rounds = int(evolution_cfg.get("max_rounds", 10))
    crossover_size = int(evolution_cfg.get("crossover_size", 2))
    crossover_n = int(evolution_cfg.get("crossover_n", 3))
    steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
    use_local = bool(exec_cfg.get("use_local", True))

    mutation_enabled = bool(evolution_cfg.get("mutation_enabled", True))
    crossover_enabled = bool(evolution_cfg.get("crossover_enabled", True))
    mutation_top_fraction = float(
        evolution_cfg.get("mutation_top_fraction", 1.0) or 1.0
    )
    parent_selection_strategy = str(
        evolution_cfg.get("parent_selection_strategy", "best")
    )
    top_percent_threshold = float(evolution_cfg.get("top_percent_threshold", 0.3))
    # T6 ADMITTED-PUSH. This was declared on EvolutionConfig and read by
    # ``_prepare_mutation_targets`` but never plumbed through from the YAML, so
    # the knob sat at its 0.0 dataclass default and every admitted winner went
    # crossover-only regardless of the configured fraction.
    admitted_push_fraction = float(
        evolution_cfg.get("admitted_push_fraction", 0.0) or 0.0
    )
    # Same trap: declared on EvolutionConfig, read by the router, but with no
    # path from the YAML. Not currently set in any config, so wiring it changes
    # nothing today -- it just stops a future value from being silently dropped.
    orthogonal_tail_fraction = float(
        evolution_cfg.get("orthogonal_tail_fraction", 0.25) or 0.0
    )
    log_root = str(logger.log_trace_path)
    parallel_enabled = bool(evolution_cfg.get("parallel_enabled", False))
    # QA_SEQUENTIAL_EVOLUTION overrides the config.
    # Feedback only teaches batches generated after it, so batches produced
    # concurrently within a round cannot learn from each other's verdicts --
    # the mechanism the net-of-cost objective depends on is silently disabled
    # by parallelism. Sequential evolution costs wall-clock and buys the signal.
    if os.environ.get("QA_SEQUENTIAL_EVOLUTION", "").lower() in ("1", "true", "yes"):
        if parallel_enabled:
            logger.info(
                "QA_SEQUENTIAL_EVOLUTION set: forcing sequential evolution so each "
                "batch sees every earlier admission verdict"
            )
        parallel_enabled = False
    fresh_start = bool(evolution_cfg.get("fresh_start", True))
    cleanup_on_finish = bool(evolution_cfg.get("cleanup_on_finish", False))

    # Generate initial directions
    planning_enabled = bool(planning_cfg.get("enabled", False))
    prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
    prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)

    if planning_enabled and initial_direction:
        directions = generate_parallel_directions(
            initial_direction=initial_direction,
            n=num_directions,
            prompt_file=prompt_path,
            max_attempts=int(planning_cfg.get("max_attempts", 5)),
            use_llm=bool(planning_cfg.get("use_llm", True)),
            seed_in_generation=bool(planning_cfg.get("seed_in_generation", True)),
        )
    elif planning_enabled:
        directions = [None] * num_directions
    else:
        directions = [initial_direction] if initial_direction else [None]

    logger.info(f"Generated {len(directions)} exploration directions")
    for i, d in enumerate(directions):
        logger.info(f"  Direction {i}: {d}")

    # Pool location. ``log_root`` is a FRESH timestamped directory on every
    # launch, so a pool written there can never be found by a restart -- which
    # silently defeats fresh_start=False (the run reloads nothing and re-mines
    # from round 0). Anchor it to the EXPERIMENT instead, which is stable across
    # restarts by construction: run.sh derives the ledger path from the same id.
    # QA_POOL_PATH overrides for one-off runs.
    _exp = os.environ.get("EXPERIMENT_ID", "")
    if os.environ.get("QA_POOL_PATH"):
        pool_save_path = Path(os.environ["QA_POOL_PATH"])
    elif _exp:
        pool_save_path = Path("data/results") / f"trajectory_pool_{_exp}.json"
    else:
        pool_save_path = Path(log_root) / "trajectory_pool.json"
    pool_save_path.parent.mkdir(parents=True, exist_ok=True)
    mutation_prompt_path = Path(__file__).parent / "prompts" / "evolution_prompts.yaml"
    informed_prompt_path = Path(__file__).parent / "prompts" / "informed_planning_prompts.yaml"

    logger.info(f"Trajectory pool path: {pool_save_path} (fresh_start={fresh_start})")

    config = EvolutionConfig(
        num_directions=len(directions),
        steps_per_loop=steps_per_loop,
        max_rounds=max_rounds,
        mutation_enabled=mutation_enabled,
        crossover_enabled=crossover_enabled,
        crossover_size=crossover_size,
        crossover_n=crossover_n,
        prefer_diverse_crossover=True,
        parent_selection_strategy=parent_selection_strategy,
        mutation_top_fraction=mutation_top_fraction,
        admitted_push_fraction=admitted_push_fraction,
        orthogonal_tail_fraction=orthogonal_tail_fraction,
        top_percent_threshold=top_percent_threshold,
        parallel_enabled=parallel_enabled,
        pool_save_path=str(pool_save_path),
        mutation_prompt_path=str(mutation_prompt_path)
        if mutation_prompt_path.exists()
        else None,
        crossover_prompt_path=str(mutation_prompt_path)
        if mutation_prompt_path.exists()
        else None,
        fresh_start=fresh_start,
        reseed_after_stale_rounds=int(evolution_cfg.get("reseed_after_stale_rounds", 1) or 0),
        growth_floor=int(evolution_cfg.get("growth_floor", 2) or 0),
        reseed_interval=int(evolution_cfg.get("reseed_interval", 3) or 0),
        directions=list(directions),
        initial_direction=initial_direction or "",
        informed_prompt_path=str(informed_prompt_path)
        if informed_prompt_path.exists()
        else None,
        seed_in_generation=bool(planning_cfg.get("seed_in_generation", True)),
    )

    controller = EvolutionController(config)

    logger.info("=" * 60)
    logger.info("Starting evolution loop")
    logger.info(
        f"Config: directions={len(directions)}, max_rounds={max_rounds}, "
        f"crossover_size={crossover_size}, crossover_n={crossover_n}"
    )
    logger.info(
        f"Phases: mutation={'on' if mutation_enabled else 'off'}, "
        f"crossover={'on' if crossover_enabled else 'off'}"
    )
    if mutation_enabled and not crossover_enabled:
        logger.info("Mode: mutation only (Original -> Mutation -> ...)")
    elif crossover_enabled and not mutation_enabled:
        logger.info("Mode: crossover only (Original -> Crossover -> ...)")
    elif mutation_enabled and crossover_enabled:
        logger.info("Mode: full evolution (Original -> Mutation -> Crossover -> ...)")
    else:
        logger.info("Mode: original only (no evolution)")
    logger.info(
        f"Parent selection: {parent_selection_strategy} | "
        f"mutation parents: top {100*mutation_top_fraction:.0f}%"
        + (
            f" (top_percent={top_percent_threshold})"
            if parent_selection_strategy == "top_percent_plus_random"
            else ""
        )
    )
    logger.info(f"Parallel execution: {'on' if parallel_enabled else 'off'}")
    logger.info("=" * 60)

    if parallel_enabled:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            tasks = controller.get_all_tasks_for_current_phase()
            if not tasks:
                logger.info("Evolution complete: no more tasks")
                break

            current_phase = tasks[0]["phase"]
            current_round = tasks[0]["round_idx"]
            logger.info(
                f"Parallel phase: phase={current_phase.value}, round={current_round}, tasks={len(tasks)}"
            )

            results = _run_tasks_parallel(
                tasks=tasks,
                directions=directions,
                step_n=steps_per_loop,
                use_local=use_local,
                user_direction=initial_direction,
                log_root=log_root,
                quality_gate_cfg=quality_gate_cfg,
            )

            completed_tasks = []
            for result in results:
                if result["success"]:
                    task = result["task"]
                    traj_data = result["traj_data"]
                    trajectory = controller.create_trajectory_from_loop_result(
                        task=task,
                        hypothesis=traj_data.get("hypothesis"),
                        experiment=traj_data.get("experiment"),
                        feedback=traj_data.get("feedback"),
                    )
                    controller.report_task_complete(task, trajectory)
                    completed_tasks.append(task)
                    logger.info(
                        f"Trajectory done: {trajectory.trajectory_id}, {_PRIMARY_METRIC}={trajectory.get_primary_metric()}"
                    )

            controller.advance_phase_after_parallel_completion(completed_tasks)

    else:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            task = controller.get_next_task()
            if task is None:
                logger.info("Evolution complete: no more tasks")
                break

            logger.info(
                f"Running task: phase={task['phase'].value}, round={task['round_idx']}, direction={task['direction_id']}"
            )

            try:
                traj_data = _run_evolution_task(
                    task=task,
                    directions=directions,
                    step_n=steps_per_loop,
                    use_local=use_local,
                    user_direction=initial_direction,
                    log_root=log_root,
                    stop_event=stop_event,
                    quality_gate_cfg=quality_gate_cfg,
                )
                trajectory = controller.create_trajectory_from_loop_result(
                    task=task,
                    hypothesis=traj_data.get("hypothesis"),
                    experiment=traj_data.get("experiment"),
                    feedback=traj_data.get("feedback"),
                )
                controller.report_task_complete(task, trajectory)
                logger.info(
                    f"Task done: trajectory_id={trajectory.trajectory_id}, {_PRIMARY_METRIC}={trajectory.get_primary_metric()}"
                )
            except Exception as e:
                logger.error(f"Task failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                continue

    state_path = Path(log_root) / "evolution_state.json"
    controller.save_state(state_path)
    best_trajs = controller.get_best_trajectories(top_n=5)
    logger.info("=" * 60)
    logger.info(f"Evolution complete. Top {len(best_trajs)} trajectories:")
    for i, t in enumerate(best_trajs):
        metric = t.get_primary_metric()
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        logger.info(
            f"  {i + 1}. {t.trajectory_id}: phase={t.phase.value}, {_PRIMARY_METRIC}={metric_str}"
        )
    logger.info(f"Pool stats: {controller.pool.get_statistics()}")
    logger.info("=" * 60)
    if cleanup_on_finish:
        logger.info("Cleaning up trajectory pool file...")
        controller.pool.cleanup_file()


@force_timeout()
def main(
    path=None,
    step_n=100,
    direction=None,
    stop_event=None,
    config_path=None,
    evolution_mode=None,
):
    """
    Autonomous alpha factor mining with optional evolution support.

    Args:
        path: Session path (for resume)
        step_n: Number of steps (default 100 = 20 loops * 5 steps/loop)
        direction: Initial direction
        stop_event: Stop event
        config_path: Run config file path
        evolution_mode: Enable evolution (None=from config, True/False=override)

    Evolution flow: Original -> Mutation -> Crossover -> Mutation -> ...

    You can continue running session by

    .. code-block:: python

        quantaalpha mine --direction "[Initial Direction]" --config_path configs/experiment.yaml

    """
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS

        logger.info("=" * 60)
        logger.info("Experiment config")
        logger.info(f"  Workspace: {RD_AGENT_SETTINGS.workspace_path}")
        logger.info(f"  Cache dir: {RD_AGENT_SETTINGS.pickle_cache_folder_path_str}")
        logger.info(f"  Cache enabled: {RD_AGENT_SETTINGS.cache_with_pickle}")
        logger.info("=" * 60)

        # Config file default: project_root/configs/
        _project_root = Path(__file__).resolve().parents[2]
        config_default = _project_root / "configs" / "experiment.yaml"
        config_file = Path(config_path) if config_path else config_default
        run_cfg = load_run_config(config_file)
        planning_cfg = (
            (run_cfg.get("planning") or {}) if isinstance(run_cfg, dict) else {}
        )
        exec_cfg = (run_cfg.get("execution") or {}) if isinstance(run_cfg, dict) else {}
        evolution_cfg = (
            (run_cfg.get("evolution") or {}) if isinstance(run_cfg, dict) else {}
        )
        quality_gate_cfg = (
            (run_cfg.get("quality_gate") or {}) if isinstance(run_cfg, dict) else {}
        )

        # Export the per-hypothesis factor count so the construction prompt can
        # state it. It reaches proposal.py through the environment because those
        # classes are built deep in the RD-Agent loop and get pickled. Without
        # this the key is inert: nothing else reads it, and the prompt used to
        # hardcode "2-3 Factors per Generation" regardless of what config said.
        factor_cfg = (run_cfg.get("factor") or {}) if isinstance(run_cfg, dict) else {}
        _fph = factor_cfg.get("factors_per_hypothesis")
        if _fph is not None:
            os.environ["QA_FACTORS_PER_HYPOTHESIS"] = str(int(_fph))
            logger.info(f"  Factors per hypothesis: {int(_fph)}")

        if evolution_mode is not None:
            use_evolution = evolution_mode
        else:
            use_evolution = bool(evolution_cfg.get("enabled", False))

        if step_n is None or step_n == 100:
            if exec_cfg.get("step_n") is not None:
                step_n = exec_cfg.get("step_n")
            else:
                max_loops = int(exec_cfg.get("max_loops", 10))
                steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
                step_n = max_loops * steps_per_loop

        use_local = os.getenv("USE_LOCAL", "True").lower()
        use_local = True if use_local in ["true", "1"] else False
        if exec_cfg.get("use_local") is not None:
            use_local = bool(exec_cfg.get("use_local"))
        exec_cfg["use_local"] = use_local

        logger.info(
            f"Use {'Local' if use_local else 'Docker container'} to execute factor backtest"
        )

        if use_evolution and path is None:
            logger.info("=" * 60)
            logger.info("Evolution mode: Original -> Mutation -> Crossover loop")
            logger.info("=" * 60)

            run_evolution_loop(
                initial_direction=direction,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
                quality_gate_cfg=quality_gate_cfg,
            )

        elif path is None:
            planning_enabled = bool(planning_cfg.get("enabled", False))
            n_dirs = int(planning_cfg.get("num_directions", 1))
            max_attempts = int(planning_cfg.get("max_attempts", 5))
            use_llm = bool(planning_cfg.get("use_llm", True))
            prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
            prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
            if planning_enabled and direction:
                directions = generate_parallel_directions(
                    initial_direction=direction,
                    n=n_dirs,
                    prompt_file=prompt_path,
                    max_attempts=max_attempts,
                    use_llm=use_llm,
                    seed_in_generation=bool(planning_cfg.get("seed_in_generation", True)),
                )
            else:
                directions = [direction] if direction else [None]

            log_root = exec_cfg.get("branch_log_root") or "log"
            log_prefix = exec_cfg.get("branch_log_prefix") or "branch"
            use_branch_logs = planning_enabled and len(directions) > 1
            parallel_execution = bool(exec_cfg.get("parallel_execution", False))

            if parallel_execution and len(directions) > 1:
                procs: list[Process] = []
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(
                            f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}"
                        )
                    p = Process(
                        target=_run_branch,
                        args=(
                            dir_text,
                            step_n,
                            use_local,
                            idx,
                            log_root if use_branch_logs else "",
                            log_prefix,
                        ),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
            else:
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(
                            f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}"
                        )
                    if use_branch_logs:
                        branch_name = f"{log_prefix}_{idx:02d}"
                        branch_log = Path(log_root) / branch_name
                        branch_log.mkdir(parents=True, exist_ok=True)
                        logger.set_trace_path(branch_log)
                    model_loop = AlphaAgentLoop(
                        ALPHA_AGENT_FACTOR_PROP_SETTING,
                        potential_direction=dir_text,
                        stop_event=stop_event,
                        use_local=use_local,
                        quality_gate_config=quality_gate_cfg,
                    )
                    model_loop.user_initial_direction = direction
                    model_loop.run(step_n=step_n, stop_event=stop_event)
        else:
            model_loop = AlphaAgentLoop.load(path, use_local=use_local)
            model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
    finally:
        logger.info("Run finished or terminated")


if __name__ == "__main__":
    fire.Fire(main)
