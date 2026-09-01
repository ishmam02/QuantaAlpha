"""A restarted mine must CONTINUE, not re-mine what the pool already holds.

The pool is persisted (``TrajectoryPool._save``) and reloads under
``fresh_start=False``, but the controller's round/phase state block is
constructed fresh every time: ``_current_round = 0``, ``_current_phase =
ORIGINAL``, ``_directions_completed = set()``. So before ``_restore_progress``
a "resumed" run reloaded 63 trajectories and then handed out ORIGINAL round 0
for direction 0 -- re-mining every original round, at ~6 min a batch, before
touching the work it was resumed to continue.

R1  fresh_start=True still ignores the pool entirely (no behaviour change)
R2  fresh_start=False restores round, phase and completed directions
R3  the next task CONTINUES (a later round) instead of restarting ORIGINAL
R4  an empty pool under fresh_start=False is harmless (first launch)
R5  restore is idempotent -- constructing twice gives the same state

R3 also covers the budget guard: the phase getters call each other on every
transition and none re-checked the round budget, so a resumed controller with
no minable parents recursed to round 1116+ with max_rounds=15 AND
QA_MAX_ROUNDS_CAP=60 both set. With the guard the chain terminates at the cap.
"""
import json
import os
import tempfile
from pathlib import Path

# run.sh sets these; without them _rounds_exhausted takes the "no target -> stop"
# branch and a resumed controller correctly returns None instead of a task, so
# R3 would test the wrong thing. Set them BEFORE the controller is constructed.
os.environ.setdefault("QA_TARGET_MINED", "150")
os.environ.setdefault("QA_MAX_ROUNDS_CAP", "60")

from quantaalpha.pipeline.evolution.controller import (
    EvolutionConfig, EvolutionController,
)
from quantaalpha.pipeline.evolution.trajectory import RoundPhase

ROOT = Path(__file__).resolve().parents[2]
DIRS = [f"direction {i}" for i in range(4)]


def _pool_file(trajectories):
    """Write a minimal but schema-real pool file."""
    data = {"trajectories": {}, "by_direction": {}, "by_phase": {},
            "saved_at": "2026-08-24T06:20:33"}
    for t in trajectories:
        data["trajectories"][t["trajectory_id"]] = {
            "trajectory_id": t["trajectory_id"],
            "direction_id": t["direction_id"],
            "round_idx": t["round_idx"],
            "phase": t["phase"],
            "hypothesis": "h",
            "expected_ic_sign": "negative",
            "hypothesis_details": {},
            "factors": [{"name": "F", "expression": "RANK($close)"}],
            "backtest_result": None,
            "backtest_metrics": {"U": 1.0, "admitted": True, "rank_ic": 0.02,
                                 "delta_mean": 0.01, "delta_se": 0.004},
            "feedback": "", "feedback_details": {},
            "parent_ids": [], "refine_actions": [], "extra_info": {},
            "created_at": t.get("created_at", f"2026-08-24T0{t['round_idx']}:00:00"),
        }
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, fh)
    fh.close()
    return fh.name


POOL = _pool_file([
    {"trajectory_id": "t0", "direction_id": 0, "round_idx": 0, "phase": "original"},
    {"trajectory_id": "t1", "direction_id": 1, "round_idx": 0, "phase": "original"},
    {"trajectory_id": "t2", "direction_id": 2, "round_idx": 0, "phase": "original"},
    {"trajectory_id": "t3", "direction_id": 3, "round_idx": 0, "phase": "original"},
    {"trajectory_id": "m1", "direction_id": 0, "round_idx": 1, "phase": "mutation"},
    {"trajectory_id": "m2", "direction_id": 1, "round_idx": 2, "phase": "mutation",
     "created_at": "2026-08-24T09:00:00"},
])


def _cfg(fresh):
    return EvolutionConfig(num_directions=4, max_rounds=15, fresh_start=fresh,
                           directions=list(DIRS), pool_save_path=POOL)


# ---------------------------------------------------------------------------
# R1 -- fresh_start=True is unchanged: the pool is ignored, state is zero.
# ---------------------------------------------------------------------------
c_fresh = EvolutionController(_cfg(True))
assert len(c_fresh.pool._trajectories) == 0, "R1: fresh_start must ignore the pool"
assert c_fresh._current_round == 0, f"R1: round {c_fresh._current_round}"
assert c_fresh._current_phase is RoundPhase.ORIGINAL
assert c_fresh._directions_completed == set()
print("R1 PASS  fresh_start=True ignores the pool (unchanged behaviour)")

# ---------------------------------------------------------------------------
# R2 -- fresh_start=False restores round, phase and completed directions.
# ---------------------------------------------------------------------------
c = EvolutionController(_cfg(False))
assert len(c.pool._trajectories) == 6, f"R2: loaded {len(c.pool._trajectories)}"
assert c._current_round == 2, f"R2: round must be the pool max (2); got {c._current_round}"
assert c._directions_completed == {0, 1, 2, 3}, (
    f"R2: all 4 directions have an ORIGINAL trajectory; got {c._directions_completed}")
assert c._current_phase is RoundPhase.MUTATION, (
    f"R2: latest trajectory is a mutation; got {c._current_phase}")
print(f"R2 PASS  restored round={c._current_round} phase={c._current_phase.value} "
      f"completed={sorted(c._directions_completed)}")

# ---------------------------------------------------------------------------
# R3 -- the resumed run must NOT restart at ORIGINAL round 0.
#
# This is the whole point of the restore. The assertion is deliberately about
# what must NOT happen rather than about a specific next task: with a synthetic
# pool the mutation/crossover paths legitimately find no minable parents and
# return None (and, since the budget-guard fix, they return it promptly instead
# of recursing forever). Either outcome is correct; re-mining round 0 is not.
# The real-pool case is exercised in the resume smoke below.
# ---------------------------------------------------------------------------
task = c.get_next_task()
if task is not None:
    assert not (task["phase"] is RoundPhase.ORIGINAL and task["round_idx"] == 0), (
        "R3: resumed run restarted at ORIGINAL round 0 -- it is re-mining the pool")
    assert task["round_idx"] >= c._current_round, (
        f"R3: task round {task['round_idx']} is behind the restored "
        f"round {c._current_round}")
    print(f"R3 PASS  next task continues: phase={task['phase'].value} "
          f"round={task['round_idx']}")
else:
    # No task, but the state must still be the RESTORED state -- not a rewind.
    assert c._current_round >= 2 and c._directions_completed == {0, 1, 2, 3}, (
        "R3: returned None AND lost the restored state")
    print(f"R3 PASS  no minable parent in the synthetic pool; state stayed at "
          f"round {c._current_round} (did not rewind to 0)")

# ---------------------------------------------------------------------------
# R4 -- an empty pool under fresh_start=False must not crash (first launch).
# ---------------------------------------------------------------------------
empty = _pool_file([])
c_empty = EvolutionController(
    EvolutionConfig(num_directions=4, max_rounds=15, fresh_start=False,
                    directions=list(DIRS), pool_save_path=empty))
assert c_empty._current_round == 0 and c_empty._directions_completed == set()
print("R4 PASS  empty pool + fresh_start=False is a normal first launch")

# ---------------------------------------------------------------------------
# R5 -- restore is deterministic. Compare two FRESHLY CONSTRUCTED controllers:
# `c` has since been driven by get_next_task, which advances round/phase by
# design, so comparing against it would test the driving, not the restore.
# ---------------------------------------------------------------------------
a1 = EvolutionController(_cfg(False))
a2 = EvolutionController(_cfg(False))
assert (a1._current_round, a1._current_phase, a1._directions_completed) == \
       (a2._current_round, a2._current_phase, a2._directions_completed), (
    "R5: restore is not deterministic")
assert a1._current_round == 2 and a1._current_phase is RoundPhase.MUTATION, (
    f"R5: restore drifted: round={a1._current_round} phase={a1._current_phase}")
print("R5 PASS  restore is idempotent and deterministic")

print("\nALL PASS")
