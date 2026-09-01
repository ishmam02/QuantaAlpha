"""The search must remember what it already measured.

M1  the "what already FAILED" memory must survive non-blocking admission
M2  ORIGINAL (reseed) rounds must be generated WITH memory, not blind

Both fail against the previous behaviour:
  - M1 because _is_admitted read the `admitted` flag, which non-blocking
    admission sets True for almost everything, silently emptying the section.
  - M2 because ORIGINAL tasks shipped `"strategy_suffix": ""` with the comment
    "No guidance for original".
"""
from dataclasses import dataclass, field

from quantaalpha.pipeline.evolution.controller import EvolutionConfig, EvolutionController
from quantaalpha.pipeline.evolution.trajectory import RoundPhase


@dataclass
class T:
    trajectory_id: str
    hypothesis: str = "h"
    factors: list = field(default_factory=lambda: [{"expression": "RANK($close)"}])
    backtest_metrics: dict = field(default_factory=dict)


def make_controller(trajs):
    ctl = EvolutionController(EvolutionConfig(
        num_directions=2, max_rounds=3, fresh_start=True,
        directions=["direction one", "direction two"]))
    for t in trajs:
        ctl.pool._trajectories[t.trajectory_id] = t
    return ctl


# --------------------------------------------------------------------------
# M1: memory survives non-blocking admission
# --------------------------------------------------------------------------
# This is what the pool looks like under `blocking: false` -- everything has
# admitted=True, and only the VERDICT distinguishes a winner from a failure.
trajs = [
    T("win", backtest_metrics={
        "U": 0.6, "admitted": True, "verdict": "admitted", "rank_ic_own": 0.031, "rank_ic": 0.028}),
    T("bad", hypothesis="momentum continues",
      factors=[{"expression": "RANK(TS_MEAN($close,20))"}],
      backtest_metrics={
          "U": 0.2, "admitted": True, "verdict": "net_harmful", "rank_ic_own": -0.028, "rank_ic": 0.028}),
    T("meh", hypothesis="volume spike predicts drift",
      factors=[{"expression": "RANK($volume)"}],
      backtest_metrics={
          "U": 0.3, "admitted": True, "verdict": "marginal", "rank_ic_own": -0.004, "rank_ic": 0.028}),
]
ctx = make_controller(trajs)._build_zoo_context()

assert "already FAILED" in ctx, (
    "M1: the failure memory vanished. Under non-blocking admission every batch "
    "carries admitted=True, so a flag-based split empties this section -- which "
    "is the only place the signed-RankIC lesson survives across batches.")
assert "net_harmful" in ctx and "marginal" in ctx, "M1: verdicts must be shown"
assert "-0.0280" in ctx, "M1: the SIGNED RankIC is the lesson; it must be shown"
# ...and it must be the FACTOR'S sign, never the composite book's. Every fixture
# above carries rank_ic=+0.028 (the book) alongside a negative rank_ic_own (the
# factor). Rendering the book number is what made 43 of 43 values shown to the
# model positive while the factors were negative 71% of the time, directly under
# the instruction "Read the RankIC SIGN as evidence".
assert "0.0280" in ctx and "+0.0280" not in ctx, (
    "M1b: the composite book RankIC leaked into the failure memory")
assert "RANK(TS_MEAN($close,20))" in ctx, "M1: the failed expression must be shown"
# And the winner must NOT be filed under failures.
failed_block = ctx.split("already FAILED")[1]
assert "RANK($close)" not in failed_block, "M1: an admitted factor leaked into failures"
print("M1 PASS  failure memory survives non-blocking admission (verdict-based split)")


# --------------------------------------------------------------------------
# M2: ORIGINAL rounds are generated WITH memory
# --------------------------------------------------------------------------
ctl = make_controller(trajs)

seq = ctl._get_original_task()
assert seq is not None and seq["phase"] == RoundPhase.ORIGINAL
assert seq["strategy_suffix"], (
    "M2 (sequential): ORIGINAL shipped an empty suffix -- reseed re-entered "
    "exploration with no memory of anything already measured.")
assert "already FAILED" in seq["strategy_suffix"], "M2: failures must reach ORIGINAL"

ctl2 = make_controller(trajs)
ctl2._current_phase = RoundPhase.ORIGINAL
par = [t for t in ctl2.get_all_tasks_for_current_phase()
       if t.get("phase") == RoundPhase.ORIGINAL]
assert par, "M2: no ORIGINAL tasks emitted on the parallel path"
assert all(t["strategy_suffix"] for t in par), (
    "M2 (parallel): at least one ORIGINAL task still ships an empty suffix")
print("M2 PASS  ORIGINAL/reseed rounds carry memory on both paths")

print("\nALL PASS")
