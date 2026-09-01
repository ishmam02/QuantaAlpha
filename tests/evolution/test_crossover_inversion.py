"""Crossover (Eq. 7) must recombine the two best parents' VALIDATED IDEAS,
not a literal equation splice.

The old crossover spliced one parent's signal sub-tree into the other's
temporal construction and shipped the spliced expression as the literal child
("these expressions ARE this round's children -- do not replace them"). The
measured result was that the far parent contributed essentially nothing
(child-vs-far-parent expression similarity 0.543 vs a 0.417 null whose p90 is
0.592 -- below chance). Eq. 7 instead selects the two best parents, locates
the construction decisions each one's measurements VALIDATED, and has the LLM
AUTHOR A NEW hypothesis + fresh expression from those ideas, with the
``check_crossover`` gate verifying the child carries vocabulary distinctive
to each parent.

These tests pin the new behaviour:

X1  build_task_extras ships recombine (not expression), no composed factors,
    crossover_parents populated for the gate.
X2  the crossover_strength_block names BOTH parents' validated strengths and
    expressions, framed as inspiration ("do not copy"), not as the child.
X3  the suffix carries two "what each parent's measurement validated" blocks
    and forbids a literal splice; no old splice/fusion phrases leak.
X4  fewer than two parents -> None (graceful; LLM-only fallback).
X5  the crossover TASK carries crossover_strength_block + crossover_parents
    through sequential and parallel serialization.
X6  the gate fires on crossover_parents regardless of refine_target and
    rejects a single-parent child while a dual-parent child passes.
"""
import os
import re
from dataclasses import dataclass, field

from quantaalpha.pipeline.evolution.crossover import CrossoverOperator
from quantaalpha.pipeline.evolution.operator_contract import (
    check_crossover, check_refine,
)
from quantaalpha.pipeline.evolution.strength_diagnosis import StrengthDirective
from quantaalpha.pipeline.evolution.trajectory import RoundPhase as _RP


@dataclass
class T:
    trajectory_id: str
    hypothesis: str = "h"
    factors: list = field(default_factory=list)
    backtest_metrics: dict = field(default_factory=dict)
    direction_id: int = 0
    phase: object = _RP.ORIGINAL
    feedback: str = ""
    round_idx: int = 0
    parent_trajectory_ids: list = field(default_factory=list)
    expected_ic_sign: str = ""


GAP = T("A", "overnight gap reverses next day",
        [{"name": "Overnight_5D",
          "expression": "RANK(TS_SUM(($open - DELAY($close,1)) / DELAY($close,1), 5))"}],
        backtest_metrics={"U": 1.2, "rank_ic": -0.045, "turnover_book": 0.03,
                          "e_signal": 0.8, "admitted": True, "delta_mean": 0.05,
                          "delta_se": 0.02, "sign_predicted": "negative"},
        expected_ic_sign="negative")
ILLIQ = T("B", "illiquidity premium",
          [{"name": "Amihud_20D",
            "expression": "RANK(TS_MEAN(ABS($close - DELAY($close,1)) / ($volume + 1), 20))"}],
          backtest_metrics={"U": 1.1, "rank_ic": 0.05, "turnover_book": 0.02,
                            "e_signal": 0.7, "admitted": True, "delta_mean": 0.04,
                            "delta_se": 0.02, "sign_predicted": "positive"},
          expected_ic_sign="positive")


def _strength(parent, dim, text):
    return StrengthDirective.from_dict({
        "verdict": "admitted", "strongest_dimension": dim,
        "refine_target": "recombine", "frozen_layers": [],
        "mechanism_hint": "m", "directive_text": text, "targets": [],
        "hypothesis_template": {}, "lineage_segments": {},
        "parent_hypothesis": parent.hypothesis,
        "parent_expected_ic_sign": parent.expected_ic_sign,
        "parent_factors": parent.factors, "raw_metrics": {},
    })


# diagnose-never-prescribe: the strength prose locates the strength and closes
# "yours to determine"; it never states what the child should be.
sA = _strength(GAP, "signal",
               "The measurements locate the strength in the overnight-gap "
               "construction (RANK(TS_SUM(...open - DELAY($close,1)...)): "
               "rank_ic -0.045, stable across seeds. The rank-over-window "
               "pattern recurred in 2 admitted ancestors. How to combine "
               "that is yours to determine.")
sB = _strength(ILLIQ, "signal",
               "The measurements locate the strength in the Amihud-style "
               "illiquidity construction (RANK(TS_MEAN(...$volume...)): "
               "rank_ic +0.050, the lowest turnover_share in the book. A "
               "validated repair on cost produced an admitted factor on this "
               "lineage. How to combine that is yours to determine.")

op = CrossoverOperator()
extras = op.build_task_extras([GAP, ILLIQ], [sA, sB])

# --------------------------------------------------------------------------
# X1: recombine target, no composed factors, crossover_parents for the gate
# --------------------------------------------------------------------------
assert extras is not None, "X1: build_task_extras returned None on two parents"
assert extras["refine_directive"]["refine_target"] == "recombine", (
    "X1: crossover must route as 'recombine', not 'expression' (the splice path)")
assert "composed_factors" not in extras, "X1: composed_factors must be gone"
assert extras["refine_factors_block"] == "", (
    "X1: crossover builds a fresh expression; no frozen-prefix block")
cx = extras["crossover_parents"]
assert cx["a"] == [GAP.factors[0]["expression"]], "X1: parent A expressions missing"
assert cx["b"] == [ILLIQ.factors[0]["expression"]], "X1: parent B expressions missing"
print("X1 PASS  recombine target, no composed factors, crossover_parents populated")

# --------------------------------------------------------------------------
# X2: the strength block names both parents, framed as inspiration not child
# --------------------------------------------------------------------------
block = extras["crossover_strength_block"]
assert block, "X2: crossover_strength_block is empty"
bl = block.lower()
assert "inspiration, not the child" in bl, "X2: must be framed as inspiration"
assert "do not copy" in bl, "X2: must tell the constructor not to copy"
assert "overnight-gap" in block or "overnight gap" in bl, "X2: parent A strength missing"
assert "amihud" in bl or "illiquidity" in bl, "X2: parent B strength missing"
assert GAP.factors[0]["expression"] in block, "X2: parent A expression missing"
assert ILLIQ.factors[0]["expression"] in block, "X2: parent B expression missing"
assert "do not replace them" not in bl and "factors to build" not in bl, (
    "X2: old splice 'these ARE the children' framing leaked")
print("X2 PASS  strength block names both parents, inspiration not child")

# --------------------------------------------------------------------------
# X3: the suffix carries two validated-strength blocks, forbids a splice
# --------------------------------------------------------------------------
suffix = op.generate_crossover_prompt_suffix([GAP, ILLIQ], [sA, sB])
low = re.sub(r"\s+", " ", suffix.lower())
assert low.count("what each parent's measurement validated") >= 1, "X3: strength header missing"
assert "parent 1" in low and "parent 2" in low, "X3: both parent blocks missing"
assert "not a literal splice" in low, "X3: the forbid-splice framing is missing"
for bad in ["mechanically composed", "composed expressions", "fusion direction",
            "fusion_logic", "innovation points", "do not replace them",
            "use or refine", "prefer building on"]:
    assert bad not in low, f"X3: old splice phrase leaked: {bad!r}"
# diagnose-never-prescribe + no market prior
for bad in ["use ", "try ", "should", "lengthen", "simplify", "blend linearly"]:
    assert bad not in low, f"X3: prescription token leaked: {bad!r}"
assert "csi300" not in low and "china" not in low and "mean-revert" not in low, (
    "X3: market prior leaked")
assert "which validated decision it inherits from which parent" in low, (
    "X3: lineage-inheritance instruction missing")
print("X3 PASS  suffix: two strength blocks, forbids splice, no prescription, no market prior")

# --------------------------------------------------------------------------
# X4: fewer than two parents -> None (graceful LLM-only fallback)
# --------------------------------------------------------------------------
assert op.build_task_extras([GAP], [sA]) is None, (
    "X4: a single parent must return None so LLM-only crossover runs")
assert op.build_task_extras([], []) is None, "X4: no parents must return None"
print("X4 PASS  fewer than two parents -> None")

# --------------------------------------------------------------------------
# X6: the gate fires on crossover_parents, rejects single-parent children
# --------------------------------------------------------------------------
a_exprs, b_exprs = cx["a"], cx["b"]
dual = "RANK(TS_MEAN(($open - DELAY($close,1)) / ($volume + 1), 10))"
assert check_crossover(dual, a_exprs, b_exprs).ok, (
    "X6: a child drawing vocabulary from BOTH parents must pass the gate")
# A child whose vocabulary is ENTIRELY parent A's distinctive tokens
# ($open, TS_SUM, 5) and none of parent B's ($volume, TS_MEAN, ABS, +, 20).
one_only = "RANK(TS_SUM($open, 5))"
res_one = check_crossover(one_only, a_exprs, b_exprs)
assert not res_one.ok, (
    f"X6: a child drawing on ONE parent only must be rejected, got {res_one}")
# refine gate still routes for the expression path (unchanged)
assert check_refine("RANK($close)", "RANK($close)").ok is False, (
    "X6: refine gate (literal-only) must still reject an unchanged parent")
print("X6 PASS  gate fires on crossover_parents; single-parent rejected, dual-parent passes")

# --------------------------------------------------------------------------
# X5: the crossover task carries the strengths + parents, serial + parallel
# --------------------------------------------------------------------------
# Avoid the LLM strength diagnosis AND the population machinery in the
# controller by patching ``_cached_strength`` to return the fixed directives
# above (the wiring -- that the SAME directive reaches the suffix and the task
# build -- is what X5 tests, not ``_cached_strength``'s internals).
from quantaalpha.pipeline.evolution.controller import (
    EvolutionConfig, EvolutionController,
)

ctl = EvolutionController(EvolutionConfig(
    num_directions=2, max_rounds=3, fresh_start=True,
    directions=["d1", "d2"]))


def _fake_cached_strength(self, parent):
    return {"A": sA, "B": sB}.get(parent.trajectory_id)


EvolutionController._cached_strength = _fake_cached_strength
for t in (GAP, ILLIQ):
    ctl.pool._trajectories[t.trajectory_id] = t
ctl._crossover_groups = [[GAP, ILLIQ]]
ctl._crossover_idx = 0
ctl._current_phase = _RP.CROSSOVER
ctl._strength_cache = {}

task = ctl._get_crossover_task()
assert task is not None, "X5: no crossover task produced"
assert task.get("crossover_strength_block"), (
    "X5: the task does not carry the strength block -- it would never reach "
    "the constructor")
assert task.get("crossover_parents", {}).get("a"), "X5: crossover_parents missing"
assert task["refine_directive"]["refine_target"] == "recombine", (
    "X5: task routed as the wrong target")
assert "composed_factors" not in task, "X5: composed_factors leaked into the task"
# The strength block in the task is the SAME directive the suffix used (no
# double-diagnose divergence): parent A's strength text appears once.
assert task["crossover_strength_block"].count("overnight-gap") + \
       task["crossover_strength_block"].count("overnight gap") >= 1

# Must survive parallel serialization too.
from quantaalpha.pipeline.factor_mining import _serialize_task_for_parallel
ser = _serialize_task_for_parallel(task)
assert ser.get("crossover_strength_block"), (
    "X5: strength block lost in parallel serialization")
assert ser.get("crossover_parents", {}).get("b"), (
    "X5: crossover_parents lost in parallel serialization")
print("X5 PASS  crossover task carries strengths + parents, serial and parallel")

# --------------------------------------------------------------------------
# X7: crossover_strengths is PERSISTED, not a dead audit key.
#
# build_task_extras emits `crossover_strengths` "for audit". A key nothing
# reads is dead config; _create_trajectory now lands it on
# trajectory.extra_info so the recombination that was actually attempted is
# inspectable after the run.
# --------------------------------------------------------------------------
assert task.get("crossover_strengths"), "X7: the task must carry the directives"

from types import SimpleNamespace
_hyp = SimpleNamespace(hypothesis="child hypothesis", expected_ic_sign="negative")
_exp = SimpleNamespace(
    sub_tasks=[SimpleNamespace(factor_name="C", factor_description="d",
                               factor_expression="RANK($close)")],
    based_experiments=[], result=None)
_fb = SimpleNamespace(decision=True, reason="", observations="",
                      hypothesis_evaluation="", new_hypothesis="")
task_for_traj = dict(task)
task_for_traj["trajectory_id"] = "CXCHILD"
traj = ctl.create_trajectory_from_loop_result(task_for_traj, _hyp, _exp, _fb)
assert traj.extra_info.get("crossover_strengths"), (
    "X7: crossover_strengths must be persisted to trajectory.extra_info, "
    "otherwise the 'audit record' is a key nothing ever reads")
assert len(traj.extra_info["crossover_strengths"]) == 2, (
    "X7: both parents' directives must be persisted")
# And the crossover's own refine_actions entry must NOT be read back as a
# validated repair (a recombination repaired no diagnosed weakness).
from quantaalpha.pipeline.evolution.strength_diagnosis import repair_actions_summary
traj.backtest_metrics = {"U": 1.0, "admitted": True}
assert repair_actions_summary(traj, []) == [], (
    "X7: a crossover child must not surface as a validated repair action")
print("X7 PASS  crossover_strengths persisted; crossover is not a 'repair'")

print("\nALL PASS")