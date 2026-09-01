"""The first-class deterministic sign-flip refine (the measured-direction fix).

Measured on the 2026-08-22 smoke: every sign-mismatch reject that cleared the
significance bar (|t| 3.03 / 4.24 / 3.59) was routed to ORTHOGONAL and lost --
the construction had a real edge, the hypothesis just labeled its direction
backwards, and the search threw the edge away instead of correcting it. This is
the most literal "learn from the measured mistake": the search measured that its
prediction was backwards and corrects it for free, re-testing the IDENTICAL
construction under the corrected direction.

The fix has three coupled pieces, each guarded here:

  E   the root ``NumberNode.__str__`` renders integral floats as INTS so a
      re-rendered AST (crossover child, ablation window-sweep, sign-flip freeze)
      executes on a real calculator (``.rolling(5.0)`` raises ``TypeError``).
      This is the historical "crossover 0/10 admits" cause.
  F   ``sign_flip_directive`` fires ONLY on a sign mismatch whose edge clears
      the bar (|t_nw| >= 3.0); produces a SIGN directive that is a refinement
      (``is_refinement`` True, ``exhausted_lever`` False -- a backwards label is
      never an exhausted lever), freezes the expression, and corrects the
      direction to the measured one. ``diagnose_parent`` short-circuits to it
      BEFORE the LLM path (so an LLM ``exhausted=true`` cannot suppress it).
  G   the proposal.py short-circuits build the child with NO LLM call: ``gen``
      returns the frozen premise + the measured correction + the corrected
      sign; ``convert`` rebuilds the experiment from the parent factors VERBATIM
      (the construction is frozen, only the direction was wrong).
"""
import math
import os

# ---------------------------------------------------------------------------
# E -- the root NumberNode fix (covers crossover + ablation + sign-flip freeze).
# ---------------------------------------------------------------------------
from quantaalpha.factors.coder import factor_ast

_n = factor_ast.NumberNode


def _s(v):
    return str(_n(v))


assert _s(5.0) == "5", f"E1: integral float must render as int, got {_s(5.0)!r}"
assert _s(-3.0) == "-3", f"E1: negative integral float must render as int, got {_s(-3.0)!r}"
assert _s(0.0) == "0", f"E1: zero must render as int, got {_s(0.0)!r}"
# Non-integral literals are UNCHANGED (1e-12, 0.5) -- only integral floats get
# the int treatment, so fractional windows/coefficients survive.
assert _s(1e-12) == "1e-12", f"E2: non-integral float must be unchanged, got {_s(1e-12)!r}"
assert _s(0.5) == "0.5", f"E2: fractional float must be unchanged, got {_s(0.5)!r}"
assert _s(-0.25) == "-0.25", f"E2: negative fractional float must be unchanged, got {_s(-0.25)!r}"
# NaN/inf fall through to str(v) (the v==v guard skips them).
assert _s(float("nan")) == "nan", f"E2: NaN must fall through, got {_s(float('nan'))!r}"
# The debug tree view still shows the raw float.
assert _n(5.0)._node_str() == "NUM(5.0)", "E2: _node_str must still show the raw float"
# End-to-end: a re-rendered AST with an integral window is executable text.
_rt = str(factor_ast.parse_expression("TS_MEAN($close, 5)"))
assert "5)" in _rt and "5.0)" not in _rt, f"E3: re-rendered AST leaked a float window: {_rt!r}"
print("E1 PASS  NumberNode renders integral floats as ints (5.0 -> '5')")
print("E2 PASS  non-integral / NaN / inf literals are unchanged; _node_str shows the raw float")
print("E3 PASS  a re-rendered AST emits an executable integral window")

# ---------------------------------------------------------------------------
# F -- sign_flip_directive fires on a sign mismatch whose edge clears the bar.
# ---------------------------------------------------------------------------
from quantaalpha.pipeline.evolution.diagnosis import (
    RefineTarget, sign_flip_directive, _SIGN_FLIP_T_BAR,
)


class _FakeParent:
    def __init__(self, hyp="premise", factors=None):
        self.hypothesis = hyp
        self.factors = factors if factors is not None else [
            {"name": "f1", "expression": "TS_MEAN($close, 5)"}]


_MISMATCH = {"sign_predicted": "positive", "sign_realized": "negative",
             "t_nw": 4.24, "rank_ic": -0.03, "RankIC": -0.03, "U": 0.1}

d = sign_flip_directive(_FakeParent(), _MISMATCH)
assert d is not None, "F1: a sign mismatch clearing the bar must fire"
assert d.refine_target is RefineTarget.SIGN, "F1: the target must be SIGN"
assert d.parent_expected_ic_sign == "negative", \
    "F1: the direction is CORRECTED to the measured (realized) sign"
assert d.frozen_layers == ["expression"], "F1: the expression is frozen verbatim"
assert d.exhausted_lever is False, "F1: a backwards label is never an exhausted lever"
assert d.is_refinement() is True, "F1: a sign-flip is a refinement (routes to REFINE, not ORTHOGONAL)"
assert d.parent_factors == [{"name": "f1", "expression": "TS_MEAN($close, 5)"}], \
    "F1: the parent factors are carried verbatim for the frozen re-test"
# Measurement-only: the directive states the measured directions + the
# correction, never a remedy, and closes with the standing "yours to determine".
assert "positive" in d.directive_text and "negative" in d.directive_text, \
    "F1: the directive must state both measured directions"
assert "yours to determine" in d.directive_text, \
    "F1: the directive must close measurement-only (no remedy)"
assert d.parent_hypothesis == "premise", "F1: the parent hypothesis is carried"
print("F1 PASS  sign mismatch + |t|>=bar -> SIGN directive, direction corrected, expression frozen")

# The bar is the multiple-testing significance level (Harvey-Liu-Zhu 3.0).
assert _SIGN_FLIP_T_BAR == 3.0, "F1: the bar mirrors admission.k_sigma"
assert abs(float(_MISMATCH["t_nw"])) >= _SIGN_FLIP_T_BAR, "F1: the firing case clears the bar"

# F2 -- negative cases return None (fall through to the LLM/table diagnosis).
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "sign_realized": "positive", "t_nw": 4.24}) is None, \
    "F2: a sign MATCH must not fire"
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "sign_realized": "negative", "t_nw": 1.0}) is None, \
    "F2: |t| below the bar must not fire (the edge is not real)"
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "t_nw": 4.24}) is None, \
    "F2: a missing realized sign must not fire"
assert sign_flip_directive(_FakeParent(), {
    "sign_realized": "negative", "t_nw": 4.24}) is None, \
    "F2: a missing predicted sign must not fire"
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "sign_realized": "negative", "t_nw": None}) is None, \
    "F2: a missing t_nw must not fire"
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "sign_realized": "negative", "t_nw": float("nan")}) is None, \
    "F2: a NaN t_nw must not fire"
assert sign_flip_directive(_FakeParent(), {
    "sign_predicted": "positive", "sign_realized": "NEGATIVE", "t_nw": 3.03}) is not None, \
    "F2: sign strings are case-insensitive"
print("F2 PASS  None on sign match / |t|<bar / absent signs / NaN t (falls through to LLM)")

# F3 -- the directive survives a dict round-trip (task serialization).
_rt = d.to_dict()
assert _rt["refine_target"] == "sign", "F3: refine_target serializes by value"
assert _rt["parent_expected_ic_sign"] == "negative", "F3: corrected sign survives"
assert _rt["exhausted_lever"] is False, "F3: exhausted_lever survives"
from quantaalpha.pipeline.evolution.diagnosis import RefinementDirective
_d2 = RefinementDirective.from_dict(_rt)
assert _d2.refine_target is RefineTarget.SIGN, "F3: from_dict restores SIGN"
assert _d2.is_refinement() is True, "F3: still a refinement after round-trip"
assert _d2.parent_expected_ic_sign == "negative", "F3: corrected sign restored"
print("F3 PASS  the SIGN directive survives to_dict / from_dict")

# F4 -- diagnose_parent short-circuits to the sign-flip BEFORE the LLM path, and
# does NOT fire on a sign match / no U. Hermetic: no credentials, so the LLM
# path would raise/fallback -- the short-circuit must return before reaching it.
from quantaalpha.pipeline.evolution.refine import RefinementOperator
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory

_op = RefinementOperator()


def _traj(metrics):
    return StrategyTrajectory(
        trajectory_id="p", direction_id="d", round_idx=0, phase="x",
        hypothesis="premise",
        factors=[{"name": "f1", "expression": "TS_MEAN($close, 5)"}],
        backtest_metrics=metrics)


_mismatch_traj = _traj(_MISMATCH)
_d = _op.diagnose_parent(_mismatch_traj)
assert _d is not None and _d.refine_target is RefineTarget.SIGN, \
    "F4: a sign-mismatch parent short-circuits to SIGN via diagnose_parent"
assert _d.parent_expected_ic_sign == "negative", \
    "F4: the short-circuit corrects the direction"
# A sign MATCH must not short-circuit -- it falls through to the LLM/table path.
# (With no credentials the LLM falls back to the table; the point is it is NOT a
# SIGN directive.)
_match_d = _op.diagnose_parent(_traj({
    "sign_predicted": "positive", "sign_realized": "positive", "t_nw": 4.24, "U": 0.1}))
assert not (_match_d is not None and _match_d.refine_target is RefineTarget.SIGN), \
    "F4: a sign match must not produce a SIGN directive"
# No U -> None (nothing to diagnose; the sign/mechanism gate never ran).
assert _op.diagnose_parent(_traj({})) is None, \
    "F4: a parent with no U returns None (no diagnosis, no sign-flip)"
print("F4 PASS  diagnose_parent short-circuits to SIGN on mismatch, not on match / no-U")

# ---------------------------------------------------------------------------
# G -- the proposal.py short-circuits build the child with NO LLM call.
# ---------------------------------------------------------------------------
from quantaalpha.factors.proposal import (
    AlphaAgentHypothesisGen, AlphaAgentHypothesis2FactorExpression,
)
from quantaalpha.core.proposal import Trace

# G1 -- gen() returns the frozen premise + measured correction + corrected sign.
_gen = AlphaAgentHypothesisGen.__new__(AlphaAgentHypothesisGen)
_gen.refine_directive = {
    "refine_target": "sign",
    "parent_hypothesis": "premise A",
    "parent_expected_ic_sign": "negative",
    "directive_text": ("MEASUREMENT: predicted positive, realized negative (|t| 4.24). "
                       "The construction is RETAINED unchanged; the pre-registered "
                       "direction is CORRECTED to negative. How to act on that is "
                       "yours to determine."),
}
_gen.parent_prefix = {"hypothesis": "premise A", "expected_ic_sign": "negative"}
_h = _gen.gen(Trace(scen=None))
assert _h.expected_ic_sign == "negative", "G1: the child carries the CORRECTED sign"
assert _h.hypothesis.startswith("premise A"), "G1: the parent premise is preserved"
assert "corrected to negative" in _h.hypothesis.lower(), \
    "G1: the measured correction is appended to the premise"
assert "yours to determine" in _h.hypothesis.lower(), \
    "G1: the correction stays measurement-only"
print("G1 PASS  gen() SIGN: frozen premise + measured correction + corrected sign, no LLM")

# G2 -- convert() rebuilds the experiment from the parent factors VERBATIM.
_fc = AlphaAgentHypothesis2FactorExpression.__new__(AlphaAgentHypothesis2FactorExpression)
_fc.refine_directive = {
    "refine_target": "sign",
    "parent_factors": [{"name": "f1", "expression": "TS_MEAN($close, 5)",
                        "description": "d", "formulation": "f", "variables": ""}],
}
_fc.parent_prefix = {"factors": [{"name": "f1", "expression": "TS_MEAN($close, 5)"}]}


class _FakeHyp:
    hypothesis = "premise A"

    def __str__(self):
        return "premise A"


_tr = Trace(scen=None)
_exp = _fc.convert(_FakeHyp(), _tr)
assert len(_exp.tasks) == 1, "G2: the frozen factor is carried as exactly one task"
assert _exp.tasks[0].factor_expression == "TS_MEAN($close, 5)", \
    "G2: the expression is frozen VERBATIM (the construction is sound)"
assert _exp.tasks[0].factor_name == "f1", "G2: the factor name is carried"
assert getattr(_exp, "hypothesis", None) is not None, \
    "G2: the hypothesis is attached (the mechanism gate reads it)"
# The frozen expression is executable: an integral window renders as an int, so
# a real calculator's .rolling(window) will accept it (the E fix at the root).
assert "5.0)" not in _exp.tasks[0].factor_expression, \
    "G2: the frozen expression carries no float-integral window"
print("G2 PASS  convert() SIGN: experiment rebuilt from parent factors verbatim, no LLM")

print("\nALL PASS")