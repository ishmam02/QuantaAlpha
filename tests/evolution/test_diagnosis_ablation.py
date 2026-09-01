"""The ablation routes the refine on the BROKEN part, and the directive carries it.

Two coupled guarantees from the per-segment ablation plan (Part B5/B3):

  * **Route on the broken part (Q2 window-trap backstop).** A decay weakness with
    an IC-neutral window must NOT hand the model a window to move -- the edge is
    not in the window, so the hint points at the CORE. A cost/turnover weakness
    with a HEALTHY core is the case the temporal lever is made for, so the hint
    DOES target the window and states the core's measured health.
  * **The directive carries the ablation.** ``diagnose_parent`` with an
    ``ablation_eval`` threads the ``SegmentAblation.summary`` onto
    ``directive.ablation_summary`` (both the LLM and table paths), and it
    survives ``to_dict``/``from_dict`` (the T5 lineage round-trip). With the eval
    absent (the flag-off default) the summary is ``""`` -- the diagnosis is
    byte-identical to the pre-ablation path (no regression).

A3  _build_target routes decay->core (IC-neutral backstop) and turnover->window (healthy core)
A4  the directive carries ablation_summary through diagnose_parent and the dict round-trip;
    with no ablation_eval the summary is "" (the no-regression default)
"""
import os
import re
from types import SimpleNamespace

import pandas as pd

from quantaalpha.core.verdict import Verdict
from quantaalpha.pipeline.evolution.diagnosis import (
    RefinementDirective, _build_target,
)
from quantaalpha.pipeline.evolution.segment_ablation import ablate, SegmentAblation
from quantaalpha.pipeline.evolution.segment_profiling import build_segments
from quantaalpha.pipeline.evolution.refine import RefinementOperator as RefineOp

TARGET = "ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))"

# Real segment profiles for the target expression (the router walks these to
# locate the temporal op + its signature; the signature must match the
# ablation's window_sensitivity key, which it does because both use _signature).
_PARENT = SimpleNamespace(
    factors=[{"name": "f0", "expression": TARGET}],
    backtest_metrics={}, extra_info={})
_SEGS = build_segments(_PARENT)


def _eval(sub):  # the handle IS the rendered sub-expression
    return sub


# --- Mock A: unstable core, IC-neutral window (the decay-backstop scenario) ---
_RIC_UNSTABLE = pd.Series([-0.02] * 10 + [0.02] * 10)  # halves disagree -> unstable


def _score_unstable(h):
    s = str(h)
    if "TS_MEAN" in s and "ZSCORE" in s:  # full expr / window variants -> flat (IC-neutral)
        return {"rank_ic": 0.030, "t_nw": 2.0, "ic_pos_frac": 0.55,
                "monotonicity": float("nan"), "turnover_solo": 0.50, "ric_series": _RIC_UNSTABLE}
    if "TS_MEAN" in s:
        return {"rank_ic": 0.028, "t_nw": 1.8, "ic_pos_frac": 0.54,
                "monotonicity": float("nan"), "turnover_solo": 0.45, "ric_series": _RIC_UNSTABLE}
    return {"rank_ic": 0.030, "t_nw": 3.2, "ic_pos_frac": 0.50,
            "monotonicity": float("nan"), "turnover_solo": 0.80, "ric_series": _RIC_UNSTABLE}


_ABL_UNSTABLE = ablate(TARGET, "positive", eval_signal=_eval, score=_score_unstable)

# --- Mock B: healthy stable core, IC-SENSITIVE window (the turnover-lever scenario) ---
_RIC_STABLE = pd.Series([0.03] * 20)  # all positive -> stable
_WIN_RIC = {1.0: 0.010, 3.0: 0.020, 5.0: 0.030, 10.0: 0.025, 20.0: 0.015}


def _window_of(s):
    m = re.search(r"TS_MEAN\(.+?,\s*([\d.]+)\)", s)
    return float(m.group(1)) if m else 5.0


def _score_healthy(h):
    s = str(h)
    if "TS_MEAN" in s and "ZSCORE" in s:  # window variant -> rank_ic VARIES (IC-sensitive)
        return {"rank_ic": _WIN_RIC.get(_window_of(s), 0.020), "t_nw": 2.5,
                "ic_pos_frac": 0.7, "monotonicity": float("nan"),
                "turnover_solo": 0.50, "ric_series": _RIC_STABLE}
    if "TS_MEAN" in s:
        return {"rank_ic": 0.030, "t_nw": 3.0, "ic_pos_frac": 0.7,
                "monotonicity": float("nan"), "turnover_solo": 0.45, "ric_series": _RIC_STABLE}
    return {"rank_ic": 0.030, "t_nw": 4.0, "ic_pos_frac": 0.75,
            "monotonicity": float("nan"), "turnover_solo": 0.80, "ric_series": _RIC_STABLE}


_ABL_HEALTHY = ablate(TARGET, "positive", eval_signal=_eval, score=_score_healthy)

# Sanity: the two mocks realize the two regimes the router branches on.
assert next(iter(_ABL_UNSTABLE.window_sensitivity.values()))["ic_neutral"] is True
assert next(iter(_ABL_HEALTHY.window_sensitivity.values()))["ic_neutral"] is False
assert _ABL_UNSTABLE.core_sign_stability["stable"] is False
assert _ABL_HEALTHY.core_sign_stability["stable"] is True

# ---------------------------------------------------------------------------
# A3a -- decay + IC-neutral window + unstable core -> the backstop points at the
# core and does NOT hand the model a window to move.
# ---------------------------------------------------------------------------
_decay = _build_target("decay", 0.10, Verdict.MARGINAL, _SEGS, {}, _ABL_UNSTABLE)
_dh = _decay["mechanism_hint"]
assert "IC-neutral" in _dh, f"A3: decay hint should name the IC-neutral window: {_dh}"
assert "core" in _dh, f"A3: decay hint should point at the core: {_dh}"
assert "slowest-moving" not in _dh, f"A3: decay backstop must NOT frame the window as the lever: {_dh}"
# The structured target follows the hint: it points at the CORE (the core's
# signature, cleared op/param), NOT the window we refused to move -- so T5
# lineage tracks the core lever, not the window.
_core_sig = _ABL_UNSTABLE.core_sign_stability["core_signature"]
assert _decay["subtree_signature"] == _core_sig, (
    f"A3: backstop must record the CORE signature, got {_decay['subtree_signature']!r}")
assert _decay["op"] is None and _decay["param"] is None, (
    f"A3: backstop must clear op/param (no window to target), got op={_decay['op']!r} param={_decay['param']!r}")
print("A3a PASS  decay + IC-neutral window -> backstop points at the core (struct + hint), not the window")

# ---------------------------------------------------------------------------
# A3b -- turnover (cost) + healthy stable core -> the temporal lever IS the
# target, and the hint states the core's measured health.
# ---------------------------------------------------------------------------
_cost = _build_target("turnover", 0.10, Verdict.MARGINAL, _SEGS, {}, _ABL_HEALTHY)
_ch = _cost["mechanism_hint"]
assert "fastest-moving" in _ch, f"A3: cost hint should target the fastest window: {_ch}"
assert "healthy" in _ch, f"A3: cost hint should state the healthy core: {_ch}"
assert "TS_MEAN" in _ch, f"A3: cost hint should name the located op: {_ch}"
print("A3b PASS  turnover + healthy core -> targets the window and states the core's health")

# ---------------------------------------------------------------------------
# A3c -- no ablation (flag off / eval failed) -> the structural heuristic, no
# regression: the decay hint falls back to "slowest-moving" (the pre-ablation
# behaviour) and mentions neither IC-neutrality nor the core.
# ---------------------------------------------------------------------------
_decay_none = _build_target("decay", 0.10, Verdict.MARGINAL, _SEGS, {}, None)
_dhn = _decay_none["mechanism_hint"]
assert "slowest-moving" in _dhn, f"A3: no-ablation decay should use the structural heuristic: {_dhn}"
assert "IC-neutral" not in _dhn, "A3: no-ablation path must not mention IC-neutrality"
print("A3c PASS  no ablation -> the structural heuristic (no regression)")
print("A3  PASS  _build_target routes decay->core / turnover->window / none->structural")

# ---------------------------------------------------------------------------
# A4 -- the directive carries ablation_summary through diagnose_parent + the
# dict round-trip; with no eval it is "" (the no-regression default).
# ---------------------------------------------------------------------------
# force the deterministic table path (hermetic) for THIS module only, and
# restore the prior value so the LLM-diagnosis gate is not silently disabled
# for every test collected after this one (it broke test_nonblocking...).
_PREV_LLM_DIAGNOSIS = os.environ.get("QA_LLM_DIAGNOSIS")
os.environ["QA_LLM_DIAGNOSIS"] = "0"

_FAKE_SUMMARY = ("the core carries the rank edge (solo rank_ic +0.0300, t_nw +3.20); "
                 "the TS_MEAN window is IC-neutral. How to fix that is yours to determine.")


def _fake_ablation_eval(parent):
    # A minimal SegmentAblation: only the summary is load-bearing for carriage.
    return SegmentAblation(summary=_FAKE_SUMMARY)


_diag_parent = SimpleNamespace(
    factors=[{"name": "f0", "expression": TARGET}],
    backtest_metrics={"U": 0.3, "verdict": "marginal", "e_turnover": 0.10,
                      "e_stability": 0.50, "rank_ic": 0.003, "turnover_book": 0.05},
    hypothesis="close location in range predicts next-day return",
    expected_ic_sign="negative", extra_info={})

op = RefineOp()
d_abl = op.diagnose_parent(_diag_parent, ablation_eval=_fake_ablation_eval)
assert d_abl is not None and d_abl.is_refinement(), "A4: the parent must yield a refinement directive"
assert d_abl.ablation_summary == _FAKE_SUMMARY, (
    f"A4: ablation_summary not carried: {d_abl.ablation_summary!r}")
# Survives the T5 lineage round-trip (to_dict -> from_dict).
_round = RefinementDirective.from_dict(d_abl.to_dict())
assert _round.ablation_summary == _FAKE_SUMMARY, "A4: ablation_summary lost in to_dict/from_dict"
assert _round.is_refinement(), "A4: the directive lost its refinement status in the round-trip"
print("A4a PASS  diagnose_parent carries ablation_summary and it survives to_dict/from_dict")

# A4b -- with NO ablation_eval (the flag-off default) the summary is "" and the
# directive is otherwise unchanged (the no-regression guarantee).
d_none = op.diagnose_parent(_diag_parent, ablation_eval=None)
assert d_none is not None and d_none.is_refinement(), "A4: the no-ablation parent must still refine"
assert d_none.ablation_summary == "", f"A4: no-ablation summary must be empty: {d_none.ablation_summary!r}"
print("A4b PASS  no ablation_eval -> ablation_summary is \"\" (the no-regression default)")
print("A4  PASS  the directive carries the ablation through diagnose_parent + the round-trip")

# Restore the LLM-diagnosis gate for the rest of the session.
if _PREV_LLM_DIAGNOSIS is None:
    os.environ.pop("QA_LLM_DIAGNOSIS", None)
else:
    os.environ["QA_LLM_DIAGNOSIS"] = _PREV_LLM_DIAGNOSIS

print("\nALL PASS")