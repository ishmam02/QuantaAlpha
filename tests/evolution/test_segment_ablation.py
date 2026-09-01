"""Per-segment expression ablation -- the "which sub-tree is broken" measurement.

For ``ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))`` the ablation must:

  * split the expression into its xs / temporal / signal-core layers (3 parts),
    each keyed by ``segment_profiling._signature`` so it matches the
    ``_build_target`` router;
  * measure the temporal window's IC-SENSITIVITY across the sweep and flag it
    IC-neutral when the rank_ic is flat (the Q2 window-trap signal);
  * measure the core's sign-stability across date-halves (``stable`` only when
    both halves agree AND ic_pos_frac >= 0.6) -- the regime-dependence signal;
  * emit a measurement-only ``summary`` (no remedy, no index name, no hardcoded
    IC; closes with "how to fix that is yours to determine").

The scorers are deterministic fakes (no real panel) so this is hermetic. The
AST round-trip guards the S5 finding: every rendered sub-tree is accepted by
``expr_parser.parse_expression`` (5 -> 5.0, 1e-12 preserved, unary minus ok).

A1  the ablation resolves 3 parts, an IC-neutral window, an unstable core, and a
    measurement-only summary
A2  every rendered sub-tree round-trips through ``expr_parser.parse_expression``
"""
import contextlib
import io
import re

import pandas as pd

from quantaalpha.factors.coder import factor_ast, expr_parser
from quantaalpha.pipeline.evolution.segment_ablation import ablate, _strip_chain
from quantaalpha.pipeline.evolution.segment_profiling import _signature

TARGET = "ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))"

# A 20-day per-date rank-IC series whose two halves disagree in sign -- the
# regime-dependent / unstable case. ic_pos_frac = 0.5, h1 mean < 0, h2 mean > 0.
_RIC_UNSTABLE = pd.Series([-0.02] * 10 + [0.02] * 10)


def _eval_signal(sub_expr):
    # The handle IS the rendered sub-expression (the score fakes off the string).
    return sub_expr


def _score(handle):
    s = str(handle)
    if "TS_MEAN" in s and "ZSCORE" in s:
        # The full expression OR a window-sweep variant -> flat rank_ic across
        # every window (IC-neutral). The parent's own turnover is the baseline.
        return {"rank_ic": 0.030, "t_nw": 2.0, "ic_pos_frac": 0.55,
                "monotonicity": float("nan"), "turnover_solo": 0.50,
                "ric_series": _RIC_UNSTABLE}
    if "TS_MEAN" in s:
        # The temporal-layer strip (no ZSCORE): the slow-moving layer.
        return {"rank_ic": 0.028, "t_nw": 1.8, "ic_pos_frac": 0.54,
                "monotonicity": float("nan"), "turnover_solo": 0.45,
                "ric_series": _RIC_UNSTABLE}
    # The signal-core (the close-location-in-range ratio): carries the rank edge
    # but its sign is unstable across the sample.
    return {"rank_ic": 0.030, "t_nw": 3.2, "ic_pos_frac": 0.50,
            "monotonicity": float("nan"), "turnover_solo": 0.80,
            "ric_series": _RIC_UNSTABLE}


abl = ablate(TARGET, "positive", eval_signal=_eval_signal, score=_score)

# A1a -- three parts (xs / temporal / signal-core), keyed by signature.
_ast = factor_ast.parse_expression(TARGET)
chain = _strip_chain(_ast)
roles = [role for _n, role in chain]
assert roles == ["xs", "temporal", "signal-core"], f"A1: expected xs/temporal/signal-core, got {roles}"
assert len(abl.per_part) == 3, f"A1: expected 3 per-part entries, got {len(abl.per_part)}"
core_sig = _signature(chain[-1][0])
assert core_sig in abl.per_part, "A1: the core signature must key a per-part entry"
print("A1a PASS  the ablation splits into xs / temporal / signal-core (3 parts)")

# A1b -- the temporal window is IC-neutral (rank_ic flat across the sweep).
assert len(abl.window_sensitivity) == 1, "A1: expected one temporal op swept"
_ws = next(iter(abl.window_sensitivity.values()))
assert _ws["op"] == "TS_MEAN", f"A1: expected TS_MEAN, got {_ws['op']}"
assert _ws["ic_neutral"] is True, "A1: the window should be IC-neutral (flat rank_ic)"
_by_win = _ws["rank_ic_by_window"]
assert max(_by_win.values()) - min(_by_win.values()) < 0.01, "A1: rank_ic not flat across windows"
print("A1b PASS  the TS_MEAN window is flagged IC-neutral (rank_ic flat across the sweep)")

# A1c -- the core's sign is unstable (halves disagree, ic_pos_frac < 0.6).
_css = abl.core_sign_stability
assert _css["stable"] is False, "A1: the core should be sign-unstable"
assert abs(_css["ic_pos_frac"] - 0.5) < 1e-6, f"A1: ic_pos_frac should be 0.5, got {_css['ic_pos_frac']}"
assert _css["sign_by_subsample"] == ["-", "+"], f"A1: halves should disagree, got {_css['sign_by_subsample']}"
print("A1c PASS  the core is sign-unstable (ic_pos_frac 0.5, halves disagree)")

# A1d -- the structural strip shows the CORE carries the rank edge.
core = abl.per_part[core_sig]
assert core.rank_ic >= 0.028, "A1: the core should carry the rank edge"
assert core.role == "signal-core", f"A1: core role, got {core.role}"
print("A1d PASS  the core carries the rank edge (solo rank_ic +0.0300, t_nw +3.20)")

# A1e -- the summary is measurement-only: no remedy, no index name, no hardcoded
# IC claim, and closes with the non-prescriptive hand-off.
_summary = abl.summary
for _bad in ("lengthen", "shorten", "simplify", "smooth", "blend", "CSI"):
    assert _bad.lower() not in _summary.lower(), f"A1: summary prescribes/assumes '{_bad}': {_summary}"
assert _summary.rstrip().endswith("How to fix that is yours to determine."), f"A1: summary must close non-prescriptively: {_summary}"
assert "IC-neutral" in _summary and "unstable" in _summary and "core" in _summary, f"A1: summary missing the measured facts: {_summary}"
print("A1e PASS  the summary is measurement-only (no remedy / no index / no hardcoded IC; closes non-prescriptively)")

print("A1  PASS  the ablation resolves 3 parts, an IC-neutral window, an unstable core, and a measurement-only summary")

# ---------------------------------------------------------------------------
# A2 -- every rendered sub-tree round-trips through expr_parser.parse_expression.
# ``expr_parser.parse_expression`` prints the expression to stdout (a Qlib-ism),
# so swallow that; we only care that it does not raise on the rendered string.
# ---------------------------------------------------------------------------
_roundtrip_ok = True
for node, _role in chain:
    rendered = str(node)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            expr_parser.parse_expression(rendered)
    except Exception as e:
        _roundtrip_ok = False
        print(f"A2 FAIL  sub-tree did not round-trip: {rendered!r} -> {e}")
        break
assert _roundtrip_ok, "A2: a rendered sub-tree failed to round-trip through expr_parser"
# The full expression itself round-trips too (5 -> 5.0, 1e-12 preserved).
with contextlib.redirect_stdout(io.StringIO()):
    expr_parser.parse_expression(str(_ast))
print("A2  PASS  every rendered sub-tree round-trips through expr_parser.parse_expression")

# ---------------------------------------------------------------------------
# A3 -- the window-sweep + structural-strip render integral windows as INTS.
# The hermetic scorers fake off the string, so they cannot see Defect D: the
# real factor calculator's temporal ops call pandas ``.rolling(window)`` which
# rejects a float window ("slice indices must be integers"). ``NumberNode.
# __str__`` renders its float value verbatim (``5.0``); ``ablate`` must render
# integral literals as ints (``5``) so the re-rendered variants execute on a
# real calculator. Guard that every sub-expression handed to ``eval_signal``
# is free of an integral-float window (``<digits>.0`` immediately before a
# ``,`` or ``)``) -- the exact pattern that broke the real panel.
# ---------------------------------------------------------------------------
import re as _re

_seen: list[str] = []


def _recording_eval(sub_expr):
    _seen.append(sub_expr)
    return _eval_signal(sub_expr)


ablate(TARGET, "positive", eval_signal=_recording_eval, score=_score)
_float_window = _re.compile(r"\d+\.0(?=[,)])")
_bad = [s for s in _seen if _float_window.search(s)]
assert not _bad, (
    f"A3: re-rendered sub-expression(s) carry a float-integral window that the "
    f"real calculator rejects: {_bad[:3]}")
# Sanity: the sweep DID emit windowed variants (otherwise the guard is vacuous).
assert any("TS_MEAN" in s for s in _seen), "A3: no TS_MEAN variant was rendered"
print("A3  PASS  the window-sweep + structural-strip render integral windows as ints (no `<int>.0)` leaked to eval_signal)")

# A3b -- the ROOT fix: ``NumberNode.__str__`` renders integral floats as ints.
# A3 guards the ablation's own ``_render``; A3b guards the root node so EVERY
# re-rendering path (crossover child, ablation, sign-flip freeze) is covered at
# the source -- a float-integral window never reaches a real calculator's
# ``.rolling(window)`` / ``.shift(periods)`` ("slice indices must be integers").
from quantaalpha.factors.coder import factor_ast as _fast
assert str(_fast.NumberNode(5.0)) == "5", "A3b: NumberNode(5.0) must render as '5' at the root"
assert str(_fast.NumberNode(0.5)) == "0.5", "A3b: non-integral floats are unchanged"
print("A3b PASS  root NumberNode renders integral floats as ints (covers every re-render path)")

print("\nALL PASS")