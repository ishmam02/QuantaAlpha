"""Per-segment expression ablation -- the AlphaEvolve "which sub-tree is broken" signal.

For a factor like ``ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))`` the
diagnosis used to know only that "the factor is weak" (a verdict + a weakest
dimension). It could not tell *which part of the expression* carries the edge,
*which temporal window* is IC-neutral, or *whether the core's sign is stable*
across the sample -- so the refine operator measurably edited the window
regardless of the diagnosed weakness (2 of 3 refine children only moved a
window, both went from positive to net_harmful).

This module measures each sub-tree + each temporal-window variant with SOLO
metrics (IC and cost SEPARATELY, no ``net_ir`` collapse) so the diagnosis can
route the refine on the BROKEN part, and the window probe reveals IC-NEUTRALITY
(which discourages widening a window for an IC weakness -- the window-trap).

**Pure logic, no qlib.** ``ablate`` takes two injected callables --
``eval_signal(sub_expr) -> handle`` and ``score(handle) -> dict`` -- and does
only AST surgery + measurement assembly. The heavy production wiring
(``EvaluationOperator`` + ``CustomFactorCalculator`` + the metrics closures)
lives in the controller's closure builder, so this module imports neither
``eval`` nor ``backtest`` and is hermetically testable with fakes.

**Hard rules (preserved).** Prompts diagnose, never prescribe: the ``summary``
states measured numbers and the factor's OWN structure (op names, the core's
fields) and closes with "how to fix that is yours to determine" -- no remedy
("lengthen", "simplify", "smooth"). No market-specific priors: no index names,
no hardcoded IC values, no continuation/reversal prior; the sign-stability is
measured generically via ``ic_pos_frac`` + sub-sample signs.

Q2 defenses baked in (see the plan):
  1. IC and cost measured SEPARATELY per part (``per_part`` has both).
  2. Route on the BROKEN part (``core_sign_stability`` / ``per_part``), not the
     best-improving edit.
  3. The window probe reveals IC-NEUTRALITY -> the window is deployed only for
     cost-with-healthy-IC (the table-path backstop in ``diagnosis._build_target``).
  4. Perturbation deltas NEVER reach a prompt -- only ``summary`` + ``per_part``
     scalars (measurements) leave this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from quantaalpha.factors.coder import factor_ast
from quantaalpha.pipeline.evolution.segment_profiling import (
    _signature,
    _func_name,
    _category_of,
)

# Windows probed for each temporal op. Purpose: measure IC-Sensitivity of the
# window (flat rank_ic across these => IC-neutral), NOT to find the
# cost-minimizing width. The factor's own window is usually in here (a sanity
# check that the original reproduces).
_WINDOW_SWEEP: tuple[int, ...] = (1, 3, 5, 10, 20)
# rank_ic range across the sweep below this => the window is IC-neutral (moving
# it does not move the edge, so it is not the lever for an IC weakness).
_IC_NEUTRAL_RANGE: float = 0.01
# ic_pos_frac at/above this in BOTH date-halves, with agreeing signs, => stable.
_SIGN_STABLE_POSFRAC: float = 0.6
# Fewer per-date ICs than this => sign-stability is unresolved (not enough data).
_SIGN_MIN_N: int = 8

_NAN = float("nan")


@dataclass
class PartMetrics:
    """SOLO metrics for one sub-tree, keyed by ``_signature(node)``.

    IC and cost are kept SEPARATE (no ``net_ir``): ``rank_ic``/``t_nw``/
    ``ic_pos_frac``/``monotonicity`` are the edge; ``turnover_solo`` is the cost.
    """
    rank_ic: float = _NAN
    t_nw: float = _NAN
    ic_pos_frac: float = _NAN
    monotonicity: float = _NAN
    turnover_solo: float = _NAN
    role: str = ""          # "xs" / "temporal" / "signal" / "gate" / "signal-core"
    op: str = ""            # function name for a FunctionNode, "" otherwise
    window: Optional[float] = None  # the first temporal window on this node, if any

    def as_dict(self) -> dict[str, Any]:
        return {"rank_ic": self.rank_ic, "t_nw": self.t_nw,
                "ic_pos_frac": self.ic_pos_frac, "monotonicity": self.monotonicity,
                "turnover_solo": self.turnover_solo, "role": self.role,
                "op": self.op, "window": self.window}


@dataclass
class SegmentAblation:
    """The result of ``ablate``. Only ``summary`` (measurement-only prose) and
    the ``per_part`` scalars are meant to leave this module for a prompt; the
    rest is for the deterministic table-path router (``diagnosis._build_target``).
    """
    per_part: dict[str, PartMetrics] = field(default_factory=dict)
    # signature -> {op, rank_ic_by_window: {win: rank_ic}, ic_neutral: bool}
    window_sensitivity: dict[str, dict] = field(default_factory=dict)
    # {ic_pos_frac, sign_by_subsample: ["+"|"-\"|"0"], mean_by_subsample,
    #  stable: bool, core_signature}
    core_sign_stability: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# AST surgery
# ---------------------------------------------------------------------------

def _first_child(node: factor_ast.Node) -> Optional[factor_ast.Node]:
    """The first non-number child -- the "inner" layer a structural strip drops to."""
    if isinstance(node, factor_ast.FunctionNode):
        for a in node.args:
            if not isinstance(a, factor_ast.NumberNode):
                return a
        return None
    if isinstance(node, factor_ast.BinaryOpNode):
        return node.left
    if isinstance(node, factor_ast.UnaryOpNode):
        return node.operand
    return None


def _role_of(node: factor_ast.Node) -> str:
    if isinstance(node, factor_ast.FunctionNode):
        return _category_of(_func_name(node))  # temporal/xs/signal/gate
    return "signal-core"  # BinaryOp / Var / Number -> the pointwise signal core


def _first_window(node: factor_ast.Node) -> Optional[float]:
    if isinstance(node, factor_ast.FunctionNode):
        for a in node.args:
            if isinstance(a, factor_ast.NumberNode):
                return float(a.value)
    return None


def _render(node: factor_ast.Node) -> str:
    """Render an AST to an executable expression with integral literals as ints.

    ``NumberNode.__str__`` renders its ``float`` value verbatim (``5.0``), and
    the factor calculator's temporal ops (TS_MEAN / TS_SUM / ...) require an
    *integer* window -- pandas ``.rolling(5.0)`` raises ``slice indices must be
    integers``. The structural strip and window-sweep re-render the AST (the
    original ``expr`` string is only used once, for the parent baseline), so
    every windowed variant would fail without this. Integral literals are
    re-rendered as ints; non-integral literals (e.g. ``1e-12``) are unchanged.

    Mirrors each node's ``__str__`` exactly so a non-windowed expression renders
    identically to ``str(node)``. Used ONLY to render the sub-expression handed
    to ``eval_signal``; the ``_signature`` keys (which must match
    ``_build_target``'s ``subtree_signature``) still use ``str(node)``, so this
    never perturbs a signature. Local to ``ablate`` (env-gated) -> the frozen
    path is byte-identical when ``QA_ABLATION_DIAGNOSIS`` is off.
    """
    if isinstance(node, factor_ast.NumberNode):
        v = node.value
        if v == v and float(v).is_integer():
            return str(int(v))
        return str(v)
    if isinstance(node, factor_ast.VarNode):
        return node.name
    if isinstance(node, factor_ast.FunctionNode):
        name = getattr(node.name, "name", node.name)
        return f"{name}({', '.join(_render(a) for a in node.args)})"
    if isinstance(node, factor_ast.BinaryOpNode):
        return f"({_render(node.left)} {node.op} {_render(node.right)})"
    if isinstance(node, factor_ast.UnaryOpNode):
        return f"({node.op}{_render(node.operand)})"
    if isinstance(node, factor_ast.ConditionalNode):
        return (f"({_render(node.condition)} ? {_render(node.true_expr)} "
                f": {_render(node.false_expr)})")
    return str(node)


def _strip_chain(ast: factor_ast.Node) -> list[tuple[factor_ast.Node, str]]:
    """The outer-first-arg chain: root -> inner layer -> ... -> core.

    Each entry is a "structural strip" variant (drop every layer above it). The
    chain descends through the cross-sectional / temporal / gate layers and
    STOPS at the first signal-core node (the pointwise arithmetic / fields) --
    the core is one semantic unit, so its numerator / denominator / operands are
    NOT separately stripped (they are probed by the field-swap family, not the
    structural strip). Cycles / repeated signatures (defensive) terminate.
    """
    out: list[tuple[factor_ast.Node, str]] = []
    seen: set[str] = set()
    node = ast
    while node is not None:
        sig = _signature(node)
        if sig in seen:
            break
        seen.add(sig)
        role = _role_of(node)
        out.append((node, role))
        if role in ("signal", "signal-core"):
            break  # the pointwise signal core is the bottom; do not sub-strip it
        nxt = _first_child(node)
        if nxt is None or isinstance(nxt, factor_ast.NumberNode):
            break
        node = nxt
    return out


def _temporal_nodes(ast: factor_ast.Node):
    """Yield ``(node, op_name, win_arg_idx, current_window)`` for each temporal op.

    Yields the LIVE ``FunctionNode`` so the window-sweep can mutate its
    ``NumberNode`` arg in place (and restore it). The signature of the un-mutated
    node is what ``_build_target``'s ``subtree_signature`` carries, so
    ``window_sensitivity`` is keyed by ``_signature(node)`` to match.
    """
    def visit(n):
        if isinstance(n, factor_ast.FunctionNode):
            if _category_of(_func_name(n)) == "temporal":
                for i, a in enumerate(n.args):
                    if isinstance(a, factor_ast.NumberNode):
                        yield (n, _func_name(n), i, float(a.value))
                        break
            for a in n.args:
                yield from visit(a)
        elif isinstance(n, factor_ast.BinaryOpNode):
            yield from visit(n.left)
            yield from visit(n.right)
        elif isinstance(n, factor_ast.UnaryOpNode):
            yield from visit(n.operand)
        elif isinstance(n, factor_ast.ConditionalNode):
            yield from visit(n.condition)
            yield from visit(n.true_expr)
            yield from visit(n.false_expr)
    yield from visit(ast)


# ---------------------------------------------------------------------------
# Scoring contract
# ---------------------------------------------------------------------------

def _empty_score() -> dict[str, Any]:
    return {"rank_ic": _NAN, "t_nw": _NAN, "ic_pos_frac": _NAN,
            "monotonicity": _NAN, "turnover_solo": _NAN,
            "ric_series": pd.Series(dtype=float)}


def _safe_score(score: Callable[[Any], dict], handle: Any) -> dict[str, Any]:
    try:
        s = score(handle)
        if not isinstance(s, dict):
            return _empty_score()
        s.setdefault("ric_series", pd.Series(dtype=float))
        return s
    except Exception:
        return _empty_score()


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------

def ablate(expr: str,
           predicted_sign: str = "",
           *,
           eval_signal: Callable[[str], Any],
           score: Callable[[Any], dict[str, Any]],
           window_sweep: tuple[int, ...] = _WINDOW_SWEEP) -> SegmentAblation:
    """Measure each sub-tree + window variant of ``expr`` with SOLO metrics.

    ``eval_signal(sub_expr) -> handle`` computes/aligns a sub-expression's signal
    to an opaque handle (the production closure returns a wide aligned frame; a
    test returns the expr itself). ``score(handle) -> dict`` returns
    ``{rank_ic, t_nw, ic_pos_frac, monotonicity, turnover_solo, ric_series}``
    (``ric_series`` is a per-date rank-IC ``pd.Series``, used only for the core's
    sign-stability). Both are injected so this module stays free of ``eval`` /
    ``backtest`` imports and is hermetically testable.

    Never raises for a bad sub-expression -- a failed variant records NaN and the
    ablation continues. Returns a ``SegmentAblation`` whose ``summary`` is
    measurement-only prose (no remedy, no market prior).
    """
    result = SegmentAblation()
    try:
        ast = factor_ast.parse_expression(expr)
    except Exception:
        result.summary = ("the expression could not be parsed; no per-part "
                          "measurement is available. How to fix that is yours to determine.")
        return result

    parent_m = _safe_score(score, eval_signal(expr))

    # --- structural strip: each layer's solo contribution, keyed by signature --
    chain = _strip_chain(ast)
    core_node = chain[-1][0] if chain else ast
    core_full: dict[str, Any] = _empty_score()
    for node, role in chain:
        if isinstance(node, factor_ast.NumberNode):
            continue
        m = _safe_score(score, eval_signal(_render(node)))
        if node is core_node:
            core_full = m
        result.per_part[_signature(node)] = PartMetrics(
            rank_ic=float(m.get("rank_ic", _NAN)), t_nw=float(m.get("t_nw", _NAN)),
            ic_pos_frac=float(m.get("ic_pos_frac", _NAN)),
            monotonicity=float(m.get("monotonicity", _NAN)),
            turnover_solo=float(m.get("turnover_solo", _NAN)),
            role=role,
            op=_func_name(node) if isinstance(node, factor_ast.FunctionNode) else "",
            window=_first_window(node))

    # --- window sweep: each temporal op's IC-sensitivity across the sweep ------
    for node, op_name, win_idx, _orig in list(_temporal_nodes(ast)):
        sig = _signature(node)
        orig_arg = node.args[win_idx]
        ric_by_win: dict[int, float] = {}
        for w in window_sweep:
            node.args[win_idx] = factor_ast.NumberNode(float(w))
            try:
                m = _safe_score(score, eval_signal(_render(ast)))
                ric_by_win[int(w)] = float(m.get("rank_ic", _NAN))
            except Exception:
                ric_by_win[int(w)] = _NAN
        node.args[win_idx] = orig_arg  # restore the original AST
        finite = [v for v in ric_by_win.values() if v == v]
        ic_neutral = (len(finite) >= 2 and (max(finite) - min(finite)) < _IC_NEUTRAL_RANGE)
        result.window_sensitivity[sig] = {
            "op": op_name, "rank_ic_by_window": ric_by_win,
            "ic_neutral": bool(ic_neutral)}

    # --- core sign-stability: ic_pos_frac + sign across date-halves -----------
    core_sig = _signature(core_node)
    ric = core_full.get("ric_series")
    if isinstance(ric, pd.Series) and len(ric) >= _SIGN_MIN_N:
        mid = len(ric) // 2
        h1, h2 = ric.iloc[:mid], ric.iloc[mid:]
        m1 = float(h1.mean()) if len(h1) else _NAN
        m2 = float(h2.mean()) if len(h2) else _NAN
        s1 = "+" if m1 > 0 else "-" if m1 < 0 else "0"
        s2 = "+" if m2 > 0 else "-" if m2 < 0 else "0"
        pf = float((ric > 0).mean())
        stable = bool(s1 == s2 and s1 != "0" and pf >= _SIGN_STABLE_POSFRAC
                      and m1 == m1 and m2 == m2)
        result.core_sign_stability = {
            "ic_pos_frac": pf, "sign_by_subsample": [s1, s2],
            "mean_by_subsample": [m1, m2], "stable": stable,
            "core_signature": core_sig}
    else:
        result.core_sign_stability = {"ic_pos_frac": _NAN, "sign_by_subsample": [],
                                      "mean_by_subsample": [], "stable": False,
                                      "core_signature": core_sig}

    result.summary = _render_summary(result, predicted_sign, parent_m, core_node)
    return result


# ---------------------------------------------------------------------------
# Summary (measurement-only prose; no remedy, no market prior)
# ---------------------------------------------------------------------------

def _render_summary(result: SegmentAblation, predicted_sign: str,
                    parent_m: dict[str, Any], core_node: factor_ast.Node) -> str:
    parts: list[str] = []
    core_sig = result.core_sign_stability.get("core_signature", "")
    core = result.per_part.get(core_sig)

    # 1. which part carries the rank edge (the core, by solo rank_ic / t_nw).
    if core is not None and core.rank_ic == core.rank_ic and core.t_nw == core.t_nw:
        parts.append(
            f"the core carries the rank edge (solo rank_ic {core.rank_ic:+.4f}, "
            f"t_nw {core.t_nw:+.2f})")

    # 2. window IC-neutrality (the window-trap signal) + turnover attribution.
    for sig, ws in result.window_sensitivity.items():
        if ws.get("ic_neutral"):
            op = ws.get("op", "")
            wins = sorted(ws.get("rank_ic_by_window", {}).keys())
            span = f"{wins[0]}-{wins[-1]}" if wins else "?"
            parts.append(
                f"the {op} window is IC-neutral (solo rank_ic flat across windows "
                f"{span})")
    pt = parent_m.get("turnover_solo", _NAN)
    if (core is not None and core.turnover_solo == core.turnover_solo
            and pt == pt and core.turnover_solo < pt):
        parts.append(
            f"the temporal layer adds the turnover (core solo turnover "
            f"{core.turnover_solo:.3f} vs the full factor's {pt:.3f})")

    # 3. core sign-stability (the regime-dependence signal). If the realized
    #    core sign disagrees with the pre-registered direction, that is stated
    #    as a measurement (a comparison), not a prior.
    css = result.core_sign_stability
    pf = css.get("ic_pos_frac", _NAN)
    subs = css.get("sign_by_subsample", [])
    if pf == pf:
        npos = sum(1 for s in subs if s == "+")
        clause = (f"the core's sign is {'stable' if css.get('stable') else 'unstable'} "
                  f"(ic_pos_frac {pf:.2f}")
        if subs:
            clause += f", positive in {npos} of {len(subs)} sub-samples"
        clause += ")"
        if predicted_sign in ("positive", "negative") and core is not None \
                and core.rank_ic == core.rank_ic and core.rank_ic != 0:
            realized = "+" if core.rank_ic > 0 else "-"
            pred = "+" if predicted_sign == "positive" else "-"
            if realized != pred:
                clause += " -- opposite to the predicted direction"
        parts.append(clause)

    if not parts:
        return ("no per-part measurement could be resolved. How to fix that is "
                "yours to determine.")
    return "; ".join(parts) + ". How to fix that is yours to determine."


__all__ = ["SegmentAblation", "PartMetrics", "ablate"]