"""AST segment profiling for expression-aware diagnosis (T2, fixes #2/#3-partial).

The refine-mutation diagnosis (``diagnosis.py``) used to be expression-blind:
``diagnose`` carried ``parent.factors`` but never read them, and ``_DIRECTIVES``
emitted a fixed menu keyed only on ``(verdict, dimension_category)``. That left
the "alter the time scale" directive unable to name *which* operator or *which*
window -- so the child got a generic "lengthen the lookback" with no pointer to
the 5-day ``TS_MEAN`` that was actually driving turnover.

This module turns a factor expression into a ``SegmentProfile``: which
sub-patterns appear (signal / temporal / xs / gate), each temporal operator's
window(s), the tree depth, and the free-parameter count -- plus per-factor
combininer credit fields (filled in T4 from ``backtest_metrics["factor_attribution"]``).
T3's expression-aware diagnosis reads these to name the **actual operator +
parameter** behind a weakness, and T5's lineage walk reads the cached
serializable subset to detect an exhausted refinement lever.

It builds on the real mutable ``factor_ast`` AST
(``quantaalpha.factors.coder.factor_ast.parse_expression``), which is read-only
here -- nothing on the execution path is touched. On parse failure the profile
falls back to a coarse regex scan of the raw string (``ast=None``), so a
malformed expression degrades gracefully instead of killing the diagnosis.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from quantaalpha.factors.coder.factor_ast import (
    BinaryOpNode,
    ConditionalNode,
    FunctionNode,
    Node,
    NumberNode,
    UnaryOpNode,
    VarNode,
    parse_expression,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Operator taxonomy -- codified from factors/coder/qa_prompts.yaml sections.
# --------------------------------------------------------------------------
# Each known operator (UPPERCASE) maps to a sub-pattern category. The category
# says what KIND of node it is, so the diagnosis can LOCATE the sub-tree behind
# a weakness (cost -> a temporal node with a small window; overfit -> deep
# nesting / many params; signal -> a shallow source). Categories:
#   signal   -- raw $fields, arithmetic, pointwise math (the signal source)
#   temporal -- a windowed time-series / smoothing / regression op (the
#               turnover-driving lever; its window arg is the #3 magnitude)
#   xs       -- cross-sectional (ranks/zscores ACROSS instruments at a time)
#   gate     -- a conditional / comparison / logical regime the signal passes
#
# MAX/MIN are intentionally NOT ``xs``: the "Mathematical Operations" section
# redefines them as element-wise pairwise MAX(A,B)/MIN(A,B), and the element-
# wise form wins at runtime, so calling them cross-sectional would mislocate
# the sub-tree. They are pointwise -> ``signal``.
OP_CATEGORY: dict[str, str] = {
    # Cross-sectional.
    "RANK": "xs", "ZSCORE": "xs", "MEAN": "xs", "STD": "xs", "SKEW": "xs",
    "KURT": "xs", "MEDIAN": "xs", "SCALE": "xs",
    # Time-series / smoothing / regression (windowed -> the cost lever).
    "DELTA": "temporal", "DELAY": "temporal", "TS_MEAN": "temporal",
    "TS_SUM": "temporal", "TS_RANK": "temporal", "TS_ZSCORE": "temporal",
    "TS_MEDIAN": "temporal", "TS_PCTCHANGE": "temporal", "TS_MIN": "temporal",
    "TS_MAX": "temporal", "TS_ARGMAX": "temporal", "TS_ARGMIN": "temporal",
    "TS_QUANTILE": "temporal", "TS_STD": "temporal", "TS_VAR": "temporal",
    "TS_CORR": "temporal", "TS_COVARIANCE": "temporal", "TS_MAD": "temporal",
    # Rolling higher moments (TS_KURT/TS_SKEW in function_lib) -- the time-series
    # counterparts of the cross-sectional SKEW/KURT above; windowed -> temporal.
    "TS_KURT": "temporal", "TS_SKEW": "temporal",
    "PERCENTILE": "temporal", "HIGHDAY": "temporal", "LOWDAY": "temporal",
    "SUMAC": "temporal", "PROD": "temporal",
    "SMA": "temporal", "WMA": "temporal", "EMA": "temporal",
    "DECAYLINEAR": "temporal",
    "RSI": "temporal", "MACD": "temporal", "BB_MIDDLE": "temporal",
    "BB_UPPER": "temporal", "BB_LOWER": "temporal",
    "REGBETA": "temporal", "REGRESI": "temporal",
    "COUNT": "temporal", "SUMIF": "temporal",
    # Pointwise math -> signal. MAX/MIN element-wise live here (see note above).
    "LOG": "signal", "SQRT": "signal", "SIGN": "signal", "EXP": "signal",
    "ABS": "signal", "INV": "signal", "FLOOR": "signal", "POW": "signal",
    "MAX": "signal", "MIN": "signal",
    # Conditional / filtering -> gate.
    "FILTER": "gate",
    # SEQUENCE is a REGBETA/REGRESI helper (a generated 1..n column), not a
    # signal source in its own right -> signal by default.
    "SEQUENCE": "signal",
}

_ARITH = {"+", "-", "*", "/"}
_COMPARE = {">", "<", ">=", "<=", "==", "!="}
_LOGICAL = {"&&", "||", "&", "|"}

_SUB_KEYS = ("signal", "temporal", "xs", "gate")


def _func_name(node: FunctionNode) -> str:
    """The function name as a string.

    ``factor_ast.create_function_node`` sets ``FunctionNode.name = tokens[0]``,
    where ``tokens[0]`` is the ``var`` parse action's ``VarNode`` (not a str).
    ``str(VarNode)`` returns the name string so rendering works, but the raw
    ``.name`` has no ``.upper()`` -- so callers that want the name string go
    through here.
    """
    nm = node.name
    return nm.name if isinstance(nm, VarNode) else str(nm)


def _category_of(func_name: str) -> str:
    """The sub-pattern category for a function name; defaults to ``signal``."""
    return OP_CATEGORY.get(func_name.upper(), "signal")


def _signature(node: Node) -> str:
    """A canonical short string for a node, for T5 lineage comparison.

    The op name is upper-cased so ``Ts_Mean`` and ``TS_MEAN`` compare equal across
    ancestors; the args render via their own ``__str__`` (lossy but stable, so
    two identical sub-trees produce identical signatures).
    """
    if isinstance(node, FunctionNode):
        return _func_name(node).upper() + "(" + ", ".join(str(a) for a in node.args) + ")"
    return str(node)


def _depth(node: Node) -> int:
    if isinstance(node, (VarNode, NumberNode)):
        return 1
    if isinstance(node, FunctionNode):
        return 1 + max((_depth(a) for a in node.args), default=0)
    if isinstance(node, BinaryOpNode):
        return 1 + max(_depth(node.left), _depth(node.right))
    if isinstance(node, UnaryOpNode):
        return 1 + _depth(node.operand)
    if isinstance(node, ConditionalNode):
        return 1 + max(_depth(node.condition), _depth(node.true_expr),
                       _depth(node.false_expr))
    return 1


def _count_numbers(node: Node) -> int:
    if isinstance(node, NumberNode):
        return 1
    if isinstance(node, VarNode):
        return 0
    if isinstance(node, FunctionNode):
        return sum(_count_numbers(a) for a in node.args)
    if isinstance(node, BinaryOpNode):
        return _count_numbers(node.left) + _count_numbers(node.right)
    if isinstance(node, UnaryOpNode):
        return _count_numbers(node.operand)
    if isinstance(node, ConditionalNode):
        return (_count_numbers(node.condition)
                + _count_numbers(node.true_expr)
                + _count_numbers(node.false_expr))
    return 0


def _walk(root: Node) -> tuple[dict[str, bool], list[dict], int, int]:
    """Classify every node, collect temporal ops, depth, and free-param count."""
    sub = {k: False for k in _SUB_KEYS}
    temporal_ops: list[dict] = []

    def visit(node: Node) -> None:
        if isinstance(node, VarNode):
            if node.name.startswith("$"):
                sub["signal"] = True
        elif isinstance(node, NumberNode):
            pass
        elif isinstance(node, FunctionNode):
            cat = _category_of(_func_name(node))
            if cat == "temporal":
                sub["temporal"] = True
                # The window(s) are the direct NumberNode args of this op
                # (e.g. TS_MEAN(expr, 20) -> [20.0]; SMA(A, 20, 5) -> [20.0, 5.0]).
                # Nested numbers (a 14 inside the signal expr) belong to the
                # inner op, recorded on its own visit.
                windows = [float(a.value) for a in node.args if isinstance(a, NumberNode)]
                arg_idx = [i for i, a in enumerate(node.args) if isinstance(a, NumberNode)]
                temporal_ops.append({
                    "op": _func_name(node).upper(),
                    "windows": windows,
                    "arg_indices": arg_idx,
                    "signature": _signature(node),
                })
            elif cat == "xs":
                sub["xs"] = True
            elif cat == "gate":
                sub["gate"] = True
            else:  # "signal" (pointwise math) or unknown -> pointwise transform
                sub["signal"] = True
            for a in node.args:
                visit(a)
        elif isinstance(node, BinaryOpNode):
            if node.op in _COMPARE or node.op in _LOGICAL:
                sub["gate"] = True
            else:  # arithmetic combines the signal pointwise
                sub["signal"] = True
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnaryOpNode):
            sub["signal"] = True
            visit(node.operand)
        elif isinstance(node, ConditionalNode):
            sub["gate"] = True
            visit(node.condition)
            visit(node.true_expr)
            visit(node.false_expr)

    visit(root)
    return sub, temporal_ops, _depth(root), _count_numbers(root)


_OP_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_DOLLAR_RE = re.compile(r"\$[A-Za-z_]")
_GATE_RE = re.compile(r"[<>!]=|&&|\|\||\?|[^A-Za-z_][<>]")


def _regex_profile(expression: str) -> tuple[dict[str, bool], list[dict], int, int]:
    """Coarse fallback when the expression does not parse.

    Scans the raw string for operator names and numbers. No tree, so depth is
    unknown (0) and windows are not located -- but the category flags and the
    free-param count are still useful, and crucially this never raises.
    """
    sub = {k: False for k in _SUB_KEYS}
    temporal_ops: list[dict] = []
    for m in _OP_RE.finditer(expression or ""):
        op = m.group(1).upper()
        cat = OP_CATEGORY.get(op)
        if cat == "temporal":
            sub["temporal"] = True
            temporal_ops.append({"op": op, "windows": [], "arg_indices": [],
                                 "signature": op + "(?)"})
        elif cat == "xs":
            sub["xs"] = True
        elif cat == "gate":
            sub["gate"] = True
        elif cat == "signal":
            sub["signal"] = True
    if _DOLLAR_RE.search(expression or ""):
        sub["signal"] = True
    if _GATE_RE.search(expression or ""):
        sub["gate"] = True
    n_free = len(_NUM_RE.findall(expression or ""))
    return sub, temporal_ops, 0, n_free


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


@dataclass
class SegmentProfile:
    """One factor's parsed structure + per-factor combiner credit.

    ``ast`` is the live ``factor_ast`` Node (or ``None`` on parse failure); it is
    NOT persisted (see ``build_segments``), only used in-process by T3 to locate
    and rewrite the weak sub-tree. The rest of the fields are plain types and
    round-trip through ``extra_info["segments"]`` for T5's lineage walk.
    """

    factor_name: str
    expression: str
    ast: Any  # Node | None
    sub_patterns: dict[str, bool] = field(default_factory=lambda: {k: False for k in _SUB_KEYS})
    # Each temporal op in the tree: its (uppercased) name, window value(s), the
    # arg positions of those windows, and a canonical signature for T5. This is
    # the plan's ``params`` field -- the #3 magnitude (the actual window behind
    # a cost weakness), made locatable.
    temporal_ops: list[dict] = field(default_factory=list)
    depth: int = 0
    n_free_params: int = 0
    # Per-factor combiner credit, filled by T4 from
    # ``backtest_metrics["factor_attribution"][expression]``. Absent (None) until
    # T4 threads the ICIR combiner's per-factor weights through.
    combiner_weight: float | None = None
    weight_stability: float | None = None
    ic_mean: float | None = None
    ic_std: float | None = None
    rank_ic: float | None = None
    turnover_share: float | None = None

    def fill_credit(self, attr: dict[str, Any]) -> None:
        """Populate the per-factor credit fields from a T4 attribution record."""
        if not attr:
            return
        # T4 keys: weight (mean), weight_stability (1 - std/|mean|), ic_mean,
        # ic_std, rank_ic, turnover_share. Accept a couple of aliases defensively.
        self.combiner_weight = _maybe_float(attr.get("weight", attr.get("combiner_weight")))
        self.weight_stability = _maybe_float(attr.get("weight_stability"))
        self.ic_mean = _maybe_float(attr.get("ic_mean"))
        self.ic_std = _maybe_float(attr.get("ic_std"))
        self.rank_ic = _maybe_float(attr.get("rank_ic"))
        self.turnover_share = _maybe_float(attr.get("turnover_share"))

    def to_serializable(self) -> dict[str, Any]:
        """The persistable subset (no live AST) for ``extra_info`` / T5 lineage."""
        return {
            "factor_name": self.factor_name,
            "expression": self.expression,
            "sub_patterns": dict(self.sub_patterns),
            "temporal_ops": [dict(t) for t in self.temporal_ops],
            "depth": self.depth,
            "n_free_params": self.n_free_params,
            "combiner_weight": self.combiner_weight,
            "weight_stability": self.weight_stability,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "rank_ic": self.rank_ic,
            "turnover_share": self.turnover_share,
        }


def parse_factor(factor_name: str, expression: str,
                 attribution: dict[str, Any] | None = None) -> SegmentProfile:
    """Parse one factor expression into a ``SegmentProfile``.

    On parse failure ``ast`` is ``None`` and the profile is built from a regex
    scan of the raw string -- graceful, never raising. ``attribution`` (a T4
    per-factor credit record) is folded in if present.
    """
    try:
        ast: Any = parse_expression(expression or "")
    except Exception as exc:  # ParseException -> ValueError; be broad, never crash
        ast = None
        logger.debug("segment_profiling: parse failed for %r (%s); regex fallback",
                     expression, exc)

    if ast is not None:
        sub, temporal_ops, depth, n_free = _walk(ast)
    else:
        sub, temporal_ops, depth, n_free = _regex_profile(expression or "")

    prof = SegmentProfile(
        factor_name=factor_name or "",
        expression=expression or "",
        ast=ast,
        sub_patterns=sub,
        temporal_ops=temporal_ops,
        depth=depth,
        n_free_params=n_free,
    )
    prof.fill_credit(attribution or {})
    return prof


def build_segments(trajectory: Any) -> list[SegmentProfile]:
    """Profile every factor on a trajectory, filling per-factor credit.

    A serializable subset of each profile (no live AST) is cached on
    ``trajectory.extra_info["segments"]`` so T5's lineage walk can read a
    parent's sub-patterns / temporal signatures after a save/reload round-trip
    without re-parsing. The live-AST profiles are returned for in-process use by
    T3; they are NOT placed in ``extra_info`` (a ``Node`` is a dataclass and
    ``asdict`` would recurse it into a plain dict, breaking it on reload).
    """
    factors = getattr(trajectory, "factors", None) or []
    metrics = getattr(trajectory, "backtest_metrics", None) or {}
    attribution = metrics.get("factor_attribution") or {}

    profiles: list[SegmentProfile] = []
    for f in factors:
        name = f.get("name", "") if isinstance(f, dict) else ""
        expr = f.get("expression", "") if isinstance(f, dict) else ""
        # factor_attribution is keyed by expression (T4); fall back to {}.
        prof = parse_factor(name, expr, attribution.get(expr, {}))
        profiles.append(prof)

    try:
        trajectory.extra_info["segments"] = [p.to_serializable() for p in profiles]
    except Exception:  # noqa: BLE001 -- a read-only/odd trajectory must not break diagnosis
        logger.debug("segment_profiling: could not cache segments on extra_info")

    return profiles


__all__ = ["OP_CATEGORY", "SegmentProfile", "parse_factor", "build_segments"]