"""Operator-coverage measurement for the mined factor population.

The factor population converges on a small operator set (measured: ``TS_MEAN``
21/21, ``RANK`` 81%, ``DELAY`` 48% of calls; only 12 of ~55 declared operators
exercised). The direction domain narrows with it -- every round-0 direction maps
onto OHLCV price-volume microstructure, and none reach regression residuals
(``REGBETA``/``REGRESI``), technical indicators (``RSI``/``MACD``) or conditional
logic (``COUNT``/``SUMIF``/``FILTER``).

This module turns a population of factor expressions into a MEASUREMENT of that
convergence -- which operators are exercised, with what call-share, and which
declared operators the population has not yet touched. It emits a text block that
is injected into two places that steer diversity:

* the reseed digest (``controller._build_reseed_digest``) -- so the informed
  directions see which operator CLASSES are exhausted and broaden toward the
  unused ones, not just toward unused signal space;
* the factor feedback (``NetCostFactorFeedback._format_metric_block``) -- so the
  within-mechanism generator sees the same coverage when it picks an operator.

It is a MEASUREMENT, not a gate. Nothing is rejected for operator overlap -- that
would admit worse factors to enforce a shape. And it states the measurement only:
the block names the unused operators as the LOCATION of the convergence, then
closes with "how to broaden that is yours to determine." It prescribes no remedy
(no "use"/"try"/"should"/"consider") and carries no market prior -- so it keeps
the standing hard rules ("prompts DIAGNOSE, never PRESCRIBE"; "NO hardcoded
market-specific priors") intact.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

from quantaalpha.factors.coder import factor_ast, function_lib as func_lib

logger = logging.getLogger(__name__)

# Operators that are NOT real factor transforms -- arithmetic, comparison and
# logical pseudo-ops dispatched as binary/unary nodes, plus ``SEQUENCE`` (a
# REGBETA/REGRESI helper column) and ``WHERE`` (a conditional ternary). These are
# excluded from the declared set so the coverage gap names genuine operators only.
_PSEUDO_OPS: frozenset[str] = frozenset({
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "GT", "LT", "GE", "LE", "EQ", "NE",
    "AND", "OR", "WHERE", "SEQUENCE", "FLOOR",
})


def _declared_operators() -> frozenset[str]:
    """Top-level uppercase names defined in ``function_lib`` minus pseudo-ops.

    Introspected ONCE at import time. Dispatch in the factor calculator is
    ``getattr(func_lib, name)``, so every name here is executable -- the
    monoculture is a generation choice, not a crash artifact, and the coverage gap
    can safely name these operators as reachable-but-unused.
    """
    out: set[str] = set()
    for name in dir(func_lib):
        if not re.match(r"^[A-Z][A-Z0-9_]*$", name):
            continue
        obj = getattr(func_lib, name)
        if not callable(obj):
            continue
        if name in _PSEUDO_OPS:
            continue
        out.add(name)
    return frozenset(out)


DECLARED_OPERATORS: frozenset[str] = _declared_operators()


def extract_operators(expr: str) -> set[str]:
    """Return the set of operator names used in ``expr`` (upper-cased).

    Degrades safely: a malformed/unparseable expression returns an empty set
    rather than raising, so a bad factor in the population never breaks the
    coverage measurement -- and never breaks the reseed or feedback that carry
    it.
    """
    if not isinstance(expr, str) or not expr.strip():
        return set()
    try:
        tree = factor_ast.parse_expression(expr)
    except Exception:  # parse_expression raises ValueError on malformed input
        return set()
    operators: set[str] = set()
    factor_ast.collect_operators(tree, operators)
    return operators


def top_exercised_operators(expressions: Iterable[str], k: int = 3) -> list[str]:
    """The ``k`` most-exercised operator names, by call-share then name.

    A structured counterpart to ``coverage_block``'s printed "Top 3" line, so the
    direction-level novelty gate can name the monoculture set without re-parsing
    the rendered block. Ranks by call count (matches ``coverage_block``'s
    ``ranked``), falling back to alphabetical for determinism. Degrades to ``[]``
    when nothing parses.
    """
    counts: Counter[str] = Counter()
    for expr in expressions:
        for op in extract_operators(expr):
            counts[op] += 1
    ranked = sorted(counts, key=lambda op: (-counts[op], op))
    if k <= 0:
        return []
    return ranked[:k]


# Words that turn a measurement into a prescription. The block is scanned for
# these so a future edit cannot accidentally start telling the generator what to
# do -- the standing hard rule is "prompts DIAGNOSE, never PRESCRIBE."
_REMEDY_TOKENS = ("use ", "try ", "should ", "consider ", "ought ", "prefer ")


def coverage_block(expressions: Iterable[str]) -> str:
    """Render a measurement-only operator-coverage block for a population.

    Over the population of factor expressions, reports per-operator factor
    coverage, call-share, the top-3 concentration, and the set of declared
    operators the population has NOT yet exercised (the LOCATION of the
    convergence). Returns ``""`` when there is nothing to measure (empty
    population, or every expression failed to parse) so callers degrade silently.

    The block states the measurement and stops. It names the unused operators as
    where the convergence sits, then closes "how to broaden that is yours to
    determine" -- never a remedy, never a market prior.
    """
    exprs = [e for e in expressions if isinstance(e, str) and e.strip()]
    if not exprs:
        return ""

    # Per-expression operator sets (factor coverage) and total call counts
    # (call-share). An operator that appears once in an expression counts once
    # toward call-share for that factor; factor coverage counts a factor once per
    # operator it uses at all.
    per_expr: list[set[str]] = []
    call_counts: Counter[str] = Counter()
    parsed = 0
    for expr in exprs:
        ops = extract_operators(expr)
        if not ops:
            continue
        parsed += 1
        per_expr.append(ops)
        for op in ops:
            call_counts[op] += 1

    if parsed == 0:
        return ""

    total_calls = sum(call_counts.values())
    factor_coverage: Counter[str] = Counter()
    for ops in per_expr:
        for op in ops:
            factor_coverage[op] += 1

    n_factors = parsed
    # Most-used first, by call-share then by factor coverage for stability.
    ranked = sorted(
        call_counts,
        key=lambda op: (-call_counts[op], -factor_coverage[op], op),
    )

    lines: list[str] = []
    lines.append(
        f"Operator coverage so far (population of {n_factors} mined factors, "
        f"{total_calls} operator calls):"
    )

    used_lines: list[str] = []
    for op in ranked:
        share = (call_counts[op] / total_calls * 100.0) if total_calls else 0.0
        cov = (factor_coverage[op] / n_factors * 100.0) if n_factors else 0.0
        used_lines.append(f"  {op} {factor_coverage[op]}/{n_factors} factors ({cov:.0f}%), {call_counts[op]} calls ({share:.0f}%)")
    lines.extend(used_lines)

    if ranked:
        top3_calls = sum(call_counts[op] for op in ranked[:3])
        top3_share = (top3_calls / total_calls * 100.0) if total_calls else 0.0
        top3_names = ", ".join(ranked[:3])
        lines.append(f"Top 3 operators ({top3_names}) = {top3_share:.0f}% of all calls.")

    used = set(call_counts)
    declared = set(DECLARED_OPERATORS)
    unused = sorted(declared - used)
    lines.append(
        f"{len(used & declared)} of {len(declared)} declared operators exercised "
        f"so far; {len(unused)} not yet used in the population:"
    )
    if unused:
        lines.append("  " + ", ".join(unused))

    lines.append(
        "The population has converged on a small operator set. How to broaden "
        "that is yours to determine."
    )

    block = "\n".join(lines)
    # Defensive: never let a prescription leak through. The block is assembled
    # from operator names and fixed framing prose, none of which carry a remedy
    # token, but a future edit might -- so assert it before returning.
    low = block.lower()
    assert not any(tok in low for tok in _REMEDY_TOKENS), (
        "operator-coverage block must not prescribe; found a remedy token")
    return block


__all__ = ["DECLARED_OPERATORS", "extract_operators", "coverage_block",
           "top_exercised_operators"]