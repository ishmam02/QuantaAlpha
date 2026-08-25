"""Structural acceptance tests for the evolution operators.

Nothing currently checks that an operator did the job it was asked to do. The
measured consequence: **every refine child changes a lookback window**,
whichever weakness the diagnosis named -- one rescue and four regressions across
two runs. Given an expression and a free-form instruction, the minimum-edit path
for a language model is to change a number, and a prompt that merely asks for a
structural edit does not stop that.

So a better diagnosis alone buys nothing. Feed a precise, well-formatted
differential diagnosis to an operator that answers every instruction by editing
one constant and you get better-labelled constant edits. These checks are the
other half: they run BEFORE the expensive evaluation, and they make the
structural edit mandatory rather than merely requested.

Two checks, both structural rather than statistical:

* ``check_refine`` rejects a child whose AST differs from its parent's ONLY in
  numeric literals. Binary, cheap, and it is precisely the observed failure.
* ``check_crossover`` rejects a child that inherits from only one parent, using
  the set difference of each parent's distinctive vocabulary rather than a
  similarity score -- token-overlap scores could not separate real inheritance
  from chance (measured child-vs-far-parent similarity 0.543 against a null of
  0.417 whose own p90 was 0.592).

Deliberately NOT included: a cap on how many windows may change, and an "edit
alphabet" of permitted moves. Both were tried and removed. They constrain what
the model may think rather than checking what it produced, and the window cap
false-positived on a live run because an operator swap read as two window
changes. The rule here is: check the artifact, never restrict the reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


@dataclass
class ContractResult:
    ok: bool
    reason: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


# --------------------------------------------------------------------------
# Canonical forms
# --------------------------------------------------------------------------
def _skeleton(node: Node) -> str:
    """Canonical string with every numeric literal replaced by ``#``.

    Two expressions share a skeleton exactly when they differ only in constants
    -- which is the definition of the edit this contract exists to reject.
    """
    if isinstance(node, NumberNode):
        return "#"
    if isinstance(node, VarNode):
        return str(node.name)
    if isinstance(node, FunctionNode):
        # `name` is normally a str but can itself be a VarNode.
        name = getattr(node.name, "name", node.name)
        return f"{name}({','.join(_skeleton(a) for a in node.args)})"
    if isinstance(node, BinaryOpNode):
        return f"({_skeleton(node.left)}{node.op}{_skeleton(node.right)})"
    if isinstance(node, UnaryOpNode):
        return f"({node.op}{_skeleton(node.operand)})"
    if isinstance(node, ConditionalNode):
        return (f"({_skeleton(node.condition)}?{_skeleton(node.true_expr)}"
                f":{_skeleton(node.false_expr)})")
    return str(node)


def _vocabulary(node: Node) -> set[str]:
    """Operators and base features in a tree. Numbers are excluded on purpose.

    Constants are what the operators fiddle with, so counting them as inherited
    vocabulary would let a child "inherit" from a parent by copying a 20.
    """
    out: set[str] = set()

    def walk(n: Node) -> None:
        if isinstance(n, VarNode):
            out.add(str(n.name))
        elif isinstance(n, FunctionNode):
            out.add(str(getattr(n.name, "name", n.name)))
            for a in n.args:
                walk(a)
        elif isinstance(n, BinaryOpNode):
            out.add(str(n.op)); walk(n.left); walk(n.right)
        elif isinstance(n, UnaryOpNode):
            out.add(str(n.op)); walk(n.operand)
        elif isinstance(n, ConditionalNode):
            out.add("?:"); walk(n.condition); walk(n.true_expr); walk(n.false_expr)
    walk(node)
    return out


def _safe_parse(expr: str) -> Node | None:
    try:
        return parse_expression(expr)
    except Exception:
        return None


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------
def check_refine(child_expr: str, parent_expr: str) -> ContractResult:
    """A refine child must differ from its parent by more than constants.

    Unparseable input PASSES. This gate exists to catch a specific known
    failure, not to become a second syntax checker that silently drops work the
    real parser would have accepted.
    """
    c, p = _safe_parse(child_expr), _safe_parse(parent_expr)
    if c is None or p is None:
        return ContractResult(True, "unparseable", "not checked")
    if _skeleton(c) == _skeleton(p):
        return ContractResult(
            False, "literal_only",
            "the child's structure is identical to the parent's; only numeric "
            "constants changed")
    return ContractResult(True, "structural")


def check_crossover(child_expr: str, parent_a_exprs: list[str],
                    parent_b_exprs: list[str]) -> ContractResult:
    """A crossover child must carry vocabulary DISTINCTIVE to each parent.

    Set difference, not similarity: only tokens unique to one parent count as
    evidence of inheritance, so shared boilerplate (``RANK``, ``$close``) cannot
    manufacture a false positive.

    When the parents share their entire vocabulary there is nothing distinctive
    to inherit and the check passes rather than failing an impossible demand.
    """
    c = _safe_parse(child_expr)
    if c is None:
        return ContractResult(True, "unparseable", "not checked")
    va, vb = set(), set()
    for e in parent_a_exprs:
        n = _safe_parse(e)
        if n is not None:
            va |= _vocabulary(n)
    for e in parent_b_exprs:
        n = _safe_parse(e)
        if n is not None:
            vb |= _vocabulary(n)

    only_a, only_b = va - vb, vb - va
    if not only_a or not only_b:
        return ContractResult(True, "parents_indistinct",
                              "parents share their vocabulary; nothing distinctive to inherit")
    cv = _vocabulary(c)
    got_a, got_b = cv & only_a, cv & only_b
    if not got_a or not got_b:
        missing = "A" if not got_a else "B"
        return ContractResult(
            False, "single_parent",
            f"the child carries nothing distinctive to parent {missing} "
            f"(from A: {sorted(got_a)}, from B: {sorted(got_b)})")
    return ContractResult(True, "recombined",
                          f"from A: {sorted(got_a)}, from B: {sorted(got_b)}")


# --------------------------------------------------------------------------
# Enforcement
# --------------------------------------------------------------------------
@dataclass
class ContractReport:
    checked: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)

    def note(self, r: ContractResult) -> None:
        self.checked += 1
        if not r.ok:
            self.rejected += 1
            self.reasons.append(r.reason)


def enforce_refine(child_exprs: list[str], parent_exprs: list[str],
                   report: ContractReport | None = None) -> tuple[list[str], list[str]]:
    """Split refine children into (accepted, rejected-with-reason).

    A child is rejected when it is literal-only against EVERY parent expression
    -- if it restructured relative to any of them, it did structural work.
    """
    ok, bad = [], []
    for ch in child_exprs:
        results = [check_refine(ch, p) for p in parent_exprs] or [ContractResult(True)]
        passed = any(r.ok for r in results)
        if report is not None:
            report.note(results[0] if not passed else ContractResult(True, "structural"))
        (ok if passed else bad).append(ch)
        if not passed:
            logger.info("operator contract [refine] REJECTED: %s -- %s",
                        ch[:70], results[0].detail)
    return ok, bad


def enforce_crossover(child_exprs: list[str], parent_a_exprs: list[str],
                      parent_b_exprs: list[str],
                      report: ContractReport | None = None) -> tuple[list[str], list[str]]:
    """Split crossover children into (accepted, rejected-with-reason)."""
    ok, bad = [], []
    for ch in child_exprs:
        r = check_crossover(ch, parent_a_exprs, parent_b_exprs)
        if report is not None:
            report.note(r)
        (ok if r.ok else bad).append(ch)
        if not r.ok:
            logger.info("operator contract [crossover] REJECTED: %s -- %s",
                        ch[:70], r.detail)
    return ok, bad


def rejection_note(reason: str) -> str:
    """The message handed BACK to the model on a rejection.

    Names what was wrong with the artifact and stops. It does not say what to do
    instead -- the diagnosis already carries the measurements, and prescribing
    the fix here would put the system's guess ahead of the model's reasoning.
    """
    return {
        "literal_only":
            "Your previous attempt changed only numeric constants; the structure "
            "of the expression was identical to the parent's. That is not a "
            "structural edit.",
        "single_parent":
            "Your previous attempt drew on only one parent. A crossover child "
            "must carry material distinctive to both.",
    }.get(reason, "Your previous attempt did not satisfy the operator contract.")


__all__ = ["ContractResult", "ContractReport", "check_refine", "check_crossover",
           "enforce_refine", "enforce_crossover", "rejection_note"]
