"""Each operator must be checked by ITS OWN contract -- or deliberately not.

Refine and crossover BOTH set refine_mode and refine_target='expression', so the
loop cannot tell them apart by those alone. It nearly didn't: crossover carried
no parent_prefix, so the contract returned early and crossover children were
never checked at all while appearing to be covered.

D1  a REFINE task applies the literal-only check
D2  a CROSSOVER task applies the both-parent check (NOT the refine check)
D3  an ORTHOGONAL task applies NO check -- a restart has no parent to match
D4  the contract never empties a batch
"""
from quantaalpha.pipeline.loop import AlphaAgentLoop


class Task:
    """Just the attributes _apply_operator_contract reads."""
    def __init__(self, exprs):
        self.sub_tasks = [type("S", (), {"factor_expression": e})() for e in exprs]


class Loop:
    _apply_operator_contract = AlphaAgentLoop._apply_operator_contract

    def __init__(self, directive=None, prefix=None, cx=None):
        self.refine_directive = directive
        self.parent_prefix = prefix
        self.crossover_parents = cx


PARENT = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"
WINDOW_EDIT = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 20))"
STRUCTURAL = "RANK(TS_MEAN(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"
EXPR_DIRECTIVE = {"refine_target": "expression", "verdict": "marginal"}

A = ["RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"]   # $open
B = ["RANK(TS_MEAN(ABS($return) / ($volume + 1), 20))"]                  # $volume/$return
ONE_PARENT = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 20))"
BOTH = "RANK(TS_MEAN(($open - DELAY($close, 1)) / ($volume + 1), 20))"

# --- D1: refine ---------------------------------------------------------------
t = Task([WINDOW_EDIT, STRUCTURAL])
Loop(EXPR_DIRECTIVE, {"factors": [{"expression": PARENT}]})._apply_operator_contract(t)
kept = [s.factor_expression for s in t.sub_tasks]
assert kept == [STRUCTURAL], f"D1: refine check did not drop the literal-only child: {kept}"
print("D1 PASS  refine task -> literal-only child dropped, structural kept")

# --- D2: crossover ------------------------------------------------------------
t = Task([ONE_PARENT, BOTH])
Loop(EXPR_DIRECTIVE, None, {"a": A, "b": B})._apply_operator_contract(t)
kept = [s.factor_expression for s in t.sub_tasks]
assert kept == [BOTH], f"D2: crossover check did not drop the single-parent child: {kept}"
# And it must be the CROSSOVER rule, not the refine rule: ONE_PARENT is a
# literal-only edit of A, so both rules reject it -- use a child that only the
# crossover rule can catch (structural vs A, but still single-parent).
# NOT TS_MEAN -- that token is distinctive to B, so a child using it HAS
# inherited from B and is a legitimate crossover (A's signal, B's temporal op).
# Wrap in ZSCORE instead: structurally different from A, but still carrying
# nothing that is distinctive to B.
only_a_structural = "ZSCORE(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"
t2 = Task([only_a_structural, BOTH])
Loop(EXPR_DIRECTIVE, None, {"a": A, "b": B})._apply_operator_contract(t2)
kept2 = [s.factor_expression for s in t2.sub_tasks]
assert kept2 == [BOTH], (
    "D2: a child that is STRUCTURAL against A but still single-parent must be "
    f"caught by the crossover rule; got {kept2}")
print("D2 PASS  crossover task -> single-parent dropped even when structurally novel")

# --- D3: orthogonal -----------------------------------------------------------
t = Task([WINDOW_EDIT, STRUCTURAL])
Loop(None, None, None)._apply_operator_contract(t)          # no directive at all
assert len(t.sub_tasks) == 2, "D3: an orthogonal restart must not be contract-checked"
print("D3 PASS  orthogonal task -> no contract applied (nothing to match against)")

# --- D4: never empties a batch ------------------------------------------------
t = Task([WINDOW_EDIT])                                      # the ONLY child is bad
Loop(EXPR_DIRECTIVE, {"factors": [{"expression": PARENT}]})._apply_operator_contract(t)
assert len(t.sub_tasks) == 1, "D4: the contract must not empty a batch"
print("D4 PASS  a batch whose every child fails is kept, not emptied")

print("\nALL PASS")
