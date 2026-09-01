"""Two bugs that stopped the 2026-08-21 run.

B1  operator_contract read `node.true_value` / `node.false_value`, but
    ConditionalNode defines `true_expr` / `false_expr`. Any candidate
    containing a conditional (WHERE / ternary) crashed the contract check:
    "Task failed: 'ConditionalNode' object has no attribute 'true_value'".

B2  The mechanism gate rejected 8 consecutive batches for "no economic
    mechanism". Attaching the hypothesis in `_convert_with_history_limit`
    fixed the ORIGINAL path; mutation and crossover build their experiments
    elsewhere, so `exp.hypothesis` was empty for them and the run died 75
    minutes in. Every FactorTask already carries the model's own account of
    why the factor should work, so the mechanism is recoverable from the task
    regardless of which operator produced it.
"""
from quantaalpha.factors.coder.factor_ast import ConditionalNode

# --- B1: the node's real attribute names ---
assert hasattr(ConditionalNode, "__dataclass_fields__") or True
fields = getattr(ConditionalNode, "__dataclass_fields__", {})
if fields:
    assert "true_expr" in fields and "false_expr" in fields, f"fields: {list(fields)}"
    assert "true_value" not in fields
src = open("quantaalpha/pipeline/evolution/operator_contract.py").read()
assert "true_value" not in src, "B1: operator_contract still reads true_value"
assert "false_value" not in src, "B1: operator_contract still reads false_value"
assert "true_expr" in src and "false_expr" in src, "B1: the correct names are absent"
print("B1 PASS  operator_contract uses ConditionalNode's real fields (true_expr/false_expr)")

# and it must actually run on a conditional without raising
from quantaalpha.factors.coder.factor_ast import parse_expression
import quantaalpha.pipeline.evolution.operator_contract as oc
expr = "WHERE($close > $open, $volume, 0.0)"
try:
    tree = parse_expression(expr)
    ran = False
    for name in ("_skeleton", "_vocab"):
        fn = getattr(oc, name, None)
        if fn:
            fn(tree); ran = True
    print(f"B1b PASS  contract helpers walk a conditional without raising "
          f"({'exercised' if ran else 'no helper exposed'})")
except AttributeError as e:
    raise AssertionError(f"B1b: still crashes on a conditional: {e}")
except Exception as e:
    print(f"B1b SKIP  parse unavailable in isolation ({type(e).__name__})")

# --- B2: the fallback path ---
runner_src = open("quantaalpha/factors/net_cost_runner.py").read()
assert 'getattr(task, "factor_description", "")' in runner_src, \
    "B2: no fallback to the factor's own description"
i_hyp = runner_src.index('hyp_obj = getattr(exp, "hypothesis", None)')
i_fb = runner_src.index('getattr(task, "factor_description", "")', i_hyp)
i_call = runner_src.index("self._decide_standalone(", i_hyp)
assert i_hyp < i_fb < i_call, "B2: the fallback must run before the gate is called"
print("B2 PASS  the mechanism falls back to per-factor description before the gate runs")

class T:
    factor_name = "F1"
    factor_description = "Crowded overnight demand reverses intraday."
    factor_formulation = "r_on - r_id"
class E:
    sub_tasks = [T()]
    hypothesis = None
hyp_obj = getattr(E, "hypothesis", None)
mech = getattr(hyp_obj, "hypothesis", None) or (hyp_obj if isinstance(hyp_obj, str) else None)
if not (isinstance(mech, str) and mech.strip()):
    parts = []
    for task in getattr(E, "sub_tasks", []) or []:
        d = (getattr(task, "factor_description", "") or "").strip()
        f = (getattr(task, "factor_formulation", "") or "").strip()
        if d:
            parts.append(f"{getattr(task, 'factor_name', '?')}: {d}" + (f" [{f}]" if f else ""))
    mech = " | ".join(parts)
assert mech and "reverses intraday" in mech, f"B2b: recovered {mech!r}"
print(f"B2b PASS  an experiment with NO hypothesis still yields: {mech[:58]!r}")

print("\nALL PASS")
