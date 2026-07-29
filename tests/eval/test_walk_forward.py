import json, pathlib, numpy as np
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import align_signal, load_factor_signal

th = load_protocol(default_protocol_path())
lib = json.loads(pathlib.Path("data/factorlib/all_factors_library_control_20260728_151227.json").read_text())
exprs = [f["factor_expression"] for f in list(lib["factors"].values())[:5]]

def run(theta, report):
    op = EvaluationOperator(theta)
    s, e, win = op._windows(report)
    panel = op._panel(s, e)
    zoo = {x: align_signal(load_factor_signal(x), panel) for x in exprs[:3]}
    cand = {x: align_signal(load_factor_signal(x), panel) for x in exprs[3:]}
    return op.evaluate(cand, zoo_signals=zoo, zoo_metrics=[], report=report)

off = replace(th, walk_forward=replace(th.walk_forward, enabled=False))
on  = replace(th, walk_forward=replace(th.walk_forward, enabled=True, folds=3))

r_off = run(off, False)
r_on  = run(on,  False)
print(f"in-loop, walk-forward OFF : net_ir={r_off['m_net_ir']:+.4f} folds={r_off.get('m_n_folds','1')}")
print(f"in-loop, walk-forward ON  : net_ir={r_on['m_net_ir']:+.4f} folds={r_on.get('m_n_folds')}")
assert r_on.get("m_n_folds") == 3, "walk-forward should average 3 folds"
assert r_off.get("m_n_folds") in (None, 1)

# the reporting window must be identical either way
rep_off = run(off, True); rep_on = run(on, True)
print(f"\nfinal-test window OFF: {rep_off['eval_window']}")
print(f"final-test window ON : {rep_on['eval_window']}")
assert rep_off["eval_window"] == rep_on["eval_window"] == "2022-01-01..2025-12-26"
assert rep_on.get("m_n_folds") in (None, 1), "report mode must not average folds"
assert abs(rep_off["m_net_ir"] - rep_on["m_net_ir"]) < 1e-9, \
    "walk-forward must not change the final-test evaluation"
print("OK  final-test scoring is byte-identical with walk-forward on or off")
print("OK  in-loop scoring averages folds; test range untouched")
