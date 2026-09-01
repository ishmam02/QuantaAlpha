
# ---------------------------------------------------------------------------
# The metrics added in the winsor/FDR/mechanism pass must reach the generator.
# A gate the generator is not told about is a gate it cannot learn from.
NEW = ["fdr_t_required", "fdr_n_tests", "capacity_cny"]

from quantaalpha.factors.net_cost_feedback import _RAW_METRICS, _METRIC_HELP
raw_keys = {k for k, _, _ in _RAW_METRICS}
missing = [k for k in NEW if k not in raw_keys]
assert not missing, f"not surfaced in feedback: {missing}"
no_help = [k for k in NEW if k not in _METRIC_HELP]
assert not no_help, f"surfaced without an explanation of what it means: {no_help}"
print(f"NEW-1 PASS  {len(NEW)} new metrics are in _RAW_METRICS and each carries a meaning")

from quantaalpha.pipeline.evolution.llm_diagnosis import _METRIC_GLOSS
gloss_keys = {k for k, _, _, _ in _METRIC_GLOSS}
missing_g = [k for k in NEW if k not in gloss_keys]
assert not missing_g, f"not in the diagnosis gloss: {missing_g}"
print("NEW-2 PASS  the same metrics are in the diagnosis gloss the LLM reads")

# They must survive RENDERING, not merely exist in a table.
sheet = {"rank_ic_neutral": 0.031, "t_nw": 3.24, "best_horizon": 5,
         "ic_pos_frac": 0.54, "monotonicity": 0.61, "rho_max": 0.42,
         "fdr_t_required": 3.61, "fdr_n_tests": 87, "capacity_cny": 4.2e8,
         "turnover_solo": 0.18}
from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback as _FB
class _Stub:
    _fmt = staticmethod(_FB._fmt)
    _per_factor_lines = _FB._per_factor_lines
txt = "\n".join(_Stub()._per_factor_lines(
    {"factor_tearsheets": {"x_expr": sheet}, "admitted_exprs": []}))
for k in NEW:
    assert k in txt or _METRIC_HELP.get(k, "@@")[:20] in txt, f"{k} vanished in rendering"
assert "3.61" in txt and "87" in txt, "the FDR numbers did not render"
assert "4.2" in txt.replace("e+08", "e8"), "capacity did not render"
print("NEW-3 PASS  the numbers survive rendering into the text the LLM is sent")
print(f"          sample: {[l for l in txt.splitlines() if 'fdr' in l][:1]}")

# No PRESCRIPTION: the surface may state what was measured, never what to do.
BANNED = ["lengthen", "shorten", "you should", "try using", "instead use", "simplify"]
low = txt.lower()
hits = [b for b in BANNED if b in low]
assert not hits, f"prescriptive language reached the generator: {hits}"
print("NEW-4 PASS  the new text diagnoses without prescribing a remedy")
