"""The diagnosis LLM must see what was measured, not two lines of it.

The bug (2026-08-21 run, 95 diagnosis records): ``llm_diagnose`` renders
``parent.backtest_metrics`` through ``_METRIC_GLOSS`` with a strict
``if key in metrics`` guard. Three defects cut the "Everything measured on it"
block to ~2 lines in production:

  (A) casing -- gloss ``rank_ic`` vs dict ``RankIC`` (CamelCase);
  (B) tearsheet dropped -- the 14 per-factor admission scalars (t_nw,
      monotonicity, sign_predicted/realized, fdr_*, capacity, ...) live in
      ``factor_tearsheets[expr]`` and were dropped by the allowlist;
  (C) gloss never extended -- U, the 7 e_*, delta_*, cost_bps, RankICIR,
      weakest_dimensions, factor_attribution are present in the dict but had
      no gloss entry, so the block skipped them.

This file fails against the previous behaviour (that is the point) and passes
once the producer flattens the tearsheet to top-level (A1) and the consumer
case-folds + extends the gloss (A2).

  C1-C5  consumer: real-cased measurement block renders the rich keys
  P1-P5  producer: _extract_metrics flattens the candidate tearsheet + dsr
"""
import pandas as pd

import quantaalpha.pipeline.evolution.llm_diagnosis as LD
from quantaalpha.pipeline.evolution.controller import EvolutionController


# ---------------------------------------------------------------------------
# C. Consumer -- _measurement_block / _population_block on a REAL-cased dict
# ---------------------------------------------------------------------------
# CamelCase RankIC (defect A), flattened tearsheet scalars (defect B), and the
# rich keys the gloss never listed (defect C). This is what a net-cost
# trajectory's backtest_metrics actually looks like after A1.
REAL = {
    "U": 0.42,
    "RankIC": 0.021,               # CamelCase -- the casing defect
    "rank_ic_neutral": 0.018,      # flattened from factor_tearsheets
    "t_nw": 3.2,                   # flattened
    "monotonicity": 0.7,           # flattened
    "sign_predicted": "negative",  # flattened -- the pre-registered direction
    "sign_realized": "positive",   # flattened -- the OUTCOME (a sign flip)
    "rho_max": 0.55, "cx": 12.0, "dsr": 0.88,
    "delta_mean": -0.08, "delta_se": 0.23, "delta_t": -0.35,
    "cost_bps": 4.7, "RankICIR": 0.34,
    "e_effectiveness": 0.22, "e_arr": 0.19, "e_stability": 0.31,
    "e_turnover": 0.10, "e_diversity": 0.45, "e_decay": 0.28, "e_overfit": 0.6,
    "weakest_dimensions": "turnover, decay",
    "factor_attribution": {"expr1": {"weight": 0.5, "rank_ic": 0.02,
                                     "turnover_share": 0.3}},
    "turnover_book": 0.055, "net_ir": -0.1, "net_arr": -0.02,
    "admitted": False, "verdict": "net_harmful",
}

block = LD._measurement_block(REAL)
lines = [l for l in block.splitlines() if l.strip()]

# C1: the block is no longer ~2 lines. ~27 gloss entries match here.
assert len(lines) > 20, f"C1: block has only {len(lines)} lines (was ~2 before)"
print(f"C1 PASS  measurement block renders {len(lines)} lines (was ~2)")

# C2: casing -- RankIC (CamelCase) resolves to the "Raw RankIC" line + value.
assert "Raw RankIC" in block, "C2: the rank_ic->RankIC casing alias did not resolve"
assert "0.0210" in block, "C2: the RankIC value did not render"
print('C2 PASS  CamelCase RankIC resolves through the gloss alias')

# C3: gloss extension -- U, the e_* vector, delta_*, cost_bps, RankICIR,
# weakest_dimensions all reach the block (selection's actual objective).
for label in ("Repository-relative utility", "Score: effectiveness",
              "Score: turnover", "Marginal contribution to book net IR",
              "t-stat of marginal contribution", "Cost in bps/day",
              "Rank IC information ratio", "Weakest scored dimensions"):
    assert label in block, f"C3: gloss extension missing {label!r}"
print("C3 PASS  U, the e_* vector, delta_*, cost_bps, RankICIR, weakest_dimensions render")

# C4: tearsheet flatten visibility -- t_nw, monotonicity, and the sign flip
# (predicted negative / realized positive) are visible to the diagnosis.
assert "t-statistic (Newey-West)" in block, "C4: t_nw label missing"
assert "Decile monotonicity" in block, "C4: monotonicity label missing"
assert "Direction the hypothesis committed to: negative" in block, \
    "C4: sign_predicted did not render with its value"
assert "Direction the measurement produced: positive" in block, \
    "C4: sign_realized did not render -- the sign flip is invisible to diagnosis"
print("C4 PASS  t_nw, monotonicity, and the predicted/realized sign flip render")

# C5: factor_attribution renders as per-factor rows, not a raw dict str.
assert "Combiner credit per factor" in block, "C5: factor_attribution label missing"
assert "weight" in block and "turnover_share" in block, \
    "C5: factor_attribution rows did not render"
assert "{'weight'" not in block, \
    "C5: factor_attribution was str()'d as a raw dict instead of rendered as rows"
print("C5 PASS  factor_attribution renders as rows, not a raw dict")

# C6: _population_block shows the real RankIC on a prior (casing fix), not
# "not measured". The population channel is the search's memory.
_Prior = type("T", (), {"backtest_metrics": {"U": 0.2, "RankIC": 0.031,
                                             "t_nw": 2.5, "delta_mean": 0.02,
                                             "turnover_book": 0.04,
                                             "admitted": True},
                        "hypothesis": "Volume-conditioned reversal.",
                        "factors": []})
pblock = LD._population_block([_Prior()])
assert "0.0310" in pblock, "C6: _population_block dropped RankIC via casing"
assert "RankIC not measured" not in pblock, \
    "C6: the prior's RankIC rendered as 'not measured' (casing defect)"
assert "2.5000" in pblock, "C6: the prior's t_nw did not render"
print("C6 PASS  _population_block shows the real RankIC + t_nw on priors")
print("C1-C6 PASS  the diagnosis sees what was measured")


# ---------------------------------------------------------------------------
# P. Producer -- _extract_metrics flattens the candidate tearsheet + carries dsr
# ---------------------------------------------------------------------------
# A _to_series-style Series: factor_tearsheets nests the per-factor admission
# scalars under the expression; dsr is a top-level key the old allowlist dropped.
EXPR = "ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))"
TEARSHEET = {
    "t_nw": 3.2, "rank_ic_neutral": 0.025, "monotonicity": 0.7,
    "q_spread": 0.0011, "ls_sharpe": 1.1, "sign_predicted": "negative",
    "sign_realized": "negative", "mechanism_validated": True,
    "fdr_t_required": 3.1, "fdr_n_tests": 47, "capacity_cny": 5.0e8,
    "best_horizon": 5, "ic_pos_frac": 0.55, "exposure_size": -0.02,
}
series = pd.Series({
    "U": 0.42, "rho_max": 0.55, "cx": 12.0, "turnover_solo": 0.04,
    "dsr": 0.88, "dsr_n_trials": 47, "RankIC": 0.021,
    "factor_tearsheets": {EXPR: TEARSHEET},
})

ctl = EvolutionController.__new__(EvolutionController)  # class-attr methods; no __init__
m = ctl._extract_metrics(series, factor_exprs=[EXPR])

# P1: the 14 tearsheet scalars are promoted to top-level (lowercase -> gloss).
assert m["t_nw"] == 3.2, f"P1: t_nw not flattened ({m.get('t_nw')})"
assert m["monotonicity"] == 0.7, "P1: monotonicity not flattened"
assert m["rank_ic_neutral"] == 0.025, "P1: rank_ic_neutral not flattened"
assert m["ic_pos_frac"] == 0.55, "P1: ic_pos_frac not flattened"
print("P1 PASS  the 14 tearsheet scalars are promoted to top-level")

# P2: typed values survive (string sign, bool flag, int horizon).
assert m["sign_predicted"] == "negative", "P2: string sign_predicted not preserved"
assert m["sign_realized"] == "negative", "P2: string sign_realized not preserved"
assert m["mechanism_validated"] is True, "P2: bool mechanism_validated not preserved"
assert m["best_horizon"] == 5, "P2: int best_horizon not preserved"
print("P2 PASS  string/bool/int tearsheet values keep their type")

# P3: dsr (top-level Series key the old allowlist dropped) now carries.
assert m["dsr"] == 0.88, f"P3: dsr dropped ({m.get('dsr')})"
assert m["dsr_n_trials"] == 47, "P3: dsr_n_trials dropped"
print("P3 PASS  dsr + dsr_n_trials survive _extract_net_cost_metrics")

# P4: the whole factor_tearsheets dict is ALSO carried (lineage/debuggability).
assert isinstance(m.get("factor_tearsheets"), dict), "P4: factor_tearsheets dict not carried"
assert m["factor_tearsheets"][EXPR]["t_nw"] == 3.2, "P4: carried dict lost the entry"
print("P4 PASS  factor_tearsheets dict carried through for lineage")

# P5: candidate selection -- match by expr first; else the strongest |t_nw|.
multi = pd.Series({"factor_tearsheets": {
    "A": {"t_nw": 1.0}, "B": {"t_nw": 5.0}, "C": {"t_nw": 2.0}}})
assert ctl._extract_metrics(multi, factor_exprs=["A"])["t_nw"] == 1.0, \
    "P5: factor_exprs match did not select entry A"
assert ctl._extract_metrics(multi, factor_exprs=["not_present"])["t_nw"] == 5.0, \
    "P5: no-match fallback did not select the strongest |t_nw| (B=5.0)"
assert ctl._extract_metrics(
    pd.Series({"factor_tearsheets": {"only": {"t_nw": 7.0}}}),
    factor_exprs=None)["t_nw"] == 7.0, "P5: single-entry fallback failed"
print("P5 PASS  candidate selection: expr-match first, then strongest, then sole")

print("\nALL PASS")