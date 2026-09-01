"""Three folds, and the per-fold result must SURVIVE the aggregation.

F1  the folds are three distinct regimes, and final_test is never touched
F2  the per-fold vector is retained, not collapsed to its mean
F3  two batches with identical means but different regime behaviour are
    distinguishable -- which is the entire reason for running folds
F4  the worst fold is identifiable (where the money is lost)

F2-F4 all fail against the previous behaviour, which reduced folds with
np.mean and kept nothing else.
"""
import numpy as np
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")

# --------------------------------------------------------------------------
# F1: the folds are the regimes claimed
# --------------------------------------------------------------------------
# Read the count from Theta rather than pinning 3. The same note below applies
# to the fold COUNT as to the years: the 2026-08-24 re-split cut folds to 2
# (a third would have trained on 409 days), and a hardcoded 3 turned this
# regime-visibility test into a tripwire on the split itself. What must hold is
# that there is MORE THAN ONE fold and the per-fold vector survives -- that is
# what the test exists for.
N_FOLDS = TH.walk_forward.folds
assert TH.walk_forward.enabled and N_FOLDS >= 2, \
    f"F1: walk-forward must run >=2 folds to expose regimes, got {N_FOLDS}"
folds = TH.splits.walk_forward_folds(N_FOLDS)
assert len(folds) == N_FOLDS, f"F1: got {len(folds)} folds, expected {N_FOLDS}"

# Assert the PROPERTY, not fixed years: Phase 4 extended the sample to 2005
# and widened each valid window to three years, so the folds now validate on
# 2012-15 / 2016-18 / 2019-21. Hardcoding years made this test a tripwire on
# any window change rather than a check that folds are distinct and ordered.
valid_years = [va[0][:4] for _, va in folds]
assert len(set(valid_years)) == N_FOLDS, f"F1: folds must be distinct, got {valid_years}"
assert valid_years == sorted(valid_years), f"F1: folds must be ordered, got {valid_years}"
prev_end = None
for _, va in folds:
    if prev_end is not None:
        assert va[0] > prev_end, f"F1: fold valid windows overlap: {prev_end} -> {va[0]}"
    prev_end = va[1]

test_start = TH.splits.final_test[0]
for tr, va in folds:
    assert tr[1] < va[0], "F1: train must end before valid"
    assert va[1] < test_start, (
        f"F1: fold valid {va} reaches into final_test ({test_start}) -- "
        "folds must be carved from train+valid only")
# Every fold trains from the same start (expanding, not sliding): a sliding
# window would confound "later regime" with "less data".
assert len({tr[0] for tr, _ in folds}) == 1, "F1: folds must expand, not slide"
print(f"F1 PASS  {len(folds)} distinct ordered folds, valid starts "
      f"{valid_years}, expanding, final_test untouched")


# --------------------------------------------------------------------------
# F2-F4: the aggregation keeps the per-fold detail
# --------------------------------------------------------------------------
def aggregate(fold_net_irs):
    """Drive _fit_and_price with stubbed folds so no qlib/data is needed."""
    op = EvaluationOperator(TH)
    books = [{"net_ir": v, "net_arr": v / 10.0, "rank_ic": 0.02,
              "turnover_book": 0.05, "cost_bps": 4.7, "mdd": -0.1}
             for v in fold_net_irs]
    seq = iter(books)
    op._book = lambda *a, **k: dict(next(seq))
    op._folds = lambda report: [
        (TH, (f"{2019 + i}-01-01", f"{2019 + i}-12-31"))
        for i in range(len(fold_net_irs))]

    import quantaalpha.eval.combiner as cm
    real = cm.fit_predict
    cm.fit_predict = lambda *a, **k: (None, None)
    try:
        return op._fit_and_price({}, {}, None, ("2019-01-01", "2021-12-31"), False)[0]
    finally:
        cm.fit_predict = real


# Same mean (+0.50), completely different propositions.
steady = aggregate([0.50, 0.50, 0.50])   # works in every regime
lucky = aggregate([1.50, 0.00, 0.00])    # worked in exactly one year

assert abs(steady["net_ir"] - lucky["net_ir"]) < 1e-9, \
    "fixture broken: the two cases must have the same mean"

assert steady.get("fold_net_ir") == [0.50, 0.50, 0.50], \
    f"F2: per-fold vector lost, got {steady.get('fold_net_ir')}"
assert lucky.get("fold_net_ir") == [1.50, 0.00, 0.00], \
    f"F2: per-fold vector lost, got {lucky.get('fold_net_ir')}"
assert steady.get("fold_windows") and len(steady["fold_windows"]) == 3, \
    "F2: fold windows must be recorded so a fold can be NAMED"
print("F2 PASS  per-fold vector retained alongside the mean")

# F3: the two are now distinguishable, and were not before.
assert steady["net_ir_fold_std"] < lucky["net_ir_fold_std"], \
    "F3: fold dispersion must separate a steady edge from a one-year edge"
assert steady["folds_positive"] == 3 and lucky["folds_positive"] == 1, \
    f"F3: folds_positive {steady['folds_positive']} vs {lucky['folds_positive']}"
assert steady["fold_sign_consistent"] is True, "F3: steady must be sign-consistent"
assert lucky["fold_sign_consistent"] is False, "F3: lucky must NOT be sign-consistent"
print("F3 PASS  same mean, different regimes -- now separable (was not)")

# F4: where is the money lost.
losing = aggregate([-0.80, 0.10, 0.60])
worst_i = losing["fold_net_ir"].index(losing["net_ir_fold_min"])
assert losing["fold_windows"][worst_i].startswith("2019"), \
    f"F4: worst fold should be 2019, got {losing['fold_windows'][worst_i]}"
assert losing["folds_positive"] == 2, "F4: two folds positive"
assert losing["net_ir_fold_min"] == -0.80, "F4: min must be the losing fold"
print("F4 PASS  the losing regime is identifiable by name")

print("\nALL PASS")
