"""The model must be TOLD the numbers it is JUDGED on.

Before this, factors were admitted per-factor on neutralized significance while
the feedback showed book aggregates plus a RAW RankIC -- a different criterion,
and one contaminated by the size exposure the gate had just stripped.

F1  the research metrics reach the text
F2  each metric carries its MEANING and direction
F3  the per-FOLD vector reaches the text (which regime failed, not just the mean)
F4  the per-FACTOR sheets reach the text, rejections included
F5  a size-dominated factor is legible AS size-dominated
F6  the text states measurements and does not prescribe fixes
"""
from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback


class Stub:
    _format_metric_block = NetCostFactorFeedback._format_metric_block
    _per_fold_lines = NetCostFactorFeedback._per_fold_lines
    _per_factor_lines = NetCostFactorFeedback._per_factor_lines
    _fmt = staticmethod(NetCostFactorFeedback._fmt)
    _normalize = staticmethod(NetCostFactorFeedback._normalize)


AMIHUD = "RANK(TS_MEAN(ABS($return) / ($volume * $vwap), 20))"
GAP = "RANK(TS_SUM(($open - DELAY($close,1)) / DELAY($close,1), 5))"

raw = {
    "U": 0.42, "n_factors": 2,
    "m_net_ir": 0.31, "m_net_arr": 0.049,
    # per-fold: the thing that used to be averaged away
    "fold_net_ir": [-0.80, 0.10, 0.60],
    "fold_windows": ["2012-12-31..2015-12-31", "2016-01-01..2018-12-31",
                     "2019-01-01..2021-12-31"],
    "admitted_exprs": [GAP],
    "factor_tearsheets": {
        AMIHUD: {"rank_ic": -0.0146, "rank_ic_neutral": -0.0015, "t_nw": -0.41,
                 "exposure_size": -0.699, "monotonicity": 0.21, "ic_pos_frac": 0.51,
                 "best_horizon": 1, "rho_max": 0.33},
        GAP: {"rank_ic": 0.0198, "rank_ic_neutral": 0.0174, "t_nw": 4.20,
              "exposure_size": -0.021, "monotonicity": 0.43, "ic_pos_frac": 0.556,
              "best_horizon": 1, "rho_max": 0.00},
    },
}
text = Stub()._format_metric_block(raw)

# --- F1: the research metrics are present ------------------------------------
for token in ("Neutralized RankIC", "Newey-West", "Decile monotonicity",
              "Correlation with SIZE", "Horizon where the edge is strongest"):
    assert token in text, f"F1: '{token}' missing from the feedback the model reads"
print("F1 PASS  neutralized RankIC, t_nw, monotonicity, size exposure, horizon all present")

# --- F2: meanings, not bare numbers ------------------------------------------
for token in ("AFTER removing size, industry and beta",
              "standard errors", "tails only", "company size"):
    assert token in text, f"F2: meaning for '{token}' not supplied"
# and no format spec leaked in place of a number (the old _fmt bug)
for bad in (".4f", "+.4f", "{:", ".2%}"):
    assert bad not in text, f"F2: a format spec leaked into the prompt: {bad!r}"
print("F2 PASS  every metric carries its meaning; no format spec leaked")

# --- F3: per-fold ------------------------------------------------------------
assert "Per-fold result" in text, "F3: the per-fold vector never reaches the model"
assert "2012-12-31..2015-12-31" in text, "F3: folds must be NAMED, not numbered"
assert "-0.8000" in text, "F3: the losing fold's value must be shown"
assert "positive in 2 of 3" in text, "F3: sign consistency must be stated"
print("F3 PASS  per-fold results named, with the losing regime visible")

# --- F4: per-factor, rejections included -------------------------------------
assert "Per-factor measurements" in text, "F4: per-factor sheets never reach the model"
assert "[KEPT]" in text and "[not kept]" in text, "F4: kept/not-kept must be marked"
assert AMIHUD[:40] in text, "F4: the REJECTED factor must appear -- that is the lesson"
print("F4 PASS  per-factor sheets present, rejected factors included")

# --- F5: a size bet is legible as one ----------------------------------------
amihud_block = text.split(AMIHUD[:40])[1][:600]
assert "-0.6990" in amihud_block or "-0.699" in amihud_block, \
    "F5: the size exposure of the size-dominated factor must be shown"
assert "-0.0015" in amihud_block, "F5: its neutralized IC (near zero) must be shown"
print("F5 PASS  a size-dominated factor shows exposure -0.699 next to neutralized IC -0.0015")

# --- F6: measurements, not instructions --------------------------------------
low = text.lower()
for banned in ("you should", "prefer ", "lengthen", "shorten the", "make it simpler",
               "aim for", "try using"):
    assert banned not in low, f"F6: the feedback prescribes a fix: {banned!r}"
print("F6 PASS  states measurements without prescribing a remedy")

print("\nALL PASS")
