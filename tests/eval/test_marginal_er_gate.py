#!/usr/bin/env python
"""Marginal-effective_rank admission gate + the ICIR/raw-RankIC surfacing.

Two things, both verified without the qlib panel:

  * ``effective_rank`` / ``spearman_abs_matrix`` (metrics.py) -- the shared
    helper the gate uses. Independent signals -> ~n; a near-clone adds ~0
    independent directions.
  * The gate decision itself, via a pure-Python mirror that holds the SAME
    rule as ``_decide_standalone``: on the novel path, reject a candidate
    whose marginal effective_rank contribution is below ``QA_MIN_MARGINAL_ER``;
    inert when the env var is unset (0.0).
  * Source-ordering: the gate sits after the ``rho_max_arg`` redundancy check
    and before the novel-path ``kept.append`` (never on the replace path).
  * The feedback gloss now renders ``rank_icir`` and ``ls_mdd`` and the
    ``rank_ic`` entry is live (the sheet writes it).

Usage::

    python tests/eval/test_marginal_er_gate.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.metrics import effective_rank, spearman_abs_matrix

DATES = pd.date_range("2020-01-01", periods=120)
NAMES = [f"S{i:03d}" for i in range(60)]


def frame(seed):
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(120, 60)),
                         index=DATES, columns=NAMES)


def er_of(signals):
    return effective_rank(spearman_abs_matrix(signals))


# --- G1: the helper counts independent bets, not factors ---------------------
def test_effective_rank_counts_bets():
    ind = {f"x{i}": frame(100 + i) for i in range(8)}           # 8 independent
    assert 7.5 < er_of(ind) <= 8.0, "G1: 8 independent signals should be ~8"
    twin = dict(ind); twin["x0b"] = ind["x0"] * 1.0001 + 1e-9   # a near-clone of x0
    assert er_of(twin) < 8.2 and er_of(twin) > 7.0, \
        "G1: a near-clone adds ~0 independent directions"
    identical = {f"x{i}": ind["x0"] for i in range(5)}          # 5 copies of one
    assert 0.9 < er_of(identical) < 1.2, "G1: 5 identical signals are ~1 bet"
    print("G1 PASS  effective_rank counts independent bets, not factors")


# --- G2: the marginal-er decision (pure-Python mirror of the gate) ----------
def marginal_admits(held, candidate, min_mer):
    """Mirror of _decide_standalone's novel-path gate. Returns (admit, marginal)."""
    if not min_mer or not held:
        return True, float("nan")
    before = er_of(held)
    after = er_of({**held, "_cand": candidate})
    marginal = after - before
    return (marginal >= min_mer), marginal


def test_marginal_er_decision():
    held = {f"h{i}": frame(200 + i) for i in range(6)}
    # an INDEPENDENT candidate: adds ~1 direction
    indep = frame(999)
    admit, m = marginal_admits(held, indep, 0.5)
    assert admit and m > 0.5, f"G2: independent candidate should admit (m={m:.2f})"
    # a REDUNDANT candidate: a blend of three held factors + little noise ->
    # <rho_bar to each individually but adds few directions
    blend = (held["h0"] + held["h1"] + held["h2"]) / 3.0 + 0.05 * frame(7)
    admit, m = marginal_admits(held, blend, 0.5)
    assert not admit, f"G2: redundant blend should be rejected (m={m:.2f})"
    assert m < 0.5, f"G2: redundant blend marginal < bar (m={m:.2f})"
    # INERT when the bar is 0 (the default, QA_MIN_MARGINAL_ER unset)
    admit, m = marginal_admits(held, blend, 0.0)
    assert admit, "G2: bar=0 (default) admits everything -- gate inert"
    print("G2 PASS  marginal-er gate: independent admits, redundant rejects, 0=off")


# --- G3: source-ordering -- the gate is on the novel path, after rho_max ------
def test_source_ordering():
    src = Path("quantaalpha/factors/net_cost_runner.py").read_text()
    i_rho = src.index("rho_max_arg(sig, _held)")
    i_mer = src.index("_marginal = _er_all - _er_held")
    i_keep = src.index("kept.append((expr, sig_raw, t, rho, h))", i_mer)
    assert i_rho < i_mer < i_keep, \
        "G3: gate must follow rho_max_arg and precede the novel kept.append"
    # the replace-path kept.append comes BEFORE the gate (it is not gated)
    i_replace = src.index("kept.append((expr, sig_raw, t, rho, h))")
    assert i_replace < i_mer, "G3: the replace-path kept.append precedes the gate"
    # default-off: env read with 0.0 default, guarded by `if _min_mer`
    assert 'os.environ.get("QA_MIN_MARGINAL_ER", "0.0")' in src, \
        "G3: threshold read from QA_MIN_MARGINAL_ER with 0.0 default"
    assert "if _min_mer and _held:" in src, "G3: gate guarded by `if _min_mer` (0.0 = off)"
    print("G3 PASS  gate is on the novel path after rho_max; default-off via env")


# --- G4: the feedback gloss now renders ICIR, ls_mdd, and rank_ic is live ----
def test_gloss_surfaces_stability():
    import quantaalpha.factors.net_cost_feedback as F
    keys = {k for k, _, _ in F._RAW_METRICS}
    assert "rank_icir" in keys, "G4: rank_icir must be in the gloss (was dropped)"
    assert "ls_mdd" in keys, "G4: ls_mdd must be in the gloss (was stored but not rendered)"
    assert "rank_ic" in keys, "G4: rank_ic (raw) gloss entry exists"
    # the sheet now writes rank_icir and rank_ic, so those gloss entries are live
    src = Path("quantaalpha/factors/net_cost_runner.py").read_text()
    assert '"rank_icir": _icir' in src or "'rank_icir': _icir" in src, \
        "G4: the per-factor sheet must write rank_icir"
    assert '"rank_ic": _raw_rank_ic' in src or "'rank_ic': _raw_rank_ic" in src, \
        "G4: the per-factor sheet must write raw rank_ic (makes the gloss entry live)"
    # diagnose-never-prescribe: the rank_icir help states bars, no remedy
    assert "yours to determine" in F._METRIC_HELP["rank_icir"].lower(), \
        "G4: rank_icir help must close 'yours to determine' (no remedy)"
    print("G4 PASS  gloss renders rank_icir + ls_mdd; rank_ic live; no remedy named")


# --- G5: the economic-bar feedback stays live and book-framed (not regressed) --
def test_economic_gap_unchanged():
    import quantaalpha.factors.net_cost_feedback as F
    sheet = {"rank_ic_neutral": 0.0216, "ic_breakeven_book": 0.0872,
             "book_composite_ic": 0.0504}
    lines = " ".join(F.economic_gap(sheet))
    assert "0.0504" in lines and "0.0872" in lines, "G5: economic_gap states the book vs bar"
    assert "not a target for any single factor" in lines, \
        "G5: economic_gap keeps the book-property framing"
    print("G5 PASS  economic_gap states the book-vs-bar, not a per-factor target")


# --- G6: a marginal-er reject is legible to the generator on a MIXED batch ----
# The point of the gate: when it rejects a factor, the generator must be TOLD --
# the number, the "not kept" mark, and the glossary lesson -- even when other
# factors in the same batch admitted. Otherwise the gate is a filter the model
# cannot learn from. (Decision.reason carries the prose only when the WHOLE
# batch is rejected; the per-factor sheet is what reaches the generator on a
# mixed batch, so marginal_er must render there.)
def test_marginal_er_rejection_is_legible():
    import quantaalpha.factors.net_cost_feedback as F
    # the per-factor loop prints every _RAW_METRICS key present in a sheet, so
    # membership guarantees a rejected factor's marginal_er renders; the help
    # (stated once before the sheets) must teach the lesson and name no remedy.
    assert "marginal_er" in {k for k, _, _ in F._RAW_METRICS}, \
        "G6: marginal_er in _RAW_METRICS so it renders per-factor"
    h = F._METRIC_HELP["marginal_er"].lower()
    assert "yours to determine" in h, "G6: marginal_er help names no remedy"
    assert "redundant at the margin" in h, "G6: help names the failure mode"
    assert "collective" in h, "G6: help distinguishes marginal-er (collective) from rho_max (pairwise)"
    # end-to-end: a MIXED batch -- one KEPT, one marginal-er-rejected -- must
    # surface the rejected factor's marginal_er NUMBER and its 'not kept' mark.
    fb = F.NetCostFactorFeedback.__new__(F.NetCostFactorFeedback)
    raw = {
        "admitted_exprs": ["GOOD"],
        "factor_tearsheets": {
            "GOOD": {"t_nw": 5.0, "marginal_er": 1.2, "rho_max": 0.10,
                     "rank_ic_neutral": 0.03, "ic_breakeven_book": 0.0872,
                     "book_composite_ic": 0.05},
            "REDUNDANT_BLEND": {"t_nw": 4.0, "marginal_er": 0.03, "rho_max": 0.40,
                     "rank_ic_neutral": 0.02, "ic_breakeven_book": 0.0872,
                     "book_composite_ic": 0.04},
        },
    }
    lines = "\n".join(fb._per_factor_lines(raw))
    assert "[not kept]" in lines and "REDUNDANT_BLEND" in lines, \
        "G6: rejected factor is marked not kept"
    assert "marginal effective rank): 0.03" in lines, \
        "G6: the rejected factor's marginal_er NUMBER renders per-factor"
    assert "redundant at the margin" in lines.lower(), \
        "G6: the glossary lesson renders before the sheets"
    print("G6 PASS  marginal-er reject legible on a mixed batch (number + 'not kept' + gloss)")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    test_effective_rank_counts_bets()
    test_marginal_er_decision()
    test_source_ordering()
    test_gloss_surfaces_stability()
    test_economic_gap_unchanged()
    test_marginal_er_rejection_is_legible()
    print("\nALL PASS")