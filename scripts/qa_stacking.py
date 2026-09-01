#!/usr/bin/env python
"""How much of its theoretical breadth benefit does the combiner deliver?

N decorrelated signals of equal quality should combine to roughly

    IC_composite ~ IC_single * sqrt(effective_rank)

That is the whole reason to hold more than one factor. Measured on the live
book: 15 factors at |IC| 0.0216 with effective rank 10.6 should reach ~0.070,
and the book delivered 0.0196 -- 28% of theoretical. Two causes are confounded
in that number:

  (a) IS->OOS decay -- 0.0216 was measured in sample and does not survive;
  (b) the combiner failing to extract the breadth that is present.

This separates them by measuring EVERY quantity on the SAME window, so decay
cancels and only the combiner's contribution remains:

    best_single   max |IC| of any one factor, that window
    equal_weight  |IC| of the plain mean of the rank-normed signals
                  (CSRankNorm -- the SAME preprocessing production uses)
    combiner      |IC| of the fitted composite the book actually trades
    ideal         best_single * sqrt(effective_rank)

``equal_weight`` is the load-bearing control: it needs no fitting at all, so if
the fitted combiner cannot beat it, the fit is not adding information -- it is
a defect, not a limit of the data.

Usage::

    python scripts/qa_stacking.py --library data/results/zoo_newgate.json
    python scripts/qa_stacking.py --zoo data/results/ledger_<id>.jsonl --report
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def label(panel) -> pd.DataFrame:
    """The forward return the book trades: open[t+2]/open[t+1] - 1."""
    o = pd.DataFrame(panel.open, index=pd.DatetimeIndex(panel.dates),
                     columns=list(panel.instruments))
    return o.shift(-2) / o.shift(-1) - 1.0


def ic_of(sig: pd.DataFrame, lab: pd.DataFrame) -> float:
    c = _cross_sectional_corr(sig, lab, "spearman").dropna()
    return float(c.mean()) if len(c) else float("nan")


def cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization -- the SAME preprocessing the
    production combiner applies (``combiner._preprocess_v2``).

    This was a z-score, which is a different model. Measured 2026-08-24 on the
    15-factor book: identical weights, identical label, identical universe mask,
    but z-scored features gave |IC| 0.0603 against the production path's 0.0510
    -- an 18% gap from preprocessing alone, which is what produced the
    contradictory "combiner is 0.89x / 1.06x its best single factor" readings.
    ``_preprocess_v2``'s own docstring warns about exactly this trap: an
    approximated preprocessing once scored Rank IC +0.0336 against the real
    +0.1179, "two different models, not two reports of one".

    Wide (T x N) here rather than the long MultiIndex the combiner uses, so the
    rank is taken across instruments within each date -- the same axis.
    """
    ranked = df.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
    return ranked - 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--report", action="store_true",
                    help="score on final_test (default: the valid window)")
    ap.add_argument("--out", default="data/results/stacking.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    if a.zoo:
        exprs = list(replay_repository(a.zoo))
    else:
        payload = json.loads(Path(a.library).read_text())
        f = payload.get("factors", payload)
        items = f.values() if isinstance(f, dict) else f
        exprs = [e.get("factor_expression") or e.get("expression") for e in items]
        exprs = [e for e in exprs if e]

    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(a.report)
    panel = op._panel(p0, p1)
    lab = label(panel)
    print(f"protocol {theta.hash} | scored {win} | {len(exprs)} factors\n")

    sig = {}
    for e in exprs:
        try:
            sig[e] = load_aligned_signal(e, panel)
        except Exception:
            pass
    keys = list(sig)
    if len(keys) < 2:
        print("need >= 2 loadable factors")
        return 2

    # Everything below is measured on the SAME window, so IS->OOS decay is
    # common to all of them and cancels in the ratios.
    lo, hi = str(win[0]), str(win[1])
    lab_w = lab.loc[lo:hi]
    sig_w = {k: v.loc[lo:hi] for k, v in sig.items()}

    ics = {k: ic_of(v, lab_w) for k, v in sig_w.items()}
    finite = {k: v for k, v in ics.items() if v == v}
    best_k = max(finite, key=lambda k: abs(finite[k]))
    best = abs(finite[best_k])
    mean_abs = float(np.mean([abs(v) for v in finite.values()]))

    # SIGN-ALIGN before averaging. Half these factors are reversal signals with
    # negative IC; averaging them raw cancels the signal and would blame the
    # combiner for an error in this script.
    aligned = {k: (v if ics.get(k, 0) >= 0 else -v) for k, v in sig_w.items()}
    ew = sum(cs_rank_norm(v).fillna(0.0) for v in aligned.values()) / len(aligned)
    ew_ic = abs(ic_of(ew, lab_w))

    # effective rank on this window
    n = len(keys)
    R = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        c = _cross_sectional_corr(sig_w[keys[i]], sig_w[keys[j]], "spearman")
        R[i, j] = R[j, i] = abs(float(c.mean())) if not c.empty else 0.0
    ev = np.linalg.eigvalsh(R)[::-1]
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    er = float(np.exp(-(p * np.log(p)).sum()))

    # the FITTED combiner the book actually trades
    res = op.evaluate(sig, zoo_signals={}, zoo_metrics=[], report=a.report)
    comb_ic = abs(float(res.get("m_rank_ic") or float("nan")))

    ideal = best * np.sqrt(er)

    print(f"{'best single factor':28s} |IC| {best:.4f}")
    print(f"{'mean |IC| across factors':28s}      {mean_abs:.4f}")
    print(f"{'equal-weight composite':28s} |IC| {ew_ic:.4f}")
    print(f"{'FITTED combiner (the book)':28s} |IC| {comb_ic:.4f}")
    print(f"{'effective rank':28s}      {er:.1f} of {n}")
    print(f"{'ideal (best * sqrt(rank))':28s} |IC| {ideal:.4f}")
    print("-" * 58)
    print(f"  stacking efficiency (combiner / ideal)      : {comb_ic / ideal:.0%}")
    print(f"  combiner vs best single                     : {comb_ic / best:.2f}x")
    print(f"  equal-weight vs best single                 : {ew_ic / best:.2f}x")
    print(f"  combiner vs equal-weight                    : {comb_ic / ew_ic:.2f}x")
    print()
    if comb_ic < best:
        print("  VERDICT: the fitted combiner is WORSE than its own best single")
        print("           factor. Holding N factors destroys signal here.")
    elif comb_ic < ew_ic:
        print("  VERDICT: the fit LOSES to an unfitted equal-weight average.")
        print("           The fitting is subtracting information, not adding it.")
    elif comb_ic / ideal < 0.5:
        print("  VERDICT: the combiner beats its constituents but captures under")
        print("           half of the available breadth.")
    else:
        print("  VERDICT: the combiner captures most of the available breadth;")
        print("           the shortfall is in the factors, not the combination.")

    Path(a.out).write_text(json.dumps({
        "window": list(win), "n_factors": n, "effective_rank": er,
        "best_single_ic": best, "mean_abs_ic": mean_abs,
        "equal_weight_ic": ew_ic, "combiner_ic": comb_ic, "ideal_ic": ideal,
        "stacking_efficiency": comb_ic / ideal,
    }, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
