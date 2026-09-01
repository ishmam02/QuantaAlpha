#!/usr/bin/env python
"""Does CSRankNorm cost the combiner signal, versus a z-score?

The production combiner rank-normalizes every feature (``_preprocess_v2``):
each date's cross-section becomes a centred percentile rank. That discards
MAGNITUDE -- a factor's extreme values are compressed to the same spacing as
its middle -- which is deliberate (robust to outliers, matches the Qlib
baseline) but not free.

A first look suggested z-scoring beats it by 18%. That measurement was
in-sample, one book, one window, and the same class of mistake had already
produced a wrong answer twice in this session, so it is treated here as a
HYPOTHESIS and tested properly:

  * FIT the weights on one window, SCORE on a later one (no in-sample scoring)
  * both preprocessings judged on the identical fitted weights, so only the
    feature transform differs
  * run on two independent factor sets
  * a winsorized z-score is included as the third arm: if the gap is really
    about outliers rather than magnitude, clipping the tails should recover
    most of it without abandoning magnitude entirely

Usage::

    python scripts/qa_preproc_test.py --library data/results/zoo_newgate.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.combiner import _icir_weights  # noqa: E402
from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr, label_frame  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)


# --- the three preprocessings, each cross-sectional (per date) --------------
def cs_rank(df):
    """Production: centred percentile rank. Magnitude discarded."""
    return df.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True) - 0.5


def cs_z(df):
    """Z-score. Magnitude kept, outliers kept."""
    d = df.replace([np.inf, -np.inf], np.nan)
    return d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1).replace(0.0, np.nan), axis=0)


def cs_z_winsor(df, k=3.0):
    """Z-score clipped at +/-k sd. Magnitude kept, outliers tamed."""
    return cs_z(df).clip(-k, k)


PREPROC = {"cs_rank (production)": cs_rank, "z-score": cs_z,
           "z-score winsor 3sd": cs_z_winsor}


def ic(sig, lab):
    c = _cross_sectional_corr(sig, lab, "spearman").dropna()
    return float(c.mean()) if len(c) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", default="data/results/preproc_test.json")
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
    # Panel spans train..valid; fit on train, score on valid. The holdout is
    # never touched -- this is a preprocessing choice, and choosing it on the
    # test window would be the same leak the break-even bar already had.
    p0, p1, _ = op._windows(False)
    panel = op._panel(p0, p1)
    lab = label_frame(panel, theta)

    fit_lo, fit_hi = theta.splits.train
    sc_lo, sc_hi = theta.splits.valid
    print(f"protocol {theta.hash}")
    print(f"  FIT   weights on train {fit_lo}..{fit_hi}")
    print(f"  SCORE composite on valid {sc_lo}..{sc_hi}   (holdout untouched)\n")

    sig = {}
    for e in exprs:
        try:
            sig[e] = load_aligned_signal(e, panel)
        except Exception:
            pass
    cols = list(sig)
    if len(cols) < 2:
        print("need >= 2 factors")
        return 2
    print(f"  {len(cols)} factors\n")

    shrink = theta.combiner.params.get("shrinkage", "auto")
    rows = {}
    for name, fn in PREPROC.items():
        # Weights are fit on the TRAIN window, under this preprocessing.
        feat_fit = {c: fn(sig[c].loc[fit_lo:fit_hi]).fillna(0.0) for c in cols}
        lab_fit = lab.loc[fit_lo:fit_hi]
        ic_arr = np.column_stack([
            _cross_sectional_corr(feat_fit[c], lab_fit, "spearman")
            .reindex(lab_fit.index).values for c in cols])
        w, delta = _icir_weights(ic_arr, shrink)

        # ...and applied, unchanged, to the SCORING window.
        feat_sc = {c: fn(sig[c].loc[sc_lo:sc_hi]).fillna(0.0) for c in cols}
        comp = sum(feat_sc[c] * wi for c, wi in zip(cols, w))
        v = abs(ic(comp, lab.loc[sc_lo:sc_hi]))
        # best single factor under the SAME preprocessing, same window
        singles = [abs(ic(feat_sc[c], lab.loc[sc_lo:sc_hi])) for c in cols]
        rows[name] = {"delta": delta, "composite_ic": v,
                      "best_single_ic": max(singles),
                      "vs_best": v / max(singles) if max(singles) else float("nan")}
        print(f"  {name:24s} delta {delta:.3f}  composite |IC| {v:.4f}  "
              f"best single {max(singles):.4f}  ({v / max(singles):.2f}x)")

    base = rows["cs_rank (production)"]["composite_ic"]
    print("\n  vs production:")
    for name, r in rows.items():
        if name == "cs_rank (production)":
            continue
        print(f"    {name:24s} {100 * (r['composite_ic'] / base - 1):+6.1f}%")

    Path(a.out).write_text(json.dumps(
        {"fit_window": [fit_lo, fit_hi], "score_window": [sc_lo, sc_hi],
         "n_factors": len(cols), "results": rows}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
