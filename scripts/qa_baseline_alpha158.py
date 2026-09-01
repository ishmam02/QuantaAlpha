#!/usr/bin/env python
"""Is the generation industry-level? Measure PUBLISHED alphas on the same window.

The mined factors carry median |IC| 0.0216 on the valid window. Whether that is
weak or normal cannot be answered from inside the system -- it needs an external
reference measured on the SAME data, the SAME window and the SAME label.

Alpha158 is that reference: a published, widely used feature set, not tuned to
this universe by anyone here. If the published factors land near 0.02 the
generator is producing industry-normal alphas and the shortfall is downstream;
if they land at 0.05+ the generator has real room.

Everything is measured exactly as the admission gate measures a mined factor --
same neutralization, same horizons, same Newey-West t -- so the two are
comparable rather than merely adjacent.

Usage::

    python scripts/qa_baseline_alpha158.py            # the 20-seed pool
    python scripts/qa_baseline_alpha158.py --full     # all 158 (slow)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.metrics import _cross_sectional_corr, label_frame_at  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)
SEEDS = Path("data/factorlib/alpha158_20_seed_pool.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--full", action="store_true",
                    help="all 158 Alpha158 features instead of the 20-seed pool")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/results/baseline_alpha158.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)          # valid -- what admission scores on
    panel = op._panel(p0, p1)
    lo, hi = str(win[0]), str(win[1])
    print(f"protocol {theta.hash} | window {win}\n", flush=True)

    # The gate tries these horizons and keeps the strongest by |t|, so the
    # baseline must be given the same freedom or it is handicapped.
    horizons = [1, 5, 20]
    labels = {h: label_frame_at(panel, theta, h).loc[lo:hi] for h in horizons}

    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

    if a.full:
        from qlib.contrib.data.loader import Alpha158DL
        fields, names = Alpha158DL.get_feature_config()
        items = list(zip(names, fields))
    else:
        items = [(r["name"], r["expression"])
                 for r in csv.DictReader(SEEDS.open())]
    if a.limit:
        items = items[: a.limit]
    print(f"  {len(items)} published factors to measure\n", flush=True)

    calc = CustomFactorCalculator()
    rows, t0 = [], time.time()
    for i, (name, expr) in enumerate(items, 1):
        try:
            sig = calc.calculate_factor(expr, panel_data=panel) \
                if "panel_data" in calc.calculate_factor.__code__.co_varnames \
                else calc.calculate_factor(expr)
            if sig is None:
                continue
            wide = sig if isinstance(sig, pd.DataFrame) else sig.unstack(level=-1)
            wide = wide.reindex(index=panel.dates, columns=panel.instruments)
            w = wide.loc[lo:hi]
            best = None
            for h, lab in labels.items():
                ic = _cross_sectional_corr(w, lab, "spearman").dropna()
                if len(ic) < 100:
                    continue
                m, sd = float(ic.mean()), float(ic.std())
                if sd <= 0:
                    continue
                t = m / (sd / np.sqrt(max(len(ic) / h, 2.0)))
                if best is None or abs(t) > abs(best[2]):
                    best = (h, m, t)
            if best:
                h, m, t = best
                rows.append({"name": name, "expr": expr, "horizon": h,
                             "ic": m, "abs_ic": abs(m), "t": t})
                print(f"  [{i:3d}/{len(items)}] |IC| {abs(m):.4f} |t| {abs(t):5.2f} "
                      f"h={h:2d}d  {name}", flush=True)
        except Exception as exc:
            print(f"  [{i:3d}/{len(items)}] FAILED {name}: "
                  f"{type(exc).__name__}", flush=True)

    if not rows:
        print("no factors measured")
        return 2
    ics = [r["abs_ic"] for r in rows]
    ts = [abs(r["t"]) for r in rows]
    print("\n" + "=" * 66)
    print(f"PUBLISHED BASELINE (Alpha158), window {win}, n={len(rows)}")
    print("=" * 66)
    print(f"  median |IC| {st.median(ics):.4f}   mean {st.mean(ics):.4f}   "
          f"best {max(ics):.4f}")
    print(f"  median |t|  {st.median(ts):.2f}     clearing |t|>=3: "
          f"{sum(1 for t in ts if t >= 3)}/{len(ts)}")
    print("-" * 66)
    print(f"  MINED factors (same window): median 0.0216, best 0.0567")
    ratio = st.median(ics) / 0.0216
    print(f"  published / mined (median): {ratio:.2f}x")
    print()
    if ratio > 1.5:
        print("  VERDICT: published factors are materially stronger -- the")
        print("           generator has real room to improve.")
    elif ratio < 0.7:
        print("  VERDICT: the mined factors are STRONGER than the published")
        print("           reference. Generation is not the constraint.")
    else:
        print("  VERDICT: mined and published factors are comparable. The")
        print("           generator is producing industry-normal alphas, so the")
        print("           shortfall is downstream of generation.")
    print(f"  elapsed {time.time() - t0:.0f}s")

    Path(a.out).write_text(json.dumps(
        {"window": list(win), "n": len(rows),
         "median_abs_ic": st.median(ics), "best_abs_ic": max(ics),
         "mined_median_abs_ic": 0.0216, "rows": rows}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
