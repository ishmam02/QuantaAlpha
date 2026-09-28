#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Split excess-vs-cap-weighted-CSI300 into factor alpha, size tilt and residual beta.

Why this exists
---------------
The benchmark is now the official cap-weighted CSI300 TOTAL-RETURN index
(``SH000300TR``, 沪深300全收益 / CSIndex H00300). That is the right yardstick, but
it imports a problem the previous equal-weight benchmark sidestepped:

    the book is structurally EQUAL-weight. ``max_weight: 0.03`` caps a
    6%-index-weight name at 3%, so the book runs a permanent ~-3% active position
    in every large constituent. That is a SIZE BET NOBODY CHOSE -- it comes from
    the position cap, not from any mined factor.

``quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml`` measured the cost of
scoring this book against a cap-weighted index at **-9.8 pp (2019), -13.1 pp (2020),
+6.0 pp (2021)** -- comparable to or larger than the alpha being searched for, and
swinging sign year to year. Reported alone, the cap-weighted excess silently credits
or debits that bet to the factors. This script separates them.

The decomposition
-----------------
Regress the book's daily excess return on the size spread and the benchmark::

    r_excess(t) = alpha + b_size * r_size(t) + b_mkt * r_bench(t) + eps(t)

    r_size(t)  = r_equal_weight(t) - r_cap_weight(t)

``r_size`` is exactly the equal-weight-minus-cap-weight spread, i.e. the return of
the very tilt the 0.03 cap imposes -- so ``b_size`` is the book's loading on it and
``b_size * mean(r_size) * 252`` is the annualized pp of excess attributable to the
cap rather than to stock selection. Both legs are dividend-inclusive and therefore
comparable: the equal-weight basket compounds adjusted (dividend-reinvested) closes,
and ``SH000300TR`` is a total-return index.

``b_mkt`` catches residual market beta -- a long-only top-50 book is not beta-1
against the index, so some excess is just under/over-exposure.

Reported annualized contributions sum to the total excess by construction
(alpha + size + beta + residual-mean).

Usage
-----
    /opt/anaconda3/envs/quantaalpha/bin/python -u scripts/qa_report_size_decomposition.py
    ... --results-dir data/results/backtest_v2_results
    ... --per-year
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
os.environ.setdefault("QLIB_PROVIDER_URI", str(REPO / "data/qlib/cn_data"))

TD = 252


def load_excess(csv: Path) -> pd.Series:
    """btv2's persisted daily excess series (already net of benchmark and cost)."""
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
    return df["daily_excess_return"].astype(float).sort_index()


def build_factors(theta, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """(r_size, r_bench): the equal-minus-cap spread, and the cap-weighted TR."""
    from dataclasses import replace
    from quantaalpha.eval.data import load_benchmark, equal_weight_benchmark

    cap = load_benchmark(theta, start, end)                       # SH000300TR
    ew = equal_weight_benchmark(replace(theta, benchmark_construction="equal"),
                                start, end)
    idx = cap.index.intersection(ew.index)
    cap, ew = cap.reindex(idx).astype(float), ew.reindex(idx).astype(float)
    size = (ew - cap).rename("size")
    return size, cap.rename("bench")


def decompose(r: pd.Series, size: pd.Series, bench: pd.Series) -> dict:
    """OLS of excess on [size, bench]; annualized contribution of each leg."""
    df = pd.concat([r.rename("r"), size, bench], axis=1).dropna()
    if len(df) < 30:
        return {"n": int(len(df)), "error": "too few overlapping days"}
    # Full-series excess, BEFORE the regression's dropna. The regression sample loses
    # any day where size/bench is NaN (notably day 1, where pct_change is undefined),
    # so its mean does not reconcile with the backtest JSON. Report both: `excess_*`
    # is the whole series (comparable to the backtest metrics), the decomposition legs
    # are on the regression sample.
    full = r.dropna()
    X = np.column_stack([np.ones(len(df)), df["size"].to_numpy(), df["bench"].to_numpy()])
    y = df["r"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b_size, b_mkt = (float(c) for c in coef)
    resid = y - X @ coef

    mean_size, mean_bench = float(df["size"].mean()), float(df["bench"].mean())
    # Annualized pp contributions. Arithmetic (mean*TD) so the legs are additive;
    # the compounded total is reported separately for reference.
    out = {
        "n": int(len(df)),
        "n_full": int(len(full)),
        # whole series -- reconciles with the backtest metrics JSON
        "excess_ann_pp": float(full.mean()) * TD * 100,
        "excess_cum_pp": float((1.0 + full).prod() - 1.0) * 100,
        # regression sample -- the legs below are additive against THIS
        "total_excess_ann_pp": float(y.mean()) * TD * 100,
        "total_excess_cum_pp": float((1.0 + y).prod() - 1.0) * 100,
        "alpha_ann_pp": a * TD * 100,
        "size_ann_pp": b_size * mean_size * TD * 100,
        "beta_ann_pp": b_mkt * mean_bench * TD * 100,
        "b_size": b_size,
        "b_mkt": b_mkt,
        "size_spread_ann_pp": mean_size * TD * 100,
        "bench_ann_pp": mean_bench * TD * 100,
        "resid_ann_pp": float(resid.mean()) * TD * 100,
        "r2": float(1.0 - resid.var() / y.var()) if y.var() > 0 else float("nan"),
        "alpha_t": float(a / (resid.std(ddof=3) / np.sqrt(len(df))))
                   if resid.std(ddof=3) > 0 else float("nan"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="data/results/backtest_v2_results")
    ap.add_argument("--protocol",
                    default="quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")
    ap.add_argument("--per-year", action="store_true")
    ap.add_argument("--out", default="data/results/report_size_decomposition.json")
    args = ap.parse_args()

    from quantaalpha.eval.protocol import load_protocol
    theta = load_protocol(args.protocol)
    print(f"benchmark {theta.benchmark} "
          f"({theta.benchmark_construction}/{theta.benchmark_basis})")

    rdir = Path(args.results_dir)
    csvs = sorted(rdir.glob("*_cumulative_excess.csv"))
    if not csvs:
        print(f"no *_cumulative_excess.csv under {rdir} -- run the btv2 backtests first")
        return 1

    # One factor build covering the union of all books' windows.
    lo = min(load_excess(c).index.min() for c in csvs)
    hi = max(load_excess(c).index.max() for c in csvs)
    print(f"building size/bench factors over {lo.date()}..{hi.date()}", flush=True)
    size, bench = build_factors(theta, str(lo.date()), str(hi.date()))
    print(f"  size spread (EW - cap) {size.mean()*TD*100:+.2f} pp/yr | "
          f"benchmark {bench.mean()*TD*100:+.2f} pp/yr\n")

    results = {}
    def label(stem: str) -> str:
        # Strip the shared prefix so the distinguishing tail survives: a head-truncated
        # name collapsed "..._194432_zoo" and "..._194432" into the same 43 chars.
        s = stem.replace("all_factors_library_", "")
        return s if len(s) <= 30 else "..." + s[-27:]

    hdr = (f"{'library':32}{'excess':>9}{'alpha':>9}{'size':>9}{'beta':>9}"
           f"{'b_size':>8}{'R2':>7}")
    print(hdr); print("-" * len(hdr))
    for c in csvs:
        name = c.name.replace("_cumulative_excess.csv", "")
        r = load_excess(c)
        d = decompose(r, size, bench)
        results[name] = d
        if "error" in d:
            print(f"{label(name):32}{'-- ' + d['error']:>40}")
            continue
        print(f"{label(name):32}{d['excess_ann_pp']:>9.2f}{d['alpha_ann_pp']:>9.2f}"
              f"{d['size_ann_pp']:>9.2f}{d['beta_ann_pp']:>9.2f}"
              f"{d['b_size']:>8.2f}{d['r2']:>7.2f}")

        if args.per_year:
            for y in sorted({t.year for t in r.index}):
                ry = r[r.index.year == y]
                dy = decompose(ry, size, bench)
                if "error" in dy:
                    continue
                print(f"    {y:<28}{dy['excess_ann_pp']:>9.2f}"
                      f"{dy['alpha_ann_pp']:>9.2f}{dy['size_ann_pp']:>9.2f}"
                      f"{dy['beta_ann_pp']:>9.2f}{dy['b_size']:>8.2f}{dy['r2']:>7.2f}")
                results[f"{name}::{y}"] = dy

    print("\nall figures are annualized pp of EXCESS over the cap-weighted CSI300 total"
          "\nreturn. 'size' is the part explained by the equal-minus-cap spread, i.e. the"
          "\nmax_weight 0.03 cap -- NOT stock selection. 'alpha' is what survives it.")

    Path(args.out).write_text(json.dumps(
        {"benchmark": theta.benchmark, "protocol_hash": theta.hash,
         "size_spread_ann_pp": float(size.mean() * TD * 100),
         "bench_ann_pp": float(bench.mean() * TD * 100),
         "books": results}, indent=2, default=lambda o: None))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
