#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-year metric table for the annual rolling refit, beside the benchmark's own.

All strategy figures are EXCESS over SH000300TR (the official cap-weighted CSI300
TOTAL-RETURN index) and net of the kappa cost model, because
``costs.net_return`` computes ``gross - benchmark - charges``. So:

  * IR is the INFORMATION ratio (mean/std*sqrt(252) of the excess series). In this
    codebase's own tables that quantity is labelled "Sharpe"; for a benchmark-
    relative book they are the same number. A TRUE Sharpe would need the absolute
    NAV series net of the risk-free rate, which the refit run did not persist.
  * MDD and therefore CRR are on the EXCESS path, not on NAV. There is no
    absolute-NAV drawdown anywhere in quantaalpha/eval -- strategy_metrics runs
    max_drawdown on r_net.
  * CRR = ARR / |MDD| (the repo's calmar convention).

Benchmark columns are the index's OWN absolute metrics, for scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

TD = 252
YEARS = range(2021, 2026)


def bench_series() -> pd.Series:
    import qlib
    qlib.init(provider_uri=str(ROOT / "data/qlib/cn_data"), region="cn")
    from qlib.data import D
    s = D.features(["SH000300TR"], ["$close"], start_time="2020-12-01",
                   end_time="2025-12-31")["$close"].droplevel("instrument")
    s.index = pd.to_datetime(s.index)
    return s.sort_index().pct_change().dropna()


def stats(x: pd.Series):
    g = float((1.0 + x).prod())
    arr = (g ** (TD / len(x)) - 1.0) * 100
    sh = float(x.mean() / x.std() * np.sqrt(TD)) if x.std() > 0 else float("nan")
    c = (1.0 + x).cumprod()
    mdd = float((c / c.cummax() - 1.0).min()) * 100
    return arr, sh, mdd, (arr / abs(mdd) if mdd else float("nan"))


def main() -> int:
    r = bench_series()
    W = json.load(open(ROOT / "data/results/walkforward_refit_2021_2025.json"))["libraries"]
    for nm in ("main_full", "original"):
        Y = W[nm]["years"]
        print(f"=== {nm} (n={W[nm]['n_factors']}) -- ANNUAL ROLLING REFIT ===")
        print("    strategy: EXCESS over SH000300TR, net of kappa    |    benchmark: ABSOLUTE")
        print(f"  {'yr':5}{'IC':>8}{'RankIC':>8}{'IR':>7}{'ARR%':>8}{'MDD%':>8}{'CRR':>7}"
              f"{'TC':>6}  |{'bmARR%':>9}{'bmShp':>7}{'bmMDD%':>8}{'bmCRR':>7}")
        for y in YEARS:
            d = Y[str(y)]
            a, m = d["arr"] * 100, d["mdd"] * 100
            ba, bs, bm, bc = stats(r[r.index.year == y])
            print(f"  {y:5}{d['ic']:>8.4f}{d['rank_ic']:>8.4f}{d['ir']:>7.2f}"
                  f"{a:>8.2f}{m:>8.1f}{a/abs(m):>7.3f}{d['tc']:>6.2f}  |"
                  f"{ba:>9.2f}{bs:>7.2f}{bm:>8.1f}{bc:>7.3f}")
        st = Y["stitched"]
        a, m = st["arr"] * 100, st["mdd"] * 100
        ic = np.mean([Y[str(y)]["ic"] for y in YEARS])
        ric = np.mean([Y[str(y)]["rank_ic"] for y in YEARS])
        tc = np.mean([Y[str(y)]["tc"] for y in YEARS])
        ba, bs, bm, bc = stats(r[r.index >= "2021-01-01"])
        print(f"  {'FULL':5}{ic:>8.4f}{ric:>8.4f}{st['ir']:>7.2f}{a:>8.2f}{m:>8.1f}"
              f"{a/abs(m):>7.3f}{tc:>6.2f}  |{ba:>9.2f}{bs:>7.2f}{bm:>8.1f}{bc:>7.3f}")
        print("    (FULL IC/RankIC/TC are means of the five yearly values; ARR/MDD/CRR/IR")
        print("     are computed on the STITCHED daily series, not averaged.)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
