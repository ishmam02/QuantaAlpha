#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Annual ROLLING REFIT: fit through Y-1, report Y, advance, repeat. No re-mining.

The question
------------
Refitting the combiner on a later window recovered 32-54% of Rank IC on the SAME
factor library (fit <=2016 vs fit <=2020, measured 2026-09-26). That was a single
step. This runs the production version of it: a fresh combiner every year, each
fitted only on data strictly before the year it is scored on.

  test 2021 <- fit 2005-01-01..2020-12-31
  test 2022 <- fit 2005-01-01..2021-12-31
  test 2023 <- fit 2005-01-01..2022-12-31
  test 2024 <- fit 2005-01-01..2023-12-31
  test 2025 <- fit 2005-01-01..2024-12-31

Every reported year is out of sample for the combiner that traded it, and the
stitched series is what an operator running an annual refit would actually have
earned. Nothing is re-mined: the factor library is fixed throughout, so any
improvement over a frozen fit is attributable to the REFIT alone.

Why the fit window EXPANDS rather than slides: dropping early history would
confound "later regime" with "less data" -- the same reasoning
``Splits.walk_forward_folds`` gives for its own expanding windows.

Efficiency: the panel and the signals are loaded ONCE and reused across years.
Only the combiner is refit (a new ``theta`` per year gives a new ``theta.hash``,
so ``combiner._PREDICTION_CACHE`` misses and genuinely refits rather than serving
a stale prediction).

Also reports, per year:
  * TC -- Clarke-de Silva-Thorley corr(alpha, w): the fraction of the alpha the
    long-only floor and the position cap let through (measured 0.37 for the
    frozen topk book, i.e. ~63% of the alpha destroyed before costs).
  * the stitched full-period ARR/IR from the concatenated per-year net returns.

Usage
-----
    FACTOR_CACHE_DIR=data/results/factor_cache_2005 \
    python -u scripts/qa_walkforward_refit.py --libs main_full --years 2021 2025
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from quantaalpha.eval.protocol import load_protocol            # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator        # noqa: E402
from quantaalpha.eval.data import load_aligned_signal           # noqa: E402
from quantaalpha.eval.metrics import _ic_block, label_frame, _to_wide  # noqa: E402
from quantaalpha.eval.portfolio import build_book               # noqa: E402
from quantaalpha.eval.execution import realized_return, fill_prices, prediction_scale  # noqa: E402
import quantaalpha.eval.costs as costs_mod                      # noqa: E402

TD = 252
LIBS = {
    "main_full": "data/factorlib/all_factors_library_meanvar_20260828_194432.json",
    "original":  "data/factorlib/all_factors_library_original_20260831_012324.json",
    "main_zoo":  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json",
}
# official cap-weighted CSI300 TOTAL RETURN (H00300), for the beta read
BENCH = {2017: 24.25, 2018: -23.64, 2019: 39.19, 2020: 29.89, 2021: -3.52,
         2022: -19.84, 2023: -9.14, 2024: 18.24, 2025: 20.98}


def annualize(r: pd.Series):
    s = pd.Series(r).astype(float).dropna()
    if len(s) < 2:
        return float("nan"), float("nan"), float("nan")
    g = float((1.0 + s).prod())
    arr = float(np.sign(g) * abs(g) ** (TD / len(s)) - 1.0) if g > 0 else -1.0
    ir = float(s.mean() / s.std() * np.sqrt(TD)) if s.std() > 0 else float("nan")
    c = (1.0 + s).cumprod()
    return arr, ir, float((c / c.cummax() - 1.0).min())


def exprs_of(path: Path) -> list[str]:
    lib = json.loads(path.read_text())
    f = lib["factors"] if isinstance(lib, dict) and "factors" in lib else lib
    if isinstance(f, dict):
        f = list(f.values())
    return [x["factor_expression"] for x in f if x.get("factor_expression")]


def transfer_coefficient(alpha: pd.DataFrame, w: pd.DataFrame) -> float:
    vals = []
    for d in w.index:
        if d not in alpha.index:
            continue
        a, v = alpha.loc[d].to_numpy(float), w.loc[d].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(v)
        if m.sum() < 10 or np.nanstd(a[m]) == 0 or np.nanstd(v[m]) == 0:
            continue
        vals.append(float(np.corrcoef(a[m], v[m])[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol",
                    default="quantaalpha/eval/protocol_csi300_lightgbm_topk_decay.yaml")
    ap.add_argument("--libs", nargs="+", default=["main_full"])
    ap.add_argument("--years", nargs=2, type=int, default=[2021, 2025],
                    metavar=("FIRST", "LAST"))
    ap.add_argument("--fit-start", default="2005-01-01")
    ap.add_argument("--out", default="data/results/walkforward_refit.json")
    a = ap.parse_args()
    y0, y1 = a.years

    base = load_protocol(a.protocol)
    print(f"protocol {a.protocol}")
    print(f"  combiner={base.combiner.model} construction={base.portfolio.construction} "
          f"benchmark={base.benchmark} ({base.benchmark_construction}/{base.benchmark_basis})")
    print(f"  ANNUAL ROLLING REFIT: fit {a.fit_start}..(Y-1), report Y, for Y in {y0}..{y1}")
    print()

    # One panel + one signal load, reused for every year.
    span = replace(base, splits=replace(base.splits,
                   train=(a.fit_start, f"{y0-1}-12-31"),
                   final_test=(f"{y0}-01-01", f"{y1}-12-31")))
    op0 = EvaluationOperator(span)
    p_start, p_end, _ = op0._windows(True)
    panel = op0._panel(p_start, p_end)
    label = label_frame(panel, span)
    y_tilde = realized_return(fill_prices(panel, span))
    sigma = costs_mod.trailing_vol(panel.close, span.costs.vol_window)
    adv = costs_mod.trailing_adv(panel, span)
    mask = op0._trade_mask(panel)
    print(f"  panel {p_start}..{p_end}: {len(panel.dates)} dates x "
          f"{len(panel.instruments)} instruments", flush=True)

    out = {"protocol": a.protocol, "benchmark": base.benchmark,
           "fit_start": a.fit_start, "years": [y0, y1], "libraries": {}}

    for lib in a.libs:
        t0 = time.time()
        exprs = exprs_of(ROOT / LIBS[lib])
        cands, fails = {}, 0
        for e in exprs:
            try:
                cands[e] = load_aligned_signal(e, panel)
            except Exception:
                fails += 1
        print(f"\n[{lib}] {len(cands)}/{len(exprs)} signals loaded ({time.time()-t0:.0f}s)",
              flush=True)
        if len(cands) < 0.90 * len(exprs):
            raise RuntimeError(f"{lib}: only {len(cands)}/{len(exprs)} signals -- "
                               f"refusing (a thin book prices as the null model)")

        rows, series = {}, []
        for Y in range(y0, y1 + 1):
            tY = replace(base, splits=replace(base.splits,
                         train=(a.fit_start, f"{Y-1}-12-31"),
                         valid=(f"{Y-1}-01-01", f"{Y-1}-12-31"),  # inert: no valid_sets
                         final_test=(f"{Y}-01-01", f"{Y}-12-31")))
            opY = EvaluationOperator(tY)
            win = (f"{Y}-01-01", f"{Y}-12-31")
            tk = time.time()
            book = opY._strategy_batch(cands, {}, panel, win, report=True)
            pred = book["prediction"]
            wide = _to_wide(pred).reindex(index=panel.dates, columns=panel.instruments)
            wide = wide.where(panel.universe)
            nr = pd.Series(book["metrics"]["_net_return_series"]).astype(float)
            nr.index = pd.to_datetime(nr.index)
            series.append(nr)

            # rebuild the SAME weights _book used, for the transfer coefficient
            wp = pred.loc[win[0]:win[1]]
            beta = 1.0
            if tY.portfolio.cost_aware_dropout or tY.portfolio.construction == "mean_variance":
                sw = tY.splits.window(getattr(tY.portfolio, "scale_split", None)
                                      or tY.combiner.fit_split)
                beta = prediction_scale(pred, y_tilde, sw)
            w, _ = build_book(wp, tY, y_tilde=y_tilde,
                              universe=panel.universe.loc[wp.index], mask=mask,
                              sigma=sigma, pred_scale=beta, close=panel.close, adv=adv)
            tc = transfer_coefficient(wp, w)

            blk = _ic_block(wide, label, win)
            arr, ir, mdd = annualize(nr)
            rows[str(Y)] = {"fit": [a.fit_start, f"{Y-1}-12-31"],
                            "ic": blk.get("ic"), "rank_ic": blk.get("rank_ic"),
                            "arr": arr, "ir": ir, "mdd": mdd, "tc": tc,
                            "days": int(len(nr)), "secs": round(time.time()-tk, 1)}
            print(f"  {Y}  fit<={Y-1}  RankIC {blk.get('rank_ic'):.4f}  "
                  f"ARR {arr*100:+.2f}%  IR {ir:+.2f}  TC {tc:.2f}  "
                  f"({time.time()-tk:.0f}s)", flush=True)

        stitched = pd.concat(series).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        arr, ir, mdd = annualize(stitched)
        rows["stitched"] = {"arr": arr, "ir": ir, "mdd": mdd,
                            "days": int(len(stitched))}
        print(f"  STITCHED {y0}-{y1}: ARR {arr*100:+.2f}%  IR {ir:+.2f}  "
              f"MDD {mdd*100:.1f}%  ({len(stitched)} days)", flush=True)
        out["libraries"][lib] = {"n_factors": len(cands), "years": rows}

    Path(a.out).write_text(json.dumps(out, indent=2, default=lambda o: None))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
