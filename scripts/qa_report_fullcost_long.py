#!/usr/bin/env python
"""Full-cost LGBM-combiner + top-k backtest, per YEAR, for each report library.

Protocol: quantaalpha/eval/protocol_csi300.yaml -- combiner.model=lightgbm
(LGBM stacking), portfolio.construction=topk_dropout (topk 50 / n_drop 5),
full kappa cost model (kappa0+kappa1+kappa2), final_test 2022-01-01..2025-12-26.

Per-year IC / Rank IC are a gap in the shipped tooling (the operator computes a
per-date _ic_series then filters underscore keys out at metrics.py:19-20), so
this calls the same _ic_block primitive per calendar year on the combined
prediction -- exactly the quantity Qlib's SigAnaRecord reports -- and groups the
book's daily net-return series by year for ARR.

    python scripts/qa_report_fullcost_yearly.py --out data/results/report_fullcost_yearly.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import load_aligned_signal
from quantaalpha.eval.metrics import _ic_block, label_frame, _to_wide

TRADING_DAYS = 243

LIBS = [
    ("main_zoo",  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json"),
    ("main_full", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("original",  "data/factorlib/all_factors_library_original_20260831_012324.json"),
]


def annualize(r: pd.Series):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan,) * 4
    arr = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ir = float(arr / vol) if vol > 0 else np.nan
    curve = (1.0 + r).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return arr, vol, ir, mdd


def exprs_of(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors
    out = [it.get("factor_expression") or it.get("expression") for it in items]
    return [e for e in out if e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--out", default="data/results/report_fullcost_yearly.json")
    ap.add_argument("--long-window", action="store_true",
                    help="score on 2016-01-01..2026-01-09 (the 10y holdout) instead "
                         "of the protocol's own final_test")
    ap.add_argument("--no-dividend-fix", action="store_true",
                    help="leave the protocol's price-return cap-weighted benchmark as-is")
    a = ap.parse_args()

    theta = load_protocol(a.protocol)
    # DIVIDEND FIX. protocol_csi300.yaml omits benchmark_basis/benchmark_construction,
    # so it defaults to a PRICE-return, CAP-weighted SH000300 benchmark. The book
    # prices with ADJUSTED closes (dividends reinvested), so a price-return benchmark
    # credits the strategy with the index's entire dividend yield as if it were alpha
    # (+4.25pp/yr measured on CSI300). And the book is structurally equal-weight
    # (max_weight caps every large constituent), so scoring it against a cap-weighted
    # index adds an unchosen size bet. Both are put on the same basis here, matching
    # protocol_csi300_meanvar_soft_linear.yaml:56-58.
    if not a.no_dividend_fix:
        theta = replace(theta, benchmark_basis="estimated_total",
                        benchmark_construction="equal")
    print(f"benchmark: basis={theta.benchmark_basis} "
          f"construction={theta.benchmark_construction}", flush=True)
    if a.long_window:
        from dataclasses import replace as _rep
        theta = _rep(theta, splits=_rep(theta.splits,
                     final_test=("2016-01-01", "2026-01-09")))
        print("LONG WINDOW: final_test forced to 2016-01-01..2026-01-09", flush=True)
    op = EvaluationOperator(theta)
    p_start, p_end, eval_window = op._windows(True)          # report=True -> final_test
    panel = op._panel(p_start, p_end)
    label = label_frame(panel, theta)
    print(f"protocol {theta.hash} | combiner={theta.combiner.model} | "
          f"construction={theta.portfolio.construction} | SCORED {eval_window}", flush=True)

    results = {"protocol_hash": theta.hash, "combiner": theta.combiner.model,
               "benchmark_basis": theta.benchmark_basis,
               "benchmark_construction": theta.benchmark_construction,
               "construction": theta.portfolio.construction,
               "eval_window": list(eval_window), "libraries": {}}

    # the no-alpha baseline (empty book) on the same window, priced the same way
    base = op._baseline({}, panel, eval_window, report=True)
    base_series = base.get("_net_return_series")
    if base_series is not None:
        base_series = pd.Series(base_series).astype(float).dropna()
        base_series.index = pd.to_datetime(base_series.index)
        b_rows = {}
        for yr, r in base_series.groupby(base_series.index.year):
            arr, _, ir, mdd = annualize(r)
            b_rows[int(yr)] = {"arr": arr, "ir": ir, "mdd": mdd, "days": int(len(r))}
        arr, _, ir, mdd = annualize(base_series)
        b_rows["full"] = {"arr": arr, "ir": ir, "mdd": mdd, "days": int(len(base_series))}
        results["baseline_no_alpha"] = b_rows
        print(f"  baseline no-alpha FULL ARR {100*arr:.2f}% IR {ir:.2f}", flush=True)

    for label_name, rel in LIBS:
        t0 = time.time()
        path = ROOT / rel
        exprs = exprs_of(path)
        cands = {}
        for e in exprs:
            try:
                cands[e] = load_aligned_signal(e, panel)
            except Exception as exc:
                print(f"    skip {str(exc)[:60]}", flush=True)
        print(f"[{label_name}] {len(cands)}/{len(exprs)} signals loaded "
              f"({time.time()-t0:.0f}s)", flush=True)
        book = op._strategy_batch(cands, {}, panel, eval_window, report=True)
        pred = book["prediction"]
        wide = _to_wide(pred).reindex(index=panel.dates, columns=panel.instruments)
        wide = wide.where(panel.universe)
        nr = pd.Series(book["metrics"]["_net_return_series"]).astype(float)
        nr.index = pd.to_datetime(nr.index)

        rows = {}
        for yr, r in nr.groupby(nr.index.year):
            blk = _ic_block(wide, label, (f"{yr}-01-01", f"{yr}-12-31"))
            arr, _, ir, mdd = annualize(r)
            rows[int(yr)] = {"ic": float(blk.get("ic", np.nan)),
                             "rank_ic": float(blk.get("rank_ic", np.nan)),
                             "icir": float(blk.get("icir", np.nan)),
                             "rank_icir": float(blk.get("rank_icir", np.nan)),
                             "arr": arr, "ir": ir, "mdd": mdd, "days": int(len(r))}
        blk = _ic_block(wide, label, eval_window)
        arr, _, ir, mdd = annualize(nr)
        rows["full"] = {"ic": float(blk.get("ic", np.nan)),
                        "rank_ic": float(blk.get("rank_ic", np.nan)),
                        "icir": float(blk.get("icir", np.nan)),
                        "rank_icir": float(blk.get("rank_icir", np.nan)),
                        "arr": arr, "ir": ir, "mdd": mdd, "days": int(len(nr))}
        m = book["metrics"]
        rows["book"] = {k: (float(m[k]) if isinstance(m.get(k), (int, float)) else None)
                        for k in ("net_ir", "net_arr", "cost_bps", "turnover_book",
                                  "mdd", "effective_rank")}
        results["libraries"][label_name] = {"library": rel, "n_factors": len(cands),
                                            "years": rows, "secs": round(time.time()-t0, 1)}
        print(f"[{label_name}] FULL ic={rows['full']['ic']:.4f} "
              f"rank_ic={rows['full']['rank_ic']:.4f} ARR={100*rows['full']['arr']:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps(results, indent=2))

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
