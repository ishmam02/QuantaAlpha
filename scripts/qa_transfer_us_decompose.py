#!/usr/bin/env python
"""Decompose the US transfer's weak result into FACTORS vs COMBINER vs CONSTRUCTION+COSTS.

Loads the cached recomputed US factor signals (qa_transfer_us.py --cache-signals) and
the US protocol, then measures, on the test window (2016-2026):

  1. PER-FACTOR quality: each factor's IC / RankIC on the US test (the raw signal).
  2. COMBINER: equal-weight (average the factors) IC/RankIC vs the LightGBM
     combined IC/RankIC -- does the combiner add value over a naive average, or
     does it overfit / degrade?
  3. CONSTRUCTION + COSTS: the GROSS (no-cost) topk book excess vs the NET (US
     realistic cost) excess -- how much of the underperformance is the cost, and
     how much is the construction + raw signal before cost?

Usage::

    QA_THREADS=8 conda run -n quantaalpha python scripts/qa_transfer_us_decompose.py \\
        --protocol quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml \\
        --cache data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \\
        --qlib-dir data/qlib/us_data --window 2016-01-01 2026-01-09 --nav 1000000000 \\
        --cache-signals /tmp/us_candidates.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval import data as data_mod
from quantaalpha.eval.data import align_signal
from quantaalpha.eval.metrics import _ic_block, label_frame
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

TD = 252


def annualize(r):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() ** (TD / len(r)) - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml")
    ap.add_argument("--cache", default="data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    ap.add_argument("--qlib-dir", default="data/qlib/us_data")
    ap.add_argument("--window", nargs=2, default=["2016-01-01", "2026-01-09"])
    ap.add_argument("--nav", type=float, default=1e9)
    ap.add_argument("--cache-signals", default="/tmp/us_candidates.pkl")
    args = ap.parse_args()

    os.environ["QLIB_PROVIDER_URI"] = str(Path(args.qlib_dir).resolve())
    os.environ["QLIB_REGION"] = "us"

    base = load_protocol(args.protocol)
    base = replace(base, splits=replace(base.splits, final_test=(args.window[0], args.window[1])))
    base = replace(base, benchmark_basis="estimated_total", benchmark_construction="index")
    th = replace(base, costs=replace(base.costs, nav=float(args.nav)))
    op = EvaluationOperator(th)
    p0, p1, win = op._windows(True)
    panel = op._panel(p0, p1)
    label = label_frame(panel, th)
    print(f"window {win} | panel {len(panel.instruments)} instr | label {label.shape}", flush=True)

    with open(args.cache_signals, "rb") as fh:
        cached, ok, fail = pickle.load(fh)
    cands = {e: (data_mod._align(s, panel) if isinstance(s, pd.DataFrame) else align_signal(s, panel))
             for e, s in cached.items()}
    print(f"loaded {len(cands)} signals", flush=True)

    # ---- 1. PER-FACTOR IC / RankIC on the test window ----
    per_ic, per_ric = [], []
    for e, s in cands.items():
        blk = _ic_block(s, label, win)
        if isinstance(blk.get("ic"), (int, float)) and blk["ic"] == blk["ic"]:
            per_ic.append(float(blk["ic"]))
        if isinstance(blk.get("rank_ic"), (int, float)) and blk["rank_ic"] == blk["rank_ic"]:
            per_ric.append(float(blk["rank_ic"]))
    per_ic = np.array(per_ic); per_ric = np.array(per_ric)
    print("\n=== 1. PER-FACTOR quality on US test (the raw factor signal) ===")
    print(f"  factors: {len(per_ic)} | IC  mean {np.mean(per_ic):+.4f} median {np.median(per_ic):+.4f} "
          f"max {np.max(per_ic):+.4f} min {np.min(per_ic):+.4f} | >0: {(per_ic > 0).mean()*100:.0f}%")
    print(f"  factors: {len(per_ric)} | RankIC mean {np.mean(per_ric):+.4f} median {np.median(per_ric):+.4f} "
          f"max {np.max(per_ric):+.4f} min {np.min(per_ric):+.4f} | >0: {(per_ric > 0).mean()*100:.0f}%")

    # ---- 2. COMBINER: equal-weight vs LightGBM ----
    # equal-weight: average the aligned factor frames
    frames = list(cands.values())
    eq = sum(frames) / len(frames)
    blk_eq = _ic_block(eq, label, win)
    print("\n=== 2. COMBINER (combined prediction IC, test window) ===")
    print(f"  equal-weight avg:  IC {float(blk_eq.get('ic', np.nan)):+.4f}  "
          f"RankIC {float(blk_eq.get('rank_ic', np.nan)):+.4f}")

    # LightGBM combined: run the combiner (fit on train, predict test)
    t0 = time.time()
    from quantaalpha.eval import combiner as combiner_mod
    pred, _ = combiner_mod.fit_predict({}, None, panel, th, candidate_signals=cands)
    from quantaalpha.eval.metrics import _to_wide
    wide = _to_wide(pred).reindex(index=panel.dates, columns=panel.instruments).where(panel.universe)
    blk_lgb = _ic_block(wide, label, win)
    print(f"  LightGBM 5-seed:  IC {float(blk_lgb.get('ic', np.nan)):+.4f}  "
          f"RankIC {float(blk_lgb.get('rank_ic', np.nan)):+.4f}  ({time.time()-t0:.0f}s)")
    print(f"  (the single best factor had RankIC {np.max(per_ric):+.4f}; equal-weight {float(blk_eq.get('rank_ic', np.nan)):+.4f})")

    # ---- 3. CONSTRUCTION + COSTS: gross (no-cost) vs net (US cost) topk book ----
    print("\n=== 3. CONSTRUCTION + COSTS (topk book excess, test window) ===")
    bench = op._benchmark(str(win[0]), str(win[1]))
    rows = {}
    for tag, costs in (("GROSS (no cost)", (0.0, 0.0, 0.0)),
                       ("NET (US realistic)", (th.costs.kappa0, th.costs.kappa1, th.costs.kappa2))):
        thc = replace(th, costs=replace(th.costs, kappa0=costs[0], kappa1=costs[1], kappa2=costs[2]))
        opc = EvaluationOperator(thc)
        book = opc._strategy_batch(cands, {}, panel, win, report=True)
        nr = pd.Series(book["metrics"]["_net_return_series"]).astype(float)
        nr.index = pd.to_datetime(nr.index)
        exc_arr = annualize(nr)
        rows[tag] = {"excess_arr": exc_arr,
                    "cost_bps": float(book["metrics"].get("cost_bps", np.nan)),
                    "turnover": float(book["metrics"].get("turnover_book", np.nan))}
        print(f"  {tag:<20}: excess {100*exc_arr:+7.2f}%/yr  cost {rows[tag]['cost_bps']:.2f}bps  "
              f"turnover {rows[tag]['turnover']:.4f}")
    print(f"\n  -> cost accounts for {100*(rows['NET (US realistic)']['excess_arr'] - rows['GROSS (no cost)']['excess_arr']):+.2f}pp/yr of the excess; "
          f"the gross (no-cost) topk book itself is {100*rows['GROSS (no cost)']['excess_arr']:+.2f}%/yr vs the cap-weight index.")
    print("\nSUMMARY: factors (per-factor RankIC) -> combiner (eq vs LightGBM) -> construction (gross) -> costs (net).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())