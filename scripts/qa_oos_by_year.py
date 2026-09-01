#!/usr/bin/env python
"""Per-YEAR net performance of a book on the test window, vs the no-alpha baseline.

A ten-year aggregate hides the thing that matters most about a mined book: an
alpha that works for two years and then dies reports the same full-sample number
as one that never worked. Decay is a property of the path, so the path is what
gets reported here -- year by year, book and baseline side by side.

Reuses the aligned-frame cache written by ``qa_eval_oneshot`` / the mine, so a
re-run costs a pickle read per factor rather than a 3.6s re-align.

Usage::

    python scripts/qa_oos_by_year.py \\
        --zoo data/results/ledger_full_20260824_002254.jsonl \\
        --protocol quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_oos_by_year")

TRADING_DAYS = 243


def annualize(r: pd.Series) -> tuple[float, float, float, float]:
    """(ARR, vol, IR, maxDD) from a daily net-return series."""
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan,) * 4
    arr = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ir = float(arr / vol) if vol > 0 else np.nan
    curve = (1.0 + r).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return arr, vol, ir, mdd


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zoo", metavar="LEDGER")
    src.add_argument("--library")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", default=None, help="write the per-year table as JSON")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    if a.zoo:
        exprs = list(replay_repository(a.zoo))
    else:
        payload = json.loads(Path(a.library).read_text())
        factors = payload.get("factors", payload)
        items = factors.values() if isinstance(factors, dict) else factors
        exprs = [e.get("factor_expression") or e.get("expression") for e in items]
        exprs = [e for e in exprs if e]

    op = EvaluationOperator(theta)
    p_start, p_end, eval_window = op._windows(True)
    panel = op._panel(p_start, p_end)
    print(f"protocol {theta.hash} | panel {p_start}..{p_end} | SCORED {eval_window}")

    t0 = time.time()
    cands = {}
    for e in exprs:
        try:
            cands[e] = load_aligned_signal(e, panel)
        except Exception as exc:
            logger.warning("skipping %.50s: %s", e, exc)
    print(f"loaded {len(cands)}/{len(exprs)} signals in {time.time() - t0:.1f}s")
    if not cands:
        return 2

    res = op.evaluate(cands, zoo_signals={}, zoo_metrics=[], report=True)
    book = res.get("_net_return_series")
    if book is None:
        print("no net-return series returned; cannot break down by year")
        return 3

    # The no-alpha baseline over the SAME window: an empty zoo priced through
    # the same construction and the same cost model.
    base = op._baseline({}, panel, eval_window, report=True)
    base_series = base.get("_net_return_series")

    book = pd.Series(book).astype(float).dropna()
    book.index = pd.to_datetime(book.index)
    if base_series is not None:
        base_series = pd.Series(base_series).astype(float).dropna()
        base_series.index = pd.to_datetime(base_series.index)

    print()
    print("=" * 92)
    print(f"{'year':>6} {'days':>5} | {'book ARR':>10} {'IR':>7} {'maxDD':>8}"
          f" | {'base ARR':>10} {'IR':>7} | {'ARR delta':>10}")
    print("-" * 92)
    rows = []
    for yr, r in book.groupby(book.index.year):
        arr, vol, ir, mdd = annualize(r)
        b_arr = b_ir = np.nan
        if base_series is not None:
            br = base_series[base_series.index.year == yr]
            if len(br):
                b_arr, _, b_ir, _ = annualize(br)
        d = arr - b_arr if np.isfinite(b_arr) else np.nan
        flag = "  <-- book beats no-alpha" if np.isfinite(d) and d > 0 else ""
        print(f"{yr:>6} {len(r):>5} | {100*arr:>9.2f}% {ir:>7.2f} {100*mdd:>7.1f}%"
              f" | {100*b_arr:>9.2f}% {b_ir:>7.2f} | {100*d:>9.2f}pp{flag}")
        rows.append({"year": int(yr), "days": int(len(r)), "book_arr": arr,
                     "book_ir": ir, "book_mdd": mdd, "base_arr": b_arr,
                     "base_ir": b_ir, "arr_delta": d})
    print("-" * 92)
    arr, vol, ir, mdd = annualize(book)
    b = annualize(base_series) if base_series is not None else (np.nan,) * 4
    print(f"{'FULL':>6} {len(book):>5} | {100*arr:>9.2f}% {ir:>7.2f} {100*mdd:>7.1f}%"
          f" | {100*b[0]:>9.2f}% {b[2]:>7.2f} | {100*(arr-b[0]):>9.2f}pp")
    print("=" * 92)

    wins = [r for r in rows if np.isfinite(r["arr_delta"]) and r["arr_delta"] > 0]
    print(f"\nyears the book beat the no-alpha baseline: {len(wins)}/{len(rows)}"
          + (f"  ({', '.join(str(r['year']) for r in wins)})" if wins else ""))
    pos = [r for r in rows if np.isfinite(r["book_arr"]) and r["book_arr"] > 0]
    print(f"years the book made money outright        : {len(pos)}/{len(rows)}"
          + (f"  ({', '.join(str(r['year']) for r in pos)})" if pos else ""))

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
