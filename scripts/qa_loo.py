#!/usr/bin/env python
"""Leave-one-out: which factors HURT the book?

For each factor f, price the book WITHOUT it and compare to the full book:

    harm(f) = net_ARR(book \\ f) - net_ARR(book)

A POSITIVE harm means the book is better off without f -- that factor is
destroying value. This is the marginal contribution measured the honest way:
against the book as it actually stands, out of sample, net of the full cost
model, rather than against an empty repository in-sample.

n+1 full evaluations, so it is only practical on a small book. The aligned
frames are cached, so the cost is the combiner fit and the pricing, not the
data.

Usage::

    python scripts/qa_loo.py --library data/results/zoo_newgate.json --report
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

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("qa_loo")
TRADING_DAYS = 243


def ann(r) -> tuple[float, float]:
    """(ARR, IR) from a daily net-return series."""
    s = pd.Series(r).astype(float).dropna()
    if s.empty:
        return float("nan"), float("nan")
    arr = float((1.0 + s).prod() ** (TRADING_DAYS / len(s)) - 1.0)
    vol = float(s.std() * np.sqrt(TRADING_DAYS))
    return arr, (arr / vol if vol > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--report", action="store_true", help="score on final_test")
    ap.add_argument("--out", default=None)
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
    print(f"protocol {theta.hash} | {len(exprs)} factors | scored {win}")

    sig = {}
    for e in exprs:
        try:
            sig[e] = load_aligned_signal(e, panel)
        except Exception as exc:
            logger.warning("skip %.50s: %s", e, exc)
    exprs = [e for e in exprs if e in sig]
    print(f"loaded {len(exprs)} signals\n")

    t0 = time.time()
    full = op.evaluate(sig, zoo_signals={}, zoo_metrics=[], report=a.report)
    full_arr, full_ir = ann(full.get("_net_return_series"))
    print(f"FULL BOOK ({len(exprs)} factors): net_ARR {100*full_arr:+.2f}%  "
          f"net_IR {full_ir:+.3f}   [{time.time()-t0:.0f}s]\n")

    base = op._baseline({}, panel, win, report=a.report)
    base_arr, base_ir = ann(base.get("_net_return_series"))
    print(f"NO-ALPHA BASELINE          : net_ARR {100*base_arr:+.2f}%  "
          f"net_IR {base_ir:+.3f}\n")

    rows = []
    for k, drop in enumerate(exprs, 1):
        sub = {e: s for e, s in sig.items() if e != drop}
        r = op.evaluate(sub, zoo_signals={}, zoo_metrics=[], report=a.report)
        arr, ir = ann(r.get("_net_return_series"))
        harm = arr - full_arr          # >0 => the book is BETTER without it
        rows.append({"expr": drop, "arr_without": arr, "ir_without": ir,
                     "harm_arr": harm, "harm_ir": ir - full_ir})
        print(f"  [{k:2d}/{len(exprs)}] without: ARR {100*arr:+7.2f}% "
              f"IR {ir:+6.3f}  harm {100*harm:+6.2f}pp   {drop[:58]}", flush=True)

    rows.sort(key=lambda r: -r["harm_arr"])
    print("\n" + "=" * 96)
    print("RANKED BY HARM (positive = the book is BETTER without this factor)")
    print("=" * 96)
    print(f"{'harm_ARR':>10} {'harm_IR':>9}   expression")
    print("-" * 96)
    for r in rows:
        flag = "  <== HARMFUL" if r["harm_arr"] > 0 else ""
        print(f"{100*r['harm_arr']:>9.2f}pp {r['harm_ir']:>+9.3f}   {r['expr'][:62]}{flag}")
    print("-" * 96)
    harmful = [r for r in rows if r["harm_arr"] > 0]
    print(f"\nfactors the book is better WITHOUT: {len(harmful)}/{len(rows)}")
    if harmful:
        tot = sum(r["harm_arr"] for r in harmful)
        print(f"  their combined (additive, approximate) drag: {100*tot:.2f}pp")
        keep = [r["expr"] for r in rows if r["harm_arr"] <= 0]
        print(f"  dropping all {len(harmful)} would leave {len(keep)} factors")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"full_arr": full_arr, "full_ir": full_ir,
             "base_arr": base_arr, "base_ir": base_ir, "rows": rows},
            indent=2, default=float))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
