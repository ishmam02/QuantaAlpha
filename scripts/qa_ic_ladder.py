#!/usr/bin/env python
"""How much IC does this construction need before net_IR turns positive?

Builds a signal of KNOWN quality by mixing the realised forward return with
noise, prices it through the real book (ICIR combiner -> mean-variance ->
full cost model), and reports net_IR against the achieved RankIC. The crossing
point is the target the generator has to hit for the system to make money.

This is the ladder the 2026-08-2x parameter audit intended to run. That attempt
keyed every noisy variant as ``"noisy"`` in the candidates dict, so the
evaluator's cache returned the first one and 1x/3x/10x noise reported IDENTICAL
numbers. Each rung here carries its own expression string, and the script
asserts the achieved ICs are distinct before reporting anything.

The oracle is LOOK-AHEAD by construction -- it is a probe of the construction's
capacity, never a tradable signal, and it never touches the mine.

Usage::

    python scripts/qa_ic_ladder.py --report
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

from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)
TRADING_DAYS = 243


def ann(r):
    s = pd.Series(r).astype(float).dropna()
    if s.empty:
        return float("nan"), float("nan")
    arr = float((1.0 + s).prod() ** (TRADING_DAYS / len(s)) - 1.0)
    vol = float(s.std() * np.sqrt(TRADING_DAYS))
    return arr, (arr / vol if vol > 0 else float("nan"))


def forward_return(panel, theta) -> pd.DataFrame:
    """The label the book actually trades: open[t+2]/open[t+1] - 1."""
    o = pd.DataFrame(panel.open, index=pd.DatetimeIndex(panel.dates),
                     columns=list(panel.instruments))
    return (o.shift(-2) / o.shift(-1) - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--report", action="store_true", help="score on final_test")
    ap.add_argument("--noise", default="0,0.5,1,2,3,5,8,12,20,32",
                    help="noise multiples; 0 = the perfect oracle")
    ap.add_argument("--out", default="data/results/ic_ladder.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(a.report)
    panel = op._panel(p0, p1)
    print(f"protocol {theta.hash} | panel {p0}..{p1} | scored {win}\n")

    fwd = forward_return(panel, theta)
    sd = float(np.nanstd(fwd.values))
    rng = np.random.default_rng(42)

    # A no-alpha reference: the same construction with a pure-noise signal.
    rungs = [float(x) for x in a.noise.split(",")]
    rows = []
    print(f"{'noise':>7} {'rank_IC':>9} {'net_IR':>9} {'net_ARR':>10} {'TC':>7}  secs")
    print("-" * 58)
    for k in rungs:
        sig = fwd + (rng.normal(0.0, sd * k, size=fwd.shape) if k > 0 else 0.0)
        # UNIQUE key per rung: the evaluator caches on the candidate dict, and a
        # shared key is exactly what invalidated the previous attempt.
        key = f"__ORACLE_noise_{k:g}__"
        t0 = time.time()
        res = op.evaluate({key: sig}, zoo_signals={}, zoo_metrics=[], report=a.report)
        # Use the operator's OWN net_ir / net_arr, not a re-derivation from the
        # return series. `_net_return_series` is not always a series -- on the
        # valid window it came back as a scalar, and `pd.Series(float).dropna()`
        # is empty, which silently produced NaN for every rung while the ICs and
        # TC looked perfectly healthy. The metrics the evaluator publishes are
        # the same numbers admission uses; recomputing them here only creates a
        # second definition that can disagree.
        ir = res.get("m_net_ir")
        arr = res.get("m_net_arr")
        try:
            ir = float(ir) if ir is not None else float("nan")
            arr = float(arr) if arr is not None else float("nan")
        except (TypeError, ValueError):
            ir = arr = float("nan")
        if ir != ir:                       # fall back only if the metric is absent
            arr, ir = ann(res.get("_net_return_series"))
        ic = res.get("m_rank_ic")
        tc = res.get("m_transfer_coefficient")
        rows.append({"noise": k, "rank_ic": ic, "net_ir": ir, "net_arr": arr,
                     "tc": tc})
        print(f"{k:>7g} {ic:>+9.4f} {ir:>+9.3f} {100*arr:>+9.2f}% {tc:>+7.3f}"
              f"  {time.time()-t0:.0f}", flush=True)

    ics = [r["rank_ic"] for r in rows if r["rank_ic"] is not None]
    assert len(set(np.round(ics, 6))) == len(ics), (
        "rungs returned IDENTICAL ICs -- the evaluator cache collapsed them, "
        "which is the bug that invalidated the previous ladder")

    # Where does net_IR cross zero? Interpolate between the bracketing rungs.
    ok = [r for r in rows if r["rank_ic"] == r["rank_ic"] and r["net_ir"] == r["net_ir"]]
    ok.sort(key=lambda r: r["rank_ic"])
    cross = None
    for lo, hi in zip(ok, ok[1:]):
        if lo["net_ir"] <= 0 < hi["net_ir"]:
            w = (0 - lo["net_ir"]) / (hi["net_ir"] - lo["net_ir"])
            cross = lo["rank_ic"] + w * (hi["rank_ic"] - lo["rank_ic"])
            break

    print("\n" + "=" * 58)
    if cross is not None:
        print(f"BREAK-EVEN RankIC (net_IR = 0): {cross:+.4f}")
        print(f"  the mine currently produces ~0.02 -> shortfall "
              f"{cross / 0.02:.1f}x")
    else:
        print("net_IR does not cross zero inside the ladder -- widen --noise")
    print("=" * 58)

    Path(a.out).write_text(json.dumps(
        {"rows": rows, "breakeven_rank_ic": cross, "window": list(win)},
        indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
