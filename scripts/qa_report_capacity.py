#!/usr/bin/env python
"""Trading capacity: does the book still work at 500M and 1B CNY?

Market impact is the only cost term that scales with size. Participation is
|dw| / (dollar ADV / NAV), so raising NAV raises participation and therefore the
quadratic impact term -- a book that is profitable at 100M can be unprofitable at
1B purely because it cannot trade its own signal.

Reports BOTH return conventions at each NAV:
  * NET EXCESS  -- w.y - benchmark - cost   (what the rest of the report shows)
  * NET         -- w.y - cost               (the raw return the book earns, with
                    the benchmark added back; this is the "net ARR" of the book)

    python scripts/qa_report_capacity.py --out data/results/report_capacity.json
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import load_aligned_signal, load_benchmark

TRADING_DAYS = 243
NAVS = [1e8, 5e8, 1e9]          # 100M, 500M, 1B CNY
LIBS = [
    ("main_zoo",  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json"),
    ("main_full", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("original",  "data/factorlib/all_factors_library_original_20260831_012324.json"),
]


def ann(r: pd.Series):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan,) * 4
    arr = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ir = float(arr / vol) if vol > 0 else np.nan
    curve = (1.0 + r).cumprod()
    return arr, vol, ir, float((curve / curve.cummax() - 1.0).min())


def exprs_of(path: Path):
    d = json.loads(path.read_text())
    f = d.get("factors", d)
    items = f.values() if isinstance(f, dict) else f
    return [e for e in (it.get("factor_expression") or it.get("expression")
                        for it in items) if e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--long-window", action="store_true")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"),
                    help="score on an explicit window, e.g. 2017-01-01 2025-12-31 "
                         "(2016 is excluded by default in the report because the "
                         "constructor blows up in the first year after its fit)")
    ap.add_argument("--out", default="data/results/report_capacity.json")
    a = ap.parse_args()

    base = load_protocol(a.protocol)
    base = replace(base, benchmark_basis="estimated_total",
                   benchmark_construction="equal")
    if a.window:
        base = replace(base, splits=replace(base.splits,
                       final_test=(a.window[0], a.window[1])))
    elif a.long_window:
        base = replace(base, splits=replace(base.splits,
                       final_test=("2016-01-01", "2026-01-09")))
    out = {"navs": NAVS, "benchmark_basis": base.benchmark_basis,
           "long_window": bool(a.long_window),
           "window": list(base.splits.final_test), "libraries": {}}

    for name, rel in LIBS:
        out["libraries"][name] = {"library": rel, "by_nav": {}}
        for nav in NAVS:
            t0 = time.time()
            theta = replace(base, costs=replace(base.costs, nav=float(nav)))
            op = EvaluationOperator(theta)
            p0, p1, win = op._windows(True)
            panel = op._panel(p0, p1)
            cands = {}
            for e in exprs_of(ROOT / rel):
                try:
                    cands[e] = load_aligned_signal(e, panel)
                except Exception:
                    pass
            book = op._strategy_batch(cands, {}, panel, win, report=True)
            m = book["metrics"]
            exc = pd.Series(m["_net_return_series"]).astype(float)
            exc.index = pd.to_datetime(exc.index)
            # add the benchmark back to recover the NET (non-excess) book return
            bench = load_benchmark(theta, win[0], win[1])
            bench.index = pd.to_datetime(bench.index)
            net = exc.add(bench.reindex(exc.index).fillna(0.0), fill_value=0.0)
            e_arr, _, e_ir, e_mdd = ann(exc)
            n_arr, _, n_ir, n_mdd = ann(net)
            row = {
                "n_factors": len(cands),
                "net_excess_arr": e_arr, "net_excess_ir": e_ir, "net_excess_mdd": e_mdd,
                "net_arr": n_arr, "net_ir_series": n_ir, "net_mdd": n_mdd,
                "cost_bps": float(m.get("cost_bps", np.nan)),
                "turnover_book": float(m.get("turnover_book", np.nan)),
                "protocol_net_ir": float(m.get("net_ir", np.nan)),
                "secs": round(time.time() - t0, 1),
            }
            out["libraries"][name]["by_nav"][f"{nav:.0f}"] = row
            print(f"[{name} @ NAV {nav/1e8:.0f}e8] net_excess_arr {100*e_arr:+7.2f}% | "
                  f"net_arr {100*n_arr:+7.2f}% | cost {row['cost_bps']:.2f}bps | "
                  f"turnover {row['turnover_book']:.4f} ({row['secs']:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
