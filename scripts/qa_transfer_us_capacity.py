#!/usr/bin/env python
"""US transfer COMPOUNDING CAPACITY: the CSI300-mined zoo recomputed on S&P 500,
full realistic cost, fund growing at its own return (compounding), at multiple NAVs
to find where the US book breaks.

Mirrors scripts/qa_report_capacity_compounding.py (CompoundingNav) but on US data:
  * the factors are RECOMPUTED on the US cache (or loaded from qa_transfer_us.py's
    --cache-signals pickle), not read from the CSI md5 cache;
  * the protocol is the US realistic LightGBM+topk one;
  * the window is 2022-2025 (the report's primary) so the per-date book loop is
    ~4x faster than the full 2016-2026 range;
  * NAVs default to $100M/$500M/$1B/$5B -- the US is ~10x more liquid than CSI, so
    a $100M fund pays negligible impact; the table finds the break point.

For each NAV x mode (fixed / compounding), the book is priced; compounding charges
each day against nav_t = nav0 * prod_{s<t}(1+r_s) (fund grows at its absolute
return), fixed charges against the constant nav0.

Usage::

    QA_THREADS=8 conda run -n quantaalpha python scripts/qa_transfer_us_capacity.py \\
        --library data/factorlib/all_factors_library_meanvar_20260828_194432.json \\
        --protocol quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml \\
        --cache data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \\
        --qlib-dir data/qlib/us_data --window 2022-01-01 2025-12-26 \\
        --cache-signals /tmp/us_candidates.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval import costs as costs_mod
from quantaalpha.eval import data as data_mod
from quantaalpha.eval.data import align_signal
from quantaalpha.eval.execution import fill_prices, realized_return
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

from scripts.qa_transfer_us import load_library, recompute_factor  # reuse

TD = 252


class CompoundingNav:
    """Wraps ``costs.cost`` so each day is charged at the fund's current (grown)
    size. Replicated from scripts/qa_report_capacity_compounding.py."""

    def __init__(self, theta, nav0, y_tilde, bench):
        self._orig = costs_mod.cost
        self.theta, self.nav0 = theta, float(nav0)
        self.y_tilde, self.bench = y_tilde, bench
        self.nav = float(nav0)
        self._prev_date = None
        self.passes = []

    def __call__(self, w, w_drift, sigma, adv, theta, offlist=None):
        date = w.name if getattr(w, "name", None) is not None else None
        if date is not None and self._prev_date is not None and date <= self._prev_date:
            self.passes.append(self.nav)
            self.nav = float(self.nav0)
        self._prev_date = date
        th = replace(theta, costs=replace(theta.costs, nav=self.nav))
        c = self._orig(w, w_drift, sigma, adv, th, offlist)
        if date is not None and date in self.y_tilde.index:
            yr = self.y_tilde.loc[date].reindex(w.index).fillna(0.0)
            gross = float((w * yr).sum())
            step = 1.0 + gross - c
            if np.isfinite(step) and step > 0.0:
                self.nav *= step
        return c


def ann(r):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan, np.nan, np.nan)
    cagr = float((1.0 + r).prod() ** (TD / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TD))
    ir = float(cagr / vol) if vol > 0 else np.nan
    cm = r.cumsum()
    mdd = float((cm.cummax() - cm).max())
    return cagr, ir, mdd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="data/factorlib/all_factors_library_meanvar_20260828_194432.json")
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml")
    ap.add_argument("--cache", default="data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    ap.add_argument("--qlib-dir", default="data/qlib/us_data")
    ap.add_argument("--window", nargs=2, default=["2022-01-01", "2025-12-26"])
    ap.add_argument("--navs", default="100000000,500000000,1000000000,5000000000",
                    help="comma list of NAVs (USD) to sweep")
    ap.add_argument("--modes", default="fixed,compounding")
    ap.add_argument("--cache-signals", default=None,
                    help="qa_transfer_us.py --cache-signals pickle (aligned frames); "
                         "re-aligned to this window's panel. If absent, recompute on US.")
    args = ap.parse_args()

    os.environ["QLIB_PROVIDER_URI"] = str(Path(args.qlib_dir).resolve())
    os.environ["QLIB_REGION"] = "us"

    navs = [float(x) for x in args.navs.split(",")]
    modes = args.modes.split(",")

    base = load_protocol(args.protocol)
    base = replace(base, splits=replace(base.splits, final_test=(args.window[0], args.window[1])))
    base = replace(base, benchmark_basis="estimated_total", benchmark_construction="equal")
    print(f"US compounding capacity | protocol {base.hash} | window {args.window} | "
          f"combiner={base.combiner.model} construction={base.portfolio.construction} "
          f"seeds={len(base.combiner.seeds)}", flush=True)

    op = EvaluationOperator(base)
    p0, p1, win = op._windows(True)
    panel = op._panel(p0, p1)
    print(f"panel {p0}..{p1}, {len(panel.instruments)} instruments", flush=True)

    # Factor signals: load + re-align the cached aligned frames, or recompute on US.
    cs = args.cache_signals
    cands: dict[str, object] = {}
    if cs and Path(cs).exists():
        with open(cs, "rb") as fh:
            cached, ok, fail = pickle.load(fh)
        for e, s in cached.items():
            # cached holds wide aligned frames; re-align onto THIS window's panel.
            cands[e] = data_mod._align(s, panel) if isinstance(s, pd.DataFrame) else align_signal(s, panel)
        print(f"loaded {len(cands)} cached signals, re-aligned to {args.window} (ok={ok} fail={fail})", flush=True)
    else:
        df = pd.read_hdf(args.cache, key="data")
        ok = fail = 0
        for name, expr in load_library(args.library):
            try:
                sig = recompute_factor(expr, df, factor_name=name)
                cands[expr] = align_signal(sig, panel)
                ok += 1
            except Exception as exc:
                fail += 1
                print(f"recompute FAILED {name} ({expr[:40]}): {exc}", flush=True)
        print(f"recomputed {ok}/{ok + fail} factors ({fail} failed)", flush=True)
    if not cands:
        print("no factors"); return 2

    y_tilde = realized_return(fill_prices(panel, base))
    bench = op._benchmark(str(win[0]), str(win[1]))

    rows = []
    for nav0 in navs:
        for mode in modes:
            t0 = time.time()
            th = replace(base, costs=replace(base.costs, nav=float(nav0)))
            opn = EvaluationOperator(th)
            hook = None
            if mode == "compounding":
                hook = CompoundingNav(th, nav0, y_tilde, bench)
                costs_mod.cost = hook
            try:
                book = opn._strategy_batch(cands, {}, panel, win, report=True)
            finally:
                if hook is not None:
                    costs_mod.cost = hook._orig
            m = book["metrics"]
            exc = pd.Series(m["_net_return_series"]).astype(float).dropna()
            exc.index = pd.to_datetime(exc.index)
            bs = pd.Series(opn._benchmark(str(win[0]), str(win[1])))
            bs.index = pd.to_datetime(bs.index)
            net = exc.add(bs.reindex(exc.index).fillna(0.0), fill_value=0.0)
            e_cagr, e_ir, e_mdd = ann(exc)
            n_cagr, _, _ = ann(net)
            nav_end = float(hook.passes[0]) if (hook and hook.passes) else float(nav0)
            rec = {"nav0": nav0, "mode": mode, "n_factors": len(cands),
                   "excess_cagr": e_cagr, "net_cagr": n_cagr, "excess_ir": e_ir,
                   "excess_mdd": e_mdd,
                   "cost_bps": float(m.get("cost_bps", np.nan)),
                   "turnover": float(m.get("turnover_book", np.nan)),
                   "rank_ic": float(m.get("rank_ic", np.nan)),
                   "nav_end": nav_end, "nav_growth_x": nav_end / float(nav0),
                   "secs": round(time.time() - t0, 1)}
            rows.append(rec)
            extra = f" nav x{rec['nav_growth_x']:.2f}" if mode == "compounding" else ""
            print(f"[${nav0 / 1e6:>4.0f}M {mode:<11}] excess {100 * e_cagr:+7.2f}% "
                  f"net {100 * n_cagr:+7.2f}% cost {rec['cost_bps']:5.2f}bps "
                  f"turnover {rec['turnover']:.4f} rank_ic {rec['rank_ic']:+.4f}{extra} "
                  f"({rec['secs']:.0f}s)", flush=True)

    print("\n=== US TRANSFER COMPOUNDING CAPACITY (CSI300 zoo -> S&P 500, "
          f"{win[0]}..{win[1]}) ===")
    print(f"{'NAV':>7} {'mode':<11} {'excess%':>8} {'net%':>8} {'cost_bps':>8} "
          f"{'turnover':>8} {'nav_x':>6}")
    for r in rows:
        print(f"${r['nav0'] / 1e6:>5.0f}M {r['mode']:<11} {100 * r['excess_cagr']:+7.2f} "
              f"{100 * r['net_cagr']:+7.2f} {r['cost_bps']:7.2f} {r['turnover']:7.4f} "
              f"{r['nav_growth_x']:6.2f}")
    Path("/tmp/us_capacity.json").write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())