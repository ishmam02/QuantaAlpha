#!/usr/bin/env python
"""Capacity with a COMPOUNDING fund, not a fixed one.

The standard capacity run prices market impact against a constant fund size: a book
labelled 100M is charged as a 100M fund on its first day and on its last, even though
years of compounding would have left it far larger. Impact grows with the size of each
order relative to the stock's daily volume, so a fund that actually grew would pay
more than the fixed-size run reports.

This reprices the same books against a fund value that grows with its own realised net
return,

    nav_t = nav_0 * prod_{s<t} (1 + r_s),

by monkey-patching the cost function that the production evaluator calls once per day.
Everything upstream -- the formulas, the merging model, the portfolio construction, the
benchmark, the trade masks -- is the production code path, untouched, so the only
difference between the two modes is the fund size carried into the impact term.

The NAV for day t depends only on returns strictly before t, so the feedback loop
(bigger fund -> higher cost -> lower return -> slower growth) resolves forward in time
with no iteration and no look-ahead.

    python scripts/qa_report_capacity_compounding.py \
        --window 2017-01-01 2025-12-31 --out data/results/report_capacity_compound.json
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval import costs as costs_mod
from quantaalpha.eval.data import load_aligned_signal
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

TRADING_DAYS = 243
NAVS = [1e8, 5e8, 1e9]
LIBS = [
    ("main_full", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("main_kept", "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json"),
    ("original",  "data/factorlib/all_factors_library_original_20260831_012324.json"),
]


def ann(r):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan,) * 3
    cagr = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    curve = (1.0 + r).cumprod()
    return cagr, (float(cagr / vol) if vol > 0 else np.nan), \
        float((curve / curve.cummax() - 1.0).min())


def exprs_of(path: Path):
    d = json.loads(path.read_text())
    f = d.get("factors", d)
    items = f.values() if isinstance(f, dict) else f
    return [e for e in (it.get("factor_expression") or it.get("expression")
                        for it in items) if e]


class CompoundingNav:
    """Wraps ``costs.cost`` so each day is charged at the fund's current size.

    The production pricer calls ``cost(w_t, w_drift_t, sigma_t, adv_t, theta)`` once
    per date, in date order. This wrapper rescales ``theta.costs.nav`` to the fund's
    running value before delegating, then advances that value by the day's realised
    net return (gross P&L less benchmark less the cost just charged).
    """

    def __init__(self, theta, nav0, y_tilde, bench):
        self._orig = costs_mod.cost
        self.theta, self.nav0 = theta, float(nav0)
        self.y_tilde, self.bench = y_tilde, bench
        self.nav = float(nav0)
        self._prev_date = None
        self.passes = []          # end-NAV of each pricing pass, for inspection

    def __call__(self, w, w_drift, sigma, adv, theta, offlist=None):
        date = w.name if getattr(w, "name", None) is not None else None
        # ``_strategy_batch`` prices more than one book (the candidate book, then the
        # zoo-only baseline), each a fresh pass over the same dates in order. A new
        # pass is detected by the date going backwards; the fund restarts at nav0 so
        # each book is charged against its own growth path rather than inheriting the
        # previous book's.
        if date is not None and self._prev_date is not None and date <= self._prev_date:
            self.passes.append(self.nav)
            self.nav = float(self.nav0)
        self._prev_date = date

        th = replace(theta, costs=replace(theta.costs, nav=self.nav))
        c = self._orig(w, w_drift, sigma, adv, th, offlist)

        # Advance the fund by this day's ABSOLUTE return after costs -- what the
        # capital actually did. The benchmark is NOT subtracted here: an investor's
        # money compounds at the portfolio's own return, not at its return relative
        # to an index, and it is the absolute size of the fund that determines how
        # much impact its orders incur.
        if date is not None and date in self.y_tilde.index:
            yr = self.y_tilde.loc[date].reindex(w.index).fillna(0.0)
            gross = float((w * yr).sum())
            step = 1.0 + gross - c
            if np.isfinite(step) and step > 0.0:
                self.nav *= step
        return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"),
                    default=["2017-01-01", "2025-12-31"])
    ap.add_argument("--out", default="data/results/report_capacity_compound.json")
    a = ap.parse_args()

    base = load_protocol(a.protocol)
    base = replace(base, benchmark_basis="estimated_total",
                   benchmark_construction="equal")
    base = replace(base, splits=replace(base.splits,
                   final_test=(a.window[0], a.window[1])))
    print(f"window {a.window} | basis {base.benchmark_basis}", flush=True)

    out = {"window": list(a.window), "navs": NAVS,
           "note": "fixed = fund size constant (the production convention); "
                   "compounding = fund grows with its own net return, so impact is "
                   "charged against the fund's actual size that day",
           "libraries": {}}

    from quantaalpha.eval.execution import realized_return, fill_prices

    for name, rel in LIBS:
        out["libraries"][name] = {"library": rel, "by_nav": {}}
        for nav0 in NAVS:
            for mode in ("fixed", "compounding"):
                t0 = time.time()
                theta = replace(base, costs=replace(base.costs, nav=float(nav0)))
                op = EvaluationOperator(theta)
                p0, p1, win = op._windows(True)
                panel = op._panel(p0, p1)
                cands = {}
                for e in exprs_of(ROOT / rel):
                    try:
                        cands[e] = load_aligned_signal(e, panel)
                    except Exception:
                        pass

                hook = None
                if mode == "compounding":
                    y_tilde = realized_return(fill_prices(panel, theta))
                    bench = op._benchmark(str(win[0]), str(win[1]))
                    hook = CompoundingNav(theta, nav0, y_tilde, bench)
                    costs_mod.cost = hook
                try:
                    book = op._strategy_batch(cands, {}, panel, win, report=True)
                finally:
                    if hook is not None:
                        costs_mod.cost = hook._orig

                m = book["metrics"]
                exc = pd.Series(m["_net_return_series"]).astype(float).dropna()
                exc.index = pd.to_datetime(exc.index)
                bs = op._benchmark(str(win[0]), str(win[1]))
                bs = pd.Series(bs); bs.index = pd.to_datetime(bs.index)
                net = exc.add(bs.reindex(exc.index).fillna(0.0), fill_value=0.0)
                e_cagr, e_ir, e_mdd = ann(exc)
                n_cagr, _, _ = ann(net)
                rec = {"n_factors": len(cands),
                       "excess_cagr": e_cagr, "net_cagr": n_cagr,
                       "excess_ir": e_ir, "excess_mdd": e_mdd,
                       "cost_bps": float(m.get("cost_bps", np.nan)),
                       "turnover": float(m.get("turnover_book", np.nan)),
                       "secs": round(time.time() - t0, 1)}
                if hook is not None:
                    # the first pass is the candidate book itself; later passes are
                    # the zoo-only baseline priced for the marginal-contribution term
                    first = hook.passes[0] if hook.passes else hook.nav
                    rec["nav_end"] = float(first)
                    rec["nav_growth_x"] = float(first) / float(nav0)
                    rec["n_pricing_passes"] = len(hook.passes) + 1
                out["libraries"][name]["by_nav"].setdefault(f"{nav0:.0f}", {})[mode] = rec
                extra = (f" nav x{rec['nav_growth_x']:.1f}"
                         if "nav_growth_x" in rec else "")
                print(f"[{name} @ {nav0/1e8:.0f}e8 {mode:<11}] "
                      f"excess {100*e_cagr:+7.2f}% net {100*n_cagr:+7.2f}% "
                      f"cost {rec['cost_bps']:.2f}bps{extra} ({rec['secs']:.0f}s)",
                      flush=True)
                Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())