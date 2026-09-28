#!/usr/bin/env python
"""US transfer FULL-COST $1B COMPOUNDING backtest -- per-year + total metrics.

The CSI300-mined zoo recomputed on S&P 500, LightGBM 5-seed + topk (cost-aware) +
US-realistic costs, COMPOUNDING at $1B (fund grows at its own return, costs against
the actual daily size), over the full 2016-2026 test window. Reports per calendar
year: IC, Rank IC, ICIR, RankICIR, ARR, IR, MDD; plus the full-window total and the
book metrics (net_ir, net_arr, cost_bps, turnover, mdd, effective_rank, nav_growth).

Mirrors scripts/qa_report_fullcost_long.py (per-year _ic_block + annualize) but on US
data with recomputed (cache-loaded) factors + the CompoundingNav wrapper from
scripts/qa_report_capacity_compounding.py.

Usage::

    QA_THREADS=8 conda run -n quantaalpha python scripts/qa_transfer_us_fullcost_yearly.py \\
        --library data/factorlib/all_factors_library_meanvar_20260828_194432.json \\
        --protocol quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml \\
        --cache data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \\
        --qlib-dir data/qlib/us_data --window 2016-01-01 2026-01-09 --nav 1000000000 \\
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
from quantaalpha.eval.metrics import _ic_block, _to_wide, label_frame
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

from scripts.qa_transfer_us import load_library, recompute_factor

TD = 252


class CompoundingNav:
    def __init__(self, theta, nav0, y_tilde, bench):
        self._orig = costs_mod.cost
        self.theta, self.nav0 = theta, float(nav0)
        self.y_tilde, self.bench = y_tilde, bench
        self.nav = float(nav0); self._prev_date = None; self.passes = []

    def __call__(self, w, w_drift, sigma, adv, theta, offlist=None):
        date = w.name if getattr(w, "name", None) is not None else None
        if date is not None and self._prev_date is not None and date <= self._prev_date:
            self.passes.append(self.nav); self.nav = float(self.nav0)
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


def annualize(r: pd.Series):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    arr = float((1.0 + r).prod() ** (TD / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TD))
    ir = float(arr / vol) if vol > 0 else np.nan
    # MDD = max % drawdown of the COMPOUNDED NAV (peak-to-trough). It is a MAX over
    # the window, not additive across years; the cumulative-sum drawdown is
    # unnormalized and would not be a %.
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    mdd = float(((peak - nav) / peak).max())
    return arr, vol, ir, mdd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="data/factorlib/all_factors_library_meanvar_20260828_194432.json")
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_sp500_lightgbm_topk.yaml")
    ap.add_argument("--cache", default="data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    ap.add_argument("--qlib-dir", default="data/qlib/us_data")
    ap.add_argument("--window", nargs=2, default=["2016-01-01", "2026-01-09"])
    ap.add_argument("--nav", type=float, default=1e9)
    ap.add_argument("--cache-signals", default=None)
    args = ap.parse_args()

    os.environ["QLIB_PROVIDER_URI"] = str(Path(args.qlib_dir).resolve())
    os.environ["QLIB_REGION"] = "us"

    base = load_protocol(args.protocol)
    base = replace(base, splits=replace(base.splits, final_test=(args.window[0], args.window[1])))
    base = replace(base, benchmark_basis="estimated_total", benchmark_construction="index")
    th = replace(base, costs=replace(base.costs, nav=float(args.nav)))
    print(f"US full-cost yearly | protocol {th.hash} | nav ${args.nav/1e6:.0f}M | window {args.window} | "
          f"combiner={th.combiner.model} x{len(th.combiner.seeds)} seeds construction={th.portfolio.construction} "
          f"compounding=True", flush=True)

    op = EvaluationOperator(th)
    p0, p1, win = op._windows(True)
    panel = op._panel(p0, p1)
    print(f"panel {p0}..{p1}, {len(panel.instruments)} instruments", flush=True)

    cs = args.cache_signals
    cands: dict[str, object] = {}
    if cs and Path(cs).exists():
        with open(cs, "rb") as fh:
            cached, ok, fail = pickle.load(fh)
        for e, s in cached.items():
            cands[e] = data_mod._align(s, panel) if isinstance(s, pd.DataFrame) else align_signal(s, panel)
        print(f"loaded {len(cands)} cached signals, re-aligned (ok={ok} fail={fail})", flush=True)
    else:
        df = pd.read_hdf(args.cache, key="data")
        for name, expr in load_library(args.library):
            try:
                cands[expr] = align_signal(recompute_factor(expr, df, factor_name=name), panel)
            except Exception as exc:
                print(f"recompute FAILED {name}: {exc}", flush=True)
    if not cands:
        print("no factors"); return 2

    y_tilde = realized_return(fill_prices(panel, th))
    bench = op._benchmark(str(win[0]), str(win[1]))
    hook = CompoundingNav(th, args.nav, y_tilde, bench)
    costs_mod.cost = hook
    t0 = time.time()
    try:
        book = op._strategy_batch(cands, {}, panel, win, report=True)
    finally:
        costs_mod.cost = hook._orig
    print(f"book priced in {time.time()-t0:.0f}s | nav_end ${hook.passes[0]/1e6:.0f}M "
          f"(x{hook.passes[0]/args.nav:.2f})", flush=True)

    pred = book["prediction"]
    wide = _to_wide(pred).reindex(index=panel.dates, columns=panel.instruments).where(panel.universe)
    label = label_frame(panel, th)
    nr = pd.Series(book["metrics"]["_net_return_series"]).astype(float)
    nr.index = pd.to_datetime(nr.index)
    bs = pd.Series(op._benchmark(str(win[0]), str(win[1]))); bs.index = pd.to_datetime(bs.index)
    net_abs = nr.add(bs.reindex(nr.index).fillna(0.0), fill_value=0.0)  # absolute (fund) return

    rows = {}
    for yr, r in nr.groupby(nr.index.year):
        blk = _ic_block(wide, label, (f"{yr}-01-01", f"{yr}-12-31"))
        arr, vol, ir, _ = annualize(r)  # excess (net of benchmark) ARR + IR
        nabs = net_abs[net_abs.index.year == yr]
        arr_abs, _, _, mdd = annualize(nabs)  # absolute fund ARR + MDD (investor drawdown)
        rows[int(yr)] = {"ic": float(blk.get("ic", np.nan)),
                         "rank_ic": float(blk.get("rank_ic", np.nan)),
                         "icir": float(blk.get("icir", np.nan)),
                         "rank_icir": float(blk.get("rank_icir", np.nan)),
                         "excess_arr": arr, "abs_arr": arr_abs, "ir": ir, "mdd": mdd,
                         "days": int(len(r))}

    blk = _ic_block(wide, label, win)
    arr, vol, ir, _ = annualize(nr)
    arr_abs, _, _, mdd = annualize(net_abs)  # absolute fund MDD (investor drawdown)
    rows["full"] = {"ic": float(blk.get("ic", np.nan)),
                    "rank_ic": float(blk.get("rank_ic", np.nan)),
                    "icir": float(blk.get("icir", np.nan)),
                    "rank_icir": float(blk.get("rank_icir", np.nan)),
                    "excess_arr": arr, "abs_arr": arr_abs, "ir": ir, "mdd": mdd,
                    "days": int(len(nr))}
    m = book["metrics"]
    rows["book"] = {k: (float(m[k]) if isinstance(m.get(k), (int, float)) and m[k] == m[k] else None)
                    for k in ("net_ir", "net_arr", "cost_bps", "turnover_book", "mdd",
                              "effective_rank", "cx", "transfer_coefficient")}
    rows["book"]["nav_end"] = float(hook.passes[0]) if hook.passes else float(args.nav)
    rows["book"]["nav_growth_x"] = rows["book"]["nav_end"] / float(args.nav)

    # ---- print the yearly table ----
    print("\n" + "=" * 94)
    print(f"US TRANSFER -- FULL COST @ ${args.nav/1e6:.0f}M COMPOUNDING | CSI300 zoo -> S&P 500 | "
          f"{win[0]}..{win[1]}")
    print("=" * 94)
    print(f"{'year':>6} {'IC':>8} {'RankIC':>8} {'ICIR':>7} {'RankICIR':>9} "
          f"{'excess%':>8} {'abs%':>8} {'IR':>7} {'MDD':>7} {'days':>5}")
    for k in sorted([k for k in rows if k != "book" and k != "full"]) + ["full"]:
        r = rows[k]
        lbl = "FULL" if k == "full" else str(k)
        print(f"{lbl:>6} {r['ic']:+8.4f} {r['rank_ic']:+8.4f} {r['icir']:+7.3f} "
              f"{r['rank_icir']:+9.3f} {100*r['excess_arr']:+7.2f} {100*r['abs_arr']:+7.2f} "
              f"{r['ir']:+7.2f} {r['mdd']:7.3f} {r['days']:5d}")
    b = rows["book"]
    print("-" * 94)
    print(f"book: net_ir={b.get('net_ir')} net_arr={b.get('net_arr')} cost_bps={b.get('cost_bps')} "
          f"turnover={b.get('turnover_book')} mdd={b.get('mdd')} eff_rank={b.get('effective_rank')} "
          f"cx={b.get('cx')} TC={b.get('transfer_coefficient')} nav_x={b.get('nav_growth_x'):.2f}")
    print("=" * 94)
    Path("/tmp/us_fullcost_yearly.json").write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())