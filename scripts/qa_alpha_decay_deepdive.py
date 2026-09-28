#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Deep dive: is the 2023-2025 shortfall alpha decay, a size/regime shift, or the
long-only constraint? And would shorting -- or re-mining -- fix it?

The claim under test
--------------------
    "A pronounced performance collapse is observed for the baselines in 2023,
     coinciding with a major regime shift in the A-share market... In 2023, the
     market rotates toward small-cap and thematic stocks... Baseline methods,
     whose factor libraries assume smooth trends and regular reversal patterns,
     fail to transfer to this out-of-distribution environment."

Four things have to be separated before that can be believed or rejected:

  1. PREDICTION QUALITY  -- did IC actually collapse in 2023, and stay collapsed?
  2. WHAT IS HELD        -- is the book buying large caps while small caps win?
  3. CONSTRAINT LOSS     -- how much alpha do long-only + the 3% cap destroy?
  4. HEADROOM            -- what would a long/short book have earned instead?

(3) is the decisive one and is measured by the CLARKE-DE SILVA-THORLEY TRANSFER
COEFFICIENT, ``mean_t corr(alpha_t, w_t)`` -- the fraction of the alpha vector the
constraints let through. ``IR = TC * IC * sqrt(N)``, so a LOW TC means the position
cap and the long-only floor are the binding constraint, NOT factor quality. The
codebase already implements it (``operator._transfer_coefficient``) but gates it to
the ICIR path, so the LightGBM+topk books never report it. This script computes it
for them.

(4) is measured as the decile spread of the SAME prediction: top-decile minus
bottom-decile equal-weighted realised return, dollar-neutral, before costs. If the
long-only book is negative while the decile spread is positive, the short leg is
where the money is and ``signed: false`` is what forbids it.

Everything runs on the protocol's own construction (LightGBM + topk_dropout) under
the FULL kappa cost model, on the CAP-WEIGHTED DIVIDEND-REINVESTED benchmark
(SH000300TR), for both train/test regimes:

  * ``--window test``  : the protocol's own final_test (train 2016-2020)
  * ``--window long``  : final_test forced to 2016-01-01..2026-01-09 (frozen pre-2016
                         train) -- a different LightGBM fit, hence different results

Usage
-----
    /opt/anaconda3/envs/quantaalpha/bin/python -u scripts/qa_alpha_decay_deepdive.py
    ... --window test --window long
    ... --libs main_full original
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
from quantaalpha.eval.execution import realized_return, fill_prices    # noqa: E402
import quantaalpha.eval.costs as costs_mod                      # noqa: E402

TD = 252

LIBS = {
    "main_full": "data/factorlib/all_factors_library_meanvar_20260828_194432.json",
    "original":  "data/factorlib/all_factors_library_original_20260831_012324.json",
    "main_zoo":  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json",
}


def annualize(r: pd.Series):
    s = pd.Series(r).astype(float).dropna()
    if len(s) < 2:
        return float("nan"), float("nan"), float("nan")
    growth = float((1.0 + s).prod())
    arr = float(np.sign(growth) * abs(growth) ** (TD / len(s)) - 1.0) if growth > 0 else -1.0
    ir = float(s.mean() / s.std() * np.sqrt(TD)) if s.std() > 0 else float("nan")
    curve = (1.0 + s).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return arr, ir, mdd


def exprs_of(path: Path) -> list[str]:
    lib = json.loads(path.read_text())
    facs = lib["factors"] if isinstance(lib, dict) and "factors" in lib else lib
    if isinstance(facs, dict):
        facs = list(facs.values())
    return [f["factor_expression"] for f in facs if f.get("factor_expression")]


def transfer_coefficient(alpha: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Per-date corr(alpha, w). Mean of this is the Clarke-de Silva-Thorley TC.

    Low TC => the long-only floor and the max_weight cap are throwing the alpha
    away, i.e. the constraint binds rather than the factors being bad.
    """
    out = {}
    for d in w.index:
        if d not in alpha.index:
            continue
        av, wv = alpha.loc[d].to_numpy(dtype=float), w.loc[d].to_numpy(dtype=float)
        m = np.isfinite(av) & np.isfinite(wv)
        if m.sum() < 10 or np.nanstd(av[m]) == 0 or np.nanstd(wv[m]) == 0:
            continue
        out[d] = float(np.corrcoef(av[m], wv[m])[0, 1])
    return pd.Series(out).sort_index()


def decile_spread(pred: pd.DataFrame, y: pd.DataFrame, q: int = 10) -> pd.Series:
    """Daily EW top-decile minus bottom-decile realised return of the SAME signal.

    This is the dollar-neutral long/short leg the book is forbidden to trade
    (``signed: false``), BEFORE costs and borrow. It is the headroom shorting would
    unlock, not a tradable result.
    """
    out = {}
    for d in pred.index:
        if d not in y.index:
            continue
        p, r = pred.loc[d], y.loc[d]
        m = p.notna() & r.notna()
        if m.sum() < q * 3:
            continue
        pm, rm = p[m], r[m]
        rank = pm.rank(pct=True)
        top, bot = rm[rank > 1 - 1.0 / q], rm[rank <= 1.0 / q]
        if len(top) and len(bot):
            out[d] = float(top.mean() - bot.mean())
    return pd.Series(out).sort_index()


def holdings_profile(w: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """Per-date: how many names held, and WHERE in the size distribution.

    ``size`` is log(circ_mv) on the panel grid (neutralize.size_frame). The book's
    size percentile is the weight-average of each held name's cross-sectional size
    RANK that day; 0.5 means it holds the median-size name on average, >0.5 a
    large-cap tilt, <0.5 small-cap.
    """
    rows = {}
    for d in w.index:
        wd = w.loc[d]
        held = wd[wd.abs() > 1e-9]
        if held.empty:
            continue
        rec = {"n_held": int(len(held)), "max_w": float(held.max()),
               "gross": float(held.abs().sum()),
               "n_at_cap": int((held >= held.max() - 1e-9).sum())}
        if d in size.index:
            sd = size.loc[d].dropna()
            if len(sd) > 20:
                rank = sd.rank(pct=True)
                common = held.index.intersection(rank.index)
                if len(common):
                    ww = held[common].abs()
                    rec["size_pctile"] = float((rank[common] * ww).sum() / ww.sum())
                    rec["univ_size_pctile"] = 0.5
        rows[d] = rec
    return pd.DataFrame(rows).T.sort_index()


# The 2005-start cache is canonical: protocol_csi300_lightgbm_topk_decay trains from
# 2005, and generate.py warns that a cache starting later "silently shortens every
# lookback that depends on it". Signals from it live in a SEPARATE cache dir
# (FACTOR_CACHE_DIR=data/results/factor_cache_2005) because factor_cache_path is
# md5(expression) alone -- it does not record which h5 produced the signal, so
# mixing 2005- and 2008-derived pickles in one directory would corrupt silently.
CSI_H5 = "data/git_ignore_folder/factor_implementation_source_data_2005/daily_pv_all.h5"
_H5 = {}


def _load_h5(path: str):
    """The 8-col factor cache, loaded once per process."""
    if path not in _H5:
        t0 = time.time()
        _H5[path] = pd.read_hdf(ROOT / path)
        print(f"  loaded factor cache {path} "
              f"({len(_H5[path]):,} rows, {time.time()-t0:.0f}s)", flush=True)
    return _H5[path]


def compute_signal(expression: str, df: pd.DataFrame) -> pd.Series:
    """Recompute one expression on the factor cache, via the SAME path the miner uses.

    Mirrors scripts/qa_transfer_us.py:compute_signal -- parse_symbol ->
    parse_expression (stdout suppressed) -> swap bare fields for df['$field'] ->
    eval with every function_lib operator in scope.
    """
    import io
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
    import quantaalpha.factors.coder.function_lib as func_lib

    expr = parse_symbol(expression, df.columns)
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        expr = parse_expression(expr)
    finally:
        sys.stdout = old
    for col in df.columns:
        if isinstance(col, str) and col.startswith("$"):
            expr = expr.replace(col[1:], f"df['{col}']")
    g = {"df": df, "np": np, "pd": pd}
    for name in dir(func_lib):
        if not name.startswith("_"):
            obj = getattr(func_lib, name)
            if callable(obj):
                g[name] = obj
    res = eval(expr, g)
    if isinstance(res, pd.DataFrame):
        res = res.iloc[:, 0]
    if not isinstance(res, pd.Series):
        res = pd.Series(res, index=df.index)
    res = res.astype(np.float64)
    if not res.index.equals(df.index):
        if res.index.duplicated().any():
            res = res[~res.index.duplicated(keep="last")]
        res = res.reindex(df.index)
    return res


def get_signal(expression: str, panel, h5_path: str):
    """Aligned signal: cache fast-path, else COMPUTE and populate the cache.

    The CSI300 md5 signal cache was cleared at some point (11 files against 150
    needed, no aligned/ subdir), so load_aligned_signal's fallback -- which only
    READS load_factor_signal -- raises FileNotFoundError for nearly every factor.
    An earlier version of this script swallowed that and priced an EMPTY book,
    producing byte-identical "results" for different libraries. Compute instead,
    and write the raw signal back to the md5 path so every other report script
    (qa_report_fullcost_*, etc.) is repaired too.
    """
    from quantaalpha.eval.data import align_signal, factor_cache_path
    try:
        return load_aligned_signal(expression, panel)
    except Exception:
        pass
    sig = compute_signal(expression, _load_h5(h5_path))
    try:
        p = factor_cache_path(expression)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".pkl.tmp")
        sig.to_pickle(tmp)
        tmp.replace(p)
    except Exception as exc:
        print(f"    (could not cache signal: {type(exc).__name__}: {exc})", flush=True)
    return align_signal(sig, panel)


def run_one(theta, op, panel, label_wide, size, lib_key: str, path: Path,
            eval_window, h5_path: str, min_frac: float = 0.90) -> dict:
    t0 = time.time()
    exprs = exprs_of(path)
    cands, fails = {}, []
    for e in exprs:
        try:
            cands[e] = get_signal(e, panel, h5_path)
        except Exception as exc:
            fails.append(f"{type(exc).__name__}: {str(exc)[:90]}")
    print(f"  [{lib_key}] {len(cands)}/{len(exprs)} signals loaded "
          f"({time.time()-t0:.0f}s)", flush=True)
    if fails:
        print(f"    {len(fails)} FAILED; first 3:", flush=True)
        for f in fails[:3]:
            print(f"      {f}", flush=True)
    # HARD GATE: an empty or thin candidate set silently prices the NULL book and
    # every library then reports identical numbers. Refuse rather than mislead.
    if len(cands) < min_frac * len(exprs):
        raise RuntimeError(
            f"{lib_key}: only {len(cands)}/{len(exprs)} signals available "
            f"(<{min_frac:.0%}). Refusing to run -- an empty/thin book would be "
            f"priced as the null model and reported as a library result.")

    book = op._strategy_batch(cands, {}, panel, eval_window, report=True)
    pred = book["prediction"]
    wide = _to_wide(pred).reindex(index=panel.dates, columns=panel.instruments)
    wide = wide.where(panel.universe)

    nr = pd.Series(book["metrics"]["_net_return_series"]).astype(float)
    nr.index = pd.to_datetime(nr.index)

    # Rebuild the SAME weights _book used, so TC and holdings describe the book
    # that produced `nr` rather than a lookalike.
    start, end = eval_window
    window_pred = pred.loc[str(start):str(end)]
    y_tilde = realized_return(fill_prices(panel, theta))
    universe = panel.universe.loc[window_pred.index]
    sigma = costs_mod.trailing_vol(panel.close, theta.costs.vol_window)
    adv = costs_mod.trailing_adv(panel, theta)
    mask = op._trade_mask(panel)
    # Match operator._book exactly: when cost_aware_dropout is on it scales the
    # prediction into expected-return units with an OLS beta fitted on the scale
    # window (NOT the eval window -- that would leak). Passing 1.0 instead would
    # rebuild a DIFFERENT book than the one whose returns we just measured, making
    # the transfer coefficient describe a portfolio that was never traded.
    beta = 1.0
    if theta.portfolio.cost_aware_dropout or theta.portfolio.construction == "mean_variance":
        from quantaalpha.eval.execution import prediction_scale
        sw = theta.splits.window(
            getattr(theta.portfolio, "scale_split", None) or theta.combiner.fit_split)
        beta = prediction_scale(pred, y_tilde, sw)
    w, _wd = build_book(window_pred, theta, y_tilde=y_tilde, universe=universe,
                        mask=mask, sigma=sigma, pred_scale=beta,
                        close=panel.close, adv=adv)

    tc = transfer_coefficient(window_pred, w)
    ls = decile_spread(wide.loc[str(start):str(end)], y_tilde)
    prof = holdings_profile(w, size)

    years = sorted({d.year for d in nr.index})
    rows = {}
    for yr in years + ["full"]:
        if yr == "full":
            r, t_, l_, p_ = nr, tc, ls, prof
            blk = _ic_block(wide, label_wide, (f"{years[0]}-01-01", f"{years[-1]}-12-31"))
        else:
            r = nr[nr.index.year == yr]
            t_ = tc[tc.index.year == yr] if len(tc) else tc
            l_ = ls[ls.index.year == yr] if len(ls) else ls
            p_ = prof[prof.index.year == yr] if len(prof) else prof
            blk = _ic_block(wide, label_wide, (f"{yr}-01-01", f"{yr}-12-31"))
        arr, ir, mdd = annualize(r)
        ls_arr, ls_ir, _ = annualize(l_) if len(l_) else (np.nan, np.nan, np.nan)
        rows[str(yr)] = {
            "ic": blk.get("ic"), "rank_ic": blk.get("rank_ic"),
            "arr": arr, "ir": ir, "mdd": mdd, "days": int(len(r)),
            "tc": float(t_.mean()) if len(t_) else float("nan"),
            "ls_spread_arr": ls_arr, "ls_spread_ir": ls_ir,
            "n_held": float(p_["n_held"].mean()) if len(p_) else float("nan"),
            "size_pctile": float(p_["size_pctile"].mean())
                           if len(p_) and "size_pctile" in p_ else float("nan"),
            "n_at_cap": float(p_["n_at_cap"].mean()) if len(p_) else float("nan"),
        }
    return {"library": str(path), "n_factors": len(cands), "years": rows,
            "secs": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml",
                    help="LightGBM + topk_dropout + full kappa costs")
    ap.add_argument("--window", action="append", choices=["test", "long"],
                    help="repeatable; default both")
    ap.add_argument("--libs", nargs="+", default=["main_full", "original"])
    ap.add_argument("--signed", action="store_true",
                    help="flip portfolio.signed=True (topk_dropout implements the "
                         "short leg) -- the direct long-only test")
    ap.add_argument("--h5", default=CSI_H5,
                    help="CSI300 8-col factor cache, used when the md5 signal "
                         "cache cannot serve an expression")
    ap.add_argument("--out", default="data/results/alpha_decay_deepdive.json")
    a = ap.parse_args()
    windows = a.window or ["test", "long"]

    base = load_protocol(a.protocol)
    base = replace(base, benchmark="SH000300TR",
                   benchmark_construction="index", benchmark_basis="price")
    if a.signed:
        base = replace(base, portfolio=replace(base.portfolio, signed=True))

    out = {"protocol": a.protocol, "benchmark": base.benchmark,
           "signed": bool(a.signed), "windows": {}}

    for wname in windows:
        theta = base
        if wname == "long":
            theta = replace(theta, splits=replace(theta.splits,
                            final_test=("2016-01-01", "2026-01-09")))
        op = EvaluationOperator(theta)
        p_start, p_end, eval_window = op._windows(True)
        panel = op._panel(p_start, p_end)
        label_wide = label_frame(panel, theta)
        from quantaalpha.eval.neutralize import size_frame
        try:
            size = size_frame(panel, getattr(theta, "market", "csi300"))
        except Exception as exc:
            print(f"  size_frame unavailable ({type(exc).__name__}); "
                  f"size profile disabled", flush=True)
            size = pd.DataFrame(index=panel.dates, columns=panel.instruments, dtype=float)

        print(f"\n=== window={wname}  eval {eval_window}  hash {theta.hash} "
              f"| {theta.combiner.model}+{theta.portfolio.construction} "
              f"| signed={theta.portfolio.signed} ===", flush=True)

        wout = {"eval_window": list(eval_window), "libraries": {}}
        for k in a.libs:
            wout["libraries"][k] = run_one(theta, op, panel, label_wide, size,
                                           k, ROOT / LIBS[k], eval_window, a.h5)
        out["windows"][wname] = wout

        # readable table
        for k, v in wout["libraries"].items():
            print(f"\n  --- {k} (n={v['n_factors']}) window={wname} ---")
            print(f"    {'year':6}{'IC':>9}{'RankIC':>9}{'ARR%':>9}{'IR':>7}"
                  f"{'TC':>7}{'L/S ARR%':>10}{'held':>7}{'size%ile':>9}")
            for yr, r in v["years"].items():
                def f(x, n=4, mul=1):
                    return "     n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
                           else f"{x*mul:.{n}f}"
                print(f"    {yr:6}{f(r['ic']):>9}{f(r['rank_ic']):>9}"
                      f"{f(r['arr'],2,100):>9}{f(r['ir'],2):>7}{f(r['tc'],2):>7}"
                      f"{f(r['ls_spread_arr'],2,100):>10}{f(r['n_held'],0):>7}"
                      f"{f(r['size_pctile'],3):>9}")

    Path(a.out).write_text(json.dumps(out, indent=2, default=lambda o: None))
    print(f"\n-> {a.out}")
    print("\nTC = corr(alpha, weights): the fraction of the alpha the long-only floor")
    print("and the 3% cap let through. IR = TC*IC*sqrt(N). L/S ARR% is the decile")
    print("spread of the SAME signal, dollar-neutral, BEFORE costs and borrow --")
    print("the headroom 'signed: false' forbids, not a tradable number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
