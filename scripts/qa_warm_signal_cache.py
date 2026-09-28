#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Recompute and cache the md5 signal pickles every eval script depends on.

Why this is needed
------------------
``quantaalpha/eval/data.py`` resolves a factor's signal in two steps:

    load_aligned_signal(expr, panel)      # aligned cache, keyed (expr, grid)
      -> load_factor_signal(expr)         # md5 pickle cache -- READ-ONLY
         -> FileNotFoundError

``load_factor_signal`` cannot compute; it only reads. So when the md5 cache is
missing -- as it now is for CSI300, which holds 11 pickles against the 150 a
library needs -- **every** consumer silently loses its factors:

    cands = {}                            # nothing loaded
    op._strategy_batch(cands, ...)        # prices the EMPTY book
    -> identical "results" for every library, which are really the null model

That failure is silent and produces plausible numbers, which is the worst kind.
This script repopulates the cache so ``qa_report_fullcost_yearly.py``,
``qa_report_fullcost_long.py``, ``qa_alpha_decay_deepdive.py`` and the rest work
again, and so the recompute is paid ONCE rather than per script.

Signals are computed by the same path the miner uses -- ``parse_symbol`` ->
``parse_expression`` -> eval with ``function_lib`` in scope -- mirroring
``scripts/qa_transfer_us.py:compute_signal`` and
``quantaalpha/backtest/custom_factor_calculator.py``.

The cached object is the RAW ``(datetime, instrument)`` Series over the whole
factor cache, which is what ``load_factor_signal`` expects. It is panel-
INDEPENDENT, so one warm-up serves every split and window.

Cost: roughly 40 s and ~170 MB per factor on the CSI300 cache (14.2M rows).
Check free disk before warming several libraries.

Usage
-----
    /opt/anaconda3/envs/quantaalpha/bin/python -u scripts/qa_warm_signal_cache.py \
        --libs data/factorlib/all_factors_library_meanvar_20260828_194432.json
    ... --libs A.json --libs B.json      # repeatable
    ... --dry-run                        # report what is missing, compute nothing
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSI_H5 = "data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5"


def exprs_of(path: Path) -> list[str]:
    lib = json.loads(path.read_text())
    facs = lib["factors"] if isinstance(lib, dict) and "factors" in lib else lib
    if isinstance(facs, dict):
        facs = list(facs.values())
    return [f["factor_expression"] for f in facs if f.get("factor_expression")]


def compute_signal(expression: str, df: pd.DataFrame) -> pd.Series:
    """Mirrors scripts/qa_transfer_us.py:compute_signal (the miner's own path)."""
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
    import quantaalpha.factors.coder.function_lib as func_lib

    expr = parse_symbol(expression, df.columns)
    old = sys.stdout
    sys.stdout = io.StringIO()          # parse_expression prints; suppress it
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libs", action="append", required=True,
                    help="library JSON path; repeatable")
    ap.add_argument("--h5", default=CSI_H5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=int, default=25,
                    help="abort if free disk falls below this while warming")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index (0-based) for parallel warming")
    ap.add_argument("--of", type=int, default=1,
                    help="total number of parallel workers. Each shard takes a "
                         "disjoint stride of the MISSING list, so workers never "
                         "compute the same expression. Each loads its own copy of "
                         "the factor cache (~1.5 GB), so size --of to free RAM.")
    a = ap.parse_args()

    from quantaalpha.eval.data import factor_cache_path

    wanted: list[str] = []
    for rel in a.libs:
        es = exprs_of(ROOT / rel)
        wanted.extend(es)
        print(f"{rel}: {len(es)} expressions")
    uniq = list(dict.fromkeys(wanted))
    missing = [e for e in uniq if not factor_cache_path(e).exists()]
    total_missing = len(missing)
    if a.of > 1:
        missing = missing[a.shard::a.of]      # disjoint stride -> no duplicated work
    print(f"\n{len(uniq)} unique expressions | {len(uniq)-total_missing} cached | "
          f"{total_missing} MISSING"
          + (f" | shard {a.shard}/{a.of} takes {len(missing)}" if a.of > 1 else ""))
    if a.dry_run or not missing:
        print("nothing to do" if not missing else "--dry-run: computing nothing")
        return 0

    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print(f"free disk {free_gb:.0f} GB | ~{len(missing)*0.17:.0f} GB needed "
          f"(~170 MB/factor)\n")

    df = pd.read_hdf(ROOT / a.h5)
    print(f"factor cache loaded: {len(df):,} rows\n", flush=True)

    ok = err = 0
    t_start = time.time()
    for i, e in enumerate(missing, 1):
        free_gb = shutil.disk_usage(ROOT).free / 2**30
        if free_gb < a.min_free_gb:
            print(f"ABORT: free disk {free_gb:.0f} GB < {a.min_free_gb} GB "
                  f"after {ok} signals", flush=True)
            break
        try:
            t0 = time.time()
            sig = compute_signal(e, df)
            p = factor_cache_path(e)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".pkl.tmp")
            sig.to_pickle(tmp)
            tmp.replace(p)               # atomic: a reader never sees a partial file
            ok += 1
            if i % 10 == 0 or i == len(missing):
                el = time.time() - t_start
                print(f"  [{i}/{len(missing)}] ok={ok} err={err} "
                      f"last {time.time()-t0:.0f}s | elapsed {el/60:.1f}m | "
                      f"free {free_gb:.0f}GB", flush=True)
        except Exception as exc:
            err += 1
            print(f"  [{i}/{len(missing)}] FAILED {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)

    print(f"\ndone: {ok} cached, {err} failed, "
          f"{(time.time()-t_start)/60:.1f} min")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
