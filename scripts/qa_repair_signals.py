#!/usr/bin/env python
"""Recompute a library's missing cached signals from their stored expressions.

``qa_compare_arms.py`` builds ``zoo_b`` with a dict comprehension over every
library entry, so ``load_factor_signal`` raising ``FileNotFoundError`` on one
missing signal aborts the whole comparison -- 155 present factors produce no
result because of the 156th. That is what happened to the finished
mean_variance arm: the comparison died at 16:57:22 on
``md5=40af314b9d019ee3215a607433bbd7b1`` and wrote nothing.

The signals were reclaimed as scratch while their factors stayed in the
library. Nothing irreplaceable was lost: a library entry carries
``factor_expression``, and the cache is keyed by ``md5(expression)``, so the
signal is a pure function of data already on disk. This recomputes it by the
same route ``factor.py`` takes -- ``parse_symbol`` then ``parse_expression``
against ``daily_pv.h5`` -- and writes the pickle back under the key the loader
will look for.

Each repaired signal is read back through ``load_factor_signal`` and
``align_signal`` before the temp file is promoted, because the cache has two
legitimate on-disk shapes and the failure mode of writing the wrong one is
silent until a consumer rejects it (see ``_read_cached_signal``). Verifying
through the real loader rather than the parser is the point: a previous cache
change passed a test that called ``align_signal`` directly and broke every
caller.

Usage::

    python scripts/qa_repair_signals.py data/factorlib/<library>.json
    python scripts/qa_repair_signals.py <library>.json --check   # report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PV = ROOT / "data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5"


def _entries(lib: dict) -> dict:
    factors = lib.get("factors", lib)
    if isinstance(factors, list):
        return {f.get("factor_id") or str(i): f for i, f in enumerate(factors)}
    return factors


def _eval_namespace() -> dict:
    """The names a generated factor.py has in scope when it evaluates.

    ``factor.py`` does a star-import of ``function_lib`` at module level, which
    a function body cannot; the namespace is assembled explicitly instead so
    the evaluated expression sees exactly the same operators (TS_MEAN, RANK,
    DELTA, ...) under the same names.
    """
    import numpy as np
    import pandas as pd

    from quantaalpha.factors.coder import function_lib

    ns = {n: getattr(function_lib, n) for n in dir(function_lib) if not n.startswith("_")}
    ns.update(np=np, pd=pd)
    return ns


def _compute(expr: str, name: str, df):
    """Evaluate one expression exactly as the generated factor.py does."""
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol

    parsed = parse_expression(parse_symbol(expr, df.columns))
    for col in df.columns:
        parsed = parsed.replace(col[1:], f"df['{col}']")
    ns = _eval_namespace()
    ns["df"] = df
    return eval(parsed, ns).astype("float64")  # noqa: S307 -- same trust as factor.py


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("library", help="path to an all_factors_library_*.json")
    ap.add_argument("--cache-dir", default=os.environ.get("FACTOR_CACHE_DIR"))
    ap.add_argument("--pv", default=str(DEFAULT_PV), help="daily_pv.h5 source")
    ap.add_argument("--protocol", default=None, help="protocol to align against")
    ap.add_argument("--check", action="store_true", help="report gaps, repair nothing")
    args = ap.parse_args()

    import pandas as pd
    from quantaalpha.eval.data import align_signal, load_factor_signal
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    cache = Path(args.cache_dir or (ROOT / "data/results/factor_cache"))
    lib = json.load(open(args.library))
    factors = _entries(lib)

    gaps = []
    for fid, it in factors.items():
        expr = (it or {}).get("factor_expression")
        name = (it or {}).get("factor_name") or fid
        if not expr:
            print(f"  !! {name}: no factor_expression -- cannot recompute")
            continue
        p = cache / f"{hashlib.md5(expr.encode()).hexdigest()}.pkl"
        if not p.exists():
            gaps.append((name, expr, p))

    print(f"library {Path(args.library).name}: {len(factors)} factor(s)")
    print(f"cache   {cache}")
    print(f"missing {len(gaps)} signal(s)")
    if not gaps:
        return 0
    for name, _, p in gaps:
        print(f"  - {name}  ({p.name})")
    if args.check:
        print("\n  --check: nothing written")
        return 1

    pv = Path(args.pv)
    if not pv.exists():
        print(f"\nsource data not found: {pv}")
        return 1
    df = pd.read_hdf(str(pv), key="data")
    if "$return" not in df.columns:
        print(f"\n{pv} has no $return column; expressions using it cannot be rebuilt")

    print()
    cache.mkdir(parents=True, exist_ok=True)
    # The same panel the comparison will align against, so a signal that
    # verifies here is one qa_compare_arms.py can actually consume.
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, _ = op._windows(True)
    pnl = op._panel(start, end)
    ok = 0
    for name, expr, p in gaps:
        try:
            sig = _compute(expr, name, df)
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            continue
        tmp = p.with_suffix(".pkl.tmp")
        sig.to_pickle(tmp)
        try:
            # Through the real loader, not the parser -- see the module docstring.
            os.replace(tmp, p)
            wide = align_signal(load_factor_signal(expr, cache_dir=cache), pnl)
            cov = float(wide.notna().to_numpy().mean())
        except Exception as exc:
            p.unlink(missing_ok=True)
            print(f"  FAIL {name}: wrote but could not load back -- {exc}")
            continue
        ok += 1
        print(f"  ok   {name}: {wide.shape[0]}x{wide.shape[1]} panel, {cov:.1%} non-NaN")

    print(f"\n  repaired {ok}/{len(gaps)}")
    return 0 if ok == len(gaps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
