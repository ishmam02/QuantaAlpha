#!/usr/bin/env python
"""Seed a run's factor library from the Alpha158(20) subset.

The paper says the initial seed factor pool is derived from Alpha158(20). Two
separate mechanisms have to be fed for that to be true, and neither was:

1. **The regulator's novelty check.** ``factor_zoo_path`` was ``null``, so
   ``match_alphazoo`` compared every candidate against an empty frame and the
   duplication gate never fired. Writing the pool as the two-column CSV the
   regulator reads fixes that; see ``quantaalpha/factors/seed_pool.py``.
2. **The starting library.** ``FactorLibraryManager`` loads its JSON from disk
   if the file is already there, so writing a seeded library before the run
   starts is all the seeding takes -- no change to the mining loop.

This does (2), and computes each seed's signal into ``factor_cache`` so the
seeded factors are immediately scoreable by ``E_Θ`` rather than being library
entries that nothing can price. That distinction has bitten this project twice:
a library records what the search *proposed*, and an entry with no cached
signal is not a factor anyone can use.

Seeds are written with ``admitted: true``. They are the pool the search starts
from, not candidates it has judged, and marking them otherwise would make the
first admission decision rank them against themselves.

Usage::

    python scripts/qa_seed_library.py --library data/factorlib/all_factors_library_<id>.json
    python scripts/qa_seed_library.py --library ... --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PV = ROOT / "data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5"


def _entry(name: str, expr: str) -> tuple[str, dict]:
    fid = hashlib.md5(f"seed::{name}::{expr}".encode()).hexdigest()[:16]
    return fid, {
        "factor_id": fid,
        "factor_name": name,
        "factor_expression": expr,
        "factor_implementation_code": "",
        "factor_description": f"Alpha158(20) seed factor {name}.",
        "factor_formulation": expr,
        "cache_location": {},
        "metadata": {
            "source": "alpha158_20_seed_pool",
            "round_number": -1,
            "evolution_phase": "seed",
            "seeded_at": datetime.now().isoformat(),
        },
        "backtest_results": {},
        "feedback": "",
        "admitted": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="library JSON to seed")
    ap.add_argument("--cache-dir", default=None, help="factor_cache dir (env FACTOR_CACHE_DIR)")
    ap.add_argument("--pv", default=str(PV))
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    import os

    import numpy as np
    import pandas as pd

    from quantaalpha.eval.data import factor_cache_path
    from quantaalpha.factors.coder import function_lib
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
    from quantaalpha.factors.seed_pool import SEED_POOL

    cache = Path(args.cache_dir or os.environ.get("FACTOR_CACHE_DIR")
                 or (ROOT / "data/results/factor_cache"))
    lib_path = Path(args.library)

    payload = {"metadata": {"created_at": datetime.now().isoformat(),
                            "last_updated": datetime.now().isoformat(),
                            "total_factors": 0, "version": "1.0"},
               "factors": {}}
    if lib_path.exists():
        payload = json.loads(lib_path.read_text())

    have = {f.get("factor_expression") for f in payload.get("factors", {}).values()}
    todo = {n: e for n, e in SEED_POOL.items() if e not in have}
    cached = sum(1 for e in SEED_POOL.values() if factor_cache_path(e, cache).exists())
    print(f"library : {lib_path}")
    print(f"  holds {len(payload.get('factors', {}))} factor(s); "
          f"{len(SEED_POOL) - len(todo)}/{len(SEED_POOL)} seeds already present")
    print(f"cache   : {cache}")
    print(f"  {cached}/{len(SEED_POOL)} seed signals cached")
    if args.check:
        return 0 if not todo and cached == len(SEED_POOL) else 1

    df = pd.read_hdf(args.pv, key="data")
    ns = {n: getattr(function_lib, n) for n in dir(function_lib) if not n.startswith("_")}
    ns.update(np=np, pd=pd, df=df)

    cache.mkdir(parents=True, exist_ok=True)
    computed = 0
    for name, expr in SEED_POOL.items():
        p = factor_cache_path(expr, cache)
        if p.exists():
            continue
        parsed = parse_expression(parse_symbol(expr, df.columns))
        for col in df.columns:
            parsed = parsed.replace(col[1:], f"df['{col}']")
        sig = eval(parsed, dict(ns)).astype("float64")  # noqa: S307 -- as factor.py does
        tmp = p.with_suffix(".pkl.tmp")
        sig.to_pickle(tmp)
        os.replace(tmp, p)
        computed += 1

    for name, expr in todo.items():
        fid, entry = _entry(name, expr)
        payload["factors"][fid] = entry
    payload["metadata"]["last_updated"] = datetime.now().isoformat()
    payload["metadata"]["total_factors"] = len(payload["factors"])
    payload["metadata"]["seeded_from"] = "alpha158_20"

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  seeded {len(todo)} factor(s), computed {computed} signal(s)")
    print(f"  library now holds {len(payload['factors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
