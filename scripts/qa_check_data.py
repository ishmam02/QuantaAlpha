#!/usr/bin/env python
"""Pre-flight check: is the market data and the factor cache usable?

Answers the two questions that silently ruin a long mine:

  1. Which price/volume fields does the Qlib data actually serve? A field the data
     lacks cannot be added to ``generate.py``.
  2. Which fields does the factor cache expose to the generator? The cache -- not the
     raw Qlib data -- is what mined formulas are computed against, so a field missing
     here means every formula referencing it is silently dropped.

Run before any long mine:

    python scripts/qa_check_data.py
"""
from __future__ import annotations
import collections
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHES = [
    ROOT / "data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
    ROOT / "data/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5",
]
# The field list is read out of generate.py rather than restated here. A hand-kept copy
# drifts, and a stale copy makes this check call a healthy cache broken -- which is
# exactly what a pre-flight check must never do. Parsed from the source with ast so that
# reading it does not import generate.py, which calls qlib.init at module level.
GENERATE = ROOT / "quantaalpha/factors/data_template/generate.py"


def generator_fields() -> list[str]:
    import ast

    tree = ast.parse(GENERATE.read_text())
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FIELDS" for t in targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit(f"no FIELDS assignment found in {GENERATE}")


FIELDS = generator_fields()
# $return is derived from $close inside generate.py, so the cache carries a column that
# the raw Qlib data does not serve. The two checks below therefore expect different sets.
EXPECTED_CACHE = FIELDS + ["$return"]

ok = True


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------- Qlib raw data
section("1. Qlib data")

provider = os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI")
if not provider:
    provider = str(ROOT / "data/qlib/cn_data")
    print(f"  QLIB_DATA_DIR unset; trying {provider}")
provider = os.path.expanduser(provider)

feat = Path(provider) / "features"
if not feat.is_dir():
    print(f"  MISSING: {feat}")
    print("  -> the data is not where .env points. Use an ABSOLUTE path (see RUNNING.md 3.1).")
    ok = False
else:
    counts: collections.Counter = collections.Counter()
    for p in glob.glob(str(feat / "*" / "*.day.bin")):
        counts[os.path.basename(p).replace(".day.bin", "")] += 1
    n_inst = len(list(feat.iterdir()))
    print(f"  {n_inst} instruments at {provider}")
    for name, n in sorted(counts.items()):
        print(f"    ${name:<10} {n}")
    served = {f"${n}" for n in counts}
    missing = [f for f in FIELDS if f not in served]
    if missing:
        print(f"  NOTE: generate.py asks for {' '.join(missing)}, which this data does not serve.")
        print("  -> remove them from generate.py, or the cache rebuild will fail.")
        ok = False

    try:
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=provider, region="cn")
        cal = D.calendar(start_time="2000-01-01", end_time="2030-01-01")
        print(f"  calendar: {len(cal)} days, {cal[0].date()} -> {cal[-1].date()}")
        if len(cal) < 3000:
            print("  WARNING: fewer than 3000 trading days -- the download looks incomplete.")
            ok = False
    except Exception as exc:                       # noqa: BLE001
        print(f"  could not read the calendar ({type(exc).__name__}: {exc})")
        ok = False

# ---------------------------------------------------------------- factor cache
section("2. Factor cache (what the generator actually sees)")

try:
    import pandas as pd
except ImportError:
    print("  pandas unavailable; skipping")
    pd = None

if pd is not None:
    found_any = False
    for cache in CACHES:
        if not cache.exists():
            print(f"  absent: {cache.relative_to(ROOT)}")
            continue
        found_any = True
        d = pd.read_hdf(cache)
        cols = list(d.columns)
        dates = d.index.get_level_values(0)
        print(f"  {cache.relative_to(ROOT)}")
        print(f"    {len(cols)} columns: {' '.join(cols)}")
        print(f"    {len(d):,} rows, {dates.min().date()} -> {dates.max().date()}")
        missing = [f for f in EXPECTED_CACHE if f not in cols]
        if missing:
            print(f"    MISSING {' '.join(missing)} -- formulas using them cannot be computed.")
            print("    -> delete the cache and re-run a mine to rebuild it (RUNNING.md 4.1.1).")
            ok = False
    if not found_any:
        print("  no cache yet -- the first mine builds one. That is expected on a fresh setup.")

# ---------------------------------------------------------------- verdict
section("Verdict")
print("  READY" if ok else "  NOT READY -- fix the items marked above before a long mine.")
sys.exit(0 if ok else 1)
