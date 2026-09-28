#!/usr/bin/env python
"""Pre-flight check: is the market data and the factor cache usable?

Answers the three questions that silently ruin a long mine:

  1. Which price/volume fields does the Qlib data actually serve? A field the data
     lacks cannot be added to ``generate.py``.
  2. Which fields does the factor cache expose to the generator? The cache -- not the
     raw Qlib data -- is what mined formulas are computed against, so a field missing
     here means every formula referencing it is silently dropped.
  3. Is the cache the reference cache? With generate.py's default fields, its content
     must match the reference fingerprint, so every setup mines against the same panel.

Run before any long mine:

    python scripts/qa_check_data.py
"""
from __future__ import annotations
import collections
import glob
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Factor code runs against the data/ copies. The repo-root copies feed the generator's
# column list and the automatic rebuild (RUNNING.md 3.2, "Why two folders"), so they
# are checked too when present, but a missing one is not an error.
CACHES = [
    ROOT / "data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
    ROOT / "data/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5",
]
MIRRORS = [ROOT / "git_ignore_folder" / c.parent.name / c.name for c in CACHES]

# Content fingerprints of the reference cache, from content_digest() below. generate.py
# with its defaults, run on the Hugging Face cn_data.zip, reproduces both exactly
# (verified 2026-09-16). HDF5 bytes differ from write to write even when the content is
# identical, so the fingerprint covers the content, not the file.
REFERENCE_FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor", "$vwap"]
REFERENCE_DIGEST = {
    "factor_implementation_source_data": "dd5541b38b2d2a69",
    "factor_implementation_source_data_debug": "b98eb53ae1fd7769",
}
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


def content_digest(df) -> str:
    """Fingerprint of a cache's column names, dtypes, index and values."""
    import numpy as np

    h = hashlib.sha256()
    h.update(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode())
    h.update(repr(list(df.index.names)).encode())
    h.update(df.index.get_level_values(0).values.astype("datetime64[ns]").astype("int64").tobytes())
    h.update("\n".join(df.index.get_level_values(1)).encode())
    values = np.ascontiguousarray(df.to_numpy())
    # NaN bit patterns can vary by platform, so hash where the NaNs are, not their bits.
    nan = np.isnan(values)
    h.update(nan.tobytes())
    h.update(np.where(nan, values.dtype.type(0), values).tobytes())
    return h.hexdigest()[:16]


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
    # A deliberate FIELDS edit (RUNNING.md 4.1.1) changes the content, so only a cache
    # built from the reference field list is held to the reference fingerprint.
    compare = FIELDS == REFERENCE_FIELDS
    found_any = False
    for cache in CACHES + MIRRORS:
        if not cache.exists():
            optional = "" if cache in CACHES else " (optional: a mine builds it if missing)"
            print(f"  absent: {cache.relative_to(ROOT)}{optional}")
            continue
        found_any = found_any or cache in CACHES
        d = pd.read_hdf(cache)
        cols = list(d.columns)
        dates = d.index.get_level_values(0)
        print(f"  {cache.relative_to(ROOT)}")
        print(f"    {len(cols)} columns: {' '.join(cols)}")
        print(f"    {len(d):,} rows, {dates.min().date()} -> {dates.max().date()}")
        missing = [f for f in EXPECTED_CACHE if f not in cols]
        if missing:
            print(f"    MISSING {' '.join(missing)} -- formulas using them cannot be computed.")
            print("    -> rebuild it with quantaalpha/factors/data_template/generate.py (RUNNING.md 3.2, Step 3).")
            ok = False
        elif compare:
            digest, want = content_digest(d), REFERENCE_DIGEST[cache.parent.name]
            if digest == want:
                print(f"    matches the reference cache (fingerprint {digest})")
            else:
                print(f"    DIFFERS from the reference cache (fingerprint {digest}, expected {want})")
                print("    -> rebuild with generate.py's defaults from the Hugging Face cn_data.zip and copy")
                print("       the result into both cache folders (RUNNING.md 3.2, Step 3).")
                ok = False
        del d
    if not compare:
        print("  generate.py FIELDS differ from the reference set; content not compared.")
    if not found_any:
        print("  no cache in data/git_ignore_folder/ yet -- build it before mining (RUNNING.md 3.2, Step 3).")

# ---------------------------------------------------------------- verdict
section("Verdict")
print("  READY" if ok else "  NOT READY -- fix the items marked above before a long mine.")
sys.exit(0 if ok else 1)
