#!/usr/bin/env python
"""Delete workspaces and pickle caches for arms that have FINISHED.

These are the largest consumers by far -- a completed arm leaves ~87 GB across
workspace_* and pickle_cache_*, against ~2 GB of compacted signals -- and the
cache compactor never touched them, so disk fell to 23 GB mid-run twice.

They are scratch: every factor a library references is recoverable from
factor_cache, and the library stores the expression besides. But "recoverable"
has to be CHECKED, not assumed. Deleting a workspace whose signals never made
it into factor_cache loses those factors for good; that happened once, costing
3 of 156 factors, because the guard was written as a separate command and the
delete ran anyway when it failed.

So: this refuses per stamp unless every factor in every library carrying that
stamp resolves in factor_cache, and it never touches a stamp whose process is
still alive.

    python scripts/qa_reap_scratch.py            # dry run
    python scripts/qa_reap_scratch.py --yes
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAMP = re.compile(r"(\d{8}_\d{6})")


def gb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    from quantaalpha.eval.data import factor_cache_path

    results = ROOT / "data/results"
    scratch: dict[str, list[pathlib.Path]] = {}
    for d in results.glob("*"):
        if not d.is_dir() or not d.name.startswith(("workspace_", "pickle_cache_")):
            continue
        m = STAMP.search(d.name)
        if m:
            scratch.setdefault(m.group(1), []).append(d)

    # A stamp still being written must never be reaped.
    try:
        live = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except Exception:
        live = ""

    freed = 0.0
    for stamp, dirs in sorted(scratch.items()):
        if stamp in live:
            print(f"  {stamp}: SKIP -- a process is still using it")
            continue

        libs = list((ROOT / "data/factorlib").glob(f"*{stamp}*.json"))
        if not libs:
            print(f"  {stamp}: no library references it; treating as abandoned scratch")
        missing = 0
        total = 0
        for lib in libs:
            factors = json.loads(lib.read_text()).get("factors", {})
            for v in factors.values():
                e = v.get("factor_expression")
                if not e:
                    continue
                total += 1
                if not factor_cache_path(e).exists():
                    missing += 1
        if missing:
            print(f"  {stamp}: REFUSING -- {missing}/{total} factor(s) are NOT in "
                  f"factor_cache; deleting would lose them permanently")
            continue

        size = sum(gb(d) for d in dirs)
        print(f"  {stamp}: {len(dirs)} dir(s), {size:.1f} GB, {total}/{total} recoverable")
        freed += size
        if args.yes:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    verb = "would free" if not args.yes else "freed"
    print(f"\n  {verb} {freed:.1f} GB | disk free now "
          f"{shutil.disk_usage(ROOT).free/1e9:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
