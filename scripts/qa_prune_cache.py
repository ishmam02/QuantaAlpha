#!/usr/bin/env python
"""Reclaim signal-cache space without losing anything a comparison needs.

The cache stores each factor's signal at full 2008-2026 x 5982-instrument
resolution -- ~228 MB apiece, against the ~14 MB the 2022-2025 CSI 300
evaluation actually reads. A paper-scale seed mines ~450 factors, so one seed
costs ~100 GB and three exhaust a 1 TB disk before the run finishes.

Two tiers, both keyed on what the libraries can still reach:

  --orphans   pickles no library references at all. Always safe: nothing can
              load them, so nothing can miss them.
  --seed S    every pickle for seed S, once its comparison artefacts exist.
              The results are already written to markdown and JSON by then, so
              the signals are only needed to RE-score, and any factor can be
              recomputed from the expression the library stores.

Dry-run by default. Refuses --seed while that seed's comparisons are missing,
because deleting signals before the numbers are produced turns a slow run into
a lost one.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cache_keys(cache: pathlib.Path) -> dict[str, pathlib.Path]:
    return {p.stem: p for p in cache.glob("*.pkl")}


def referenced() -> set[str]:
    keys = set()
    for f in (ROOT / "data/factorlib").glob("*.json"):
        try:
            factors = json.loads(f.read_text())["factors"]
        except Exception:
            continue
        for v in factors.values():
            e = v.get("factor_expression")
            if e:
                keys.add(hashlib.md5(e.encode()).hexdigest())
    return keys


def gb(paths) -> float:
    return sum(p.stat().st_size for p in paths) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orphans", action="store_true", help="delete unreferenced pickles")
    ap.add_argument("--seed", help="delete a finished seed's whole cache")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    victims: list[pathlib.Path] = []

    if args.orphans:
        keep = referenced()
        for cache in (ROOT / "data/results").glob("factor_cache*"):
            on_disk = cache_keys(cache)
            orph = [p for k, p in on_disk.items() if k not in keep]
            if orph:
                print(f"  {cache.name}: {len(orph)}/{len(on_disk)} orphaned, {gb(orph):.1f} GB")
            victims += orph

    if args.seed:
        cache = ROOT / f"data/results/factor_cache_s{args.seed}"
        if not cache.exists():
            print(f"  no cache for seed {args.seed}")
        else:
            done = glob.glob(str(ROOT / "data/results/paper_*/arm_comparison_*.md"))
            if not done:
                print(f"  REFUSING --seed {args.seed}: no comparison artefacts exist yet.")
                print( "  Deleting signals before the numbers are produced turns a slow")
                print( "  run into a lost one. Run the comparisons first.")
                return 1
            files = list(cache.glob("*.pkl"))
            print(f"  {cache.name}: {len(files)} pickles, {gb(files):.1f} GB "
                  f"(comparisons present: {len(done)})")
            victims += files

    if not victims:
        print("  nothing to reclaim")
        return 0
    print(f"\n  TOTAL: {len(victims)} files, {gb(victims):.1f} GB")
    if not args.yes:
        print("  dry run -- re-run with --yes to delete")
        return 0
    for p in victims:
        try:
            p.unlink()
        except OSError:
            pass
    print(f"  deleted. free now: "
          f"{__import__('shutil').disk_usage(ROOT).free/1e9:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
