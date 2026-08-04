#!/usr/bin/env python
"""Drop stale ``factor.execute`` memo entries from a *live* run's pickle cache.

``qa_reap_scratch.py`` only touches stamps whose process has exited, which is
right for workspaces but leaves the running arm's cache to grow without bound.
Measured on the Arm B topk re-mine: ``pickle_cache_<stamp>/…coder.factor.execute``
reached 20 GB in 3.4 hours -- about 6 GB/hour, on a disk with 7 hours of runway
and 11.6 hours of mining left. The run would have stopped on a full disk around
03:30 with no other symptom.

**Why dropping these is safe, and why it is limited to one folder.**
``cache_with_pickle`` (quantaalpha/core/utils.py) is pure memoization: on a miss
it calls the wrapped function and re-pickles the result. For ``FactorFBWorkspace.execute``
the key is ``md5(data_type + code_dict["factor.py"])`` with no post-process hook,
so a miss costs exactly one re-execution of that factor's code -- deterministic,
tens of seconds, no correctness consequence.

That argument does **not** transfer to the cache's other folders, so this tool
will not touch them by default:

  * ``…net_cost_runner.develop`` memoizes a whole E_Theta evaluation. It is
    ~1 MB against the execute folder's 20 GB, and a miss costs a full re-score.
    Keeping it is free and losing it is expensive -- the exact inverse trade.
  * ``utils.env.run`` is 8 KB.

A ``.pkl`` is written only after the wrapped call returns, so an entry older
than the threshold is a *completed* memo; no lock check is needed. Unlinking a
file another process is mid-``pickle.load`` on is safe on POSIX -- the reader
holds the inode until it closes.

The age floor exists because the one place identical code recurs is the coder's
debug-retry loop, which turns over in minutes. An entry untouched for an hour
and a half will almost certainly never be asked for again.

Usage::

    python scripts/qa_prune_pickle_cache.py --yes
    python scripts/qa_prune_pickle_cache.py            # dry run
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("DATA_RESULTS_DIR", ROOT / "data/results"))

# Only this folder. See the module docstring for why the safety argument is
# specific to it rather than to pickle caches in general.
PRUNABLE = "quantaalpha.factors.coder.factor.execute"


def _gb(n: int) -> float:
    return n / 1073741824.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yes", action="store_true", help="actually delete")
    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=90.0,
        help="keep entries touched more recently than this (default 90)",
    )
    ap.add_argument(
        "--folder",
        default=PRUNABLE,
        help=f"cache subfolder to prune (default {PRUNABLE})",
    )
    args = ap.parse_args()

    cutoff = time.time() - args.min_age_minutes * 60.0
    total, freed, kept_fresh = 0, 0, 0

    caches = sorted(p for p in RESULTS.glob("pickle_cache_*") if p.is_dir())
    if not caches:
        print("no pickle_cache_* directories found")
        return 0

    for cache in caches:
        folder = cache / args.folder
        if not folder.is_dir():
            continue
        entries = sorted(folder.glob("*.pkl"))
        f_freed, f_kept, f_n = 0, 0, 0
        for pkl in entries:
            try:
                st = pkl.stat()
            except FileNotFoundError:
                continue
            total += st.st_size
            if st.st_mtime > cutoff:
                kept_fresh += 1
                f_kept += 1
                continue
            f_freed += st.st_size
            f_n += 1
            if args.yes:
                try:
                    pkl.unlink()
                except FileNotFoundError:
                    pass
        freed += f_freed
        if entries:
            verb = "freed" if args.yes else "would free"
            print(
                f"  {cache.name}: {f_n} stale / {len(entries)} entries, "
                f"{verb} {_gb(f_freed):.1f} GB ({f_kept} kept, younger than "
                f"{args.min_age_minutes:.0f} min)"
            )

    verb = "reclaimed" if args.yes else "reclaimable"
    print(
        f"\n  {verb} {_gb(freed):.1f} GB of {_gb(total):.1f} GB "
        f"({kept_fresh} entries kept as fresh)"
    )
    if not args.yes and freed:
        print("  dry run -- pass --yes to delete")

    try:
        st = os.statvfs(RESULTS)
        print(f"  disk free now {_gb(st.f_bavail * st.f_frsize):.0f} GB")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
