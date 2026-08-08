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

So: this refuses per stamp when deletion would destroy something, and it never
touches a stamp whose process is still alive.

"Would destroy something" is the test that matters, and it is narrower than
"every library factor is cached". A library records what the search *proposed*,
not what it computed: factors whose expression pipes a cross-sectional operator
into a time-series one -- ``TS_MEAN(STD($return), 10)``, where STD is
groupby('datetime') and TS_MEAN is groupby('instrument') -- raise
``KeyError: 'instrument'`` and never produce a signal at all. They carry no
implementation code and no ``result_h5_path``, and they are permanently absent
from factor_cache through no fault of any deletion.

Refusing on those would be refusing forever: the stalled top-k arm had 2 such
entries out of 173 and they would have pinned 27.6 GB of dead scratch on the
disk for the rest of the experiment. So an uncached factor blocks the reap only
when a ``result.h5`` for it still exists in the scratch about to be deleted --
that is the case where bytes are actually lost, and the fix is to sync it
first, not to delete it anyway.

    python scripts/qa_reap_scratch.py            # dry run
    python scripts/qa_reap_scratch.py --yes
"""

from __future__ import annotations

import argparse
import json
import pathlib
import stat as statmod
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAMP = re.compile(r"(\d{8}_\d{6})")


def gb(path: pathlib.Path) -> float:
    """Bytes this tree would actually release, in GB.

    ``lstat`` rather than ``stat``, and symlinks skipped: every factor workspace
    holds a ``daily_pv.h5`` symlink to the shared source panel, and following
    those counted the same ~300 MB file once per workspace. On the stalled
    top-k arm that reported 100.2 GB against 27.6 GB really on disk -- deleting
    a symlink frees the link, never the target.
    """
    total = 0
    for f in path.rglob("*"):
        if f.is_symlink():
            continue
        try:
            st = f.lstat()
        except OSError:
            continue
        if statmod.S_ISREG(st.st_mode):
            total += st.st_size
    return total / 1e9


def _result_h5(entry: dict) -> pathlib.Path | None:
    """The factor's ``result.h5`` if one is actually on disk, else ``None``.

    ``cache_location`` carries ``result_h5_path`` only for factors that executed;
    for the rest it holds just ``workspace_path``/``factor_dir``, so the path is
    derived and then checked rather than trusted.
    """
    cl = entry.get("cache_location") or {}
    candidates = []
    if cl.get("result_h5_path"):
        candidates.append(pathlib.Path(cl["result_h5_path"]))
    if cl.get("workspace_path") and cl.get("factor_dir"):
        candidates.append(pathlib.Path(cl["workspace_path"]) / cl["factor_dir"] / "result.h5")
    for c in candidates:
        p = c if c.is_absolute() else ROOT / c
        if p.exists():
            return p
    return None


def _release_cached_h5(stamp, dirs, factor_cache_path, min_age_minutes, apply):
    """Delete result.h5 inside a LIVE stamp, but only where it is redundant.

    A finished stamp can be reaped wholesale; a running one cannot, and that
    asymmetry is what makes disk the binding constraint on a long mine. The
    workspace grows ~7 GB/hour and nothing may touch it until the arm exits, so
    a 20-hour arm needs 141 GB of headroom that this machine does not have.

    But "the process is alive" is not the same as "every byte is load-bearing".
    Once a factor's signal is in factor_cache its result.h5 is redundant, and
    every consumer says so: library.py syncs FROM it and has already done so,
    runner.py treats a missing one as a recompute, and load_factor_signal only
    falls back to h5 when the cache misses. So the same question the reap guard
    asks -- would deleting this destroy anything -- has a per-file answer inside
    a live stamp.

    Two things keep it safe. A file is only released when its factor resolves in
    factor_cache, verified per file rather than per stamp. And files younger
    than ``min_age_minutes`` are left alone, because a factor written moments
    ago may not have been synced yet and the miner may still be reading it.
    """
    import json
    import time

    released, n = 0, 0
    cutoff = time.time() - min_age_minutes * 60.0
    cached: set[str] = set()
    for lib in (ROOT / "data/factorlib").glob(f"*{stamp}*.json"):
        try:
            factors = json.loads(lib.read_text()).get("factors", {})
        except Exception:
            continue
        for v in factors.values():
            e = v.get("factor_expression")
            if e and factor_cache_path(e).exists():
                loc = v.get("cache_location") or {}
                h5 = loc.get("result_h5_path")
                if not h5 and loc.get("workspace_path") and loc.get("factor_dir"):
                    h5 = str(pathlib.Path(loc["workspace_path"]) / loc["factor_dir"] / "result.h5")
                if h5:
                    cached.add(h5)

    for h5 in cached:
        p = pathlib.Path(h5)
        p = p if p.is_absolute() else ROOT / p
        try:
            st = p.lstat()
        except OSError:
            continue
        if not statmod.S_ISREG(st.st_mode) or st.st_mtime > cutoff:
            continue
        released += st.st_size
        n += 1
        if apply:
            try:
                p.unlink()
            except OSError:
                pass
    verb = "released" if apply else "could release"
    print(f"  {stamp}: LIVE -- {verb} {n} redundant result.h5 "
          f"({released/1e9:.1f} GB); workspace kept, signals already cached")
    return released / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="also release cached result.h5 inside a RUNNING stamp")
    ap.add_argument("--min-age-minutes", type=float, default=30.0,
                    help="leave result.h5 files younger than this alone (default 30)")
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
            if args.live:
                freed += _release_cached_h5(stamp, dirs, factor_cache_path,
                                            args.min_age_minutes, args.yes)
            else:
                print(f"  {stamp}: SKIP -- a process is still using it")
            continue

        libs = list((ROOT / "data/factorlib").glob(f"*{stamp}*.json"))
        if not libs:
            print(f"  {stamp}: no library references it; treating as abandoned scratch")
        # Keyed by expression, because a stamp has both a full library and a
        # _zoo subset and every admitted factor appears in both -- counting per
        # file reported one at-risk factor as "2/173".
        at_risk: dict[str, str] = {}    # uncached AND still on disk here -> real loss
        never_built: set[str] = set()   # uncached AND nowhere on disk -> nothing to lose
        seen: set[str] = set()
        for lib in libs:
            factors = json.loads(lib.read_text()).get("factors", {})
            for v in factors.values():
                e = v.get("factor_expression")
                if not e or e in seen:
                    continue
                seen.add(e)
                if factor_cache_path(e).exists():
                    continue
                if _result_h5(v):
                    at_risk[e] = v.get("factor_name") or "unnamed"
                else:
                    never_built.add(e)
        total = len(seen)
        if at_risk:
            names = list(at_risk.values())
            print(f"  {stamp}: REFUSING -- {len(names)}/{total} factor(s) have a "
                  f"result.h5 here and no factor_cache entry; deleting would lose "
                  f"them permanently: {', '.join(names[:5])}"
                  + (" ..." if len(names) > 5 else ""))
            continue

        size = sum(gb(d) for d in dirs)
        note = ""
        if never_built:
            note = (f", {len(never_built)} never produced a signal at all "
                    f"(not at risk from this deletion)")
        print(f"  {stamp}: {len(dirs)} dir(s), {size:.1f} GB, "
              f"{total - len(never_built)}/{total} in factor_cache{note}")
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
