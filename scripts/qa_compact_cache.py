#!/usr/bin/env python
"""Rewrite cached factor signals aligned to the evaluation panel.

A signal is cached exactly as computed: a (datetime, instrument) Series over
the full 2008-2026 x 5982-instrument grid, ~228 MB. Every consumer immediately
calls ``align_signal`` to reduce it to the 2427 x 622 evaluation panel -- 12 MB.
Storing the raw form costs 18.8x the space for data nothing reads.

Rewriting is safe because ``align_signal`` short-circuits on an already-wide
frame, so a consumer that aligns a compacted signal gets the identical result.
Verified per file before the replace, not assumed.

Runs against a LIVE mining process:
  * skips files written in the last 60s, which may still be being flushed
  * writes to a temp file in the same directory and os.replace()s it, so a
    reader either sees the old file or the new one, never a partial
  * a failure on one file skips that file and leaves it untouched

    python scripts/qa_compact_cache.py            # dry run
    python scripts/qa_compact_cache.py --yes      # rewrite
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALIGNED_MAX_MB = 40.0     # anything smaller is already compacted
FRESH_SECONDS = 60.0      # leave files the miner may still be writing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    args = ap.parse_args()

    import pandas as pd
    from quantaalpha.eval.data import align_signal
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    theta = load_protocol(default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, _ = op._windows(True)          # widest window any consumer uses
    panel = op._panel(start, end)
    print(f"panel {start}..{end}  {len(panel.dates)}x{len(panel.instruments)}")

    caches = sorted((ROOT / "data/results").glob("factor_cache*"))
    saved = done = skipped = failed = 0
    now = time.time()

    for cache in caches:
        files = sorted(cache.glob("*.pkl"), key=lambda p: -p.stat().st_size)
        print(f"\n{cache.name}: {len(files)} pickles")
        for p in files:
            st = p.stat()
            if st.st_size / 1e6 <= ALIGNED_MAX_MB:
                skipped += 1
                continue
            if now - st.st_mtime < FRESH_SECONDS:
                skipped += 1          # the miner may still be writing this one
                continue
            if not args.yes:
                saved += st.st_size
                done += 1
                if args.limit and done >= args.limit:
                    break
                continue
            try:
                wide = align_signal(pd.read_pickle(p), panel)
                fd, tmp = tempfile.mkstemp(suffix=".pkl", dir=str(cache))
                os.close(fd)
                wide.to_pickle(tmp)
                # Verify before replacing: a consumer must get the same frame.
                back = align_signal(pd.read_pickle(tmp), panel)
                if not back.fillna(-9e9).equals(wide.fillna(-9e9)):
                    os.unlink(tmp)
                    failed += 1
                    continue
                new = os.path.getsize(tmp)
                os.replace(tmp, p)     # atomic: readers see old or new, never half
                saved += st.st_size - new
                done += 1
            except Exception as exc:
                failed += 1
                print(f"  skip {p.name[:16]}: {type(exc).__name__}")
                try: os.unlink(tmp)
                except Exception: pass
            if args.limit and done >= args.limit:
                break

    verb = "would reclaim" if not args.yes else "reclaimed"
    print(f"\n{done} file(s) {verb} {saved/1e9:.1f} GB "
          f"({skipped} already small or too fresh, {failed} skipped on error)")
    print(f"free now: {shutil.disk_usage(ROOT).free/1e9:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
