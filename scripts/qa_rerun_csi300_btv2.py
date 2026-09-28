#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Re-run the CSI300 btv2 backtests AFTER the train/test contamination fix.

Why this exists
---------------
`PrecomputedDataHandler.fetch` (backtest/runner.py:392) used to slice a segment
only when the selector was a ``tuple``. ``DatasetH.prepare("<seg>")`` passes
``self.segments[seg]``, and YAML ``segments: train: ["2005-01-01", "2012-12-31"]``
deserializes to a **list** -- so no segment was ever date-clamped, every fetch
returned the full range, and the model trained on its own test window. The
reported "test" IC was in-sample. Fix: ``isinstance(selector, (tuple, list))``.

The bug is market-agnostic (it is purely list-vs-tuple), and all three CSI300
configs use list segments, so every stored CSI300 btv2 result was contaminated.
Two smoking guns, both verified 2026-09-26:

  * ``backtest_v2_long`` (test 2016-2026) and ``backtest_v2_1725`` (test
    2017-2025) reported BIT-IDENTICAL IC 0.13375613931109762 / Rank IC
    0.12949665672324426 despite different test windows -- impossible under
    correct slicing, exactly what "never sliced" produces.
  * IC ~0.125-0.134 against the known CSI300 OOS composite ~0.0166 (~8x high),
    and the per-year IC was nearly flat -- the memorization signature.

The pre-fix results are quarantined under
``data/results/_leaked/btv2_csi300_preleakfix_20260926/``; this driver rebuilds
the canonical ``data/results/backtest_v2_{results,long,1725}`` paths so the
dashboard and ``scripts/qa_report_bt2_yearly.py`` read trustworthy numbers.

What it runs
------------
3 windows x 3 libraries = 9 backtests, SEQUENTIALLY. Each backtest already asks
LightGBM for 20 threads (configs/backtest*.yaml) on an 8-core box, so running
them concurrently would only thrash. Nothing about the configs is modified --
the point is the SAME configuration with only the leak fixed, so the clean
numbers are comparable to the quarantined ones.

Resumable: a (window, library) pair whose metrics JSON already exists is
skipped, so an interrupted run can be relaunched without repeating work.

Usage
-----
    /opt/anaconda3/envs/quantaalpha/bin/python -u scripts/qa_rerun_csi300_btv2.py
    ... --only long              # one window
    ... --force                  # ignore existing outputs
    ... --dry-run                # print the commands only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/opt/anaconda3/envs/quantaalpha/bin/python"
ENTRY = "quantaalpha/backtest/run_backtest.py"

# (key, config, mlflow experiment prefix, output_dir) -- output_dir is declared by
# the config's experiment.output_dir; repeated here so the driver can check for
# existing results without parsing YAML.
WINDOWS = [
    ("results", "configs/backtest.yaml",      "rep",   "data/results/backtest_v2_results"),
    ("long",    "configs/backtest_long.yaml", "long",  "data/results/backtest_v2_long"),
    ("1725",    "configs/backtest_1725.yaml", "w1725", "data/results/backtest_v2_1725"),
]

# Library stems, in the order the original batch ran them (cheapest first, so a
# fast smoke signal arrives before the 150-factor runs).
LIBS = [
    "all_factors_library_meanvar_20260828_194432_zoo",
    "all_factors_library_meanvar_20260828_194432",
    "all_factors_library_original_20260831_012324",
]


def metrics_path(out_dir: str, lib: str) -> Path:
    """Where runner.py writes this library's metrics (output_name = json stem)."""
    return REPO / out_dir / f"{lib}_backtest_metrics.json"


def run_one(cfg: str, prefix: str, out_dir: str, lib: str, dry: bool) -> dict:
    lib_json = f"data/factorlib/{lib}.json"
    cmd = [
        PY, "-u", ENTRY,
        "-c", cfg,
        "-s", "custom",          # MUST be set: -j alone leaves type at alpha158_20
        "-j", lib_json,
        "-e", f"{prefix}_{lib}_fixed",   # _fixed keeps clean mlruns apart from leaked
    ]
    print(f"\n{'='*72}\n[{prefix}] {lib}\n  {' '.join(cmd)}\n{'='*72}", flush=True)
    if dry:
        return {"skipped": "dry-run"}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"[{prefix}] {lib}: {'OK' if ok else f'FAILED rc={proc.returncode}'} "
          f"in {dt/60:.1f} min", flush=True)
    return {"returncode": proc.returncode, "minutes": round(dt / 60, 2), "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=[w[0] for w in WINDOWS], action="append",
                    help="run only these window(s); repeatable")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the metrics JSON already exists")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    os.chdir(REPO)
    windows = [w for w in WINDOWS if not args.only or w[0] in args.only]

    print("CSI300 btv2 re-run AFTER the train/test contamination fix")
    print(f"  repo    : {REPO}")
    print(f"  windows : {[w[0] for w in windows]}")
    print(f"  libs    : {len(LIBS)}")
    print(f"  total   : {len(windows) * len(LIBS)} backtests, sequential")

    results: dict[str, dict] = {}
    t_start = time.time()
    for key, cfg, prefix, out_dir in windows:
        for lib in LIBS:
            tag = f"{key}/{lib}"
            mp = metrics_path(out_dir, lib)
            if mp.exists() and not args.force:
                print(f"\n[skip] {tag} -- {mp.relative_to(REPO)} already exists", flush=True)
                results[tag] = {"skipped": "exists"}
                continue
            results[tag] = run_one(cfg, prefix, out_dir, lib, args.dry_run)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*72}\nSUMMARY  ({elapsed:.1f} min total)\n{'='*72}")
    failed = []
    for tag, r in results.items():
        if r.get("skipped"):
            print(f"  skip  {tag}  ({r['skipped']})")
        elif r.get("ok"):
            print(f"  OK    {tag}  {r['minutes']} min")
        else:
            print(f"  FAIL  {tag}  rc={r.get('returncode')}")
            failed.append(tag)

    # Clean IC/ARR per window, for the leak-vs-clean comparison.
    print(f"\n{'='*72}\nCLEAN METRICS\n{'='*72}")
    print(f"{'window':9} {'library':46} {'IC':>8} {'RankIC':>8} {'ARR%':>8}")
    for key, _cfg, _p, out_dir in windows:
        for lib in LIBS:
            mp = metrics_path(out_dir, lib)
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text()).get("metrics", {})
                arr = m.get("annualized_return")
                print(f"{key:9} {lib:46} {m.get('IC', float('nan')):>8.4f} "
                      f"{m.get('Rank IC', float('nan')):>8.4f} "
                      f"{(arr * 100 if arr is not None else float('nan')):>8.2f}")
            except Exception as exc:  # pragma: no cover - reporting only
                print(f"{key:9} {lib:46}  <unreadable: {type(exc).__name__}>")

    out = REPO / "data/results/csi300_btv2_rerun_summary.json"
    out.write_text(json.dumps({"elapsed_minutes": round(elapsed, 2),
                               "runs": results}, indent=2))
    print(f"\n-> {out.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
