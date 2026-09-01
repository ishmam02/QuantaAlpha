#!/usr/bin/env python
"""Is the post-optimization batch speed actually live in production?

Compares the run's OWN timings before and after the aligned-signal cache landed,
and checks them against the prediction that motivated the change:

    before:  factor_backtest = 100s + 5.0s * |zoo|     (measured, 59 batches)
    after :  factor_backtest = 100s + 0.01s * |zoo|    (predicted)
    => batch time becomes FLAT in zoo size, ~3.1 min

A claim of "the speedup is live" needs three things to hold together, so all
three are reported:

  1. the per-batch wall time fell,
  2. `factor_backtest` specifically fell (that is what was optimized -- if some
     other stage moved, the cause is something else),
  3. the SLOPE against |zoo| collapsed. This is the real test: the fix removed
     a per-incumbent cost, so its signature is that batch time stops growing
     with the book. A lower intercept alone would not prove it.

Usage:  python scripts/qa_batch_speed.py <EXPERIMENT_ID> [--split-at "HH:MM"]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STAGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^|]*\|[^|]*\| "
                      r"quantaalpha\.log\.time:wrapper:\d+ - (\w+) took ([\d.]+)s")

# Measured on this machine before the fix (59 live batches).
BASELINE_INTERCEPT = 100.0
BASELINE_SLOPE = 5.0
PREDICTED_SLOPE = 0.010
OTHER_STAGES = 87.2      # calculate + construct + propose + feedback + init


def parse_sessions(log_root: Path) -> pd.DataFrame:
    """Per-stage durations from the STRUCTURED log tree, not the text log.

    Each task writes ``<task>/__session__/0/{0_factor_propose ... 4_feedback}``
    and the file mtimes are the stage boundaries. This is the only surviving
    record of the pre-restart run: the relaunch used ``>`` and truncated the
    text log (110k lines -> 1k). It also keeps working across restarts, which
    the text log does not.
    """
    rows = []
    for d in sorted(log_root.glob("*/__session__/0")):
        files = sorted(d.glob("*_*"),
                       key=lambda q: int(q.name.split("_")[0])
                       if q.name.split("_")[0].isdigit() else 99)
        if len(files) < 2:
            continue
        stamps = [(f.name, f.stat().st_mtime) for f in files]
        for (n1, t1), (n2, t2) in zip(stamps, stamps[1:]):
            dur = float(t2 - t1)
            # mtimes are not guaranteed monotonic within a session dir (a stale
            # or re-touched file can precede its predecessor). A negative gap is
            # not a measurement, and one -7333s outlier is enough to drag a mean
            # to nonsense, so drop rather than clamp. Measured: 1 of 253.
            if dur < 0:
                continue
            rows.append((pd.Timestamp(t2, unit="s"),
                         n2.split("_", 1)[1], dur))
    return pd.DataFrame(rows, columns=["ts", "stage", "secs"])


def parse_stages(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open(errors="ignore") as fh:
        for line in fh:
            m = STAGE_RE.match(line)
            if m:
                rows.append((pd.Timestamp(m.group(1)), m.group(2), float(m.group(3))))
    return pd.DataFrame(rows, columns=["ts", "stage", "secs"])


def load_ledger(exp: str) -> pd.DataFrame:
    p = ROOT / f"data/results/ledger_{exp}.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["ts", "zoo_size"])
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return pd.DataFrame({
        "ts": pd.to_datetime([r["ts"] for r in rows]).tz_localize(None),
        "zoo_size": [r.get("zoo_size") or 0 for r in rows],
    })


def fit(zoo, secs):
    """Slope + intercept with a CI on the slope. n<3 gives no usable fit."""
    z, s = np.asarray(zoo, float), np.asarray(secs, float)
    ok = np.isfinite(z) & np.isfinite(s)
    z, s = z[ok], s[ok]
    if len(z) < 3 or z.std() == 0:
        return None
    # Theil-Sen: median of pairwise slopes. A least-squares fit on this data is
    # not trustworthy -- backtest durations are heavy-tailed (a single 547s
    # REGBETA factor sits beside a 60s one), and one outlier moves polyfit
    # enough to change the verdict.
    idx = np.argsort(z)
    z, s = z[idx], s[idx]
    pair = [(s[j] - s[i]) / (z[j] - z[i])
            for i in range(len(z)) for j in range(i + 1, len(z)) if z[j] != z[i]]
    a = float(np.median(pair)) if pair else 0.0
    b = float(np.median(s - a * z))
    resid = s - (a * z + b)
    se = np.sqrt((resid ** 2).sum() / max(1, len(z) - 2)) / (z.std() * np.sqrt(len(z)))
    return {"slope": float(a), "intercept": float(b), "n": int(len(z)),
            "slope_ci": (float(a - 1.96 * se), float(a + 1.96 * se))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_id")
    ap.add_argument("--split-at", default=None,
                    help='ISO time the optimization went live, e.g. "2026-08-24 08:42"')
    ap.add_argument("--new-log-root", default=None,
                    help="log/<stamp> dir the POST-fix run writes to. Preferred over "
                         "--split-at: a restart opens a NEW log root, and the old "
                         "root keeps receiving writes from the run that was still "
                         "draining, so a wall-clock split mislabels them as post-fix.")
    a = ap.parse_args()

    log = ROOT / f"data/results/{a.experiment_id}.log"
    if not log.exists():
        sys.exit(f"no log at {log}")

    stages = parse_stages(log)
    # The text log is truncated on any restart that used `>`; the structured
    # session tree is not. Merge both and de-duplicate on (ts, stage).
    for root in sorted((ROOT / "log").glob("*")):
        if root.is_dir():
            sess = parse_sessions(root)
            if not sess.empty:
                sess["log_root"] = root.name
                stages = pd.concat([stages, sess], ignore_index=True)
    if "log_root" not in stages.columns:
        stages["log_root"] = ""
    if stages.empty:
        sys.exit("no stage timings found (neither text log nor session tree)")
    stages = stages.drop_duplicates(subset=["ts", "stage"]).sort_values("ts")
    ledger = load_ledger(a.experiment_id)

    split = pd.Timestamp(a.split_at) if a.split_at else None
    bt = stages[stages.stage == "factor_backtest"].copy()

    # Attach the zoo size in force when each backtest ran (last ledger row before it).
    if not ledger.empty and not bt.empty:
        ledger = ledger.sort_values("ts")
        bt = pd.merge_asof(bt.sort_values("ts"), ledger, on="ts", direction="backward")
    else:
        bt["zoo_size"] = np.nan

    def window(df, lo=None, hi=None):
        d = df
        if lo is not None:
            d = d[d.ts >= lo]
        if hi is not None:
            d = d[d.ts < hi]
        return d

    print(f"experiment: {a.experiment_id}")
    print(f"backtests timed: {len(bt)}"
          + (f"   split at {split}" if split is not None else "   (no split given)"))
    print()

    groups = [("ALL", window(bt))]
    if a.new_log_root:
        # Split by LOG ROOT, which is unambiguous: each launch opens its own.
        post = bt[bt.log_root == a.new_log_root]
        pre = bt[bt.log_root != a.new_log_root]
        groups = [("BEFORE fix", pre), ("AFTER fix ", post)]
    elif split is not None:
        groups = [("BEFORE fix", window(bt, hi=split)),
                  ("AFTER fix ", window(bt, lo=split))]

    print(f"{'window':12s} {'n':>4} {'mean_bt':>9} {'median':>8} {'zoo_lo':>7} {'zoo_hi':>7}"
          f" {'slope s/factor':>15}")
    print("-" * 72)
    fits = {}
    for name, d in groups:
        if d.empty:
            print(f"{name:12s} {0:4d}   (no data)")
            continue
        f = fit(d.zoo_size, d.secs)
        fits[name] = f
        slope = f"{f['slope']:+.3f}" if f else "n/a"
        print(f"{name:12s} {len(d):4d} {d.secs.mean():8.1f}s {d.secs.median():7.1f}s "
              f"{d.zoo_size.min():7.0f} {d.zoo_size.max():7.0f} {slope:>15}")

    print()
    print(f"reference (measured pre-fix): backtest = {BASELINE_INTERCEPT:.0f}s "
          f"+ {BASELINE_SLOPE:.1f}s x |zoo|")
    print(f"prediction (post-fix)       : backtest = {BASELINE_INTERCEPT:.0f}s "
          f"+ {PREDICTED_SLOPE:.3f}s x |zoo|  -> batch ~"
          f"{(BASELINE_INTERCEPT + OTHER_STAGES) / 60:.1f} min, flat in |zoo|")
    print()

    after = fits.get("AFTER fix ") or fits.get("ALL")
    if not after:
        print("VERDICT: not enough post-fix batches yet to fit a slope "
              "(need >= 3 at differing zoo sizes). Re-run once more batches land.")
        return

    lo, hi = after["slope_ci"]
    print(f"post-fix slope: {after['slope']:+.3f} s/factor  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  (n={after['n']})")
    if hi < BASELINE_SLOPE / 2:
        print("VERDICT: the per-incumbent cost is GONE -- the slope is well below the "
              f"pre-fix {BASELINE_SLOPE:.1f} s/factor. The speedup is live.")
    elif lo > BASELINE_SLOPE / 2:
        print("VERDICT: the slope is still near its pre-fix value. The cache is NOT "
              "taking effect in production -- check that the aligned cache directory "
              "is being written and that the panel grid matches.")
    else:
        print("VERDICT: inconclusive -- the CI spans both hypotheses. More batches "
              "across a wider zoo range are needed before claiming anything.")

    # Wall-clock per batch, the number the user actually feels.
    if not ledger.empty:
        led = ledger.sort_values("ts")
        gaps = led.ts.diff().dt.total_seconds().div(60).dropna()
        if split is not None:
            pre = gaps[led.ts.iloc[1:].values < split.to_datetime64()]
            post = gaps[led.ts.iloc[1:].values >= split.to_datetime64()]
            print()
            if len(pre):
                print(f"wall-clock per batch BEFORE: mean {pre.mean():.1f} min (n={len(pre)})")
            if len(post):
                print(f"wall-clock per batch AFTER : mean {post.mean():.1f} min (n={len(post)})")
            else:
                print("wall-clock per batch AFTER : no completed batch yet since the restart")


if __name__ == "__main__":
    main()
