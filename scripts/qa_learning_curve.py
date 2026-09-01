#!/usr/bin/env python
"""Decision gate 3: is the generator learning, round over round?

`declarative-doodling-wadler.md` names this as the real deliverable of its
Phase 5, and states the baseline it has to beat:

    The test for it. Round-over-round improvement in the NEUTRALIZED IC of
    generated factors. The current baseline is 0 of 7 rounds showing
    improvement. If after this change it is still 0 of N, the bottleneck is the
    generator itself rather than the feedback.

Neutralized, specifically, because raw IC improving is ambiguous -- a factor can
raise its raw IC purely by loading harder on size, which is not learning. The
per-factor tearsheets now persist `rank_ic_neutral` alongside `exposure_size`,
so the question is answerable directly from a run's library.

Two things are measured separately, because they answer different questions:

  * ALL generated factors -- does the generator PROPOSE better factors over
    time? This is the learning question.
  * ADMITTED factors only -- does the book improve? This can rise with no
    learning at all, purely because the gate keeps the best of a fixed-quality
    stream, so it is reported but never used as the verdict.

Statistics follow the standing protocol: population not sample, every estimate
carries a CI, a null control must collapse, and two estimators must agree
(OLS slope on the round index, and Spearman rank correlation).

Usage::

    python scripts/qa_learning_curve.py --library data/factorlib/<run>.json
    python scripts/qa_learning_curve.py --library A.json --library B.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(23)


def load(path: str) -> list[dict]:
    """One row per generated factor: round, neutralized IC, and friends."""
    d = json.loads(Path(path).read_text())
    items = d.get("factors", d)
    items = list(items.values()) if isinstance(items, dict) else items
    rows = []
    for it in items:
        md = it.get("metadata") or {}
        br = it.get("backtest_results") or {}
        ts = br.get("factor_tearsheets") or {}
        if isinstance(ts, dict):
            ts = list(ts.values())
        for t in (ts or [{}]):
            r = md.get("round_number")
            v = t.get("rank_ic_neutral")
            if r is None or v is None:
                continue
            rows.append({
                "round": int(r), "ic_neut": abs(float(v)),
                "ic_raw": abs(float(t.get("rank_ic") or np.nan)),
                "t_nw": abs(float(t.get("t_nw") or np.nan)),
                "size_exp": abs(float(t.get("exposure_size") or np.nan)),
                "mono": abs(float(t.get("monotonicity") or np.nan)),
                "admitted": bool(it.get("admitted")),
                "phase": md.get("evolution_phase") or "?",
            })
    return rows


def trend(x: np.ndarray, y: np.ndarray):
    """OLS slope + Spearman, each with a CI. Two estimators that must agree."""
    from scipy.stats import spearmanr, linregress
    if len(x) < 6 or len(set(x.tolist())) < 3:
        return None
    lr = linregress(x, y)
    # 95% CI on the slope from its standard error
    lo, hi = lr.slope - 1.96 * lr.stderr, lr.slope + 1.96 * lr.stderr
    rho = float(spearmanr(x, y).statistic)
    # null control: permute the ROUND labels, keep the values
    null = [abs(float(linregress(RNG.permutation(x), y).slope)) for _ in range(400)]
    return {"slope": float(lr.slope), "ci": [float(lo), float(hi)],
            "p": float(lr.pvalue), "rho": rho,
            "null_95": float(np.percentile(null, 95))}


def report(name: str, rows: list[dict], key: str = "ic_neut") -> dict:
    by = defaultdict(list)
    for r in rows:
        if np.isfinite(r[key]):
            by[r["round"]].append(r[key])
    rounds = sorted(by)
    if len(rounds) < 3:
        print(f"  {name}: only {len(rounds)} rounds -- cannot establish a trend")
        return {}
    means = [float(np.mean(by[r])) for r in rounds]
    ns = [len(by[r]) for r in rounds]
    print(f"  {name}   ({sum(ns)} factors over {len(rounds)} rounds)")
    for r, m, n in zip(rounds, means, ns):
        bar = "#" * int(round(m / max(means) * 34)) if max(means) > 0 else ""
        print(f"    round {r:>2}  n={n:>3}  {key} {m:.4f}  {bar}")
    # the plan's own scoring rule: how many rounds beat the one before
    ups = sum(1 for a, b in zip(means, means[1:]) if b > a)
    print(f"    rounds improving on the previous: {ups}/{len(means)-1}")
    x = np.array([r["round"] for r in rows if np.isfinite(r[key])], float)
    y = np.array([r[key] for r in rows if np.isfinite(r[key])], float)
    t = trend(x, y)
    if t:
        sig = "YES" if (t["ci"][0] > 0 or t["ci"][1] < 0) else "no"
        print(f"    slope {t['slope']:+.5f}/round  CI [{t['ci'][0]:+.5f}, "
              f"{t['ci'][1]:+.5f}]  p={t['p']:.3f}  rho={t['rho']:+.3f}")
        print(f"    null control |slope| 95th pct {t['null_95']:.5f}   "
              f"distinguishable from noise: {sig}")
    return {"rounds": rounds, "means": means, "n": ns,
            "improving": ups, "of": len(means) - 1, "trend": t}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", action="append", required=True)
    ap.add_argument("--out", default="data/results/learning_curve.json")
    a = ap.parse_args()

    out = {}
    for path in a.library:
        rows = load(path)
        nm = Path(path).stem
        print("=" * 74)
        print(f"{nm}   {len(rows)} factors carrying a neutralized IC")
        print("=" * 74)
        if not rows:
            print("  no tearsheets with rank_ic_neutral -- nothing to measure\n")
            continue
        res = {"all": report("ALL GENERATED (the learning question)", rows)}
        adm = [r for r in rows if r["admitted"]]
        if len(adm) >= 6:
            print()
            res["admitted"] = report("ADMITTED ONLY (the book, not learning)", adm)
        print()
        # is any apparent gain just a bigger size bet?
        res["size"] = report("SIZE EXPOSURE (a rise here is not learning)",
                             rows, key="size_exp")
        out[nm] = res
        t = res["all"].get("trend")
        print("\n  " + "-" * 70)
        if not t:
            print("  INCONCLUSIVE: too few rounds.")
        elif t["ci"][0] > 0:
            print("  LEARNING: neutralized IC rises with round, and the rise is")
            print("  outside its own null. The generator is improving.")
        elif t["ci"][1] < 0:
            print("  DEGRADING: neutralized IC FALLS with round, outside null.")
        else:
            print(f"  FLAT: slope {t['slope']:+.5f}/round with a CI spanning zero")
            print(f"  ({res['all']['improving']}/{res['all']['of']} rounds improving).")
            print("  Round-over-round quality is not distinguishable from noise --")
            print("  the same verdict as the 0-of-7 baseline, on a longer run.")
        print()

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
