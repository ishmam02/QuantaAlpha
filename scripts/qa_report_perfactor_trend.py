#!/usr/bin/env python
"""Does each system produce better formulas as it goes? Indexed by FORMULA, not round.

Two earlier framings were unusable:

  * indexing by ROUND bakes in the budget asymmetry -- one round is ten formulas in
    main and thirty in original, so "per round" measures different things in each;
  * re-scoring both libraries on one shared window measures how well each system's
    formulas survive an era neither was tuned for, not whether the search improved.

This uses the common axis both libraries actually share: they contain exactly 150
formulas each, mined in order. Progress is measured against that ordinal position
(1..150), and every formula is scored with the metric its OWN system recorded in its
OWN library file, on its own evaluation window. Absolute levels are therefore never
compared across systems -- only the SLOPE of each system against itself, which is the
question "does it get better as it goes?".

Metrics available in both libraries: Rank IC and IC (magnitude). Main additionally
stores a size-neutralised Rank IC and a t-statistic per factor, reported for main only.

    python scripts/qa_report_perfactor_trend.py --out data/results/report_perfactor_trend.json
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Location of the ORIGINAL-branch worktree (the paper baseline). Defaults to a sibling
# directory named qa_orig_mine; override with QA_ORIG_DIR when it lives elsewhere.
ORIG_DIR = Path(os.environ.get("QA_ORIG_DIR", str(ROOT.parent / "qa_orig_mine")))

RNG = np.random.default_rng(23)
LIBS = {
    "main": ROOT / "data/factorlib/all_factors_library_meanvar_20260828_194432.json",
    "original": ORIG_DIR / ("data/factorlib/"
                     "all_factors_library_original_20260831_012324.json"),
}


def rows(path: Path, is_main: bool):
    """One row per formula, in mining order, with that library's own metrics."""
    d = json.loads(path.read_text())
    f = d["factors"]
    items = list(f.values()) if isinstance(f, dict) else f
    out = []
    for i, it in enumerate(items):
        br = it.get("backtest_results") or {}
        md = it.get("metadata") or {}
        r = {"i": i + 1, "round": md.get("round_number"),
             "phase": md.get("evolution_phase")}

        def num(v):
            try:
                x = float(v)
                return abs(x) if np.isfinite(x) else None
            except (TypeError, ValueError):
                return None

        r["rank_ic"] = num(br.get("RankIC") if is_main else br.get("Rank IC"))
        r["ic"] = num(br.get("IC"))
        if is_main:
            r["admitted"] = bool(it.get("admitted"))
            ts = br.get("factor_tearsheets") or {}
            t = list(ts.values())[0] if isinstance(ts, dict) and ts else {}
            r["rank_ic_neutral"] = num(t.get("rank_ic_neutral"))
            r["t_nw"] = num(t.get("t_nw"))
        out.append(r)
    return out


def trend(x, y):
    """OLS slope with CI, Spearman, and a permutation null. Two estimators must agree."""
    from scipy.stats import linregress, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return None
    lr = linregress(x, y)
    null = [abs(float(linregress(RNG.permutation(x), y).slope)) for _ in range(400)]
    n95 = float(np.percentile(null, 95))
    # first vs last third, a shape-free check on the same data
    k = len(x) // 3
    order = np.argsort(x)
    early, late = y[order][:k], y[order][-k:]
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(late, early, alternative="greater") if k >= 5 else None
    return {"n": int(len(x)), "slope_per_formula": float(lr.slope),
            "slope_per_100": float(lr.slope * 100),
            "ci_per_100": [float((lr.slope - 1.96 * lr.stderr) * 100),
                           float((lr.slope + 1.96 * lr.stderr) * 100)],
            "p": float(lr.pvalue), "rho": float(spearmanr(x, y).statistic),
            "beats_null": bool(abs(lr.slope) > n95),
            "first_third_mean": float(np.mean(early)),
            "last_third_mean": float(np.mean(late)),
            "mw_p_late_gt_early": float(mw.pvalue) if mw else None}


def running(y, k=25):
    """Trailing mean over the last k formulas -- the curve the figure plots."""
    y = np.asarray(y, float)
    out = []
    for i in range(len(y)):
        w = y[max(0, i - k + 1): i + 1]
        w = w[np.isfinite(w)]
        out.append(float(np.mean(w)) if len(w) else np.nan)
    return out


def cum_best(y):
    out, best = [], -np.inf
    for v in y:
        if np.isfinite(v):
            best = max(best, v)
        out.append(float(best) if np.isfinite(best) else np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/report_perfactor_trend.json")
    a = ap.parse_args()

    out = {"axis": "formula index 1..150 (mining order), the one axis both libraries "
                   "share; each system scored with its OWN stored metric on its OWN "
                   "window, so only slopes are compared, never levels",
           "systems": {}}

    for name, path in LIBS.items():
        rs = rows(path, name == "main")
        rec = {"library": str(path), "n": len(rs), "index": [r["i"] for r in rs],
               "metrics": {}}
        keys = ["rank_ic", "ic"] + (["rank_ic_neutral", "t_nw"] if name == "main" else [])
        for k in keys:
            y = [r.get(k) for r in rs]
            y = [np.nan if v is None else v for v in y]
            if np.isfinite(y).sum() < 10:
                continue
            rec["metrics"][k] = {
                "values": y, "running25": running(y), "cum_best": cum_best(y),
                "trend": trend([r["i"] for r in rs], y),
            }
        if name == "main":
            adm = [r["i"] for r in rs if r.get("admitted")]
            rec["admitted_index"] = adm
            # quality of the KEPT formulas, in the order they were kept
            ky = [r["rank_ic"] for r in rs if r.get("admitted")]
            ky = [np.nan if v is None else v for v in ky]
            rec["kept_rank_ic"] = {"values": ky, "cum_mean":
                                   list(np.nancumsum(ky) / np.arange(1, len(ky) + 1)),
                                   "cum_best": cum_best(ky),
                                   "trend": trend(list(range(1, len(ky) + 1)), ky)}
        out["systems"][name] = rec

        print(f"=== {name} ({len(rs)} formulas) ===")
        for k, v in rec["metrics"].items():
            t = v["trend"]
            if not t:
                continue
            print(f"  {k:<16} slope {t['slope_per_100']:+.4f} per 100 formulas "
                  f"CI[{t['ci_per_100'][0]:+.4f},{t['ci_per_100'][1]:+.4f}] "
                  f"p={t['p']:.3f} rho={t['rho']:+.3f} null-beat={t['beats_null']}")
            print(f"  {'':<16} first third {t['first_third_mean']:.4f} -> "
                  f"last third {t['last_third_mean']:.4f} "
                  f"(MW p={t['mw_p_late_gt_early']:.3f})")
        if name == "main" and rec.get("kept_rank_ic", {}).get("trend"):
            t = rec["kept_rank_ic"]["trend"]
            print(f"  KEPT set        slope {t['slope_per_100']:+.4f} per 100 kept "
                  f"p={t['p']:.3f}  first {t['first_third_mean']:.4f} -> "
                  f"last {t['last_third_mean']:.4f}")
        print()

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())