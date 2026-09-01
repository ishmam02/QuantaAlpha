#!/usr/bin/env python
"""Stage B — the learning comparison, per the user's 2026-08-31 definition.

Learning is NOT "constantly better than before" (impossible). It is:
  1. No anti-learning  -- per-round MEDIAN factor quality is flat-or-rising.
  2. Zoo mean rises    -- the cumulative admitted-zoo mean quality rises over rounds.
  3. Best bar rises    -- the cumulative best factor seen-so-far rises over rounds.
The zoo is the lens.  Lineage (beat-parent) is evidence, not the verdict.

Per-factor quality (each mine's native metric; see caveat below):
  main  -> |rank_ic_neutral| from factor_tearsheets (size-neutralized solo rank IC
           on the gate's valid window 2013-2015) ; fallback |U|.
  orig  -> |Rank IC| from backtest_results (the Qlib backtest IC the original
           branch records on its 2021 test window).
CAVEAT: the two mines use different quality metrics (the original branch has no
factor_tearsheets / no neutralized IC and no gate).  The comparison is therefore
on the SHAPE of each curve (rising vs flat), not absolute levels.  A robustness
re-evaluation of the original's factors on the gate's valid window (to obtain a
comparable rank_ic_neutral) is noted in the report as an extension.

Admission: main has a top-level `admitted` flag (the gate); the original branch
has no admission system, so every original factor is "in the zoo" (admitted=True).

Output: data/results/report_learning.json
  {main: {rounds:[...], median:[...], zoo_mean:[...], zoo_best:[...], n:[...],
          metric, n_factors, n_admitted},
   original: {...}}
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Location of the ORIGINAL-branch worktree (the paper baseline). Defaults to a sibling
# directory named qa_orig_mine; override with QA_ORIG_DIR when it lives elsewhere.
ORIG_DIR = Path(os.environ.get("QA_ORIG_DIR", str(ROOT.parent / "qa_orig_mine")))

MAIN_LIB = ROOT / "data/factorlib/all_factors_library_meanvar_20260828_194432.json"
ORIG_LIB = ORIG_DIR / ("data/factorlib/all_factors_library_original_20260831_012324.json")
OUT = ROOT / "data/results/report_learning.json"


def factor_rows(path: Path, is_main: bool) -> list[dict]:
    d = json.loads(path.read_text())
    factors = d["factors"]
    items = list(factors.values()) if isinstance(factors, dict) else factors
    rows = []
    for it in items:
        md = it.get("metadata") or {}
        br = it.get("backtest_results") or {}
        r = md.get("round_number")
        if r is None:
            continue
        # quality
        q = None
        if is_main:
            ts = br.get("factor_tearsheets") or {}
            if isinstance(ts, dict) and ts:
                t = list(ts.values())[0]
                if isinstance(t, dict) and t.get("rank_ic_neutral") is not None:
                    q = abs(float(t["rank_ic_neutral"]))
            if q is None and br.get("U") is not None:
                try:
                    q = abs(float(br["U"]))
                except (TypeError, ValueError):
                    pass
        else:
            v = br.get("Rank IC")
            if v is not None:
                try:
                    q = abs(float(v))
                except (TypeError, ValueError):
                    pass
        if q is None:
            continue
        admitted = bool(it.get("admitted")) if is_main else True
        rows.append({"round": int(r), "quality": float(q), "admitted": admitted,
                     "phase": md.get("evolution_phase")})
    return rows


def curve(rows: list[dict], metric_name: str) -> dict:
    by_round = defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)
    rounds = sorted(by_round)
    # zoo = admitted (main) or all (orig, since admitted=True for every row)
    zoo = [r for r in rows if r["admitted"]]
    zoo_by_round = defaultdict(list)
    for r in zoo:
        zoo_by_round[r["round"]].append(r)
    median, mean, n = [], [], []
    zoo_cum_mean, zoo_cum_best = [], []
    cum_zoo = []
    for rd in rounds:
        qs = np.array([r["quality"] for r in by_round[rd]], float)
        median.append(float(np.median(qs)))
        mean.append(float(np.mean(qs)))
        n.append(int(len(qs)))
        # cumulative zoo up to and including this round
        cum_zoo = [r for r in zoo if r["round"] <= rd]
        zq = np.array([r["quality"] for r in cum_zoo], float)
        zoo_cum_mean.append(float(np.mean(zq)) if len(zq) else np.nan)
        zoo_cum_best.append(float(np.max(zq)) if len(zq) else np.nan)
    # trend (OLS slope + sign) on each series vs round
    def slope(y):
        y = np.array(y, float)
        x = np.array(rounds, float)
        m = np.isfinite(y)
        if m.sum() < 3:
            return None
        x1, y1 = x[m], y[m]
        b = np.polyfit(x1, y1, 1)[0]
        return float(b)
    return {
        "metric": metric_name,
        "rounds": rounds,
        "n": n,
        "median": median,
        "mean": mean,
        "zoo_mean": zoo_cum_mean,
        "zoo_best": zoo_cum_best,
        "n_zoo": len(zoo),
        "n_factors": len(rows),
        "slope_median": slope(median),
        "slope_zoo_mean": slope(zoo_cum_mean),
        "slope_zoo_best": slope(zoo_cum_best),
    }


def main_system_level(ledger_path: Path) -> dict:
    """Cumulative-zoo composite trajectory from the run's ledger (the system-level
    learning signal: the book's composite rank_ic + U as the gate admits factors).
    `metrics.rank_ic` is the cumulative book's composite rank IC at each decision;
    `U` is the gate utility; `zoo_size` is the running admitted count."""
    rows = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    adm = [r for r in rows if r.get("admitted")]
    zoo_size, rank_ic, U, ts = [], [], [], []
    for r in adm:
        m = r.get("metrics") or {}
        if r.get("zoo_size") is None:
            continue
        zoo_size.append(int(r["zoo_size"]))
        rank_ic.append(float(m.get("rank_ic", float("nan"))))
        U.append(float(r.get("U", float("nan"))))
        ts.append(str(r.get("ts", "")))
    def slope(y):
        y = np.array(y, float); x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 3 else None
    return {
        "zoo_size": zoo_size, "rank_ic": rank_ic, "U": U, "ts": ts,
        "n_points": len(zoo_size),
        "slope_rank_ic": slope(rank_ic), "slope_U": slope(U),
        "rank_ic_start": rank_ic[0] if rank_ic else None,
        "rank_ic_end": rank_ic[-1] if rank_ic else None,
        "U_start": U[0] if U else None, "U_end": U[-1] if U else None,
    }


if __name__ == "__main__":
    main_rows = factor_rows(MAIN_LIB, True)
    orig_rows = factor_rows(ORIG_LIB, False)
    out = {
        "main": curve(main_rows, "|rank_ic_neutral| (valid 2013-2015)"),
        "original": curve(orig_rows, "|Rank IC| (Qlib test 2021)"),
        "caveat": ("Main uses size-neutralized solo rank IC on the gate's valid "
                   "window; original uses raw rank IC on its Qlib test window (the "
                   "original branch has no factor_tearsheets / no gate). Compare the "
                   "SHAPE (rising vs flat), not absolute levels."),
    }
    out["main"]["system_level"] = main_system_level(
        ROOT / "data/results/ledger_meanvar_20260828_194432.jsonl")
    OUT.write_text(json.dumps(out, indent=2))
    for name, c in out.items():
        if not isinstance(c, dict):
            continue
        print(f"{name}: {c['n_factors']} factors ({c['n_zoo']} in zoo) over rounds {c['rounds']}")
        print(f"  slope median={c['slope_median']:+.2e}  zoo_mean={c['slope_zoo_mean']:+.2e}  zoo_best={c['slope_zoo_best']:+.2e}")
        print(f"  median by round: " + ", ".join(f"r{r}:{v:.4f}" for r, v in zip(c['rounds'], c['median'])))
        print(f"  zoo_best by round: " + ", ".join(f"r{r}:{v:.4f}" for r, v in zip(c['rounds'], c['zoo_best'])))
    print(f"\nwrote {OUT}")