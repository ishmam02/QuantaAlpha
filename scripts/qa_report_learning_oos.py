#!/usr/bin/env python
"""DO THE FACTORS GET BETTER AS THE SYSTEM LEARNS?

The decisive test, and the one the report leads with. Every factor of both mines is
re-scored on the SAME out-of-sample window (2022-01-01..2025-12-26) with the SAME
measure -- solo |Rank IC| of the factor's own signal against the protocol label --
and plotted against the round that produced it.

Why this and not the libraries' stored metrics: main stores size-neutralised solo
rank IC on its gate's 2013-2015 validation window, original stores raw Qlib rank IC
on its own 2021 test window. Those are different measures on different windows and
must never be compared. Re-scoring both on one window makes "did the factors get
better?" answerable.

Reported per mine:
  * per-round mean / median solo |Rank IC|      -- is the generator improving?
  * OLS slope vs round + Spearman + a permutation null  -- is it distinguishable
    from noise? (two estimators must agree; the null must collapse)
  * cumulative best-so-far                       -- (L3) does the bar rise?
  * cumulative admitted-zoo mean (main only)     -- (L2) does the book improve?

    python scripts/qa_report_learning_oos.py --out data/results/report_learning_oos.json
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import load_aligned_signal
from quantaalpha.eval.metrics import label_frame, _cross_sectional_corr

RNG = np.random.default_rng(23)
LIBS = [
    ("main", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("original", "data/factorlib/all_factors_library_original_20260831_012324.json"),
]


def rows_of(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    factors = d["factors"]
    items = list(factors.values()) if isinstance(factors, dict) else factors
    out = []
    for it in items:
        md = it.get("metadata") or {}
        e = it.get("factor_expression") or it.get("expression")
        if not e or md.get("round_number") is None:
            continue
        out.append({"expr": e, "round": int(md["round_number"]),
                    "phase": md.get("evolution_phase"),
                    "admitted": bool(it.get("admitted"))})
    return out


def trend(x, y):
    """OLS slope with CI + Spearman + permutation null. Two estimators must agree."""
    from scipy.stats import linregress, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6 or len(set(x.tolist())) < 3:
        return None
    lr = linregress(x, y)
    rho = float(spearmanr(x, y).statistic)
    null = [abs(float(linregress(RNG.permutation(x), y).slope)) for _ in range(400)]
    return {"n": int(len(x)), "slope": float(lr.slope),
            "ci": [float(lr.slope - 1.96 * lr.stderr), float(lr.slope + 1.96 * lr.stderr)],
            "p": float(lr.pvalue), "rho": rho,
            "null_95": float(np.percentile(null, 95)),
            "beats_null": bool(abs(lr.slope) > np.percentile(null, 95))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--out", default="data/results/report_learning_oos.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol)
    theta = replace(theta, benchmark_basis="estimated_total",
                    benchmark_construction="equal")
    op = EvaluationOperator(theta)
    p_start, p_end, eval_window = op._windows(True)
    panel = op._panel(p_start, p_end)
    label = label_frame(panel, theta)
    print(f"scoring every factor on {eval_window} (solo |Rank IC| vs the protocol label)",
          flush=True)

    out = {"eval_window": list(eval_window), "measure": "solo |Rank IC| on the OOS window",
           "mines": {}}
    for name, rel in LIBS:
        t0 = time.time()
        rows = rows_of(ROOT / rel)
        for r in rows:
            try:
                s = load_aligned_signal(r["expr"], panel)
                c = _cross_sectional_corr(s, label, "spearman")
                c = c.loc[eval_window[0]:eval_window[1]]
                r["q"] = abs(float(c.mean())) if len(c) else np.nan
            except Exception:
                r["q"] = np.nan
        rows = [r for r in rows if np.isfinite(r.get("q", np.nan))]
        rounds = sorted({r["round"] for r in rows})
        per_round, cum_best, cum_zoo_mean = [], [], []
        best, zoo_vals = 0.0, []
        for rd in rounds:
            qs = [r["q"] for r in rows if r["round"] == rd]
            zq = [r["q"] for r in rows if r["round"] == rd and r["admitted"]]
            zoo_vals += zq
            best = max([best] + qs)
            per_round.append({"round": rd, "n": len(qs),
                              "mean": float(np.mean(qs)), "median": float(np.median(qs)),
                              "max": float(np.max(qs)),
                              "n_admitted": len(zq)})
            cum_best.append(float(best))
            cum_zoo_mean.append(float(np.mean(zoo_vals)) if zoo_vals else np.nan)
        # trends: per-factor (round -> q) and per-round-mean
        t_factor = trend([r["round"] for r in rows], [r["q"] for r in rows])
        t_mean = trend(rounds, [p["mean"] for p in per_round])
        t_best = trend(rounds, cum_best)
        t_zoo = trend(rounds, cum_zoo_mean) if any(np.isfinite(cum_zoo_mean)) else None
        # first-half vs second-half (robust to the round-index gap)
        half = len(rounds) // 2
        early = [r["q"] for r in rows if r["round"] in rounds[:half]]
        late = [r["q"] for r in rows if r["round"] in rounds[half:]]
        from scipy.stats import mannwhitneyu
        mw = mannwhitneyu(late, early, alternative="greater") if early and late else None
        out["mines"][name] = {
            "library": rel, "n_factors": len(rows), "rounds": rounds,
            "per_round": per_round, "cum_best": cum_best, "cum_zoo_mean": cum_zoo_mean,
            "trend_per_factor": t_factor, "trend_round_mean": t_mean,
            "trend_cum_best": t_best, "trend_cum_zoo_mean": t_zoo,
            "early_mean": float(np.mean(early)) if early else None,
            "late_mean": float(np.mean(late)) if late else None,
            "early_n": len(early), "late_n": len(late),
            "mw_p_late_gt_early": float(mw.pvalue) if mw else None,
            "secs": round(time.time() - t0, 1),
        }
        print(f"[{name}] {len(rows)} factors, {len(rounds)} rounds "
              f"({time.time()-t0:.0f}s)", flush=True)
        if t_factor:
            print(f"   per-factor slope {t_factor['slope']:+.2e}/round "
                  f"CI[{t_factor['ci'][0]:+.2e},{t_factor['ci'][1]:+.2e}] "
                  f"p={t_factor['p']:.3f} rho={t_factor['rho']:+.3f} "
                  f"beats_null={t_factor['beats_null']}", flush=True)
        print(f"   early mean {out['mines'][name]['early_mean']:.4f} -> "
              f"late mean {out['mines'][name]['late_mean']:.4f} "
              f"(MW p={out['mines'][name]['mw_p_late_gt_early']:.4f})", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2))

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
