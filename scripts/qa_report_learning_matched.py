#!/usr/bin/env python
"""Learning trend on a DECAY-MATCHED horizon (the split-asymmetry control).

The two mines selected their factors on different eras:

    main      combiner fit 2005-2012, gate validated on 2013-2015
    original  Qlib train 2016-2019, valid 2020, scored on 2021

So a common 2022-2025 report window sits ~7 years after main's selection window
but only ~1 year after original's. Alpha decays, so that window silently handicaps
main: it asks main's factors to survive seven years of drift and original's to
survive one. A learning trend measured there confounds "did the generator improve?"
with "how far has each arm's edge decayed?".

This measures the per-round trend on each mine's OWN horizon, at matched decay
distance -- the window immediately AFTER each mine's selection window:

    main      2016-2018   (0-3y after its 2013-2015 validation window)
    original  2022-2024   (0-3y after its 2021 test window)

plus, for reference, each mine scored on the other's horizon. The question
"do the factors get better as the system learns?" is then asked of each mine on
the era its own search was actually pointed at.

    python scripts/qa_report_learning_matched.py --out data/results/report_learning_matched.json
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

# each mine's own post-selection horizon, matched on DECAY DISTANCE (0-3y out)
HORIZONS = {
    "main_own":     ("2016-01-01", "2018-12-31"),   # 0-3y after valid 2013-2015
    "original_own": ("2022-01-01", "2024-12-31"),   # 0-3y after test  2021
}
LIBS = {
    "main": "data/factorlib/all_factors_library_meanvar_20260828_194432.json",
    "original": "data/factorlib/all_factors_library_original_20260831_012324.json",
}
OWN = {"main": "main_own", "original": "original_own"}


def rows_of(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    factors = d["factors"]
    items = list(factors.values()) if isinstance(factors, dict) else factors
    out = []
    for it in items:
        md = it.get("metadata") or {}
        e = it.get("factor_expression") or it.get("expression")
        if e and md.get("round_number") is not None:
            out.append({"expr": e, "round": int(md["round_number"]),
                        "admitted": bool(it.get("admitted"))})
    return out


def trend(x, y):
    from scipy.stats import linregress, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6 or len(set(x.tolist())) < 3:
        return None
    lr = linregress(x, y)
    null = [abs(float(linregress(RNG.permutation(x), y).slope)) for _ in range(400)]
    return {"n": int(len(x)), "slope": float(lr.slope),
            "ci": [float(lr.slope - 1.96 * lr.stderr), float(lr.slope + 1.96 * lr.stderr)],
            "p": float(lr.pvalue), "rho": float(spearmanr(x, y).statistic),
            "null_95": float(np.percentile(null, 95)),
            "beats_null": bool(abs(lr.slope) > np.percentile(null, 95))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--out", default="data/results/report_learning_matched.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol)
    theta = replace(theta, benchmark_basis="estimated_total",
                    benchmark_construction="equal")
    op = EvaluationOperator(theta)
    # panel must span every horizon we score on
    lo = min(w[0] for w in HORIZONS.values())
    hi = max(w[1] for w in HORIZONS.values())
    panel = op._panel(lo, hi)
    label = label_frame(panel, theta)
    print(f"panel {lo}..{hi}; horizons {HORIZONS}", flush=True)

    out = {"horizons": {k: list(v) for k, v in HORIZONS.items()},
           "rationale": ("each mine scored 0-3y after ITS OWN selection window, so both "
                         "face the same decay distance: main selects on valid 2013-2015, "
                         "original on test 2021"),
           "mines": {}}

    for mine, rel in LIBS.items():
        t0 = time.time()
        rows = rows_of(ROOT / rel)
        sigs = {}
        for r in rows:
            try:
                sigs[r["expr"]] = load_aligned_signal(r["expr"], panel)
            except Exception:
                pass
        print(f"[{mine}] {len(sigs)}/{len(rows)} signals ({time.time()-t0:.0f}s)", flush=True)
        per_h = {}
        for hname, win in HORIZONS.items():
            vals = []
            for r in rows:
                s = sigs.get(r["expr"])
                if s is None:
                    continue
                c = _cross_sectional_corr(s, label, "spearman")
                c = c.loc[win[0]:win[1]]
                q = abs(float(c.mean())) if len(c) else np.nan
                if np.isfinite(q):
                    vals.append({"round": r["round"], "q": q, "admitted": r["admitted"]})
            if not vals:
                continue
            rounds = sorted({v["round"] for v in vals})
            per_round = [{"round": rd,
                          "n": sum(1 for v in vals if v["round"] == rd),
                          "mean": float(np.mean([v["q"] for v in vals if v["round"] == rd])),
                          "median": float(np.median([v["q"] for v in vals if v["round"] == rd]))}
                         for rd in rounds]
            best, cum_best = 0.0, []
            for rd in rounds:
                best = max([best] + [v["q"] for v in vals if v["round"] == rd])
                cum_best.append(float(best))
            half = len(rounds) // 2
            early = [v["q"] for v in vals if v["round"] in rounds[:half]]
            late = [v["q"] for v in vals if v["round"] in rounds[half:]]
            from scipy.stats import mannwhitneyu
            mw = mannwhitneyu(late, early, alternative="greater") if early and late else None
            per_h[hname] = {
                "window": list(win), "n_factors": len(vals), "rounds": rounds,
                "per_round": per_round, "cum_best": cum_best,
                "trend_per_factor": trend([v["round"] for v in vals], [v["q"] for v in vals]),
                "trend_round_mean": trend(rounds, [p["mean"] for p in per_round]),
                "early_mean": float(np.mean(early)) if early else None,
                "late_mean": float(np.mean(late)) if late else None,
                "mw_p_late_gt_early": float(mw.pvalue) if mw else None,
                "overall_mean": float(np.mean([v["q"] for v in vals])),
            }
            tag = "  <-- OWN horizon" if hname == OWN[mine] else ""
            t = per_h[hname]["trend_per_factor"]
            print(f"  [{mine} @ {hname} {win[0][:4]}-{win[1][:4]}] mean |RankIC| "
                  f"{per_h[hname]['overall_mean']:.4f} | slope "
                  f"{(t['slope'] if t else float('nan')):+.2e} p={(t['p'] if t else float('nan')):.3f} "
                  f"beats_null={(t['beats_null'] if t else None)} | early "
                  f"{per_h[hname]['early_mean']:.4f} -> late {per_h[hname]['late_mean']:.4f} "
                  f"(MW p={per_h[hname]['mw_p_late_gt_early']:.4f}){tag}", flush=True)
        out["mines"][mine] = {"library": rel, "own_horizon": OWN[mine], "horizons": per_h}
        Path(a.out).write_text(json.dumps(out, indent=2))

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
