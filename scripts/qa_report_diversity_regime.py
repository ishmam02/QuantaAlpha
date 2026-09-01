#!/usr/bin/env python
"""Two design claims, tested on both libraries over the same window.

CLAIM 1 -- DIVERSITY / REDUNDANCY. main gates on marginal contribution and runs
explicit redundancy checks (pairwise vs the book, within-batch, and a marginal
effective-rank check), so its library should be less internally redundant and carry
more independent directions than an ungated one. Measured as: the pairwise
|Spearman| distribution between factors, the share of pairs above the gate's
threshold, the effective rank of the correlation matrix, and the spread of operators
the expressions actually use.

CLAIM 2 -- REGIME ROBUSTNESS. main's feedback reports each factor's IC separately on
crash and rally days, so its search can see and act on a regime-concentrated edge.
Measured per factor exactly as the mine does it: bin each day's cross-sectional IC by
that day's equal-weight universe return, take the mean IC over the bottom 20% of days
(crash) and the top 20% (rally).

Both are computed on the same out-of-sample window for both libraries, so the
comparison is like-for-like.

    python scripts/qa_report_diversity_regime.py --out data/results/report_diversity_regime.json
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import load_aligned_signal
from quantaalpha.eval.metrics import label_frame, _cross_sectional_corr, _abs_spearman

LIBS = [
    ("main_zoo",  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json"),
    ("main_full", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("original",  "data/factorlib/all_factors_library_original_20260831_012324.json"),
]
# operator families as they appear in the expression language
OPS = ["TS_MEAN","TS_STD","TS_MAX","TS_MIN","TS_SUM","TS_RANK","TS_CORR","TS_COV",
       "TS_SKEW","TS_KURT","TS_QUANTILE","TS_ARGMAX","TS_ARGMIN","TS_DELTA","TS_DELAY",
       "TS_ZSCORE","TS_PCTCHANGE","TS_MEDIAN","TS_PROD","TS_DECAYLINEAR","TS_REGBETA",
       "RANK","ZSCORE","SCALE","SIGN","LOG","ABS","POW","SQRT","EXP",
       "PERCENTILE","CORR","COV","MAX","MIN","MEAN","STD","SUM","DELTA","DELAY",
       "RSI","MACD","BB","ATR","OBV","COUNT","IF","WHERE","CLIP","NEG"]


def factors_of(path: Path):
    d = json.loads(path.read_text())
    f = d.get("factors", d)
    items = list(f.values()) if isinstance(f, dict) else f
    out = []
    for it in items:
        e = it.get("factor_expression") or it.get("expression")
        if e:
            out.append(e)
    return out


def operator_profile(exprs):
    """Which operator families the library actually uses, and how evenly."""
    per_factor, counts = [], Counter()
    for e in exprs:
        toks = set(re.findall(r"[A-Z_]{2,}", e.upper()))
        used = {o for o in OPS if o in toks}
        per_factor.append(len(used))
        counts.update(used)
    total = sum(counts.values())
    if total == 0:
        return {}
    p = np.array([c / total for c in counts.values()], float)
    shannon = float(-(p * np.log(p)).sum())
    return {
        "distinct_operators": len(counts),
        "operator_entropy": shannon,
        "effective_operators": float(np.exp(shannon)),   # perplexity
        "top5_share": float(sum(sorted(counts.values(), reverse=True)[:5]) / total),
        "mean_ops_per_factor": float(np.mean(per_factor)),
        "counts": dict(counts.most_common(12)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--gate-protocol",
                    default="quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml",
                    help="the protocol the ADMISSION GATE ran under; its rho_bar is "
                         "the threshold both libraries are judged against")
    ap.add_argument("--out", default="data/results/report_diversity_regime.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol)
    theta = replace(theta, benchmark_basis="estimated_total",
                    benchmark_construction="equal")
    # Redundancy is judged with the GATE's OWN threshold and primitive, so both
    # libraries are held to the identical bar the admission gate applied: rho_bar
    # from the protocol the gate actually ran under, and the same |Spearman| the
    # gate computes (_abs_spearman). The report protocol carries a different
    # rho_bar, so it is read from the gate's protocol explicitly.
    gate_theta = load_protocol(a.gate_protocol)
    op = EvaluationOperator(theta)
    p_start, p_end, win = op._windows(True)
    panel = op._panel(p_start, p_end)
    label = label_frame(panel, theta)
    rho_bar = float(getattr(gate_theta.gates, "rho_bar", 0.60) or 0.60)
    print(f"window {win} | gate rho_bar {rho_bar} (from {gate_theta.hash})", flush=True)

    out = {"eval_window": list(win), "rho_bar": rho_bar,
           "gate_protocol": a.gate_protocol, "gate_hash": gate_theta.hash,
           "libraries": {}}

    for name, rel in LIBS:
        t0 = time.time()
        exprs = factors_of(ROOT / rel)
        sigs = {}
        for e in exprs:
            try:
                sigs[e] = load_aligned_signal(e, panel)
            except Exception:
                pass
        keys = list(sigs)
        n = len(keys)
        print(f"[{name}] {n}/{len(exprs)} signals ({time.time()-t0:.0f}s)", flush=True)

        # ---- pairwise redundancy ----
        # Same quantity the gate computes (|mean daily cross-sectional Spearman|),
        # but each factor is ranked ONCE and the full matrix comes from a single
        # matmul per day-block rather than n(n-1)/2 independent pairwise calls.
        # Cross-sectional Spearman == Pearson on per-day ranks, so ranking rows,
        # centring and normalising them turns every pair into a dot product.
        import pandas as pd
        cols = panel.instruments
        idx = panel.dates
        R = np.full((n, len(idx), len(cols)), np.nan)
        for k, e in enumerate(keys):
            w = sigs[e]
            if isinstance(w, pd.Series):
                w = w.unstack()
            w = w.reindex(index=idx, columns=cols)
            w = w.where(panel.universe)
            r = w.rank(axis=1)                      # per-day cross-sectional ranks
            r = r.sub(r.mean(axis=1), axis=0)       # centre each day
            nrm = np.sqrt((r ** 2).sum(axis=1))
            r = r.div(nrm.replace(0.0, np.nan), axis=0)
            R[k] = r.to_numpy(dtype=float)
        Rz = np.nan_to_num(R, nan=0.0)
        valid = (~np.isnan(R)).any(axis=2)          # (n, days) which days a factor has
        # daily correlation between every pair, then average over shared days
        num = np.einsum("idc,jdc->ijd", Rz, Rz)     # (n, n, days)
        shared = (valid[:, None, :] & valid[None, :, :])
        with np.errstate(invalid="ignore"):
            M = np.abs(np.where(shared, num, np.nan))
            M = np.nanmean(M, axis=2)
        M = np.nan_to_num(M, nan=0.0)
        np.fill_diagonal(M, 1.0)
        iu = np.triu_indices(n, k=1)
        pair = M[iu]
        ev = np.linalg.eigvalsh(M)
        ev = ev[ev > 1e-12]
        p_ = ev / ev.sum()
        er = float(np.exp(-(p_ * np.log(p_)).sum()))

        # ---- regime-conditional IC, per factor, exactly as the mine bins it ----
        mkt = label.where(panel.universe).mean(axis=1)
        crash, rally, allic, ratio = [], [], [], []
        for e in keys:
            ic = _cross_sectional_corr(sigs[e], label, "spearman").dropna()
            ic = ic.loc[win[0]:win[1]]
            m = mkt.reindex(ic.index)
            good = m.notna() & ic.notna()
            ic2, m2 = ic[good], m[good]
            if len(ic2) < 100:
                continue
            lo, hi = float(m2.quantile(0.20)), float(m2.quantile(0.80))
            c = float(ic2[m2 <= lo].mean()); r = float(ic2[m2 >= hi].mean())
            t = float(ic2.mean())
            crash.append(c); rally.append(r); allic.append(t)
            if abs(t) > 1e-9:
                ratio.append(abs(c) / abs(t))

        def st(v):
            v = np.array([x for x in v if np.isfinite(x)], float)
            return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                    "sd": float(v.std())} if v.size else {}

        # sign-aligned crash/rally magnitude (a factor's edge may be negative)
        crash_abs = [abs(x) for x in crash]
        rally_abs = [abs(x) for x in rally]
        # does the edge SURVIVE a crash? share of factors whose |crash IC| >= |overall IC|
        survive = float(np.mean([abs(c) >= abs(t) for c, t in zip(crash, allic)
                                 if np.isfinite(c) and np.isfinite(t)]))

        out["libraries"][name] = {
            "library": rel, "n_factors": n,
            "diversity": {
                "pairwise_abs_spearman": st(pair),
                "share_pairs_above_rho_bar": float((pair >= rho_bar).mean()),
                "p90_pairwise": float(np.percentile(pair, 90)) if pair.size else None,
                "max_pairwise": float(pair.max()) if pair.size else None,
                "effective_rank": er,
                "rank_density": er / n,
                "operators": operator_profile(exprs),
            },
            "regime": {
                "ic_crash": st(crash), "ic_rally": st(rally),
                "abs_ic_crash": st(crash_abs), "abs_ic_rally": st(rally_abs),
                "abs_ic_all": st([abs(x) for x in allic]),
                "crash_to_all_ratio": st(ratio),
                "share_edge_survives_crash": survive,
            },
            "secs": round(time.time() - t0, 1),
        }
        d = out["libraries"][name]
        print(f"   diversity: mean|rho| {d['diversity']['pairwise_abs_spearman']['mean']:.4f} "
              f"| >=rho_bar {100*d['diversity']['share_pairs_above_rho_bar']:.1f}% "
              f"| eff.rank {er:.1f} ({100*er/n:.0f}% of n) "
              f"| distinct ops {d['diversity']['operators'].get('distinct_operators')}"
              f" (eff {d['diversity']['operators'].get('effective_operators',0):.1f})", flush=True)
        print(f"   regime   : |IC| crash {d['regime']['abs_ic_crash']['mean']:.4f} "
              f"rally {d['regime']['abs_ic_rally']['mean']:.4f} "
              f"all {d['regime']['abs_ic_all']['mean']:.4f} "
              f"| crash/all {d['regime']['crash_to_all_ratio']['median']:.2f} "
              f"| survives crash {100*survive:.0f}%", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
