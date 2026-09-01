#!/usr/bin/env python
"""Is main_zoo's lower composite IC a QUALITY deficit or a COUNT artifact?

The LightGBM combiner is fed one column per factor. More columns give it more to
stack, and stacking gains do not require the added columns to be individually
good -- so a 56-factor book and a 150-factor book are NOT comparable on composite
IC, and the raw comparison silently penalises the gated arm for being selective.

This isolates the two effects:

  (A) COUNT-MATCHED books. Score main_full, original and main_zoo at the SAME
      factor count n=56, drawing random n-subsets (fixed seeds) of the two
      150-factor libraries. If main's factors are individually better, main's
      count-matched book beats original's at equal n.

  (B) COUNT CURVE. Score n = 10, 20, 40, 56, 80, 120, 147 subsets of ONE library
      to show composite IC rising with n on its own -- i.e. how much of the
      150-vs-56 gap is explained by count alone.

  (C) SOLO factor quality. Per-factor |Rank IC| on the same window for every
      factor of every library -- the count-free measure of "are these factors
      better?"

    python scripts/qa_report_count_control.py --out data/results/report_count_control.json
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.data import load_aligned_signal
from quantaalpha.eval.metrics import _ic_block, label_frame, _to_wide, _cross_sectional_corr

TRADING_DAYS = 243
LIBS = [
    ("main_zoo",  "data/factorlib/all_factors_library_meanvar_20260828_194432_zoo.json"),
    ("main_full", "data/factorlib/all_factors_library_meanvar_20260828_194432.json"),
    ("original",  "data/factorlib/all_factors_library_original_20260831_012324.json"),
]


def exprs_of(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors
    return [e for e in (it.get("factor_expression") or it.get("expression")
                        for it in items) if e]


def arr_of(nr) -> float:
    r = pd.Series(nr).astype(float).dropna()
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0) if len(r) else np.nan


def score(op, cands, panel, eval_window, label):
    book = op._strategy_batch(cands, {}, panel, eval_window, report=True)
    wide = _to_wide(book["prediction"]).reindex(index=panel.dates,
                                               columns=panel.instruments)
    wide = wide.where(panel.universe)
    blk = _ic_block(wide, label, eval_window)
    return {"n": len(cands),
            "ic": float(blk.get("ic", np.nan)),
            "rank_ic": float(blk.get("rank_ic", np.nan)),
            "arr": arr_of(book["metrics"].get("_net_return_series")),
            "net_ir": float(book["metrics"].get("net_ir", np.nan))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--out", default="data/results/report_count_control.json")
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args()

    theta = load_protocol(a.protocol)
    theta = replace(theta, benchmark_basis="estimated_total",
                    benchmark_construction="equal")   # dividend fix
    op = EvaluationOperator(theta)
    p_start, p_end, eval_window = op._windows(True)
    panel = op._panel(p_start, p_end)
    label = label_frame(panel, theta)
    print(f"protocol {theta.hash} | basis={theta.benchmark_basis} | window {eval_window}",
          flush=True)

    sig = {}
    for name, rel in LIBS:
        d = {}
        for e in exprs_of(ROOT / rel):
            try:
                d[e] = load_aligned_signal(e, panel)
            except Exception:
                pass
        sig[name] = d
        print(f"  {name}: {len(d)} signals", flush=True)

    out = {"protocol_hash": theta.hash, "eval_window": list(eval_window),
           "benchmark_basis": theta.benchmark_basis}

    # ---- (C) SOLO per-factor |Rank IC| -- count-free quality ----
    solo = {}
    for name, d in sig.items():
        vals = []
        for e, s in d.items():
            c = _cross_sectional_corr(s, label, "spearman")
            c = c.loc[eval_window[0]:eval_window[1]] if len(c) else c
            if len(c):
                vals.append(abs(float(c.mean())))
        vals = [v for v in vals if np.isfinite(v)]
        solo[name] = {"n": len(vals), "mean": float(np.mean(vals)),
                      "median": float(np.median(vals)),
                      "p75": float(np.percentile(vals, 75)),
                      "p90": float(np.percentile(vals, 90)),
                      "max": float(np.max(vals)), "values": vals}
        print(f"[solo] {name}: n={len(vals)} mean |RankIC|={np.mean(vals):.4f} "
              f"median={np.median(vals):.4f} p90={np.percentile(vals,90):.4f}", flush=True)
    out["solo_quality"] = solo
    Path(a.out).write_text(json.dumps(out, indent=2))

    # ---- (A) COUNT-MATCHED at n = |main_zoo| ----
    n_match = len(sig["main_zoo"])
    matched = {"n": n_match, "arms": {}}
    matched["arms"]["main_zoo"] = {"draws": [score(op, sig["main_zoo"], panel,
                                                   eval_window, label)]}
    print(f"[matched n={n_match}] main_zoo done", flush=True)
    for name in ("main_full", "original"):
        keys = list(sig[name])
        draws = []
        for k in range(a.draws):
            rng = np.random.default_rng(1000 + k)
            pick = rng.choice(len(keys), size=min(n_match, len(keys)), replace=False)
            sub = {keys[i]: sig[name][keys[i]] for i in pick}
            r = score(op, sub, panel, eval_window, label)
            draws.append(r)
            print(f"[matched n={n_match}] {name} draw {k}: rank_ic={r['rank_ic']:+.4f} "
                  f"arr={100*r['arr']:+.2f}%", flush=True)
        matched["arms"][name] = {"draws": draws,
            "rank_ic_mean": float(np.mean([d["rank_ic"] for d in draws])),
            "rank_ic_sd": float(np.std([d["rank_ic"] for d in draws])),
            "arr_mean": float(np.mean([d["arr"] for d in draws]))}
        Path(a.out).write_text(json.dumps({**out, "count_matched": matched}, indent=2))
    out["count_matched"] = matched

    # ---- (B) COUNT CURVE on main_full and original ----
    curve = {}
    for name in ("main_full", "original"):
        keys = list(sig[name])
        pts = []
        for n in (10, 20, 40, 56, 80, 120, len(keys)):
            if n > len(keys):
                continue
            rng = np.random.default_rng(7)
            pick = rng.choice(len(keys), size=n, replace=False)
            sub = {keys[i]: sig[name][keys[i]] for i in pick}
            r = score(op, sub, panel, eval_window, label)
            pts.append(r)
            print(f"[curve] {name} n={n}: rank_ic={r['rank_ic']:+.4f} "
                  f"arr={100*r['arr']:+.2f}%", flush=True)
        curve[name] = pts
        Path(a.out).write_text(json.dumps({**out, "count_curve": curve}, indent=2))
    out["count_curve"] = curve

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
