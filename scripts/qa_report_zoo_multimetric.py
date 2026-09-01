#!/usr/bin/env python
"""(L2) zoo mean rises and (L3) best bar rises -- across the FULL quality vector.

Judging the zoo on rank IC alone is too narrow: a gate that trades a little IC for
breadth, robustness or lower cost is doing its job, and a single-metric read scores
that as "flat". The learning definition is therefore evaluated on every dimension
the protocol actually measures.

Two families, both as the zoo GROWS (x = admission index / zoo size):

  BOOK-LEVEL (the accumulated zoo, from each admission's ledger row)
      U, delta_t, and the seven protocol dimension scores
      (effectiveness, arr, stability, turnover, diversity, overfit, decay),
      plus the book's own metrics (rank_ic, icir, rank_icir, ic_pos_frac,
      persistence_ratio, decay_slope, is_oos_gap, rho_max, rho_within).

  FACTOR-LEVEL (the admitted factors themselves, from their tearsheets)
      t_nw, rank_ic_neutral, rank_icir, monotonicity, q_spread, ls_sharpe,
      ic_crash, ic_rally, ic_pos_frac, capacity_cny, turnover_solo, exposure_size.

For each: (L2) mean of everything admitted so far, (L3) best-so-far, each with an
OLS slope, Spearman rho and a 400-draw permutation null. Metrics where LOWER is
better are sign-flipped so "rises" always means "improves".

    python scripts/qa_report_zoo_multimetric.py --out data/results/report_zoo_multimetric.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/results/ledger_meanvar_20260828_194432.jsonl"
RNG = np.random.default_rng(23)

# lower-is-better -> negate so "up" always means "better"
LOWER_BETTER = {"turnover", "overfit", "decay", "rho_max", "rho_within",
                "decay_slope", "is_oos_gap", "turnover_solo", "exposure_size",
                "ls_mdd", "ic_breakeven_book", "ic_breakeven_solo"}
# metrics whose decline is NOT a quality loss but a property of a growing book:
# capacity falls as the book holds more names; crash/rally IC are regime splits of a
# signal whose mean is what the gate targets; persistence is measured per-factor and
# dilutes as breadth grows. Flagged rather than scored.
CONTEXT_ONLY = {"capacity_cny", "ic_crash", "ic_rally", "persistence_ratio"}
# magnitude matters, not sign
ABS_METRICS = {"rank_ic", "rank_ic_neutral", "t_nw", "ic", "icir", "rank_icir",
               "ic_crash", "ic_rally", "delta_t"}


def orient(name: str, v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if name in ABS_METRICS:
        x = abs(x)
    if name in LOWER_BETTER:
        x = -x
    return x


def trend(y):
    from scipy.stats import linregress, spearmanr
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    if m.sum() < 6 or len(set(y[m].tolist())) < 3:
        return None
    xx, yy = x[m], y[m]
    lr = linregress(xx, yy)
    null = [abs(float(linregress(RNG.permutation(xx), yy).slope)) for _ in range(400)]
    n95 = float(np.percentile(null, 95))
    return {"n": int(m.sum()), "slope": float(lr.slope), "p": float(lr.pvalue),
            "rho": float(spearmanr(xx, yy).statistic), "null_95": n95,
            "beats_null": bool(abs(lr.slope) > n95),
            "first": float(yy[0]), "last": float(yy[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/report_zoo_multimetric.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    adm = [r for r in rows if r.get("admitted")]
    adm.sort(key=lambda r: (r.get("zoo_size") if r.get("zoo_size") is not None else 0))
    print(f"{len(adm)} admissions from {LEDGER.name}")

    # ---------- BOOK-LEVEL series, one point per admission ----------
    book_series: dict[str, list] = {}
    def push(k, v):
        x = orient(k, v)
        book_series.setdefault(k, []).append(x if x is not None else np.nan)

    for r in adm:
        push("U", r.get("U"))
        push("delta_t", r.get("delta_t"))
        for k, v in (r.get("e") or {}).items():
            push(f"e_{k}", v)
        m = r.get("metrics") or {}
        for k in ("rank_ic", "ic", "icir", "rank_icir", "ic_pos_frac",
                  "persistence_ratio", "decay_slope", "is_oos_gap",
                  "rho_max", "rho_within", "delta_net_ir", "delta_net_arr"):
            if k in m:
                push(k, m[k])

    # ---------- FACTOR-LEVEL: each admitted factor's own tearsheet ----------
    fac_vals: dict[str, list] = {}
    for r in adm:
        ts = r.get("factor_tearsheets") or {}
        best = None
        for t in (ts.values() if isinstance(ts, dict) else []):
            if isinstance(t, dict):
                best = t
                break
        if not best:
            continue
        for k in ("t_nw", "rank_ic_neutral", "rank_icir", "monotonicity", "q_spread",
                  "ls_sharpe", "ic_crash", "ic_rally", "ic_pos_frac", "capacity_cny",
                  "turnover_solo", "exposure_size", "ic_breakeven_solo"):
            if k in best:
                fac_vals.setdefault(k, []).append(orient(k, best[k]))

    out = {"n_admissions": len(adm), "book": {}, "factor": {}}

    def summarise(series: dict, bucket: str):
        for k, vals in series.items():
            v = np.array([np.nan if x is None else x for x in vals], float)
            if np.isfinite(v).sum() < 6:
                continue
            run_mean, run_best, acc, best = [], [], [], -np.inf
            for x in v:
                if np.isfinite(x):
                    acc.append(x)
                    best = max(best, x)
                run_mean.append(float(np.mean(acc)) if acc else np.nan)
                run_best.append(float(best) if np.isfinite(best) else np.nan)
            out[bucket][k] = {
                "context_only": k in CONTEXT_ONLY,
                "n": int(np.isfinite(v).sum()),
                "L2_cum_mean": trend(run_mean),
                "L3_cum_best": trend(run_best),
                "raw_first": float(v[np.isfinite(v)][0]),
                "raw_last": float(v[np.isfinite(v)][-1]),
                "oriented": (k in LOWER_BETTER and "lower-is-better (negated)")
                            or (k in ABS_METRICS and "magnitude") or "raw",
            }

    summarise(book_series, "book")
    summarise(fac_vals, "factor")

    # ---------- verdict tally ----------
    def tally(bucket, key):
        """Values are already oriented at ingest (lower-is-better metrics were
        negated), so a positive slope always means "improving" -- no second
        sign correction here."""
        rise = flat = fall = 0
        rising, falling = [], []
        for k, v in out[bucket].items():
            if k in CONTEXT_ONLY:
                continue
            t = v.get(key)
            if not t:
                continue
            sig = t["p"] < 0.05 and t["beats_null"]
            if sig and t["slope"] > 0:
                rise += 1; rising.append(k)
            elif sig and t["slope"] < 0:
                fall += 1; falling.append(k)
            else:
                flat += 1
        return {"rises": rise, "flat": flat, "declines": fall,
                "rising": rising, "falling": falling}

    out["verdict"] = {
        "L2_zoo_mean": {"book": tally("book", "L2_cum_mean"),
                        "factor": tally("factor", "L2_cum_mean")},
        "L3_best_bar": {"book": tally("book", "L3_cum_best"),
                        "factor": tally("factor", "L3_cum_best")},
    }

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    for bucket in ("book", "factor"):
        print(f"\n===== {bucket.upper()}-LEVEL =====")
        print(f"{'metric':<22} {'L2 mean':>26}   {'L3 best':>26}")
        for k, v in sorted(out[bucket].items()):
            def fmt(t):
                if not t:
                    return f"{'--':>26}"
                mark = "RISES" if (t["slope"] > 0 and t["p"] < .05 and t["beats_null"]) \
                    else ("FALLS" if (t["slope"] < 0 and t["p"] < .05 and t["beats_null"]) else "flat ")
                return f"{mark} {t['slope']:+.2e} p={t['p']:.3f}"
            print(f"{k:<22} {fmt(v['L2_cum_mean']):>26}   {fmt(v['L3_cum_best']):>26}")
    print("\nVERDICT:", json.dumps(out["verdict"], indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
