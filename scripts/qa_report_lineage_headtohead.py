#!/usr/bin/env python
"""Head-to-head learning comparison from BOTH mines' trajectory pools.

CORRECTION to an earlier analysis: the original branch DOES write a trajectory pool
-- at log/<experiment_id>/trajectory_pool.json inside its worktree, not under
data/results/ where main writes its own. Its 37 parent references all resolve within
that pool, so the original's lineage IS traceable and per-operator child-vs-parent is
measurable for both arms.

(The library's trajectory_id lives in a different id space than the pool's, because
generate_id() hashes a microsecond timestamp and is called once in factor_mining.py
and again in controller.py. That breaks library->pool joins, not pool-internal
lineage.)

Measured per mine, on each mine's OWN metrics (never cross-compared):

  LINEAGE   mutation / crossover children vs their parents, against the operator's
            chance baseline (1 parent -> 50%; 2 parents beat-both -> ~33%)
  L1        per-round median flat-or-rising (no anti-learning)
  L2        cumulative kept-set mean rises -- on EVERY metric the pool carries
  L3        cumulative best-so-far rises   -- on EVERY metric the pool carries

    python scripts/qa_report_lineage_headtohead.py --out data/results/report_headtohead.json
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Location of the ORIGINAL-branch worktree (the paper baseline). Defaults to a sibling
# directory named qa_orig_mine; override with QA_ORIG_DIR when it lives elsewhere.
ORIG_DIR = Path(os.environ.get("QA_ORIG_DIR", str(ROOT.parent / "qa_orig_mine")))

POOLS = {
    "main": ROOT / "data/results/trajectory_pool_meanvar_20260828_194432.json",
    "original": ORIG_DIR / ("log/"
                     "2026-08-31_05-23-28-565698/trajectory_pool.json"),
}
RNG = np.random.default_rng(23)

LOWER_BETTER = {"max_drawdown", "rho_max", "rho_within", "turnover", "decay_slope",
                "is_oos_gap", "e_turnover", "e_overfit", "e_decay"}
ABS_METRICS = {"IC", "RankIC", "ICIR", "RankICIR", "ic", "rank_ic", "icir",
               "rank_icir", "rank_ic_neutral", "t_nw"}
# metrics both pools carry (each mine still scored only against itself)
CORE = ["RankIC", "IC", "RankICIR", "ICIR", "annualized_return",
        "information_ratio", "max_drawdown"]


def orient(name, v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if name in ABS_METRICS:
        x = abs(x)
    if name in LOWER_BETTER:
        x = -x          # negated so "up" always means "better"
    return x


def trend(y):
    from scipy.stats import linregress, spearmanr
    y = np.asarray(y, float); x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    if m.sum() < 6 or len(set(y[m].tolist())) < 3:
        return None
    xx, yy = x[m], y[m]
    lr = linregress(xx, yy)
    null = [abs(float(linregress(RNG.permutation(xx), yy).slope)) for _ in range(400)]
    n95 = float(np.percentile(null, 95))
    return {"slope": float(lr.slope), "p": float(lr.pvalue),
            "rho": float(spearmanr(xx, yy).statistic),
            "beats_null": bool(abs(lr.slope) > n95),
            "first": float(yy[0]), "last": float(yy[-1]), "n": int(m.sum())}


def load(pool_path: Path):
    d = json.loads(pool_path.read_text())
    tr = d.get("trajectories", d)
    nodes = {}
    for tid, t in tr.items():
        if not isinstance(t, dict):
            continue
        bm = t.get("backtest_metrics") or {}
        nodes[tid] = {
            "id": tid,
            "round": t.get("round_idx"),
            "phase": t.get("phase"),
            "parents": list(t.get("parent_ids") or []),
            "metrics": bm,
            "admitted": bool(bm.get("admitted")) if "admitted" in bm else None,
        }
    return nodes


def sign_test(deltas):
    from scipy.stats import binomtest
    d = [x for x in deltas if np.isfinite(x) and x != 0]
    if len(d) < 5:
        return None
    w = sum(1 for x in d if x > 0)
    return {"n": len(d), "wins": w, "win_rate": w / len(d),
            "p_gt_chance": float(binomtest(w, len(d), 0.5, alternative="greater").pvalue),
            "mean_delta": float(np.mean(d))}


def analyse(nodes, metric_keys):
    out = {"n_nodes": len(nodes)}
    resolvable = sum(1 for n in nodes.values() for p in n["parents"] if p in nodes)
    out["resolvable_parent_links"] = resolvable
    out["lineage_available"] = resolvable > 0

    # ---- lineage per operator, per metric ----
    ops = {}
    for phase in ("mutation", "crossover"):
        per_metric = {}
        for mk in metric_keys:
            deltas, beat_all = [], []
            for tid, n in nodes.items():
                if n["phase"] != phase:
                    continue
                q = orient(mk, n["metrics"].get(mk))
                if q is None:
                    continue
                pq = [orient(mk, nodes[p]["metrics"].get(mk))
                      for p in n["parents"] if p in nodes]
                pq = [x for x in pq if x is not None]
                if not pq:
                    continue
                deltas.append(q - max(pq))
                beat_all.append(all(q > x for x in pq))
            st = sign_test(deltas)
            if st:
                per_metric[mk] = {**st, "beat_all_rate": float(np.mean(beat_all)),
                                  "chance_beat_all": 0.5 if phase == "mutation" else 1/3}
        if per_metric:
            ops[phase] = per_metric
    out["lineage"] = ops

    # ---- L1 / L2 / L3 per metric ----
    rounds = sorted({n["round"] for n in nodes.values() if n["round"] is not None})
    L = {}
    for mk in metric_keys:
        med, cum_mean, cum_best, acc, best = [], [], [], [], -np.inf
        for rd in rounds:
            qs = [orient(mk, n["metrics"].get(mk))
                  for n in nodes.values() if n["round"] == rd]
            qs = [q for q in qs if q is not None]
            if not qs:
                med.append(np.nan); cum_mean.append(np.nan); cum_best.append(np.nan); continue
            med.append(float(np.median(qs)))
            acc += qs
            best = max([best] + qs)
            cum_mean.append(float(np.mean(acc)))
            cum_best.append(float(best))
        L[mk] = {"L1_median": trend(med), "L2_cum_mean": trend(cum_mean),
                 "L3_cum_best": trend(cum_best)}
    out["rounds"] = rounds
    out["definition"] = L

    def tally(key):
        r = f = fl = 0; rise=[]; fall=[]
        for mk, v in L.items():
            t = v.get(key)
            if not t:
                continue
            sig = t["p"] < 0.05 and t["beats_null"]
            if sig and t["slope"] > 0: r += 1; rise.append(mk)
            elif sig and t["slope"] < 0: f += 1; fall.append(mk)
            else: fl += 1
        return {"rises": r, "flat": fl, "declines": f, "rising": rise, "falling": fall}

    out["verdict"] = {"L1_no_anti_learning": tally("L1_median"),
                      "L2_zoo_mean": tally("L2_cum_mean"),
                      "L3_best_bar": tally("L3_cum_best")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/report_headtohead.json")
    a = ap.parse_args()
    res = {}
    for mine, path in POOLS.items():
        if not path.exists():
            print(f"[{mine}] MISSING {path}"); continue
        nodes = load(path)
        keys = [k for k in CORE if any(k in n["metrics"] for n in nodes.values())]
        r = analyse(nodes, keys)
        r["pool"] = str(path); r["metrics_used"] = keys
        res[mine] = r
        print(f"\n===== {mine.upper()} =====")
        print(f"  pool: {path.name}")
        print(f"  {r['n_nodes']} trajectories | {r['resolvable_parent_links']} "
              f"resolvable parent links | metrics: {keys}")
        for phase, pm in (r.get("lineage") or {}).items():
            print(f"  -- {phase} --")
            for mk, v in pm.items():
                print(f"     {mk:<20} beat-parent {v['win_rate']*100:5.1f}% "
                      f"(n={v['n']:>3}, p={v['p_gt_chance']:.3f}) | beat-all "
                      f"{v['beat_all_rate']*100:5.1f}% vs chance "
                      f"{v['chance_beat_all']*100:.0f}%")
        for L, v in r["verdict"].items():
            print(f"  {L}: rises={v['rises']} flat={v['flat']} declines={v['declines']}"
                  + (f" | falling: {v['falling']}" if v['falling'] else ""))
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
