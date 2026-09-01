#!/usr/bin/env python
"""DO MUTATION, CROSSOVER AND RESEED MAKE THE FACTORS BETTER?

The question is per-OPERATOR and WITHIN each library, so absolute levels are
irrelevant: each mine is judged against itself. Cross-library differences in cost
model, split era or stored metrics therefore do not enter -- only whether an
operator's children improve on the parents that operator was given, and whether the
learning definition is satisfied round over round.

Learning definition (fixed in advance):
  (L1) no anti-learning -- per-round median quality flat or rising, never declining
  (L2) the zoo improves -- accumulated kept-set mean rises
  (L3) the bar rises    -- cumulative best-so-far increases

Per operator (mutation / crossover / reseed-origin) we report:
  * child vs parent win rate, against the operator's own chance baseline
    (1 parent -> 50%; 2 parents, beat BOTH -> ~33%, beat EITHER -> ~67%)
  * the paired child-minus-parent delta with a sign test and a Wilcoxon test
  * per-round trend of each operator's output

Quality is each library's OWN stored per-factor metric (main: neutralised solo
rank IC from its tearsheets; original: its Qlib Rank IC) -- the comparison is
within-library, so the two never share an axis.

    python scripts/qa_report_operator_learning.py --out data/results/report_operator_learning.json
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

RNG = np.random.default_rng(23)

ROOT = Path(__file__).resolve().parents[1]
LIBS = {
    "main": ("data/factorlib/all_factors_library_meanvar_20260828_194432.json",
             "data/results/trajectory_pool_meanvar_20260828_194432.json"),
    "original": ("data/factorlib/all_factors_library_original_20260831_012324.json", None),
}


def quality(it: dict, is_main: bool):
    br = it.get("backtest_results") or {}
    if is_main:
        ts = br.get("factor_tearsheets") or {}
        if isinstance(ts, dict) and ts:
            t = list(ts.values())[0]
            if isinstance(t, dict) and t.get("rank_ic_neutral") is not None:
                return abs(float(t["rank_ic_neutral"]))
        v = br.get("rank_ic")
        return abs(float(v)) if v is not None else None
    v = br.get("Rank IC")
    try:
        return abs(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_nodes(lib_path: Path, pool_path: Path | None, is_main: bool):
    """trajectory_id -> {round, phase, parents, q, admitted}. Uses the pool for
    main (its parent_ids all resolve); the library's embedded metadata otherwise."""
    lib = json.loads(lib_path.read_text())["factors"]
    items = list(lib.values()) if isinstance(lib, dict) else lib
    q_by_tid, adm, meta = {}, {}, {}
    for it in items:
        md = it.get("metadata") or {}
        tid = md.get("trajectory_id")
        if not tid:
            continue
        q = quality(it, is_main)
        if q is not None:
            q_by_tid[tid] = max(q_by_tid.get(tid, 0.0), q)
        adm[tid] = adm.get(tid, False) or bool(it.get("admitted"))
        meta.setdefault(tid, {"round": md.get("round_number"),
                              "phase": md.get("evolution_phase"),
                              "parents": list(md.get("parent_trajectory_ids") or [])})
    nodes = {}
    if pool_path and Path(pool_path).exists():
        pool = json.loads(Path(pool_path).read_text())["trajectories"]
        for tid, t in pool.items():
            if not isinstance(t, dict):
                continue
            bm = t.get("backtest_metrics") or {}
            q = q_by_tid.get(tid)
            if q is None and bm.get("rank_ic_neutral") is not None:
                q = abs(float(bm["rank_ic_neutral"]))
            nodes[tid] = {"round": t.get("round_idx"), "phase": t.get("phase"),
                          "parents": list(t.get("parent_ids") or []),
                          "q": q, "admitted": adm.get(tid, False),
                          "verdict": bm.get("verdict")}
    else:
        for tid, m in meta.items():
            nodes[tid] = {**m, "q": q_by_tid.get(tid), "admitted": adm.get(tid, False),
                          "verdict": None}
    return nodes


def sign_test(deltas):
    """P(child > parent) with a binomial sign test against 50%."""
    from scipy.stats import binomtest, wilcoxon
    d = [x for x in deltas if np.isfinite(x) and x != 0]
    if len(d) < 5:
        return None
    wins = sum(1 for x in d if x > 0)
    bt = binomtest(wins, len(d), 0.5, alternative="greater")
    try:
        w = wilcoxon(d, alternative="greater")
        wp = float(w.pvalue)
    except Exception:
        wp = None
    return {"n": len(d), "wins": wins, "win_rate": wins / len(d),
            "p_binom_gt_half": float(bt.pvalue), "p_wilcoxon": wp,
            "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d))}


def analyse(nodes: dict):
    out = {"n_nodes": len(nodes)}
    resolvable = sum(1 for n in nodes.values()
                     for p in n["parents"] if p in nodes)
    out["resolvable_parent_links"] = resolvable
    out["lineage_available"] = resolvable > 0
    if resolvable == 0:
        out["lineage_note"] = (
            "parent_trajectory_ids are RECORDED but resolve to nothing: the original "
            "branch mints a trajectory id from a microsecond timestamp in two "
            "independent places (factor_mining.py:159 writes the library's "
            "trajectory_id; controller.py:737 mints the id that becomes a child's "
            "parent_trajectory_ids), so the same logical trajectory carries two "
            "different ids and the two id spaces are disjoint (38 parent refs vs 62 "
            "trajectory ids, 0 overlap, in the untrimmed 184-factor library). "
            "Per-operator child-vs-parent cannot be computed; the round-over-round "
            "learning definition (L1/L2/L3) still can, and is reported below.")

    # ---- per-operator child-vs-parent (needs resolvable lineage) ----
    ops = {}
    if resolvable == 0:
        ops = None
    for phase in (("mutation", "crossover") if resolvable else ()):
        deltas, both, either, rows = [], [], [], []
        for tid, n in nodes.items():
            if n["phase"] != phase or n["q"] is None:
                continue
            pq = [nodes[p]["q"] for p in n["parents"]
                  if p in nodes and nodes[p]["q"] is not None]
            if not pq:
                continue
            deltas.append(n["q"] - max(pq))
            both.append(all(n["q"] > x for x in pq))
            either.append(any(n["q"] > x for x in pq))
            rows.append({"tid": tid, "round": n["round"], "q": n["q"],
                         "parents_q": pq, "admitted": n["admitted"]})
        if not rows:
            continue
        ops[phase] = {
            "n": len(rows),
            "vs_best_parent": sign_test(deltas),
            "beat_all_parents_rate": float(np.mean(both)),
            "beat_any_parent_rate": float(np.mean(either)),
            "chance_beat_all": 0.5 if phase == "mutation" else 1/3,
            "chance_beat_any": 0.5 if phase == "mutation" else 2/3,
            "admitted_rate": float(np.mean([r["admitted"] for r in rows])),
        }
        # fix-from-failure: parent not admitted -> child admitted
        fixes = [(nodes[p]["admitted"], n["admitted"])
                 for tid, n in nodes.items() if n["phase"] == phase
                 for p in n["parents"] if p in nodes]
        failed = [(a, b) for a, b in fixes if not a]
        if failed:
            ops[phase]["fix_from_failure_rate"] = float(np.mean([b for _, b in failed]))
            ops[phase]["fix_from_failure_n"] = len(failed)
    out["operators"] = ops

    # ---- per-PHASE round-over-round output quality (no lineage needed) ----
    # "are the factors this operator produces getting better over the run?"
    from scipy.stats import linregress, spearmanr
    phase_trend = {}
    for phase in ("original", "mutation", "crossover"):
        pts = [(n["round"], n["q"]) for n in nodes.values()
               if n["phase"] == phase and n["q"] is not None and n["round"] is not None]
        if len(pts) < 6 or len({p[0] for p in pts}) < 3:
            continue
        xs = np.array([p[0] for p in pts], float); ys = np.array([p[1] for p in pts], float)
        lr = linregress(xs, ys)
        null = [abs(float(linregress(RNG.permutation(xs), ys).slope)) for _ in range(400)]
        rounds_p = sorted({p[0] for p in pts})
        half = len(rounds_p) // 2
        early = [y for x, y in pts if x in rounds_p[:half]]
        late = [y for x, y in pts if x in rounds_p[half:]]
        from scipy.stats import mannwhitneyu
        mw = mannwhitneyu(late, early, alternative="greater") if early and late else None
        phase_trend[phase] = {
            "n": len(pts), "slope": float(lr.slope), "p": float(lr.pvalue),
            "rho": float(spearmanr(xs, ys).statistic),
            "null_95": float(np.percentile(null, 95)),
            "beats_null": bool(abs(lr.slope) > np.percentile(null, 95)),
            "early_mean": float(np.mean(early)) if early else None,
            "late_mean": float(np.mean(late)) if late else None,
            "mw_p_late_gt_early": float(mw.pvalue) if mw else None,
        }
    out["phase_output_trend"] = phase_trend

    # ---- L1/L2/L3 per round ----
    by_round = defaultdict(list)
    for n in nodes.values():
        if n["round"] is not None and n["q"] is not None:
            by_round[n["round"]].append(n)
    rounds = sorted(by_round)
    med, mean_, best_c, zoo_c, kept = [], [], [], [], []
    best = 0.0
    for rd in rounds:
        qs = [n["q"] for n in by_round[rd]]
        med.append(float(np.median(qs))); mean_.append(float(np.mean(qs)))
        best = max([best] + qs); best_c.append(float(best))
        kept += [n["q"] for n in by_round[rd] if n["admitted"]]
        zoo_c.append(float(np.mean(kept)) if kept else np.nan)

    def slope(y):
        from scipy.stats import linregress, spearmanr
        yy = np.asarray(y, float); xx = np.asarray(rounds, float)
        m = np.isfinite(yy)
        if m.sum() < 3:
            return None
        lr = linregress(xx[m], yy[m])
        return {"slope": float(lr.slope), "p": float(lr.pvalue),
                "rho": float(spearmanr(xx[m], yy[m]).statistic)}

    out["rounds"] = rounds
    out["median_per_round"] = med
    out["mean_per_round"] = mean_
    out["cum_best"] = best_c
    out["cum_zoo_mean"] = zoo_c
    out["L1_no_anti_learning"] = {"trend": slope(med),
                                  "verdict": None}
    t = out["L1_no_anti_learning"]["trend"]
    out["L1_no_anti_learning"]["verdict"] = (
        "PASS (flat or rising)" if t and (t["slope"] >= 0 or t["p"] > 0.05)
        else "FAIL (declining)" if t else "insufficient rounds")
    t2 = slope(zoo_c)
    out["L2_zoo_improves"] = {"trend": t2,
        "verdict": ("PASS (rising)" if t2 and t2["slope"] > 0 and t2["p"] < 0.05
                    else "flat" if t2 else "no zoo (no gate)")}
    t3 = slope(best_c)
    out["L3_bar_rises"] = {"trend": t3, "first": best_c[0] if best_c else None,
        "last": best_c[-1] if best_c else None,
        "verdict": ("PASS (rising)" if best_c and best_c[-1] > best_c[0] else "flat")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/report_operator_learning.json")
    a = ap.parse_args()
    out = {"question": "do mutation / crossover / reseed make the factors better?",
           "note": ("within-library comparison only -- each mine is judged against "
                    "itself, so differing cost models, splits and stored metrics "
                    "across the two libraries do not enter"),
           "mines": {}}
    for mine, (lib, pool) in LIBS.items():
        nodes = load_nodes(ROOT / lib, ROOT / pool if pool else None, mine == "main")
        res = analyse(nodes)
        out["mines"][mine] = res
        print(f"=== {mine} ===")
        print(f"  nodes {res['n_nodes']}, resolvable parent links "
              f"{res['resolvable_parent_links']}")
        if not res.get("lineage_available"):
            print("  LINEAGE NOT RESOLVABLE (ids minted twice; see lineage_note) --"
                  " child-vs-parent cannot be computed; round-over-round below")
        for phase, o in (res.get("operators") or {}).items():
            v = o["vs_best_parent"]
            print(f"  {phase}: n={o['n']} beat-best-parent "
                  f"{v['win_rate']*100:.0f}% (p={v['p_binom_gt_half']:.3f}) | "
                  f"beat-all {o['beat_all_parents_rate']*100:.0f}% "
                  f"(chance {o['chance_beat_all']*100:.0f}%) | "
                  f"admitted {o['admitted_rate']*100:.0f}%"
                  + (f" | fix-from-failure {o.get('fix_from_failure_rate',0)*100:.0f}%"
                     if o.get("fix_from_failure_n") else ""))
        for phase, t in (res.get("phase_output_trend") or {}).items():
            print(f"  [{phase} output] n={t['n']} slope {t['slope']:+.2e} "
                  f"p={t['p']:.3f} beats_null={t['beats_null']} | early "
                  f"{t['early_mean']:.4f} -> late {t['late_mean']:.4f} "
                  f"(MW p={t['mw_p_late_gt_early']:.3f})")
        for k in ("L1_no_anti_learning", "L2_zoo_improves", "L3_bar_rises"):
            print(f"  {k}: {res[k]['verdict']}"
                  + (f"  (slope {res[k]['trend']['slope']:+.2e}, p={res[k]['trend']['p']:.3f})"
                     if res[k].get("trend") else ""))
        print()
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
