#!/usr/bin/env python
"""Post-mine analysis: direction diversity, operator coverage, and whether the
evolution operators are actually LEARNING.

Answers, from the run's own artifacts (ledger + factor library + log):

  A. Are the INITIAL and RESEED directions diverse?
  B. Are the operators used across all phases diverse, and how much of the
     library vocabulary is reached?
  C. Mutation -- for each of REFINE / ORTHOGONAL / ADMITTED-PUSH: does the child
     beat its parent? Compare parent vs child HYPOTHESIS and EXPRESSION.
  D. Crossover -- does the child inherit from BOTH parents and improve?

Measurement protocol (enforced, after three errors in one session):
  * population, not sample -- every trial in the ledger, admitted AND rejected
  * every statistic carries a CI (bootstrap, 2000 draws)
  * a null control is mandatory -- "better than chance" needs a chance model
  * two estimators must agree before a claim is made

Usage:  python scripts/qa_analyze_mine.py <EXPERIMENT_ID> [--json out.json]
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OP_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\s*\(")
FIELD_RE = re.compile(r"\$[a-z_]+")
WIN_RE = re.compile(r",\s*(\d+)\s*\)")

# Window summarizers are interchangeable plumbing; shape operators carry the
# mechanism. Reporting only the raw top-operator share conflates the two and
# reads as monoculture when the denominators simply all need an average.
SMOOTHERS = {"TS_MEAN", "TS_MEDIAN", "TS_SUM", "TS_STD", "TS_VAR", "TS_ZSCORE",
             "MEAN", "MEDIAN", "STD", "SMA", "EMA", "WMA"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def bootstrap_ci(vals, stat=np.mean, n=2000, seed=0):
    """Percentile bootstrap CI. Every statistic in this report carries one."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return (float("nan"),) * 3
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(stat(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ops_of(expr):
    return set(OP_RE.findall(expr or ""))


def mean_jaccard(sets):
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if (a | b)]
    return (statistics.mean(js), js) if js else (float("nan"), [])


def library_vocab():
    from quantaalpha.factors.coder import function_lib as FL
    return sorted(set(re.findall(r"^def ([A-Z][A-Z0-9_]*)\(",
                                 inspect.getsource(FL), re.M)))


def prompt_vocab():
    """Operators the CONSTRUCT prompt actually shows. Coverage must be scored
    against this, not against every symbol the library happens to define -- a
    model cannot select an operator it was never shown."""
    import yaml
    d = yaml.safe_load((ROOT / "quantaalpha/factors/prompts/prompts.yaml").read_text())
    desc = d["function_lib_description"]
    shown = sorted(set(re.findall(r"\*\*([A-Z][A-Z0-9_]{1,})\(", desc)))
    return shown or sorted(set(re.findall(r"\b([A-Z][A-Z0-9_]{1,})\s*\(", desc)))


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_ledger(exp):
    p = ROOT / f"data/results/ledger_{exp}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_library(exp):
    p = ROOT / f"data/factorlib/all_factors_library_{exp}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("factors", {})


def load_log(exp):
    p = ROOT / f"data/results/{exp}.log"
    return p.read_text(errors="ignore") if p.exists() else ""


def all_exprs(ledger):
    out = []
    for r in ledger:
        out += (r.get("factor_exprs") or []) + (r.get("rejected_exprs") or [])
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------
# A. direction diversity (initial + reseed)
# --------------------------------------------------------------------------
def directions(log):
    init = re.findall(r"Direction \d+: (.+)", log)
    reseed_blocks = re.findall(
        r"Generated (\d+) informed directions?|informed-direction", log)
    fired = len(re.findall(r"reseed(?:ing)?[^\n]*fired|_do_reseed[^\n]*reseed", log, re.I))
    failed = len(re.findall(r"Informed-direction generation returned nothing", log))
    return init, reseed_blocks, fired, failed


def direction_report(log):
    init, _, fired, failed = directions(log)
    # Reseed directions are logged the same way after a reseed; split on the
    # reseed marker so "initial" and "reseed" are not pooled.
    marker = log.find("_do_reseed")
    pre = log[:marker] if marker > 0 else log
    post = log[marker:] if marker > 0 else ""
    init_dirs = list(dict.fromkeys(re.findall(r"Direction \d+: (.+)", pre)))
    reseed_dirs = list(dict.fromkeys(re.findall(r"Direction \d+: (.+)", post)))

    def lexical_diversity(ds):
        """Distinct content words / total -- a crude but estimator-independent
        check that the directions are not restatements of one another."""
        if not ds:
            return float("nan"), 0
        toks = [set(re.findall(r"[a-z]{5,}", d.lower())) for d in ds]
        mj, _ = mean_jaccard(toks)
        return mj, len(ds)

    return {
        "initial": {"n": len(init_dirs),
                    "mean_pairwise_word_jaccard": lexical_diversity(init_dirs)[0],
                    "samples": [d[:110] for d in init_dirs[:3]]},
        "reseed": {"n": len(reseed_dirs),
                   "mean_pairwise_word_jaccard": lexical_diversity(reseed_dirs)[0],
                   "samples": [d[:110] for d in reseed_dirs[:3]]},
        "reseed_attempts_failed": failed,
    }


# --------------------------------------------------------------------------
# B. operator diversity + library coverage
# --------------------------------------------------------------------------
def operator_report(exprs, null_pool=None):
    lib, shown = library_vocab(), prompt_vocab()
    sets = [ops_of(e) for e in exprs]
    used = Counter()
    for s in sets:
        used.update(s)
    n = len(exprs)
    smooth_used = sum(1 for s in sets if s & SMOOTHERS)
    smooth_counts = Counter()
    shape_counts = Counter()
    for s in sets:
        smooth_counts.update(s & SMOOTHERS)
        shape_counts.update(s - SMOOTHERS)
    mj, _ = mean_jaccard(sets)
    shape_mj, _ = mean_jaccard([s - SMOOTHERS for s in sets])

    top = used.most_common(1)[0] if used else ("-", 0)
    top_smooth = smooth_counts.most_common(1)[0] if smooth_counts else ("-", 0)

    rep = {
        "n_expressions": n,
        "distinct_operators_used": len(used),
        "library_defines": len(lib),
        "prompt_exposes": len(shown),
        "coverage_vs_prompt_pct": 100.0 * len(set(used) & set(shown)) / max(1, len(shown)),
        "coverage_vs_library_pct": 100.0 * len(set(used) & set(lib)) / max(1, len(lib)),
        "unused_but_shown": sorted(set(shown) - set(used)),
        "top_operator": {"op": top[0], "share_pct": 100.0 * top[1] / max(1, n)},
        "smoother_slot": {
            "expressions_using_a_smoother_pct": 100.0 * smooth_used / max(1, n),
            "top_smoother": top_smooth[0],
            "top_smoother_share_of_slot_pct": 100.0 * top_smooth[1] / max(1, smooth_used),
        },
        "distinct_shape_operators": len(shape_counts),
        "mean_pairwise_jaccard_all_ops": mj,
        "mean_pairwise_jaccard_shape_only": shape_mj,
        "distinct_fields": len({f for e in exprs for f in FIELD_RE.findall(e)}),
        "distinct_windows": sorted({int(w) for e in exprs for w in WIN_RE.findall(e)}),
    }
    if null_pool:
        # NULL CONTROL: resample the same number of expressions from prior real
        # mines. "More diverse" only means anything against how this system
        # normally behaves.
        rng = np.random.default_rng(0)
        pool = [ops_of(e) for e in null_pool]
        null = []
        for _ in range(2000):
            idx = rng.choice(len(pool), min(n, len(pool)), replace=True)
            null.append(mean_jaccard([pool[i] for i in idx])[0])
        null = np.array([x for x in null if np.isfinite(x)])
        rep["null_control"] = {
            "pool_size": len(pool),
            "null_mean_jaccard": float(null.mean()),
            "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
            "p_null_le_observed": float((null <= mj).mean()),
            "verdict": ("more diverse than this system's history"
                        if mj < np.percentile(null, 2.5) else
                        "less diverse (monoculture)"
                        if mj > np.percentile(null, 97.5) else
                        "within historical range"),
        }
    return rep


# --------------------------------------------------------------------------
# C/D. learning: parent vs child, per operator
# --------------------------------------------------------------------------
# The factor library nests lineage under `metadata` and per-factor scores under
# `factor_metrics`. Reading them off the top level returns None for everything
# and the report then says "no learning" for a run that learned -- the same
# silent-empty failure mode as the broken parent links. Resolved once here.
def _meta(f, *keys, default=None):
    md = f.get("metadata") or {}
    for k in keys:
        if k in md:
            return md[k]
        if k in f:
            return f[k]
    return default


def _metric(rec, keys=("delta_mean", "t_nw", "rank_ic", "ic_mean")):
    """The factor's own score. Prefers the admission delta, falls back to the
    per-factor rank IC. Signed |IC| is used for magnitude comparisons because a
    reversal factor's edge is its |IC|, not its signed IC."""
    fm = (rec.get("factor_metrics") or {}) if isinstance(rec, dict) else {}
    for k in keys:
        v = rec.get(k, fm.get(k))
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    return None


def _abs_metric(rec):
    m = _metric(rec)
    return None if m is None else abs(m)


def learning_report(library):
    """Parent->child deltas per operator, from the factor library's lineage.

    The library is the only artifact carrying BOTH the lineage link and the
    per-factor metrics, so a broken parent link shows up here as an
    unresolvable id rather than as a silently dropped pair (which is how the
    lineage bug went unnoticed for 27 trajectories).
    """
    by_id = {}
    for fid, f in library.items():
        by_id[str(fid)] = f
        tid = _meta(f, "trajectory_id", default="")
        if tid:
            by_id[str(tid)] = f

    rows = defaultdict(list)
    unresolved = 0
    pairs = []
    for fid, f in library.items():
        phase = str(_meta(f, "evolution_phase", "phase", default="") or "").lower()
        op = str(_meta(f, "mutation_kind", "refine_target", "operator",
                       default="") or phase or "unknown").lower()
        parents = _meta(f, "parent_trajectory_ids", "parent_ids", default=[]) or []
        if not parents:
            continue
        pm, cm = [], _abs_metric(f)
        parent_exprs, parent_hyps = [], []
        for p in parents:
            pf = by_id.get(str(p))
            if pf is None:
                unresolved += 1
                continue
            parent_exprs.append((pf.get("factor_expression") or "")[:160])
            parent_hyps.append(str(_meta(pf, "hypothesis", default=""))[:200])
            m = _abs_metric(pf)
            if m is not None:
                pm.append(m)
        if not pm or cm is None:
            continue
        best_parent = max(pm)
        rows[op].append(cm - best_parent)
        c_expr = (f.get("factor_expression") or "")[:160]
        pairs.append({
            "op": op, "child": str(fid), "delta_vs_best_parent": cm - best_parent,
            "child_abs_metric": cm, "best_parent_abs_metric": best_parent,
            "child_expr": c_expr,
            "parent_exprs": parent_exprs,
            "child_hypothesis": str(_meta(f, "hypothesis", default=""))[:200],
            "parent_hypotheses": parent_hyps,
            # Did the operator change the CONSTRUCTION, or only a constant?
            # A refine that only edits a window is the failure mode measured in
            # [[qa-refine-only-edits-windows]], and it is invisible in the delta.
            "ops_added": sorted(ops_of(c_expr) - set().union(*[ops_of(e) for e in parent_exprs]) ) if parent_exprs else [],
            "ops_dropped": sorted(set().union(*[ops_of(e) for e in parent_exprs]) - ops_of(c_expr)) if parent_exprs else [],
        })

    out = {"unresolved_parent_links": unresolved, "by_operator": {}, "pairs": pairs}
    for op, deltas in sorted(rows.items()):
        m, lo, hi = bootstrap_ci(deltas)
        out["by_operator"][op] = {
            "n": len(deltas),
            "mean_delta_vs_best_parent": m,
            "ci95": [lo, hi],
            "frac_better_than_parent": float(np.mean([d > 0 for d in deltas])),
            # A claim of "learning" requires the CI to exclude zero. Anything
            # else is a direction, not a result.
            "beats_parent_significantly": bool(lo > 0),
        }
    return out


def crossover_inheritance(library):
    """Does the crossover child carry vocabulary distinctive to EACH parent?
    Re-runs the production gate offline over the finished library."""
    from quantaalpha.pipeline.evolution.operator_contract import check_crossover
    by_id = {}
    for fid, f in library.items():
        by_id[str(fid)] = f
        tid = _meta(f, "trajectory_id", default="")
        if tid:
            by_id[str(tid)] = f
    res = {"checked": 0, "recombined": 0, "single_parent": 0,
           "parents_indistinct": 0, "unresolved": 0, "examples": []}
    for fid, f in library.items():
        parents = _meta(f, "parent_trajectory_ids", "parent_ids", default=[]) or []
        if len(parents) < 2:
            continue
        pa, pb = by_id.get(str(parents[0])), by_id.get(str(parents[1]))
        if not pa or not pb:
            res["unresolved"] += 1
            continue
        r = check_crossover(f.get("factor_expression", ""),
                            [pa.get("factor_expression", "")],
                            [pb.get("factor_expression", "")])
        res["checked"] += 1
        res[r.reason if r.reason in res else "recombined"] = \
            res.get(r.reason, 0) + 1
        if len(res["examples"]) < 5:
            res["examples"].append({
                "child": (f.get("factor_expression") or "")[:140],
                "parent_a": (pa.get("factor_expression") or "")[:140],
                "parent_b": (pb.get("factor_expression") or "")[:140],
                "verdict": r.reason, "detail": r.detail,
            })
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_id")
    ap.add_argument("--json", default=None)
    ap.add_argument("--null-from", nargs="*", default=[],
                    help="prior experiment ids (or ledger paths) for the null control")
    a = ap.parse_args()

    ledger, library, log = (load_ledger(a.experiment_id),
                            load_library(a.experiment_id),
                            load_log(a.experiment_id))
    exprs = all_exprs(ledger)

    null_pool = []
    for src in a.null_from:
        p = Path(src)
        if not p.exists():
            p = ROOT / f"data/results/ledger_{src}.jsonl"
        if p.exists():
            null_pool += all_exprs([json.loads(l) for l in p.read_text().splitlines() if l.strip()])

    report = {
        "experiment_id": a.experiment_id,
        "batches_evaluated": len(ledger),
        "library_factors": len(library),
        "admitted_batches": sum(1 for r in ledger if r.get("admitted")),
        "A_directions": direction_report(log),
        "B_operators": operator_report(exprs, null_pool or None),
        "C_learning": learning_report(library),
        "D_crossover": crossover_inheritance(library),
    }

    print(json.dumps(report, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
