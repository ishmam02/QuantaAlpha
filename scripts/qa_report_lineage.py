#!/usr/bin/env python
"""Extract trajectory-level lineage from two factor libraries for the report's
circular trajectory graphs.

Both libraries carry, per factor in `metadata`: `trajectory_id`,
`parent_trajectory_ids`, `round_number`, `evolution_phase`.  Main (fph=1) has one
factor per trajectory; original (fph=3) has ~3 factors per trajectory, so factors
are grouped by `trajectory_id` to give trajectory-level nodes for both mines.

Admission: main has a top-level `admitted` flag (the gate); the original branch
has no admission system, so every original factor is "kept" (admitted=True).

Quality (for the beat-parent highlight, per-mine native metric):
  main  -> |rank_ic_neutral| from factor_tearsheets (fallback |U|, |RankIC|)
  orig  -> |Rank IC| (the Qlib backtest IC the original branch records)

Outputs:
  data/results/report_lineage_main.json
  data/results/report_lineage_original.json
each: {is_main, n_nodes, n_edges, n_orphan_parents, nodes:{id:...}, edges:[...]}
node: {id, round, phase, parents, admitted, n_factors, best_quality, mean_quality, beat_parent}
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Location of the ORIGINAL-branch worktree (the paper baseline). Defaults to a sibling
# directory named qa_orig_mine; override with QA_ORIG_DIR when it lives elsewhere.
ORIG_DIR = Path(os.environ.get("QA_ORIG_DIR", str(ROOT.parent / "qa_orig_mine")))

MAIN_LIB = ROOT / "data/factorlib/all_factors_library_meanvar_20260828_194432.json"
ORIG_LIB = ORIG_DIR / ("data/factorlib/all_factors_library_original_20260831_012324.json")
OUT_MAIN = ROOT / "data/results/report_lineage_main.json"
OUT_ORIG = ROOT / "data/results/report_lineage_original.json"


def quality(it: dict, is_main: bool) -> float:
    br = it.get("backtest_results") or {}
    if is_main:
        ts = br.get("factor_tearsheets") or {}
        if isinstance(ts, dict) and ts:
            t = list(ts.values())[0]
            if isinstance(t, dict) and t.get("rank_ic_neutral") is not None:
                return abs(float(t["rank_ic_neutral"]))
        for k in ("U", "RankIC", "Rank IC"):
            if br.get(k) is not None:
                try:
                    return abs(float(br[k]))
                except (TypeError, ValueError):
                    pass
        return 0.0
    else:
        v = br.get("Rank IC")
        try:
            return abs(float(v)) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0


def extract(path: Path, is_main: bool, out_path: Path) -> dict:
    d = json.loads(path.read_text())
    factors = d["factors"]
    items = list(factors.values()) if isinstance(factors, dict) else factors
    by_traj: dict[str, list] = defaultdict(list)
    for it in items:
        tid = (it.get("metadata") or {}).get("trajectory_id")
        if tid:
            by_traj[tid].append(it)

    nodes: dict[str, dict] = {}
    for tid, fs in by_traj.items():
        md = fs[0].get("metadata") or {}
        parents = md.get("parent_trajectory_ids") or []
        if is_main:
            admitted = any(bool(f.get("admitted")) for f in fs)
        else:
            admitted = True  # no gate on the original branch
        qs = [quality(f, is_main) for f in fs]
        nodes[tid] = {
            "id": tid,
            "round": md.get("round_number"),
            "phase": md.get("evolution_phase"),
            "parents": parents,
            "admitted": admitted,
            "n_factors": len(fs),
            "best_quality": max(qs) if qs else 0.0,
            "mean_quality": (sum(qs) / len(qs)) if qs else 0.0,
        }

    # beat_parent: child best_quality > best parent best_quality
    for tid, n in nodes.items():
        pq = [nodes[p]["best_quality"] for p in n["parents"] if p in nodes]
        n["beat_parent"] = bool(pq and n["best_quality"] > max(pq))

    edges = []
    orphan = 0
    for tid, n in nodes.items():
        for p in n["parents"]:
            if p in nodes:
                edges.append({"parent": p, "child": tid})
            else:
                orphan += 1

    out = {
        "is_main": is_main,
        "library": str(path),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_orphan_parents": orphan,
        "nodes": nodes,
        "edges": edges,
    }
    out_path.write_text(json.dumps(out, indent=2))
    n_adm = sum(1 for n in nodes.values() if n["admitted"])
    n_beat = sum(1 for n in nodes.values() if n["beat_parent"])
    print(f"{out_path.name}: {len(nodes)} nodes ({n_adm} admitted, {n_beat} beat-parent), "
          f"{len(edges)} edges, {orphan} orphan parents")
    return out


def extract_main_from_pool(pool_path: Path, lib_path: Path, out_path: Path) -> dict:
    """Main's lineage from the TRAJECTORY POOL -- 161 trajectories with fully
    resolvable `parent_ids` (the library holds only the 150 surviving factors, so
    19 of its parent references point at trajectories evicted from the library).
    Admission is joined in from the library (factor.metadata.trajectory_id ->
    factor.admitted)."""
    pool = json.loads(pool_path.read_text())["trajectories"]
    lib = json.loads(lib_path.read_text())["factors"]
    lib_items = list(lib.values()) if isinstance(lib, dict) else lib
    admitted_tids, quality_tid = set(), {}
    for it in lib_items:
        tid = (it.get("metadata") or {}).get("trajectory_id")
        if not tid:
            continue
        if it.get("admitted"):
            admitted_tids.add(tid)
        q = quality(it, True)
        quality_tid[tid] = max(quality_tid.get(tid, 0.0), q)

    nodes = {}
    for tid, t in pool.items():
        if not isinstance(t, dict):
            continue
        bm = t.get("backtest_metrics") or {}
        q = quality_tid.get(tid)
        if q is None:
            v = bm.get("rank_ic_neutral")
            try:
                q = abs(float(v)) if v is not None else 0.0
            except (TypeError, ValueError):
                q = 0.0
        nodes[tid] = {
            "id": tid,
            "round": t.get("round_idx"),
            "phase": t.get("phase"),
            "parents": list(t.get("parent_ids") or []),
            "admitted": tid in admitted_tids,
            "n_factors": len(t.get("factors") or []),
            "best_quality": float(q),
            "mean_quality": float(q),
            "U": bm.get("U"),
        }
    for tid, n in nodes.items():
        pq = [nodes[p]["best_quality"] for p in n["parents"] if p in nodes]
        n["beat_parent"] = bool(pq and n["best_quality"] > max(pq))
    edges, orphan = [], 0
    for tid, n in nodes.items():
        for pid in n["parents"]:
            if pid in nodes:
                edges.append({"parent": pid, "child": tid})
            else:
                orphan += 1
    out = {"is_main": True, "source": str(pool_path), "n_nodes": len(nodes),
           "n_edges": len(edges), "n_orphan_parents": orphan,
           "nodes": nodes, "edges": edges}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{out_path.name}: {len(nodes)} nodes "
          f"({sum(1 for n in nodes.values() if n['admitted'])} admitted, "
          f"{sum(1 for n in nodes.values() if n['beat_parent'])} beat-parent), "
          f"{len(edges)} edges, {orphan} orphan parents")
    return out


def extract_original_from_pool(pool_path: Path, out_path: Path) -> dict:
    """The original branch DOES write a trajectory pool -- under
    log/<experiment_id>/ in its own worktree, not data/results/. All 37 of its
    parent references resolve inside it, so its lineage is fully traceable."""
    pool = json.loads(pool_path.read_text())["trajectories"]
    nodes = {}
    for tid, t in pool.items():
        if not isinstance(t, dict):
            continue
        bm = t.get("backtest_metrics") or {}
        try:
            q = abs(float(bm.get("RankIC"))) if bm.get("RankIC") is not None else 0.0
        except (TypeError, ValueError):
            q = 0.0
        nodes[tid] = {"id": tid, "round": t.get("round_idx"), "phase": t.get("phase"),
                      "parents": list(t.get("parent_ids") or []),
                      "admitted": True,          # no gate on the paper baseline
                      "n_factors": len(t.get("factors") or []),
                      "best_quality": q, "mean_quality": q}
    for tid, n in nodes.items():
        pq = [nodes[p]["best_quality"] for p in n["parents"] if p in nodes]
        n["beat_parent"] = bool(pq and n["best_quality"] > max(pq))
    edges, orphan = [], 0
    for tid, n in nodes.items():
        for pid in n["parents"]:
            (edges.append({"parent": pid, "child": tid}) if pid in nodes
             else None) or (0 if pid in nodes else 0)
            if pid not in nodes:
                orphan += 1
    out = {"is_main": False, "source": str(pool_path), "n_nodes": len(nodes),
           "n_edges": len(edges), "n_orphan_parents": orphan,
           "nodes": nodes, "edges": edges}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{out_path.name}: {len(nodes)} nodes "
          f"({sum(1 for n in nodes.values() if n['beat_parent'])} beat-parent), "
          f"{len(edges)} edges, {orphan} orphan parents")
    return out


if __name__ == "__main__":
    extract_main_from_pool(
        ROOT / "data/results/trajectory_pool_meanvar_20260828_194432.json",
        MAIN_LIB, OUT_MAIN)
    extract_original_from_pool(
        ORIG_DIR / ("log/"
             "2026-08-31_05-23-28-565698/trajectory_pool.json"), OUT_ORIG)