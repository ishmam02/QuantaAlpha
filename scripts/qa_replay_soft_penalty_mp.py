"""Multiprocessing twin of qa_replay_soft_penalty.py.

WHY THIS EXISTS
---------------
The threads version (qa_replay_soft_penalty.py) uses a ThreadPoolExecutor, but
the 5 seed-evals are pure-Python book-building + data-prep that hold the GIL, so
they SERIALIZE under one GIL: ~1 core, ~6.8 min/batch at folds=1, and folds=3
would be ~3x that (~13 h). This version uses a ProcessPoolExecutor so each
seed-eval runs in its own PROCESS with its own GIL -> the 5 evals run truly
parallel (~82 s/batch at folds=1, ~3-5 min/batch at folds=3).

folds=3 is the faithful, apples-to-apples counterfactual: run 1 admitted the 23
under walk_forward.folds=3, so the soft-penalty replay must use the same
evaluation to answer "how many of the 113 would have been admitted". The threads
version could only afford folds=1 (a conservative lower bound); this one makes
folds=3 practical (~2.7 h alongside run 1, non-destructive).

Safety: identical to the threads version -- read-only on the run-1 ledger, writes
to a SEPARATE file, no portfolio.py edit, soft theta is the same new meanvar
variant (cost_in_objective=true, trade_penalty=5.0). The start method is `fork`:
the main process loads the panel + aligns all signals ONCE, and the workers
inherit that state via copy-on-write (one cold start shared, not N redundant
ones). This is safe ONLY because the IC combiner never imports LightGBM/OpenMP --
the `spawn` this file used previously existed solely to dodge a
fork-after-OpenMP deadlock, and that constraint is gone with the linear combiner.
Do NOT switch this back to `spawn` (5x redundant cold start spiked load to 80+
and crashed the box) NOR use `fork` with `model: lightgbm` (fork-after-OpenMP
deadlock).

Usage:
  python scripts/qa_replay_soft_penalty_mp.py --limit 3      # smoke test
  python scripts/qa_replay_soft_penalty_mp.py                # full folds=3 replay
  python scripts/qa_replay_soft_penalty_mp.py --folds 1      # fast conservative bound
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from quantaalpha.eval.admission import decide
from quantaalpha.eval.protocol import load_protocol

LEDGER = "data/results/ledger_treatment_mean_s42.jsonl"
SOFT_YAML = "quantaalpha/eval/protocol_csi300_meanvar_soft.yaml"
OUT = "data/results/replay_soft_penalty_mp_s42.jsonl"

# ---- shared globals (populated ONCE in main() before the pool, inherited by
# forked workers via copy-on-write -- NOT reloaded per worker) ----
_G: dict = {}
# per-worker operator cache. Empty at fork time; each worker fills its own
# (a write here triggers copy-on-write, so it stays private to that worker).
_OPS: dict[int, object] = {}


def _build_shared(yaml_path: str, folds, all_exprs: list[str]) -> None:
    """Load the panel + align all signals ONCE, in the main process.

    With ``fork`` (safe now that the IC combiner removes LightGBM/OpenMP -- the
    only reason this script ever used ``spawn`` was to dodge a
    fork-after-OpenMP deadlock) the workers inherit this state via
    copy-on-write, so there is ONE cold start shared across all workers rather
    than N redundant ones. That keeps peak memory and disk load at ~1x instead
    of ~Nx, which is what makes the run safe on a memory-limited box.
    """
    from quantaalpha.eval.data import align_signal, load_factor_signal
    from quantaalpha.eval.operator import EvaluationOperator

    theta = load_protocol(yaml_path)
    if folds is not None:
        theta = replace(theta, walk_forward=replace(theta.walk_forward, folds=int(folds)))
    _G["theta"] = theta

    # Build the panel + trade mask once (seed-independent) on a throwaway op.
    base = replace(theta, combiner=replace(theta.combiner, seeds=(42,)))
    op0 = EvaluationOperator(base)
    p_start, p_end, _ = op0._windows(False)
    panel = op0._panel(p_start, p_end)
    mask = op0._trade_mask(panel)
    _G["panel"] = panel
    _G["p_start"] = p_start
    _G["p_end"] = p_end
    _G["mask_key"] = (panel.dates[0], panel.dates[-1], len(panel.instruments))
    _G["mask"] = mask

    # Pre-align every factor signal that appears in any batch (from the MD5 cache).
    sigs: dict[str, object] = {}
    missing = 0
    for e in all_exprs:
        try:
            sigs[e] = align_signal(load_factor_signal(e), panel)
        except Exception:  # noqa: BLE001
            missing += 1
    _G["sigs"] = sigs
    print(f"[main] shared panel {panel.dates[0]}..{panel.dates[-1]} "
          f"{len(panel.instruments)} inst; signals {len(sigs)}/{len(all_exprs)} "
          f"({missing} missing)", flush=True)


def _get_op(seed: int):
    if seed not in _OPS:
        from quantaalpha.eval.operator import EvaluationOperator
        theta = replace(_G["theta"], combiner=replace(_G["theta"].combiner, seeds=(int(seed),)))
        op = EvaluationOperator(theta)
        op._panels[(_G["p_start"], _G["p_end"])] = _G["panel"]
        op._masks[_G["mask_key"]] = _G["mask"]
        _OPS[seed] = op
    return _OPS[seed]


def _worker_eval(seed, cand_exprs, zoo_exprs, zoo_metrics):
    """Evaluate one seed's marginal contribution. Returns the metrics dict
    (with m_delta_net_ir etc.) or {"_error": ...} on failure."""
    try:
        op = _get_op(seed)
        sigs = _G["sigs"]
        cands = {e: sigs[e] for e in cand_exprs if e in sigs}
        zs = {e: sigs[e] for e in zoo_exprs if e in sigs}
        return op.evaluate(cands, zoo_signals=zs, zoo_metrics=list(zoo_metrics))
    except Exception as ex:  # noqa: BLE001 -- one bad seed must not kill the run
        return {"_error": f"{type(ex).__name__}: {str(ex)[:160]}", "m_delta_net_ir": None}


def load_ledger_batches(path: str) -> list[dict]:
    batches = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        exprs = list(o.get("factor_exprs") or o.get("rejected_exprs") or [])
        batches.append(
            {
                "idx": len(batches),
                "exprs": exprs,
                "hard_admitted": bool(o.get("admitted")),
                "hard_zoo_size": o.get("zoo_size"),
            }
        )
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=SOFT_YAML)
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=5, help="parallel seed processes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None,
                    help="override walk_forward.folds (default: protocol's 3 = faithful)")
    args = ap.parse_args()

    theta = load_protocol(args.protocol)
    if args.folds is not None:
        theta = replace(theta, walk_forward=replace(theta.walk_forward, folds=int(args.folds)))
    if not theta.portfolio.cost_in_objective or theta.portfolio.trade_penalty <= 0:
        raise SystemExit(
            f"soft protocol not active: cost_in_objective={theta.portfolio.cost_in_objective} "
            f"trade_penalty={theta.portfolio.trade_penalty}")
    folds_used = theta.walk_forward.folds
    test_seeds = [int(s) for s in theta.admission.test_seeds]
    print(f"soft theta hash={theta.hash} cost_in_objective={theta.portfolio.cost_in_objective} "
          f"trade_penalty={theta.portfolio.trade_penalty} folds={folds_used}")
    print(f"test_seeds={test_seeds} k_sigma={theta.admission.k_sigma} "
          f"min_size={theta.admission.min_size} mode={theta.admission.mode}")

    batches = load_ledger_batches(args.ledger)
    if args.limit:
        batches = batches[: args.limit]
    print(f"replaying {len(batches)} batches from {args.ledger}")

    all_exprs = sorted({e for b in batches for e in b["exprs"]})
    print(f"produced: {len(all_exprs)} distinct factor exprs across {len(batches)} batches")

    # Build the panel + align all signals ONCE in this main process; the forked
    # workers inherit it via copy-on-write (one cold start shared, not N).
    _build_shared(args.protocol, args.folds, all_exprs)

    ctx = mp.get_context("fork")
    nw = min(args.workers, len(test_seeds))
    print(f"launching {nw} worker processes (fork, COW-shared panel+signals)...",
          flush=True)
    t0 = time.time()

    zoo_exprs: list[str] = []
    zoo_metrics: list[dict] = []
    results = []

    with ProcessPoolExecutor(
        max_workers=nw,
        mp_context=ctx,
    ) as ex:
        for b in batches:
            cand_exprs = [e for e in b["exprs"]]
            fut_map = {ex.submit(_worker_eval, s, cand_exprs, list(zoo_exprs), list(zoo_metrics)): s
                       for s in test_seeds}
            deltas, first_r = [], None
            for fut, s in fut_map.items():
                r = fut.result()
                if r is None:
                    r = {"m_delta_net_ir": None}
                if r.get("_error"):
                    print(f"    [seed {s}] eval failed: {r['_error']}", flush=True)
                if first_r is None and not r.get("_error"):
                    first_r = r
                deltas.append(r.get("m_delta_net_ir"))
            if first_r is None:
                first_r = {}
            bm = {k[2:]: v for k, v in first_r.items() if k.startswith("m_")}
            d = decide(deltas, len(zoo_exprs), theta, metrics=bm)
            zoo_before = len(zoo_exprs)
            if d.admit:
                for e in b["exprs"]:
                    if e not in zoo_exprs:
                        zoo_exprs.append(e)
                zoo_metrics.append(bm)
            rec = {
                "batch": b["idx"],
                "exprs": b["exprs"],
                "hard_admitted": b["hard_admitted"],
                "soft_admitted": d.admit,
                "reason": d.reason,
                "delta_mean": d.mean, "delta_se": d.se, "delta_t": d.t_stat,
                "delta_per_seed": list(d.deltas),
                "soft_zoo_size_before": zoo_before,
                "soft_zoo_size_after": len(zoo_exprs),
                "net_ir": bm.get("net_ir"), "base_net_ir": bm.get("base_net_ir"),
                "turnover_book": bm.get("turnover_book"), "cost_bps": bm.get("cost_bps"),
                "rho_max": bm.get("rho_max"),
            }
            results.append(rec)
            dt = time.time() - t0
            print(f"  batch {b['idx']:>2} | zoo {zoo_before:>3}->{len(zoo_exprs):>3} | "
                  f"{'ADMIT' if d.admit else 'reject '} | t={d.t_stat:>7.2f} mean={d.mean:>+.5f} "
                  f"turn={bm.get('turnover_book'):.4f} | hard={'ADMIT' if b['hard_admitted'] else 'reject '} "
                  f"| {dt:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    soft_batches = sum(1 for r in results if r["soft_admitted"])
    soft_factors = len(zoo_exprs)
    hard_batches = sum(1 for r in results if r["hard_admitted"])
    hard_factors = sum(len(r["exprs"]) for r in results if r["hard_admitted"])
    turns = [r["turnover_book"] for r in results if r["turnover_book"] is not None]
    avg_turn = sum(turns) / len(turns) if turns else float("nan")
    flipped_pos = sum(1 for r in results if r["soft_admitted"] and not r["hard_admitted"])
    flipped_neg = sum(1 for r in results if not r["soft_admitted"] and r["hard_admitted"])
    print(f"\n=== RESULT (soft penalty, trade_penalty={theta.portfolio.trade_penalty}, folds={folds_used}) ===")
    print(f"  soft-admitted: {soft_batches} batches / {soft_factors} factors")
    print(f"  hard-admitted: {hard_batches} batches / {hard_factors} factors")
    print(f"  produced:      {len(all_exprs)} factors across {len(batches)} batches")
    print(f"  avg turnover_book across batches: {avg_turn:.4f}")
    print(f"  flips vs hard cap: {flipped_pos} hard-reject -> soft-admit, "
          f"{flipped_neg} hard-admit -> soft-reject")
    print(f"  wrote {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()