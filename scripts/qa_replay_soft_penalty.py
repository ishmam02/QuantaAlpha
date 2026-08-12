"""Replay the hard-cap run's batches under a SOFT-penalty mean-variance
construction and count how many of the produced factors would have been
admitted to the zoo.

The hard-cap run (treatment_mean_s42) froze at 23/150 because turnover was
pinned at 0.0198 every batch by the turnover_cap=0.02 *clamp*
(portfolio.py:493-494), cornering the optimizer. The soft variant prices
turnover in the objective instead (cost_in_objective=true + trade_penalty=kappa,
routing to mv_weights_costed, which skips the clamp at portfolio.py:492) so the
optimizer CHOOSES turnover as an outcome of whether trading pays.

This re-measures each ledger batch's marginal contribution (5 test_seeds,
k_sigma=1.0, the exact `decide()` gate) under the soft construction, growing the
zoo sequentially as the soft penalty admits -- the genuine counterfactual.

Safety:
  * Read-only on the run-1 ledger (data/results/ledger_treatment_mean_s42.jsonl).
  * Writes to a SEPARATE file (data/results/replay_soft_penalty_s42.jsonl).
  * No edit to portfolio.py; the frozen topk protocol (f4f03cd5e5013329) and the
    control arm are untouched. The soft theta is a new meanvar variant
    (hash 0919959dbe69945b at trade_penalty=5.0).
  * Factor signals load from the MD5 cache (100% complete) -- no recomputation.

Usage:
  python scripts/qa_replay_soft_penalty.py --calibrate          # turnover check on the 23-zoo
  python scripts/qa_replay_soft_penalty.py                      # full sequential replay
  python scripts/qa_replay_soft_penalty.py --workers 5          # parallel seeds (default 5)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from quantaalpha.eval.admission import decide
from quantaalpha.eval.data import align_signal, load_factor_signal
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import load_protocol

LEDGER = "data/results/ledger_treatment_mean_s42.jsonl"
SOFT_YAML = "quantaalpha/eval/protocol_csi300_meanvar_soft.yaml"
OUT = "data/results/replay_soft_penalty_s42.jsonl"


def load_ledger_batches(path: str) -> list[dict]:
    batches = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        # candidates = factor_exprs (admitted) or rejected_exprs (rejected).
        exprs = list(o.get("factor_exprs") or o.get("rejected_exprs") or [])
        batches.append(
            {
                "idx": len(batches),
                "names": list(o.get("factor_names", [])),
                "exprs": exprs,
                "hard_admitted": bool(o.get("admitted")),
                "hard_zoo_size": o.get("zoo_size"),
                "hard_delta_t": o.get("delta_t"),
            }
        )
    return batches


def build_seed_operators(theta, test_seeds, workers):
    """One EvaluationOperator per seed; share one panel + trade mask across them."""
    ops = {}
    panel = p_start = p_end = mask = None
    for seed in test_seeds:
        th = replace(theta, combiner=replace(theta.combiner, seeds=(int(seed),)))
        op = EvaluationOperator(th)
        if panel is None:
            p_start, p_end, _ = op._windows(False)
            panel = op._panel(p_start, p_end)
        op._panels[(p_start, p_end)] = panel  # share the 227 MB panel
        ops[int(seed)] = op
    # build the trade mask once on the shared panel, share to every op
    mask = ops[test_seeds[0]]._trade_mask(panel)
    key = (panel.dates[0], panel.dates[-1], len(panel.instruments))
    for op in ops.values():
        op._masks[key] = mask
    return ops, panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=SOFT_YAML)
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=5, help="parallel seeds per batch")
    ap.add_argument("--limit", type=int, default=None, help="only first N batches")
    ap.add_argument("--folds", type=int, default=None,
                    help="override walk_forward.folds (lower = faster, noisier delta)")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the soft book's turnover on the hard-cap admitted zoo, then exit")
    args = ap.parse_args()

    theta = load_protocol(args.protocol)
    if args.folds is not None:
        theta = replace(theta, walk_forward=replace(theta.walk_forward, folds=int(args.folds)))
    if not theta.portfolio.cost_in_objective or theta.portfolio.trade_penalty <= 0:
        raise SystemExit(
            f"soft protocol not active: cost_in_objective={theta.portfolio.cost_in_objective} "
            f"trade_penalty={theta.portfolio.trade_penalty} -- this would replicate the hard cap.")
    test_seeds = list(theta.admission.test_seeds)
    print(f"soft theta hash={theta.hash} cost_in_objective={theta.portfolio.cost_in_objective} "
          f"trade_penalty={theta.portfolio.trade_penalty}")
    print(f"test_seeds={test_seeds} k_sigma={theta.admission.k_sigma} "
          f"min_size={theta.admission.min_size} mode={theta.admission.mode}")

    batches = load_ledger_batches(args.ledger)
    if args.limit:
        batches = batches[: args.limit]
    print(f"replaying {len(batches)} batches from {args.ledger}")

    ops, panel = build_seed_operators(theta, test_seeds, args.workers)
    print(f"panel {panel.dates[0]}..{panel.dates[-1]}, {len(panel.instruments)} instruments; "
          f"shared across {len(ops)} seed ops")

    # Pre-align every factor signal that appears in any batch (from the MD5 cache).
    all_exprs = sorted({e for b in batches for e in b["exprs"]})
    aligned, missing = {}, []
    for e in all_exprs:
        try:
            aligned[e] = align_signal(load_factor_signal(e), panel)
        except Exception as ex:  # noqa: BLE001
            missing.append((e, str(ex)[:80]))
    print(f"signals: {len(aligned)}/{len(all_exprs)} aligned, {len(missing)} missing")
    for e, ex in missing[:5]:
        print(f"  missing {e[:50]}.. : {ex}")

    # ---- calibration: turnover of the 23-factor hard-cap zoo under the soft penalty
    if args.calibrate:
        hard_zoo = []
        for b in batches:
            if b["hard_admitted"]:
                hard_zoo.extend(e for e in b["exprs"] if e in aligned)
        zoo_sigs = {e: aligned[e] for e in hard_zoo}
        print(f"\n[CALIBRATE] hard-cap admitted zoo = {len(zoo_sigs)} factors; "
              f"building the soft-penalty book (seed {test_seeds[0]}, walk-forward)...")
        op = ops[test_seeds[0]]
        r = op.evaluate(dict(zoo_sigs), zoo_signals={}, zoo_metrics=[])
        print(f"  net_ir={r.get('m_net_ir'):.4f}  turnover_book={r.get('m_turnover_book'):.4f}  "
              f"cost_bps={r.get('m_cost_bps'):.4f}  (target turnover ~0.02)")
        return

    # ---- full sequential replay: grow the soft-penalty zoo batch by batch
    zoo_signals: dict[str, object] = {}
    zoo_metrics: list[dict] = []
    results = []
    t0 = time.time()
    seed_list = [int(s) for s in test_seeds]

    def eval_seed(seed, cands, zs, zm):
        try:
            return ops[seed].evaluate(cands, zoo_signals=zs, zoo_metrics=zm)
        except Exception as ex:  # noqa: BLE001 -- one bad seed must not kill the run
            print(f"    [seed {seed}] eval failed: {type(ex).__name__}: {str(ex)[:120]}")
            return None

    for b in batches:
        cands = {e: aligned[e] for e in b["exprs"] if e in aligned}
        if not cands:
            print(f"  batch {b['idx']:>2}: no candidate signals -- skip")
            continue
        deltas, first_r = [], None
        with ThreadPoolExecutor(max_workers=min(args.workers, len(seed_list))) as ex:
            futs = [ex.submit(eval_seed, s, dict(cands), dict(zoo_signals), list(zoo_metrics))
                    for s in seed_list]
            for f in futs:  # submission order == seed order
                r = f.result()
                if first_r is None:
                    first_r = r
                deltas.append(r.get("m_delta_net_ir"))
        bm = {k[2:]: v for k, v in first_r.items() if k.startswith("m_")}
        d = decide(deltas, len(zoo_signals), theta, metrics=bm)
        zoo_before = len(zoo_signals)
        if d.admit:
            for e in b["exprs"]:
                if e in aligned:
                    zoo_signals[e] = aligned[e]
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
            "soft_zoo_size_after": len(zoo_signals),
            "net_ir": bm.get("net_ir"), "base_net_ir": bm.get("base_net_ir"),
            "turnover_book": bm.get("turnover_book"), "cost_bps": bm.get("cost_bps"),
            "rho_max": bm.get("rho_max"),
        }
        results.append(rec)
        dt = time.time() - t0
        print(f"  batch {b['idx']:>2} | zoo {zoo_before:>3}->{len(zoo_signals):>3} | "
              f"{'ADMIT' if d.admit else 'reject '} | t={d.t_stat:>7.2f} mean={d.mean:>+.5f} "
              f"turn={bm.get('turnover_book'):.4f} | hard={'ADMIT' if b['hard_admitted'] else 'reject '} "
              f"| {dt:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    soft_batches = sum(1 for r in results if r["soft_admitted"])
    soft_factors = sum(len(r["exprs"]) for r in results if r["soft_admitted"])
    hard_batches = sum(1 for r in results if r["hard_admitted"])
    hard_factors = sum(len(r["exprs"]) for r in results if r["hard_admitted"])
    turns = [r["turnover_book"] for r in results if r["turnover_book"] is not None]
    avg_turn = sum(turns) / len(turns) if turns else float("nan")
    print(f"\n=== RESULT (soft penalty, trade_penalty={theta.portfolio.trade_penalty}) ===")
    print(f"  soft-admitted: {soft_batches} batches / {soft_factors} factors")
    print(f"  hard-admitted: {hard_batches} batches / {hard_factors} factors")
    print(f"  produced:      {len(all_exprs)} factors across {len(batches)} batches")
    print(f"  avg turnover_book across batches: {avg_turn:.4f} (target ~0.02)")
    # flips vs hard cap
    flipped_pos = sum(1 for r in results if r["soft_admitted"] and not r["hard_admitted"])
    flipped_neg = sum(1 for r in results if not r["soft_admitted"] and r["hard_admitted"])
    print(f"  flips vs hard cap: {flipped_pos} hard-reject -> soft-admit, "
          f"{flipped_neg} hard-admit -> soft-reject")
    print(f"  wrote {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()