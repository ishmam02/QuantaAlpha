#!/usr/bin/env python
"""Counterfactual: which factors would the NEW redundancy gate have admitted?

Replays a run's admitted factors in the order they were decided, applying the
patched rule instead of the one that ran:

  * the comparison set is the repository AS IT WILL STAND -- every factor kept
    so far -- not an often-empty repository. Under ``admission.mode: standalone``
    each factor is its own batch, so batch-mates never met and the first admits
    were compared against nothing (rho_max = 0.00).
  * a duplicate may only displace an incumbent if it is stronger by a REAL
    margin (``QA_REPLACE_MARGIN``, default 1.0). Any margin at all used to
    qualify, so a +0.01 |t| edge swapped one near-clone for another: 20 of 38
    admits on the 2026-08-24 run were replacements that added no new direction.

Everything else about each factor's decision is held FIXED at what the run
measured (|t_nw|, monotonicity, sign, capacity). This isolates the redundancy
rule: it cannot re-mine, and it cannot admit anything the original gate
rejected -- so the counterfactual zoo is a SUBSET of the real one and the
comparison is clean.

Usage::

    python scripts/qa_replay_gate.py --ledger data/results/ledger_<id>.jsonl \\
        --out data/results/zoo_newgate.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("qa_replay_gate")


def rho_vs(sig, held: dict) -> tuple[float, str | None]:
    """Max |rank corr| of ``sig`` against everything currently held."""
    best, arg = 0.0, None
    for expr, other in held.items():
        c = _cross_sectional_corr(sig, other, "spearman")
        if c.empty:
            continue
        r = abs(float(c.mean()))
        if r > best:
            best, arg = r, expr
    return best, arg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--rho-bar", type=float, default=None,
                    help="default: Θ.gates.rho_bar")
    ap.add_argument("--margin", type=float,
                    default=float(os.environ.get("QA_REPLACE_MARGIN", "1.0")))
    ap.add_argument("--out", default=None, help="write the surviving expressions as JSON")
    ap.add_argument("--corr-cache", default="data/results/zoo_corr.npz",
                    help="precomputed pairwise |rank corr| matrix; built on first run")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    rho_bar = a.rho_bar if a.rho_bar is not None else float(theta.gates.rho_bar)

    rows = [json.loads(l) for l in Path(a.ledger).read_text().splitlines() if l.strip()]

    # Decisions in the order they were made, with the |t| the run measured.
    decisions: list[tuple[str, float]] = []
    for r in rows:
        if not r.get("admitted"):
            continue
        sheets = r.get("factor_tearsheets") or {}
        for expr in (r.get("factor_exprs") or []):
            t = (sheets.get(expr) or {}).get("t_nw")
            if isinstance(t, (int, float)) and t == t:
                decisions.append((expr, abs(float(t))))
    print(f"replaying {len(decisions)} admission decisions "
          f"(rho_bar={rho_bar}, replace margin={a.margin})\n")

    op = EvaluationOperator(theta)
    p0, p1, _ = op._windows(True)
    panel = op._panel(p0, p1)

    uniq = list(dict.fromkeys(e for e, _ in decisions))
    sig: dict[str, object] = {}
    for expr in uniq:
        try:
            sig[expr] = load_aligned_signal(expr, panel)
        except Exception as exc:
            logger.warning("no signal for %.50s: %s", expr, exc)
    order = [e for e in uniq if e in sig]

    # The pairwise matrix is the whole cost of this replay: O(n^2) cross-
    # sectional Spearman over a 5045x933 panel. It depends only on the factor
    # set, not on the gate being tested, so it is computed ONCE and cached --
    # sweeping rho_bar or the margin then costs nothing.
    cache = Path(a.corr_cache)
    R = None
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if list(z["names"]) == order:
                R = z["R"]
                print(f"  reusing cached correlation matrix ({cache})")
        except Exception:
            R = None
    if R is None:
        print(f"  computing {len(order)}x{len(order)} correlation matrix "
              f"({len(order) * (len(order) - 1) // 2} pairs)...", flush=True)
        R = np.eye(len(order))
        for i, j in itertools.combinations(range(len(order)), 2):
            c = _cross_sectional_corr(sig[order[i]], sig[order[j]], "spearman")
            R[i, j] = R[j, i] = abs(float(c.mean())) if not c.empty else 0.0
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, R=R, names=np.array(order, dtype=object))
        print(f"  cached to {cache}")
    pos = {e: i for i, e in enumerate(order)}

    held: dict[str, object] = {}
    score: dict[str, float] = {}
    kept, dropped, replaced = [], [], []

    for expr, t in decisions:
        if expr not in sig:
            continue
        if expr in held:                       # already in the book
            continue
        if held:
            i = pos[expr]
            cand = [(R[i, pos[h]], h) for h in held]
            rho, near = max(cand) if cand else (0.0, None)
        else:
            rho, near = 0.0, None
        if rho < rho_bar:
            held[expr] = sig[expr]
            score[expr] = t
            kept.append((expr, t, rho))
            continue
        # duplicate: only a decisive improvement displaces the incumbent
        inc = score.get(near, float("-inf"))
        if t > inc + a.margin:
            held.pop(near, None)
            score.pop(near, None)
            held[expr] = sig[expr]
            score[expr] = t
            replaced.append((near, expr, inc, t, rho))
        else:
            dropped.append((expr, t, rho, near, inc))

    print(f"  kept (new direction)       : {len(kept)}")
    print(f"  replaced an incumbent      : {len(replaced)}")
    print(f"  DROPPED as redundant       : {len(dropped)}")
    print(f"  -> counterfactual zoo size : {len(held)}   (real zoo was "
          f"{len({e for e, _ in decisions})})\n")

    if dropped:
        print("  dropped (would NOT be admitted now):")
        for expr, t, rho, near, inc in dropped[:12]:
            print(f"    |t|={t:5.2f} rho={rho:.3f} vs |t|={inc:5.2f} incumbent")
            print(f"       {expr[:92]}")
            print(f"       ~= {str(near)[:92]}")

    # Effective rank before and after -- the number the whole exercise is about.
    def eff_rank(exprs):
        n = len(exprs)
        if n < 2:
            return float(n)
        idx = [pos[e] for e in exprs]
        sub = R[np.ix_(idx, idx)]
        ev = np.linalg.eigvalsh(sub)[::-1]
        ev = ev[ev > 0]
        p = ev / ev.sum()
        return float(np.exp(-(p * np.log(p)).sum()))

    all_exprs = order
    new_exprs = list(held)
    print(f"\n  effective rank BEFORE: {eff_rank(all_exprs):.1f} of {len(all_exprs)}")
    print(f"  effective rank AFTER : {eff_rank(new_exprs):.1f} of {len(new_exprs)}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"factors": {f"f{i}": {"factor_expression": e, "factor_name": f"f{i}"}
                         for i, e in enumerate(new_exprs)}}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
