#!/usr/bin/env python
"""Re-score an existing factor library under ``E_Θ`` — no LLM spend.

This is the manual-validation driver for §7 of the implementation plan. It
walks a factor library, growing the repository one factor at a time exactly as
a mining run would, and reports the metric vector, the gate verdict and ``U``
for each. Because it touches no LLM, it is the cheapest way to answer "is the
new engine actually working?" before committing hours to a full arm.

It also implements the three adversarial probes, each of which catches a real
class of bug:

* ``--zero-cost``     negative control: with every κ and β at zero the net
                      metrics must collapse onto the gross ones and
                      ``cost_bps`` must be exactly 0.
* ``--kappa2 X``      sensitivity: raising the capacity knob must reshuffle the
                      ``U`` ranking. If it does not, costs are not reaching the
                      objective.
* ``--rho-bar X``     gate kill switch: an impossible redundancy ceiling must
                      reject everything. If anything survives, the gate is not
                      wired up.

Usage::

    python scripts/qa_eval_probe.py --library data/factorlib/all_factors_library.json
    python scripts/qa_eval_probe.py --zero-cost --limit 4
    python scripts/qa_eval_probe.py --kappa2 0.1
    python scripts/qa_eval_probe.py --rho-bar 0.0001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import factor_cache_path, load_factor_signal  # noqa: E402
from quantaalpha.eval.ledger import Ledger  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_eval_probe")


def load_library(path: Path) -> list[tuple[str, str]]:
    """Return ``[(name, expression)]`` for every factor with a cached signal."""
    payload = json.loads(path.read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors

    out: list[tuple[str, str]] = []
    for entry in items:
        expr = entry.get("factor_expression") or entry.get("expression") or ""
        name = entry.get("factor_name") or entry.get("name") or "unnamed"
        if not expr:
            continue
        if not factor_cache_path(expr).exists():
            logger.warning("no cached signal for %s; skipping", name)
            continue
        out.append((name, expr))
    return out


def build_theta(args: argparse.Namespace):
    theta = load_protocol(args.protocol or default_protocol_path())
    if args.zero_cost:
        theta = replace(
            theta,
            costs=replace(theta.costs, kappa0=0.0, kappa1=0.0, kappa2=0.0, beta_per_day=0.0),
        )
    if args.kappa2 is not None:
        theta = replace(theta, costs=replace(theta.costs, kappa2=args.kappa2))
    if args.rho_bar is not None:
        theta = replace(theta, gates=replace(theta.gates, rho_bar=args.rho_bar))
    if args.gamma_ic is not None:
        theta = replace(theta, gates=replace(theta.gates, gamma_ic=args.gamma_ic))
    if args.gamma_ir is not None:
        theta = replace(theta, gates=replace(theta.gates, gamma_ir=args.gamma_ir))
    return theta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default="data/factorlib/all_factors_library.json")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--ledger", default=None, help="write a JSONL trial record here")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N factors")
    parser.add_argument("--zero-cost", action="store_true", help="negative control")
    parser.add_argument("--kappa2", type=float, default=None, help="override the capacity knob")
    parser.add_argument("--rho-bar", type=float, default=None, help="override the redundancy ceiling")
    parser.add_argument("--gamma-ic", type=float, default=None, help="override the RankIC floor")
    parser.add_argument("--gamma-ir", type=float, default=None, help="override the RankICIR floor")
    parser.add_argument("--admit-all", action="store_true",
                        help="put every factor in the zoo, not just admissible ones (diagnostic)")
    parser.add_argument("--report", action="store_true", help="score on final_test instead of the OOS proxy")
    args = parser.parse_args()

    library_path = Path(args.library)
    if not library_path.exists():
        logger.error("library not found: %s", library_path)
        return 2

    theta = build_theta(args)
    factors = load_library(library_path)
    if args.limit:
        factors = factors[: args.limit]
    if not factors:
        logger.error("no factors with cached signals in %s", library_path)
        return 2

    logger.info("theta=%s | factors=%d | window=%s", theta.hash, len(factors),
                "final_test" if args.report else "search_oos")

    op = EvaluationOperator(theta)
    ledger = Ledger(args.ledger) if args.ledger else None

    zoo_signals: dict[str, object] = {}
    zoo_metrics: list[dict] = []
    rows = []

    for i, (name, expr) in enumerate(factors, 1):
        signal = load_factor_signal(expr)
        started = time.time()
        res = op.evaluate(signal, expr, zoo_signals, zoo_metrics, report=args.report)
        elapsed = time.time() - started

        rows.append((name, res))
        if ledger:
            ledger.append({
                "factor_name": name, "factor_expr": expr,
                "theta_hash": res["theta_hash"], "zoo_hash": res["zoo_hash"],
                "zoo_size": res["zoo_size"],
                "metrics": {k[2:]: v for k, v in res.items() if k.startswith("m_")},
                "e": {k[2:]: v for k, v in res.items() if k.startswith("e_")},
                "U": res["U"], "feasible": res["feasible"], "failed_gates": res["failed_gates"],
            })

        # The repository accumulates only ADMISSIBLE factors, matching
        # NetCostFactorRunner. --admit-all reproduces the pre-gate behaviour for
        # diagnostics; it is not what a real run does.
        if res.get("feasible") or args.admit_all:
            zoo_signals[expr] = signal
            zoo_metrics.append({k[2:]: v for k, v in res.items() if k.startswith("m_")})

        logger.info("[%d/%d] %.1fs %s | admitted=%s |zoo|=%d",
                    i, len(factors), elapsed, name[:36],
                    bool(res.get("feasible")) or args.admit_all, len(zoo_signals))

    _print_table(rows, theta)
    return 0


def _fmt(v, spec="{:+.4f}"):
    try:
        if v is None or v != v:
            return "n/a"
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _print_table(rows, theta) -> None:
    header = f"{'factor':<40} {'U':>7} {'feas':>5} {'rankIC':>8} {'netIR':>8} {'netARR':>8} {'cost_bps':>9} {'rho':>6} {'TO':>6}"
    print()
    print("=" * len(header))
    print(f"E_theta re-scoring  |  protocol {theta.hash}  |  kappa2={theta.costs.kappa2}  rho_bar={theta.gates.rho_bar}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in rows:
        print(
            f"{name[:40]:<40} {_fmt(r.get('U'), '{:.4f}'):>7} {str(r.get('feasible')):>5} "
            f"{_fmt(r.get('m_rank_ic')):>8} {_fmt(r.get('m_net_ir')):>8} {_fmt(r.get('m_net_arr')):>8} "
            f"{_fmt(r.get('m_cost_bps'), '{:.2f}'):>9} {_fmt(r.get('m_rho_max'), '{:.3f}'):>6} "
            f"{_fmt(r.get('m_turnover_book'), '{:.3f}'):>6}"
        )
    print("-" * len(header))

    feasible = sum(1 for _, r in rows if r.get("feasible"))
    costs = [r.get("m_cost_bps") for _, r in rows if r.get("m_cost_bps") is not None]
    print(f"feasible: {feasible}/{len(rows)}   mean cost: {_fmt(sum(costs)/len(costs) if costs else None, '{:.2f}')} bps/day")
    hashes = {r.get("theta_hash") for _, r in rows}
    sizes = [r.get("zoo_size") for _, r in rows]
    print(f"theta hashes: {hashes}  (expect exactly one)")
    print(f"zoo_size path: {sizes}  (must grow)")
    gates: dict[str, int] = {}
    for _, r in rows:
        for g in r.get("failed_gates") or []:
            gates[g] = gates.get(g, 0) + 1
    print(f"gate failures: {gates or 'none'}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
