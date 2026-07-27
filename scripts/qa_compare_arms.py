#!/usr/bin/env python
"""Head-to-head: score both arms' factor libraries under **one** ``E_Θ``.

The comparison only means something if a single evaluation engine scores both
factor sets. Scoring Arm A with the old objective and Arm B with the new one
would compare *engines*, not *objectives*, which is the mistake this script
exists to prevent.

Usage::

    python scripts/qa_compare_arms.py \\
        --arm-a data/factorlib/all_factors_library_armA.json \\
        --arm-b data/factorlib/all_factors_library_armB.json \\
        --out docs/arm_comparison.md

Interpretation — read before drawing conclusions:

* **Expected:** Arm A wins on raw IC/RankIC; Arm B wins on net IR / net ARR and
  shows lower mean turnover and lower ρ_max. That gap *is* the §3.2 claim.
* **If Arm B also wins on raw RankIC, be suspicious, not pleased.** Likeliest
  causes, in order: the feasibility gate is not filtering (check the
  ``failed_gates`` frequency in the ledger); ρ_max was computed against an
  empty zoo throughout (``zoo_size`` must grow); or Arm B read Arm A's pickle
  cache (the two ``theta_hash`` and ``PICKLE_CACHE_FOLDER_PATH_STR`` values
  must differ).
* **If the arms produce near-identical factors**, the objective is not reaching
  the generator — check that the summarizer swap actually took effect by
  reading a logged prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_factor_signal  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_eval_probe import load_library  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_compare_arms")

# The published numbers, for the third reference column.
PAPER = {
    "ic": 0.0472,
    "rank_ic": 0.0459,
    "qlib_arr": 0.0468,
    "qlib_ir": 0.6453,
    "mdd": -0.1180,
}


def score_arm(label: str, library: Path, theta, report: bool) -> list[dict]:
    """Score one arm's library, growing the zoo exactly as a run would."""
    factors = load_library(library)
    if not factors:
        logger.warning("%s: no factors with cached signals in %s", label, library)
        return []

    op = EvaluationOperator(theta)
    zoo_signals: dict[str, object] = {}
    zoo_metrics: list[dict] = []
    rows = []

    for i, (name, expr) in enumerate(factors, 1):
        signal = load_factor_signal(expr)
        res = op.evaluate(signal, expr, zoo_signals, zoo_metrics, report=report)
        res["factor_name"] = name
        rows.append(res)
        zoo_signals[expr] = signal
        zoo_metrics.append({k[2:]: v for k, v in res.items() if k.startswith("m_")})
        logger.info("%s [%d/%d] %s", label, i, len(factors), name[:44])

    return rows


def _mean(rows: list[dict], key: str):
    vals = [r.get(key) for r in rows]
    vals = [float(v) for v in vals if v is not None and v == v]
    return statistics.fmean(vals) if vals else None


def _fmt(v, spec="{:+.4f}"):
    if v is None:
        return "—"
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def build_table(a: list[dict], b: list[dict], theta) -> str:
    lines = [
        f"# Arm A vs Arm B under one `E_Θ`  (protocol `{theta.hash}`)",
        "",
        f"Both libraries scored by the **same** evaluation engine. "
        f"Cost coefficients: κ₀={theta.costs.kappa0}, κ₁={theta.costs.kappa1}, "
        f"κ₂={theta.costs.kappa2}. Portfolio: top-{theta.portfolio.topk}/drop-"
        f"{theta.portfolio.n_drop}, signed={theta.portfolio.signed}.",
        "",
        "| Metric | Arm A (RankIC) | Arm B (`U`) | Paper (GPT-5.2) |",
        "| :--- | ---: | ---: | ---: |",
    ]

    def row(label, key, spec="{:+.4f}", paper=None):
        lines.append(
            f"| {label} | {_fmt(_mean(a, key), spec)} | {_fmt(_mean(b, key), spec)} | "
            f"{_fmt(paper, spec) if paper is not None else '—'} |"
        )

    row("IC", "m_ic", paper=PAPER["ic"])
    row("RankIC", "m_rank_ic", paper=PAPER["rank_ic"])
    row("ICIR", "m_icir")
    row("RankICIR", "m_rank_icir")
    row("**Net IR** (E_Θ)", "m_net_ir")
    row("**Net ARR** (E_Θ)", "m_net_arr", "{:+.2%}")
    row("MDD", "m_mdd", "{:+.2%}", PAPER["mdd"])
    row("Mean turnover (book)", "m_turnover_book", "{:.4f}")
    row("Mean turnover (solo)", "m_turnover_solo", "{:.4f}")
    row("Mean ρ_max", "m_rho_max", "{:.4f}")
    row("Mean cx", "m_cx", "{:.0f}")
    row("IS→OOS gap", "m_is_oos_gap")
    row("Cost (bps/day)", "m_cost_bps", "{:.2f}")
    row("Utility U", "U", "{:.4f}")

    fa = sum(1 for r in a if r.get("feasible"))
    fb = sum(1 for r in b if r.get("feasible"))
    lines += [
        f"| Feasible / total | {fa}/{len(a)} | {fb}/{len(b)} | — |",
        "",
        f"Qlib standalone backtest (comparability anchor) — run separately: "
        f"paper reports ARR {PAPER['qlib_arr']:.2%} / IR {PAPER['qlib_ir']:.4f}.",
        "",
        "## Sanity checks",
        "",
    ]

    for label, rows_ in (("Arm A", a), ("Arm B", b)):
        if not rows_:
            lines.append(f"- **{label}**: no factors scored.")
            continue
        hashes = {r.get("theta_hash") for r in rows_}
        sizes = [r.get("zoo_size") for r in rows_]
        grew = sizes == sorted(sizes) and len(set(sizes)) > 1
        lines.append(
            f"- **{label}**: {len(rows_)} factors | theta hashes {hashes} "
            f"(expect exactly one) | zoo_size grows: {grew}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-a", required=True, help="control-arm factor library JSON")
    parser.add_argument("--arm-b", required=True, help="treatment-arm factor library JSON")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--out", default=None, help="write the markdown table here")
    parser.add_argument(
        "--search-window", action="store_true",
        help="score on the OOS proxy instead of final_test (default is final_test)",
    )
    args = parser.parse_args()

    theta = load_protocol(args.protocol or default_protocol_path())
    report = not args.search_window
    logger.info("scoring both arms under theta=%s on %s",
                theta.hash, "final_test" if report else "search_oos")

    a = score_arm("ArmA", Path(args.arm_a), theta, report)
    b = score_arm("ArmB", Path(args.arm_b), theta, report)

    table = build_table(a, b, theta)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
