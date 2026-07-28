#!/usr/bin/env python
"""Head-to-head: Arm A vs Arm B vs the LightGBM baseline, under one ``E_Θ``.

Three design choices make this as honest as I know how to make it.

**1. Mined factors ONLY -- the four base features are excluded from both arms.**
The published setup feeds the combiner ``base features + mined factors``, and a
measured ablation showed the base features alone reach net ARR -5.85% while the
combination reaches -4.49%: most of the level is the hand-specified features,
not the mining. Including them in an arm-vs-arm table hands both arms the same
large common component and compresses the difference actually under test. Here
each arm is scored on *its own factors and nothing else*, with the
base-features-only model reported separately as the floor.

**2. The LightGBM baseline is a row, not an assumption.**
Neither arm is interesting unless it beats a gradient-boosted tree on four
hand-written features. That number belongs in the table.

**3. Every figure carries a seed spread.**
The combiner is stochastic; measured seed-to-seed sd on net ARR is ~1.2pp
against arm differences that may be smaller. Reporting a single draw would let
noise masquerade as an effect, so every configuration is run over several seeds
and reported as mean +/- sd. A difference inside the spread is reported as not
resolvable, not as a win.

Usage::

    python scripts/qa_compare_arms.py --arm-a <libA.json> --arm-b <libB.json>
    python scripts/qa_compare_arms.py ... --seeds 42,1,7 --out comparison.md

Both libraries are scored by the SAME operator on the SAME window with the SAME
Θ. Scoring each arm with its own engine would compare engines, not objectives.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval import combiner as C  # noqa: E402
from quantaalpha.eval.data import align_signal, load_factor_signal  # noqa: E402
from quantaalpha.eval.metrics import prediction_metrics  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_compare_arms")

PAPER = {"ic": 0.0472, "rank_ic": 0.0459, "qlib_arr": 0.0468, "qlib_ir": 0.6453, "mdd": -0.1180}

BASELINE = "LightGBM baseline (base features only)"
ARM_A = "Arm A — RankIC objective (mined only)"
ARM_B = "Arm B — net-of-cost U (mined only)"


def load_library(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors
    out = []
    for entry in items:
        expr = entry.get("factor_expression") or entry.get("expression") or ""
        name = entry.get("factor_name") or entry.get("name") or "unnamed"
        if expr:
            out.append((name, expr))
    return out


def score(op, theta, panel, window, factors, seeds, label):
    """Fit and price one design matrix across seeds; return per-seed metrics."""
    rows = []
    for seed in seeds:
        th = replace(theta, combiner=replace(theta.combiner, seeds=(seed,)))
        C.clear_cache()
        pred = C.fit_predict(factors, None, panel, th)
        m = dict(op._book(pred, panel, window))
        m.update(prediction_metrics(pred, panel, th, window))
        rows.append(m)
        logger.info("%-10s seed=%-5s net_arr=%+.2f%% net_ir=%+.4f", label, seed,
                    100 * m.get("net_arr", float("nan")), m.get("net_ir", float("nan")))
    return rows


def agg(rows, key):
    vals = [r.get(key) for r in rows]
    vals = [float(v) for v in vals if v is not None and v == v]
    if not vals:
        return None, None
    return st.fmean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(mean, sd, spec="{:+.4f}"):
    if mean is None:
        return "—"
    if not sd:
        return spec.format(mean)
    return f"{spec.format(mean)} ± {spec.format(sd).lstrip('+')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default="42,1,7", help="comma-separated combiner seeds")
    ap.add_argument("--search-window", action="store_true",
                    help="score on the OOS proxy instead of final_test")
    ap.add_argument("--with-base", action="store_true",
                    help="ALSO report base features + mined factors (deployment "
                         "configuration) alongside the mined-only comparison")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, window = op._windows(not args.search_window)
    panel = op._panel(start, end)

    # Base features OFF: each arm is judged on its own mined factors alone.
    th_nobase = replace(theta, combiner=replace(theta.combiner, base_features=()))

    fa, fb = load_library(Path(args.arm_a)), load_library(Path(args.arm_b))
    logger.info("Arm A: %d factors | Arm B: %d factors | theta=%s | window=%s..%s",
                len(fa), len(fb), theta.hash, window[0], window[1])
    if not fa or not fb:
        logger.error("One of the libraries has no factors with cached signals.")
        return 2

    zoo_a = {e: align_signal(load_factor_signal(e), panel) for _, e in fa}
    zoo_b = {e: align_signal(load_factor_signal(e), panel) for _, e in fb}

    res = {
        BASELINE: score(op, theta,    panel, window, {},    seeds, "baseline"),
        ARM_A:    score(op, th_nobase, panel, window, zoo_a, seeds, "armA"),
        ARM_B:    score(op, th_nobase, panel, window, zoo_b, seeds, "armB"),
    }
    res_base = {}
    if args.with_base:
        res_base = {
            ARM_A: score(op, theta, panel, window, zoo_a, seeds, "armA+base"),
            ARM_B: score(op, theta, panel, window, zoo_b, seeds, "armB+base"),
        }

    lines = [
        f"# Arm A vs Arm B vs baseline — one `E_Θ` (`{theta.hash}`)",
        "",
        f"Window **{window[0]} .. {window[1]}** · seeds `{seeds}` · "
        f"κ₀={theta.costs.kappa0} κ₁={theta.costs.kappa1} κ₂={theta.costs.kappa2} · "
        f"top-{theta.portfolio.topk}/drop-{theta.portfolio.n_drop}",
        "",
        f"Arm A contributed **{len(fa)}** factors, Arm B **{len(fb)}**. Both arms are "
        "scored on their mined factors **only** — the four base features are "
        "excluded — so the table measures what mining produced rather than what the "
        "hand-written features contribute to both alike.",
        "",
        "All figures are mean ± sd across seeds, net of the full cost model.",
        "",
        "| Metric | Baseline (no mining) | Arm A (RankIC) | Arm B (net-of-cost U) | Paper |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    def row(label, key, spec="{:+.4f}", paper=None):
        c = [fmt(*agg(res[k], key), spec) for k in (BASELINE, ARM_A, ARM_B)]
        lines.append(f"| {label} | {c[0]} | {c[1]} | {c[2]} | "
                     f"{spec.format(paper) if paper is not None else '—'} |")

    row("IC", "ic", paper=PAPER["ic"])
    row("Rank IC", "rank_ic", paper=PAPER["rank_ic"])
    row("ICIR", "icir")
    row("Rank ICIR", "rank_icir")
    row("**Net IR**", "net_ir")
    row("**Net ARR**", "net_arr", "{:+.2%}")
    row("Max drawdown", "mdd", "{:+.2%}", PAPER["mdd"])
    row("Turnover (book)", "turnover_book", "{:.4f}")
    row("Cost (bps/day)", "cost_bps", "{:.2f}")
    row("IS→OOS gap", "is_oos_gap")
    lines.append(f"| Factor count | 0 | {len(fa)} | {len(fb)} | — |")

    if res_base:
        lines += [
            "",
            "### Deployment configuration (base features + mined factors)",
            "",
            "How the factors are actually used. The mined-only table above isolates "
            "what each arm contributed; this one shows the book you would run.",
            "",
            "| Metric | Arm A + base | Arm B + base |",
            "| :--- | ---: | ---: |",
        ]
        for label, key, spec in (("Rank IC", "rank_ic", "{:+.4f}"),
                                 ("Net IR", "net_ir", "{:+.4f}"),
                                 ("Net ARR", "net_arr", "{:+.2%}"),
                                 ("Max drawdown", "mdd", "{:+.2%}")):
            lines.append(f"| {label} | {fmt(*agg(res_base[ARM_A], key), spec)} | "
                         f"{fmt(*agg(res_base[ARM_B], key), spec)} |")

    a_m, a_s = agg(res[ARM_A], "net_arr")
    b_m, b_s = agg(res[ARM_B], "net_arr")
    z_m, _ = agg(res[BASELINE], "net_arr")
    lines += ["", "## Is the difference resolvable?", ""]
    if None in (a_m, b_m):
        lines.append("- Insufficient data to compare.")
    else:
        diff, noise = b_m - a_m, max(a_s or 0.0, b_s or 0.0)
        lines.append(f"- Arm B − Arm A net ARR: **{diff:+.2%}**")
        lines.append(f"- Seed noise (sd): **{noise:.2%}**")
        if noise and abs(diff) < 2 * noise:
            lines.append(
                f"- **{abs(diff)/noise:.1f}× the noise — NOT resolvable.** This difference "
                "is consistent with seed variance alone and should not be reported as "
                "one arm beating the other."
            )
        elif noise:
            lines.append(f"- **{abs(diff)/noise:.1f}× the noise** — the difference survives "
                         "seed variance at this sample size.")
        if z_m is not None:
            lines.append(
                f"- Versus no mining at all: Arm A {a_m - z_m:+.2%}, Arm B {b_m - z_m:+.2%}. "
                "An arm that does not clear the baseline has not demonstrated value."
            )

    lines += [
        "",
        "## Reading this table",
        "",
        "- **Expected:** Arm A leads on raw IC / Rank IC; Arm B leads on net IR and net "
        "ARR with lower turnover. That gap is the frictionless-vs-tradeable claim.",
        "- **If Arm B also leads on raw Rank IC**, be suspicious rather than pleased — "
        "check the libraries are distinct and that the ledger carries one `theta_hash`.",
        "- **Factor counts differ**, and more factors generally help the combiner "
        "regardless of how they were selected; a win driven by count alone is not a win "
        "for the objective.",
        "- **Mined-only is a harsher test than deployment.** These factors are always "
        "used alongside the four base features, and measurement shows the base "
        "features alone outperform the mined factors alone. Excluding them is the "
        "right way to compare the two arms *against each other*, but the mined-only "
        "figures are not the book anyone would run — pass `--with-base` for that view.",
        "- The Qlib standalone backtest remains the anchor for the paper's published "
        "numbers — run it per arm with `--factor-source custom`. It charges a flat fee "
        "only, so it will read far better than the columns above.",
    ]

    table = "\n".join(lines)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
