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

# Above this share of a library missing its signals, the cause is the cache
# rather than a few factors that never compiled, and the comparison would be
# reporting on a library that no longer resembles what the arm mined.
MAX_MISSING_SHARE = 0.10

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


def build_zoo(entries, panel, label):
    """Align every factor that has a cached signal; report the ones that don't.

    Returns ``(zoo, missing)`` so the caller decides what a gap means. Nothing
    is silently dropped: an excluded factor changes the design matrix and
    therefore the result, so it has to reach the log and the summary.
    """
    zoo, missing = {}, []
    for name, expr in entries:
        try:
            zoo[expr] = align_signal(load_factor_signal(expr), panel)
        except FileNotFoundError:
            missing.append((name, expr))
    return zoo, missing


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

    zoo_a, miss_a = build_zoo(fa, panel, "Arm A")
    zoo_b, miss_b = build_zoo(fb, panel, "Arm B")

    # A library entry is a record of what the search proposed, which is not the
    # same as what it managed to compute. The mean_variance arm admitted three
    # factors whose expressions pipe a cross-sectional operator into a
    # time-series one (DELTA(STD($return), 5)); they never executed, so they
    # carry no implementation code, no result_h5_path and no cached signal,
    # while all 153 siblings carry all three. The comprehension that used to
    # build these dicts raised on the first of them and destroyed the whole
    # comparison -- 153 computed factors reported nothing because of 3 that
    # were never computable. Missing signals are now dropped loudly and
    # counted, and only an implausible number of them aborts the run.
    for label, missing, total in (("Arm A", miss_a, len(fa)), ("Arm B", miss_b, len(fb))):
        if not missing:
            continue
        share = len(missing) / max(total, 1)
        logger.warning(
            "%s: %d/%d factor(s) have no cached signal and are EXCLUDED: %s",
            label, len(missing), total, ", ".join(n for n, _ in missing[:8])
            + (" ..." if len(missing) > 8 else ""))
        if share > MAX_MISSING_SHARE:
            logger.error(
                "%s is missing %.0f%% of its signals (>%.0f%%). That is a broken "
                "cache rather than a few uncomputable factors -- refusing to "
                "report a comparison on a library this incomplete. Run "
                "scripts/qa_repair_signals.py to see which and why.",
                label, 100 * share, 100 * MAX_MISSING_SHARE)
            return 2

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
    # The count that priced the book, not the count the library claims. These
    # differ whenever an arm admitted a factor it could not compute, and factor
    # count moves the combiner independently of factor quality -- so a reader
    # comparing the arms needs the number actually used.
    lines.append(f"| Factor count (scored) | 0 | {len(zoo_a)} | {len(zoo_b)} | — |")
    if miss_a or miss_b:
        lines.append(f"| Excluded, no signal | — | {len(miss_a)} | {len(miss_b)} | — |")

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

    if miss_a or miss_b:
        lines += [
            "",
            "### Excluded factors",
            "",
            "Admitted to the library but never computed — no implementation code, no "
            "`result_h5_path`, no cached signal — so they could not be priced and are "
            "absent from the design matrix. They are listed because excluding a factor "
            "changes the result, not as a footnote.",
            "",
        ]
        for label, missing in (("Arm A", miss_a), ("Arm B", miss_b)):
            for name, _ in missing:
                lines.append(f"- {label}: `{name}`")

    table = "\n".join(lines)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        logger.info("wrote %s", args.out)

        # Raw per-seed rows alongside the prose table. The report generator
        # builds its LaTeX from these rather than from parsed markdown, so the
        # document and the measurement cannot drift apart.
        raw = Path(args.out).with_suffix(".raw.json")
        raw.write_text(json.dumps({
            "theta_hash": theta.hash,
            "window": list(window),
            "seeds": list(seeds),
            "n_factors": {"arm_a": len(fa), "arm_b": len(fb)},
            "costs": {"kappa0": theta.costs.kappa0, "kappa1": theta.costs.kappa1,
                      "kappa2": theta.costs.kappa2},
            "portfolio": {"topk": theta.portfolio.topk, "n_drop": theta.portfolio.n_drop},
            "results": {k: v for k, v in res.items()},
            "results_with_base": {k: v for k, v in res_base.items()},
        }, indent=2, default=str))
        logger.info("wrote %s", raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
