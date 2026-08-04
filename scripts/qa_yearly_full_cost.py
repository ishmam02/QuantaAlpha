#!/usr/bin/env python
"""Year-by-year IC, Rank IC and net ARR under the FULL cost model.

``qa_backtest_all.py`` already breaks the flat-fee path down by calendar year.
That path charges a commission and nothing else, so its yearly returns say what
the factors would earn if trading were free of slippage and impact. This does
the same decomposition under ``E_Θ`` -- κ₀ commission, κ₁ volatility-scaled
slippage, κ₂ super-linear impact, β borrow -- which is the number that decides
whether an edge survives contact with the market.

The composite prediction is fitted **once per (configuration, seed)** and then
re-priced on each calendar year. Refitting per year would be a different
experiment: the combiner would see a different training set for each row and
the years would stop being comparable to each other or to the full-period
figure. Only the pricing window moves.

A year is scored only if it holds at least ``--min-days`` trading days, so a
truncated final year cannot masquerade as a full one.

Usage::

    python scripts/qa_yearly_full_cost.py \
        --arm-a data/factorlib/all_factors_library_control_<stamp>.json \
        --arm-b data/factorlib/all_factors_library_treatment_<stamp>_zoo.json \
        --protocol <protocol.yaml> --seeds 42,1,7
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_yearly_full_cost")

BASELINE = "baseline (no mining)"
ARM_A = "Arm A (RankIC)"
ARM_B = "Arm B (net-of-cost U)"


def _fmt(vals: list[float], spec: str, scale: float = 1.0, suffix: str = "") -> str:
    """mean ± sd, formatted.

    ``suffix`` carries the unit rather than the format spec, because "%" is not
    a valid conversion for a float and ``f"{x:+.2f%}"`` raises at render time --
    after every fit has already been paid for.
    """
    good = [v * scale for v in vals if v is not None and v == v]
    if not good:
        return "—"
    mean = st.mean(good)
    sd = st.pstdev(good) if len(good) > 1 else 0.0
    return f"{mean:{spec}}{suffix} ± {sd:{spec.replace('+', '')}}{suffix}"


def _cached_count(entries) -> int:
    from quantaalpha.eval.data import factor_cache_path
    return sum(1 for _n, e in entries if factor_cache_path(e).exists())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--min-days", type=int, default=100)
    ap.add_argument("--out", default="data/results/yearly_full_cost.md")
    ap.add_argument("--only", choices=["baseline", "arm-a", "arm-b"],
                    help="compute a single configuration and write a partial JSON")
    ap.add_argument("--render", action="store_true",
                    help="merge the partial JSONs written by --only and emit the table")
    args = ap.parse_args()

    import pandas as pd

    from quantaalpha.eval import combiner as C
    from quantaalpha.eval.metrics import _ic_block, label_frame
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/qa_compare_arms.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp"] = cmp_mod
    spec.loader.exec_module(cmp_mod)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, window = op._windows(True)
    panel = op._panel(start, end)

    # Base features off, exactly as the headline comparison does, so these rows
    # decompose that table rather than describing a different model.
    th_nobase = replace(theta, combiner=replace(theta.combiner, base_features=()))

    # One zoo at a time. Holding Arm A's 150 aligned signals and Arm B's 153
    # together is ~3.6 GB before LightGBM sees a row, and on this 16 GB box that
    # is what got the process SIGKILLed twice with no traceback. --only bounds
    # peak memory to a single configuration; --render stitches the partials.
    fa = cmp_mod.load_library(Path(args.arm_a))
    fb = cmp_mod.load_library(Path(args.arm_b))
    zoo_a, miss_a, zoo_b, miss_b = {}, [], {}, []
    if args.only in (None, "arm-a"):
        zoo_a, miss_a = cmp_mod.build_zoo(fa, panel, "Arm A")
    if args.only in (None, "arm-b"):
        zoo_b, miss_b = cmp_mod.build_zoo(fb, panel, "Arm B")
    logger.info("Arm A %d/%d | Arm B %d/%d aligned | theta=%s | only=%s",
                len(zoo_a), len(fa), len(zoo_b), len(fb), theta.hash, args.only)

    dates = pd.to_datetime(pd.Index(panel.dates))
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    in_win = dates[(dates >= lo) & (dates <= hi)]
    years = []
    for y in sorted({d.year for d in in_win}):
        n = int(((in_win.year == y)).sum())
        if n >= args.min_days:
            years.append((str(y), n))
        else:
            logger.warning("%s has only %d trading days in the window; skipped", y, n)

    if args.render:
        # Merge the per-configuration partials. Nothing is recomputed: the fits
        # are the expensive part and they are already on disk.
        results = {}
        for key in ("baseline", "arm-a", "arm-b"):
            f = Path(args.out).with_suffix(f".{key}.json")
            if not f.exists():
                logger.error("missing partial %s -- run --only %s first", f, key)
                return 2
            results.update(json.loads(f.read_text()))
        configs = [(n, None, None) for n in (BASELINE, ARM_A, ARM_B) if n in results]
        logger.info("rendering from partials: %s", ", ".join(n for n, _, _ in configs))
        return _render(args, theta, window, years, results, configs,
                       len(zoo_a) or _cached_count(fa), len(fa),
                       len(zoo_b) or _cached_count(fb), len(fb), miss_b, seeds)

    label = label_frame(panel, theta)
    # The baseline keeps the base features -- it IS the four hand-written
    # expressions with no mining, so stripping them leaves an empty design
    # matrix and CSRankNorm has nothing to group. The arms drop them so the
    # table measures what mining produced. Same split as qa_compare_arms.py.
    all_configs = [("baseline", BASELINE, {}, theta),
                   ("arm-a", ARM_A, zoo_a, th_nobase),
                   ("arm-b", ARM_B, zoo_b, th_nobase)]
    configs = [(n, f, t) for k, n, f, t in all_configs if args.only in (None, k)]
    # results[config][period][metric] -> list over seeds
    results: dict[str, dict[str, dict[str, list]]] = {}

    for name, factors, base_th in configs:
        results[name] = {}
        for seed in seeds:
            th = replace(base_th, combiner=replace(base_th.combiner, seeds=(seed,)))
            C.clear_cache()
            pred = C.fit_predict(factors, None, panel, th)

            wide = pred
            if not isinstance(wide.index, pd.DatetimeIndex) or wide.shape[1] != len(panel.instruments):
                from quantaalpha.eval.metrics import _to_wide
                wide = _to_wide(pred)
            wide = wide.reindex(index=panel.dates, columns=panel.instruments).where(panel.universe)

            periods = [(f"{y}", (f"{y}-01-01", f"{y}-12-31")) for y, _ in years]
            periods.append(("full", window))
            for pname, pwin in periods:
                book = dict(op._book(pred, panel, pwin, th))
                ics = _ic_block(wide, label, pwin)
                slot = results[name].setdefault(pname, {})
                for k, v in (("ic", ics["ic"]), ("rank_ic", ics["rank_ic"]),
                             ("net_arr", book.get("net_arr")), ("net_ir", book.get("net_ir")),
                             ("turnover", book.get("turnover_book")),
                             ("cost_bps", book.get("cost_bps"))):
                    slot.setdefault(k, []).append(v)
            logger.info("%-24s seed=%-3s full-period net_arr=%+.2f%%", name, seed,
                        100 * (results[name]["full"]["net_arr"][-1] or float("nan")))

    return _render(args, theta, window, years, results, configs,
                   len(zoo_a), len(fa), len(zoo_b), len(fb), miss_b, seeds)


def _render(args, theta, window, years, results, configs,
            n_a, tot_a, n_b, tot_b, miss_b, seeds) -> int:
    import json as _json
    # Written before rendering on purpose: the first version of this script
    # crashed in the markdown formatter after all nine fits had completed and
    # took every number with it. The raw dump is the compute; the table is a
    # view of it.
    raw_path = Path(args.out).with_suffix(f".{args.only}.json" if args.only else ".raw.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(_json.dumps(results, indent=2, default=str))
    logger.info("wrote %s (raw, before rendering)", raw_path)
    if args.only:
        logger.info("partial run complete; re-run with --render once all three exist")
        return 0

    cols = [y for y, _ in years] + ["full"]
    L = [
        "# Year by year under the FULL cost model (`E_Θ`)",
        "",
        f"Protocol `{theta.hash}` · window **{window[0]} .. {window[1]}** · seeds `{seeds}` · "
        f"κ₀={theta.costs.kappa0} κ₁={theta.costs.kappa1} κ₂={theta.costs.kappa2}",
        "",
        "Costs charged: commission **and** volatility-scaled slippage, super-linear "
        "impact and borrow. The flat-fee table in `backtest_v2_*.md` charges the "
        "commission alone; the difference between the two is the whole point of the "
        "exercise, so the same years are shown here for a like-for-like read.",
        "",
        "The composite is fitted **once per configuration per seed** and re-priced on "
        "each year — only the pricing window moves, so the rows are comparable to each "
        "other and compose into the full-period column. Mined factors only (base "
        "features excluded), matching the headline comparison.",
        "",
        f"Arm A scored **{n_a}** of {tot_a} factors, Arm B **{n_b}** of {tot_b}"
        + (f" ({len(miss_b)} never computed)" if miss_b else "") + ".",
        "",
        "Trading days per year: " + ", ".join(f"{y} ({n})" for y, n in years) + ".",
        "",
    ]

    for metric, title, spec, scale, suffix in (
        ("ic", "Annual IC", "+.4f", 1.0, ""),
        ("rank_ic", "Annual Rank IC", "+.4f", 1.0, ""),
        ("net_arr", "Annual net ARR (after full costs)", "+.2f", 100.0, "%"),
        ("net_ir", "Annual net IR", "+.4f", 1.0, ""),
    ):
        L += [f"## {title}", "",
              "| Config | " + " | ".join(c.replace("full", "Full period") for c in cols) + " |",
              "| :--- | " + " | ".join("---:" for _ in cols) + " |"]
        for name, _, _th in configs:
            row = [_fmt(results[name].get(c, {}).get(metric, []), spec, scale, suffix) for c in cols]
            L.append(f"| {name} | " + " | ".join(row) + " |")
        L.append("")

    L += ["## Turnover and cost (why the arms are hard to separate here)", "",
          "| Config | Turnover (book) | Cost (bps/day) |", "| :--- | ---: | ---: |"]
    for name, _, _th in configs:
        L.append(f"| {name} | {_fmt(results[name]['full']['turnover'], '.4f')} | "
                 f"{_fmt(results[name]['full']['cost_bps'], '.2f')} |")
    L += ["",
          "`mean_variance` enforces `turnover_cap` on **every** configuration, so if "
          "these rows are equal the budget is binding for all of them and neither "
          "objective can express a turnover advantage. That is a property of the "
          "construction, not a finding about the objectives.",
          ""]

    out = "\n".join(L)
    print(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out + "\n")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
