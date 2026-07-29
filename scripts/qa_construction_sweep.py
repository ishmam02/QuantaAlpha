#!/usr/bin/env python
"""Score the same libraries under both portfolio constructions.

``g`` is not a detail of the reporting layer -- it decides whether the cost
model can act at all. Under top-k dropout turnover is pinned at ``n_drop/topk``
(measured 0.1050-0.1060 across every configuration), so a capacity-aware
objective has nothing to express itself through. Under the mean--variance
member turnover is a budget the optimiser trades against.

Measured on a real library, this is not a second-order difference:

    topk_dropout (fixed n_drop)   net IR -0.2738   turnover 0.1064   4.31 bps
    topk_dropout (cost-aware)     net IR -0.2738   turnover 0.1064   4.31 bps
    mean_variance (cap 0.05)      net IR -0.1075   turnover 0.0486   1.63 bps
    mean_variance (cap 0.02)      net IR -0.0354   turnover 0.0195   0.66 bps

Two things follow. Cost-aware dropout is *identical* to the fixed quota on real
predictions -- at the measured 7.5x gain/cost ratio a hurdle of 1.0 clears on
every swap -- so a per-swap hurdle and a turnover budget are not substitutes.
And net IR improves monotonically as the book is forbidden from trading, which
says the strategy's problem is turnover itself.

Needs no new mining: the construction affects scoring, so existing libraries
can be re-priced. (It also affects what Arm B *mines*, since its in-loop
evaluation runs through the same construction -- that is a separate and much
more expensive experiment, and this script does not claim to answer it.)

Usage::

    python scripts/qa_construction_sweep.py --library A.json --label "Arm A" \\
                                            --library B.json --label "Arm B"
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantaalpha.eval import combiner as C  # noqa: E402
from quantaalpha.eval.data import align_signal, load_factor_signal  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402


def variants(theta):
    """(label, theta) for every construction worth reporting."""
    p = theta.portfolio
    return [
        ("topk_dropout (fixed n_drop)",
         replace(theta, portfolio=replace(p, construction="topk_dropout",
                                          cost_aware_dropout=False))),
        ("topk_dropout (cost-aware)",
         replace(theta, portfolio=replace(p, construction="topk_dropout",
                                          cost_aware_dropout=True))),
        ("mean_variance (turnover 0.05)",
         replace(theta, portfolio=replace(p, construction="mean_variance",
                                          turnover_cap=0.05))),
        ("mean_variance (turnover 0.02)",
         replace(theta, portfolio=replace(p, construction="mean_variance",
                                          turnover_cap=0.02))),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", action="append", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="data/results/construction_sweep.md")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    labels = args.label or [Path(p).stem for p in args.library]
    if len(labels) < len(args.library):
        labels += [Path(p).stem for p in args.library[len(labels):]]

    theta0 = load_protocol(default_protocol_path())
    theta0 = replace(theta0, walk_forward=replace(theta0.walk_forward, enabled=False))
    op = EvaluationOperator(theta0)
    start, end, window = op._windows(True)
    panel = op._panel(start, end)

    rows: list[tuple[str, str, dict]] = []
    for label, lib in zip(labels, args.library):
        path = Path(lib) if Path(lib).is_absolute() else ROOT / lib
        if not path.exists():
            print(f"skipping {label}: {path} not found")
            continue
        payload = json.loads(path.read_text())
        factors = payload.get("factors", payload)
        items = factors.values() if isinstance(factors, dict) else factors
        signals = {}
        for f in items:
            e = f.get("factor_expression")
            if not e:
                continue
            try:
                signals[e] = align_signal(load_factor_signal(e), panel)
            except (FileNotFoundError, OSError, ValueError):
                continue
        if not signals:
            print(f"skipping {label}: no cached signals")
            continue
        print(f"\n{label}: {len(signals)} factor(s)")
        for vlabel, th in variants(theta0):
            per_seed = []
            for seed in seeds:
                t = replace(th, combiner=replace(th.combiner, seeds=(seed,)))
                C.clear_cache()
                pred = C.fit_predict(signals, None, panel, t)
                per_seed.append(op._book(pred, panel, window, t))
            agg = lambda k: st.fmean([r[k] for r in per_seed if r.get(k) == r.get(k)])
            m = {k: agg(k) for k in ("net_ir", "net_arr", "turnover_book", "cost_bps", "mdd")}
            rows.append((label, vlabel, m))
            print(f"   {vlabel:<32} net_ir={m['net_ir']:+.4f} "
                  f"net_arr={100*m['net_arr']:+.2f}% turnover={m['turnover_book']:.4f} "
                  f"cost={m['cost_bps']:.2f}")

    lines = [
        "# Portfolio construction sweep", "",
        f"Window **{window[0]} .. {window[1]}** · seeds `{seeds}`. Same libraries, "
        "same costs, same protocol — only `g` changes.",
        "",
        "Under top-k dropout turnover is pinned at `n_drop/topk`, so the cost model "
        "applies to a fixed trade volume and a capacity-aware objective has nothing "
        "to act through. The mean–variance member makes turnover a budget the "
        "optimiser trades against. This is the comparison that decides whether the "
        "capacity question is answerable at all.",
        "",
        "| Library | Construction | Net IR | Net ARR | Turnover | Cost (bps/day) | Max DD |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, vlabel, m in rows:
        lines.append(f"| {label} | {vlabel} | {m['net_ir']:+.4f} | {100*m['net_arr']:+.2f}% | "
                     f"{m['turnover_book']:.4f} | {m['cost_bps']:.2f} | {100*m['mdd']:+.2f}% |")

    lines += ["", "## Reading this", "",
              "- **If the two `topk_dropout` rows are identical**, the cost-aware hurdle "
              "is not binding: the calibrated gain exceeds the modelled cost on every "
              "swap, so the construction is still the fixed quota in disguise.",
              "- **If net IR improves as the turnover budget tightens**, the book is "
              "being destroyed by its own trading rather than by weak signal — which is "
              "a statement about the construction, not about the factors.",
              "- Only the `mean_variance` rows can support a claim about capacity. The "
              "`topk_dropout` rows are there because they are what the published "
              "numbers use."]

    text = "\n".join(lines)
    print()
    print(text)
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
