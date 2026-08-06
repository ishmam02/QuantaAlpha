#!/usr/bin/env python
"""Cost model x construction, for every feature set, from one set of fits.

The two existing tables each hold one cell of a 2x2 and they differ in *both*
axes at once: the flat-fee backtest runs Qlib's ``TopkDropoutStrategy``, while
``E_Θ`` runs mean-variance under the full cost model. So the distance between
"+23.55%" and "-0.60%" mixes the cost model with the construction, and neither
number tells you which did the work.

This fills the grid:

                    top-k dropout        mean-variance
    flat fee        (Qlib's setup)       (missing until now)
    full costs      (missing)            (the E_Θ headline)

**One fit, four books.** ``combiner.fit_predict`` depends on ``Θ.combiner`` and
not on ``Θ.portfolio`` or ``Θ.costs``, so the composite prediction is fitted once
per (feature set, seed) and each of the four variants re-prices it. The fit is
the expensive part.

The flat-fee model here is ``κ₀`` alone at 0.0020 -- Qlib's ``open_cost`` 0.0005
plus ``close_cost`` 0.0015 -- with slippage, impact and borrow set to zero. That
makes our top-k/flat-fee cell directly comparable with Qlib's own backtest,
which is worth having as a cross-check on the engine rather than only as a
result.

Usage::

    python scripts/qa_cost_construction_matrix.py --only arm-b ...
    python scripts/qa_cost_construction_matrix.py --render
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_cost_matrix")

KEYS = ["baseline", "seed", "arm-a", "arm-b"]
LABEL = {"baseline": "baseline (no mining)", "seed": "alpha158_20 (seed)",
         "arm-a": "Arm A (RankIC)", "arm-b": "Arm B (net-of-cost U)"}
VARIANTS = [("flat", "topk_dropout"), ("flat", "mean_variance"),
            ("full", "topk_dropout"), ("full", "mean_variance")]


def _ms(vals):
    good = [v for v in vals if v is not None and v == v]
    if not good:
        return float("nan"), float("nan")
    return st.mean(good), (st.pstdev(good) if len(good) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a")
    ap.add_argument("--arm-b")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--only", choices=KEYS)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="data/results/cost_construction_matrix.md")
    args = ap.parse_args()

    import importlib.util

    from quantaalpha.eval import combiner as C
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/qa_compare_arms.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp"] = cmp_mod
    spec.loader.exec_module(cmp_mod)
    fw_spec = importlib.util.spec_from_file_location("fw", ROOT / "scripts/qa_four_way.py")
    fw = importlib.util.module_from_spec(fw_spec)
    sys.modules["fw"] = fw
    fw_spec.loader.exec_module(fw)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, window = op._windows(True)

    if args.render:
        blobs = {}
        for k in KEYS:
            f = Path(args.out).with_suffix(f".{k}.json")
            if not f.exists():
                logger.error("missing partial %s", f)
                return 2
            blobs[k] = json.loads(f.read_text())
        return _render(args, theta, window, seeds, blobs)

    panel = op._panel(start, end)
    th_nobase = replace(theta, combiner=replace(theta.combiner, base_features=()))

    key = args.only or "baseline"
    if key == "baseline":
        factors, base_th = {}, theta
    elif key == "seed":
        factors, base_th = fw._seed_zoo(panel, start, end), th_nobase
    else:
        lib = Path(args.arm_a if key == "arm-a" else args.arm_b)
        factors, missing = cmp_mod.build_zoo(cmp_mod.load_library(lib), panel, LABEL[key])
        if missing:
            logger.warning("%s: %d excluded", LABEL[key], len(missing))
        base_th = th_nobase
    logger.info("%s: %d factor(s)", LABEL[key], len(factors))

    # Qlib's commission and nothing else.
    flat_costs = replace(theta.costs, kappa1=0.0, kappa2=0.0, beta_per_day=0.0,
                         beta_offlist=0.0)

    out: dict[str, dict[str, list]] = {}
    for seed in seeds:
        th_fit = replace(base_th, combiner=replace(base_th.combiner, seeds=(seed,)))
        C.clear_cache()
        t0 = time.time()
        pred = C.fit_predict(factors, None, panel, th_fit)
        for cost_kind, construction in VARIANTS:
            th = replace(th_fit,
                         portfolio=replace(th_fit.portfolio, construction=construction),
                         costs=(flat_costs if cost_kind == "flat" else theta.costs))
            book = dict(EvaluationOperator(th)._book(pred, panel, window, th))
            slot = out.setdefault(f"{cost_kind}/{construction}", {})
            for k2 in ("net_arr", "net_ir", "mdd", "turnover_book", "cost_bps"):
                slot.setdefault(k2, []).append(book.get(k2))
        # Built whole and logged with no lazy args: the message carries literal
        # "%" from the formatted percentages, which %-style interpolation would
        # try to read as format specifiers.
        cells = "  ".join(
            f"{c}/{g[:4]}={100 * out[f'{c}/{g}']['net_arr'][-1]:+.2f}%"
            for c, g in VARIANTS)
        logger.info("  seed=%s  %s  (%.0fs)", seed, cells, time.time() - t0)

    p = Path(args.out).with_suffix(f".{key}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n_factors": len(factors), "metrics": out}, indent=2, default=str))
    logger.info("wrote %s", p)
    return 0


def _render(args, theta, window, seeds, blobs) -> int:
    port = theta.portfolio
    L = [f"# Cost model × construction (`{theta.hash}`)", "",
         f"Window **{window[0]} .. {window[1]}** · seeds `{seeds}`. "
         f"Mean-variance is Σ={port.covariance}, λ={port.risk_aversion:g}, "
         f"turnover cap {port.turnover_cap:g}, max w {port.max_weight:g}; "
         f"top-k is {port.topk}/{port.n_drop}, the published setup.", "",
         "Flat fee is κ₀=0.0020 alone (Qlib's open+close commission) with slippage, "
         "impact and borrow zeroed. Full charges all of them. Every cell comes from "
         "the same fits, so cost model and construction are the only things moving.", ""]

    for metric, title, spec, scale, suf in (("net_arr", "Net ARR", "+.2f", 100.0, "%"),
                                            ("net_ir", "Net IR", "+.4f", 1.0, ""),
                                            ("turnover_book", "Turnover", ".4f", 1.0, "")):
        L += [f"## {title}", "",
              "| Config | flat / top-k | flat / mean-var | full / top-k | full / mean-var |",
              "| :--- | ---: | ---: | ---: | ---: |"]
        for k in KEYS:
            cells = []
            for cost_kind, g in VARIANTS:
                m, _ = _ms(blobs[k]["metrics"][f"{cost_kind}/{g}"][metric])
                cells.append(f"{scale*m:{spec}}{suf}" if m == m else "—")
            L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
        L.append("")

    out = "\n".join(L)
    print(out)
    Path(args.out).write_text(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
