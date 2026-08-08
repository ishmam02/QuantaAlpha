#!/usr/bin/env python
"""Reprice a finished run's books under a flat fee, for reference only.

The objective and the headline both charge the full cost model. This charges
``κ₀`` alone -- Qlib's ``open_cost`` + ``close_cost``, with slippage, impact and
borrow zeroed -- on the *same* prediction and the *same* construction, so the
distance between "what it earns if trading were frictionless" and "what it earns
in fact" is one legible number instead of something inferred across tables.

**Same engine throughout.** Qlib's own backtest is not used here and the two
should not be mixed: it cannot charge slippage or impact (its exchange models a
commission and price limits, nothing else), so a flat-fee figure from it is not
the same experiment priced differently -- it is a different engine whose
headline this repository has separately found unsupportable from these factors
(best individual |RankIC| 0.0505 against a claimed 0.1179).

This is a reference, not a result. The number it prints is what the book would
earn if slippage and impact did not exist, which they do.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--construction", default="topk_dropout")
    ap.add_argument("--arm-b", default=None, help="library; newest treatment zoo if omitted")
    ap.add_argument("--protocol", default="quantaalpha/eval/protocol_csi300.yaml")
    ap.add_argument("--combiner-seeds", default="42,1,7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib.util

    from quantaalpha.eval import combiner as C
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import load_protocol

    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/qa_compare_arms.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp"] = cmp_mod
    spec.loader.exec_module(cmp_mod)

    lib = args.arm_b
    if lib is None:
        cands = sorted((ROOT / "data/factorlib").glob("*treatment*_zoo.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("no treatment library found; nothing to reprice")
            return 1
        lib = str(cands[0])

    theta = load_protocol(args.protocol)
    theta = replace(theta, portfolio=replace(theta.portfolio, construction=args.construction))
    flat = replace(theta.costs, kappa1=0.0, kappa2=0.0, beta_per_day=0.0, beta_offlist=0.0)

    op = EvaluationOperator(theta)
    start, end, window = op._windows(True)
    panel = op._panel(start, end)
    zoo, missing = cmp_mod.build_zoo(cmp_mod.load_library(Path(lib)), panel, "Arm B")
    if not zoo:
        print(f"{lib}: no scoreable factors")
        return 1

    out = {"library": lib, "construction": args.construction, "theta": theta.hash,
           "engine": theta.combiner.engine, "n_factors": len(zoo),
           "n_missing": len(missing), "window": list(window), "seeds": {}}
    for s in (int(x) for x in args.combiner_seeds.split(",")):
        base = replace(theta, combiner=replace(theta.combiner, base_features=(), seeds=(s,)))
        C.clear_cache()
        pred = C.fit_predict(zoo, None, panel, base)
        row = {}
        for label, costs in (("flat", flat), ("full", theta.costs)):
            th = replace(base, costs=costs)
            b = dict(EvaluationOperator(th)._book(pred, panel, window, th))
            row[label] = {k: b.get(k) for k in
                          ("net_arr", "net_ir", "mdd", "turnover_book", "cost_bps")}
        out["seeds"][s] = row
        print(f"  seed {s}: flat net_arr={row['flat']['net_arr']:+.2%}  "
              f"full net_arr={row['full']['net_arr']:+.2%}  "
              f"gap={row['flat']['net_arr'] - row['full']['net_arr']:+.2%} "
              f"(slippage + impact)", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
