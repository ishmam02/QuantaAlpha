#!/usr/bin/env python
"""Choose the mean-variance construction from Arm B's own mined factors.

Every knob in ``g`` -- the risk model, risk aversion, the turnover budget, the
position cap -- was set by argument rather than by measurement. This picks them
from data.

**Tuned on the validation split, never on the test split.** Selecting a
construction by its 2022-2025 performance and then reporting that same window
is choosing Θ on the test set: it breaks Property 1 (the protocol is supposed
to be frozen before the numbers are seen) and Property 3 (the search cannot see
the test window). The sweep therefore scores on ``splits.oos_window`` -- 2021 --
and the winner is priced on ``final_test`` exactly once, at the end, as a
confirmation rather than as a selection criterion.

**One fit, many books.** ``combiner.fit_predict`` depends on ``Θ.combiner`` and
not on ``Θ.portfolio``, so the composite prediction is fitted once per seed and
every configuration re-prices that same prediction. The fit is the expensive
part; this turns a 54-configuration sweep from hours into minutes.

Ranking is on net IR rather than net ARR: both are reported, but ARR over a
single year is one equity path, while IR normalises by the volatility of that
path and is correspondingly steadier across seeds. A configuration chosen on
the noisier statistic is a configuration chosen on noise.

Usage::

    python scripts/qa_mv_sweep.py --arm-b <library.json> --protocol <p.yaml>
"""

from __future__ import annotations

import argparse
import itertools
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
logger = logging.getLogger("qa_mv_sweep")


def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--out", default="data/results/mv_sweep.md")
    ap.add_argument("--confirm-on-test", action="store_true",
                    help="price the winning configuration on final_test once")
    args = ap.parse_args()

    import importlib.util

    from quantaalpha.eval import combiner as C
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/qa_compare_arms.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp"] = cmp_mod
    spec.loader.exec_module(cmp_mod)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)

    # report=False -> the OOS proxy (2021). The test split is not reachable here.
    start, end, valid_window = op._windows(False)
    panel = op._panel(start, end)
    logger.info("tuning window (validation): %s .. %s", *valid_window)

    th_nobase = replace(theta, combiner=replace(theta.combiner, base_features=()))
    fb = cmp_mod.load_library(Path(args.arm_b))
    zoo, missing = cmp_mod.build_zoo(fb, panel, "Arm B")
    logger.info("Arm B: %d aligned, %d missing", len(zoo), len(missing))

    grid = []
    for lam, cap, mw in itertools.product((5.0, 10.0, 25.0),
                                          (0.004, 0.008, 0.02),
                                          (0.03, 0.05)):
        grid.append({"covariance": "diagonal", "risk_aversion": lam,
                     "turnover_cap": cap, "max_weight": mw})
        for k in (5, 10):
            grid.append({"covariance": "factor", "n_factors": k, "risk_aversion": lam,
                         "turnover_cap": cap, "max_weight": mw})
    logger.info("%d configurations x %d seed(s)", len(grid), len(seeds))

    # One fit per seed; every configuration re-prices the same prediction.
    preds = {}
    for seed in seeds:
        th = replace(th_nobase, combiner=replace(th_nobase.combiner, seeds=(seed,)))
        C.clear_cache()
        t0 = time.time()
        preds[seed] = C.fit_predict(zoo, None, panel, th)
        logger.info("fitted seed %s in %.0fs", seed, time.time() - t0)

    results = []
    for i, cfg in enumerate(grid, 1):
        rows = []
        t0 = time.time()
        for seed in seeds:
            th = replace(
                th_nobase,
                combiner=replace(th_nobase.combiner, seeds=(seed,)),
                portfolio=replace(th_nobase.portfolio, **cfg),
            )
            op2 = EvaluationOperator(th)
            rows.append(dict(op2._book(preds[seed], panel, valid_window, th)))
        ir_m, ir_s = _agg(rows, "net_ir")
        arr_m, arr_s = _agg(rows, "net_arr")
        to_m, _ = _agg(rows, "turnover_book")
        c_m, _ = _agg(rows, "cost_bps")
        mdd_m, _ = _agg(rows, "mdd")
        results.append({**cfg, "net_ir": ir_m, "net_ir_sd": ir_s,
                        "net_arr": arr_m, "net_arr_sd": arr_s,
                        "turnover": to_m, "cost_bps": c_m, "mdd": mdd_m})
        logger.info("[%2d/%d] %-8s k=%-3s lam=%-5s cap=%-6s mw=%-5s -> net_ir=%+.4f "
                    "net_arr=%+.2f%% turnover=%.4f (%.0fs)",
                    i, len(grid), cfg["covariance"], cfg.get("n_factors", "-"),
                    cfg["risk_aversion"], cfg["turnover_cap"], cfg["max_weight"],
                    ir_m, 100 * arr_m, to_m, time.time() - t0)
        Path(args.out).with_suffix(".raw.json").write_text(json.dumps(results, indent=2))

    results.sort(key=lambda r: (-r["net_ir"] if r["net_ir"] == r["net_ir"] else 1e9))
    best = results[0]

    L = ["# Mean-variance construction, chosen on Arm B's mined factors", "",
         f"Tuned on the **validation** split `{valid_window[0]} .. {valid_window[1]}` — "
         "never on the test split, so this choice does not consume `final_test`. "
         f"Seeds `{seeds}`, Arm B's {len(zoo)} scoreable factors, base features excluded.",
         "",
         "Ranked by net IR. Net ARR over a single year is one equity path; IR "
         "normalises by that path's volatility and is steadier across seeds, so "
         "ranking on ARR would be ranking partly on noise.",
         "",
         "| # | Σ | k | λ | turnover cap | max w | net IR | net ARR | turnover | cost bps | MDD |",
         "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for i, r in enumerate(results[:20], 1):
        L.append(f"| {i} | {r['covariance']} | {r.get('n_factors','—')} | {r['risk_aversion']:g} | "
                 f"{r['turnover_cap']:g} | {r['max_weight']:g} | "
                 f"**{r['net_ir']:+.4f}** ± {r['net_ir_sd']:.4f} | {100*r['net_arr']:+.2f}% | "
                 f"{r['turnover']:.4f} | {r['cost_bps']:.2f} | {100*r['mdd']:+.1f}% |")

    diag_best = max((r for r in results if r["covariance"] == "diagonal"),
                    key=lambda r: r["net_ir"], default=None)
    if diag_best:
        L += ["", f"Best **diagonal** configuration: net IR {diag_best['net_ir']:+.4f} "
              f"(λ={diag_best['risk_aversion']:g}, cap={diag_best['turnover_cap']:g}, "
              f"max w={diag_best['max_weight']:g}) — "
              f"{best['net_ir'] - diag_best['net_ir']:+.4f} versus the winner."]
    L += ["", "## Chosen", "",
          "```yaml", "portfolio:"]
    for k, v in best.items():
        if k in ("covariance", "n_factors", "risk_aversion", "turnover_cap", "max_weight"):
            L.append(f"  {k}: {v}")
    L.append("```")

    out = "\n".join(L)
    print(out)
    Path(args.out).write_text(out + "\n")
    logger.info("wrote %s", args.out)

    if args.confirm_on_test:
        logger.info("confirming the winner on final_test (once)")
        t_start, t_end, test_window = op._windows(True)
        panel_t = op._panel(t_start, t_end)
        zoo_t, _ = cmp_mod.build_zoo(fb, panel_t, "Arm B")
        rows = []
        for seed in seeds:
            th = replace(
                th_nobase,
                combiner=replace(th_nobase.combiner, seeds=(seed,)),
                portfolio=replace(th_nobase.portfolio,
                                  **{k: v for k, v in best.items()
                                     if k in ("covariance", "n_factors", "risk_aversion",
                                              "turnover_cap", "max_weight")}),
            )
            C.clear_cache()
            pred = C.fit_predict(zoo_t, None, panel_t, th)
            rows.append(dict(EvaluationOperator(th)._book(pred, panel_t, test_window, th)))
        ir_m, ir_s = _agg(rows, "net_ir")
        arr_m, arr_s = _agg(rows, "net_arr")
        to_m, _ = _agg(rows, "turnover_book")
        confirm = (f"\n## Confirmation on `final_test` ({test_window[0]}..{test_window[1]})\n\n"
                   f"net IR **{ir_m:+.4f}** ± {ir_s:.4f} · net ARR **{100*arr_m:+.2f}%** "
                   f"± {100*arr_s:.2f}% · turnover {to_m:.4f}\n\n"
                   "Reported once, after the choice was made on validation. It confirms "
                   "the configuration; it did not select it.\n")
        print(confirm)
        Path(args.out).write_text(out + "\n" + confirm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
