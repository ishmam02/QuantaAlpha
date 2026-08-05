#!/usr/bin/env python
"""Baseline vs Alpha158 seed vs Arm A vs Arm B under one ``E_Θ``.

Four feature sets, one protocol, one cost model:

  baseline      the four hand-written base features, no mining at all
  alpha158_20   Qlib's own 20-factor Alpha158 subset -- an external reference
                that owes nothing to this pipeline
  Arm A         the RankIC arm's mined factors
  Arm B         the net-of-cost arm's mined factors

The seed library is evaluated through Qlib's own expression engine rather than
translated into the mining DSL. ``Ref``/``Mean``/``Std``/``Greater`` are Qlib
operators with Qlib semantics, and a hand translation would be reporting a
paraphrase of the reference rather than the reference.

Every configuration is priced by the same ``g``, so the comparison is of
feature sets and nothing else. Base features are excluded from the three
non-baseline rows for the same reason as in ``qa_compare_arms.py``: they are
common to all of them and including them measures what they contribute, not
what mining did.

``--only`` computes one configuration per process. Arm A's 150 aligned signals
and Arm B's 153 together exceed what this box can hold alongside LightGBM, and
the symptom is a silent SIGKILL rather than an error.

Usage::

    python scripts/qa_four_way.py --only baseline   ...
    python scripts/qa_four_way.py --render --out report.md
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
logger = logging.getLogger("qa_four_way")

KEYS = ["baseline", "seed", "arm-a", "arm-b"]
LABEL = {"baseline": "baseline (no mining)", "seed": "alpha158_20 (seed)",
         "arm-a": "Arm A (RankIC)", "arm-b": "Arm B (net-of-cost U)"}


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def _seed_zoo(panel, start, end):
    """The Alpha158 subset as aligned signals, via Qlib's own engine."""
    from qlib.data import D

    from quantaalpha.backtest.factor_loader import FactorLoader
    from quantaalpha.eval.data import _align, _init_qlib

    _init_qlib()
    exprs = FactorLoader.ALPHA158_20_FACTORS
    fields = list(exprs.values())
    raw = D.features(D.instruments("csi300"), fields, start_time=start, end_time=end)
    zoo = {}
    for name, expr in exprs.items():
        col = raw[expr]
        wide = col.unstack(level="instrument")
        wide.index = __import__("pandas").to_datetime(wide.index)
        zoo[expr] = _align(wide.sort_index(), panel)
    return zoo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a")
    ap.add_argument("--arm-b")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--only", choices=KEYS)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="data/results/four_way.md")
    args = ap.parse_args()

    import importlib.util

    import pandas as pd

    from quantaalpha.eval import combiner as C
    from quantaalpha.eval.metrics import _ic_block, _to_wide, label_frame
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol

    spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/qa_compare_arms.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp"] = cmp_mod
    spec.loader.exec_module(cmp_mod)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    theta = load_protocol(args.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, window = op._windows(True)

    if args.render:
        results, counts = {}, {}
        for k in KEYS:
            f = Path(args.out).with_suffix(f".{k}.json")
            if not f.exists():
                logger.error("missing partial %s", f)
                return 2
            blob = json.loads(f.read_text())
            results[LABEL[k]] = blob["metrics"]
            counts[LABEL[k]] = blob["n_factors"]
        return _render(args, theta, window, seeds, results, counts)

    panel = op._panel(start, end)
    th_nobase = replace(theta, combiner=replace(theta.combiner, base_features=()))
    label = label_frame(panel, theta)

    key = args.only or "baseline"
    if key == "baseline":
        factors, base_th = {}, theta
    elif key == "seed":
        factors, base_th = _seed_zoo(panel, start, end), th_nobase
    else:
        lib = Path(args.arm_a if key == "arm-a" else args.arm_b)
        entries = cmp_mod.load_library(lib)
        factors, missing = cmp_mod.build_zoo(entries, panel, LABEL[key])
        if missing:
            logger.warning("%s: %d factor(s) excluded (never computed)", LABEL[key], len(missing))
        base_th = th_nobase
    logger.info("%s: %d factor(s)", LABEL[key], len(factors))

    dates = pd.to_datetime(pd.Index(panel.dates))
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    in_win = dates[(dates >= lo) & (dates <= hi)]
    years = [str(y) for y in sorted({d.year for d in in_win})
             if int((in_win.year == y).sum()) >= 100]

    out: dict[str, dict[str, list]] = {}
    for seed in seeds:
        th = replace(base_th, combiner=replace(base_th.combiner, seeds=(seed,)))
        C.clear_cache()
        t0 = time.time()
        pred = C.fit_predict(factors, None, panel, th)
        wide = _to_wide(pred).reindex(index=panel.dates,
                                      columns=panel.instruments).where(panel.universe)
        for pname, pwin in ([(y, (f"{y}-01-01", f"{y}-12-31")) for y in years]
                            + [("full", window)]):
            book = dict(op._book(pred, panel, pwin, th))
            ics = _ic_block(wide, label, pwin)
            slot = out.setdefault(pname, {})
            for k2, v in (("ic", ics["ic"]), ("rank_ic", ics["rank_ic"]),
                          ("icir", ics["icir"]), ("rank_icir", ics["rank_icir"]),
                          ("net_arr", book.get("net_arr")), ("net_ir", book.get("net_ir")),
                          ("mdd", book.get("mdd")), ("turnover", book.get("turnover_book")),
                          ("cost_bps", book.get("cost_bps"))):
                slot.setdefault(k2, []).append(v)
        logger.info("  seed=%-3s net_arr=%+.2f%% net_ir=%+.4f (%.0fs)", seed,
                    100 * out["full"]["net_arr"][-1], out["full"]["net_ir"][-1],
                    time.time() - t0)

    p = Path(args.out).with_suffix(f".{key}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n_factors": len(factors), "metrics": out}, indent=2, default=str))
    logger.info("wrote %s", p)
    return 0


def _render(args, theta, window, seeds, results, counts) -> int:
    cols = [c for c in next(iter(results.values())) if c != "full"] + ["full"]
    port = theta.portfolio
    L = [f"# Baseline vs seed vs Arm A vs Arm B — one `E_Θ` (`{theta.hash}`)", "",
         f"Window **{window[0]} .. {window[1]}** · seeds `{seeds}` · "
         f"construction `{port.construction}` "
         f"(Σ={port.covariance}, λ={port.risk_aversion:g}, "
         f"turnover cap {port.turnover_cap:g}, max w {port.max_weight:g})", "",
         "Full cost model: commission, volatility-scaled slippage, super-linear "
         "impact and borrow.", "",
         "| Config | factors | " + " | ".join(c.replace("full", "**Full**") for c in cols) + " |"]
    L.append("| :--- | ---: | " + " | ".join("---:" for _ in cols) + " |")
    for metric, title, spec, scale, suf in (
            ("rank_ic", "Rank IC", "+.4f", 1.0, ""),
            ("net_arr", "Net ARR", "+.2f", 100.0, "%"),
            ("net_ir", "Net IR", "+.4f", 1.0, "")):
        L.append(f"| **{title}** | | " + " | ".join("" for _ in cols) + " |")
        for name, per in results.items():
            cells = []
            for c in cols:
                vals = [v for v in per.get(c, {}).get(metric, []) if v is not None and v == v]
                cells.append(f"{scale*st.mean(vals):{spec}}{suf}" if vals else "—")
            L.append(f"| {name} | {counts[name]} | " + " | ".join(cells) + " |")
    out = "\n".join(L)
    print(out)
    Path(args.out).write_text(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
