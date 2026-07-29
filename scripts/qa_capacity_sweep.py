#!/usr/bin/env python
"""Net IR versus capital — at what size does the strategy die?

Impact (κ₂ in Eq. 7) is super-linear in participation, so a book that clears
its costs at ¥100m need not clear them at ¥1bn. Everything else in this
repository is measured at a single fixed ``nav``, which answers "is it
profitable?" but never "how much money can it hold?" -- and for a
*capacity-aware* objective that second question is the whole claim.

This needs no new mining: it re-scores libraries that already exist, varying
only ``Θ.costs.nav``. Cheap enough to run on every arm, and the resulting curve
is the strongest real-world evidence the objective can offer -- if the
treatment arm's book decays more slowly with size than the control's, that is
capacity awareness doing something no IC statistic can show.

Usage::

    python scripts/qa_capacity_sweep.py --library data/factorlib/<lib>.json
    python scripts/qa_capacity_sweep.py --library A.json --label "Arm A" \\
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

DEFAULT_NAVS = "1e7,1e8,1e9,1e10"


def load_library(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors
    return [e for e in (f.get("factor_expression") for f in items) if e]


def money(nav: float) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if nav >= scale:
            return f"¥{nav / scale:g}{suffix}"
    return f"¥{nav:g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", action="append", required=True)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--navs", default=DEFAULT_NAVS)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="data/results/capacity_sweep.md")
    args = ap.parse_args()

    navs = [float(x) for x in args.navs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    labels = args.label or [Path(p).stem for p in args.library]
    if len(labels) < len(args.library):
        labels += [Path(p).stem for p in args.library[len(labels):]]

    theta = load_protocol(default_protocol_path())
    op = EvaluationOperator(theta)
    start, end, window = op._windows(True)
    panel = op._panel(start, end)

    results: dict[str, dict[float, dict[str, float]]] = {}
    for label, lib_path in zip(labels, args.library):
        exprs = load_library(ROOT / lib_path if not Path(lib_path).is_absolute() else Path(lib_path))
        signals = {}
        for e in exprs:
            try:
                signals[e] = align_signal(load_factor_signal(e), panel)
            except (FileNotFoundError, OSError, ValueError):
                continue
        if not signals:
            print(f"skipping {label}: no cached signals")
            continue
        print(f"{label}: {len(signals)} factor(s)")
        results[label] = {}
        for nav in navs:
            rows = []
            for seed in seeds:
                th = replace(theta,
                             costs=replace(theta.costs, nav=nav),
                             combiner=replace(theta.combiner, seeds=(seed,)))
                C.clear_cache()
                pred = C.fit_predict(signals, None, panel, th)
                rows.append(op._book(pred, panel, window, th))
            agg = lambda k: st.fmean([r[k] for r in rows if r.get(k) == r.get(k)])
            results[label][nav] = {k: agg(k) for k in ("net_ir", "net_arr", "cost_bps", "turnover_book")}
            m = results[label][nav]
            print(f"   nav={money(nav):>8}  net_ir={m['net_ir']:+.4f}  "
                  f"net_arr={100*m['net_arr']:+.2f}%  cost={m['cost_bps']:.2f}bps/day")

    lines = [
        "# Capacity: net IR versus capital", "",
        f"Window **{window[0]} .. {window[1]}** · seeds `{seeds}` · "
        f"κ₀={theta.costs.kappa0} κ₁={theta.costs.kappa1} κ₂={theta.costs.kappa2} "
        f"(exponent {theta.costs.impact_exponent})",
        "",
        "Only `nav` varies. Impact is super-linear in participation, so this is "
        "the axis on which a capacity-aware objective should distinguish itself: "
        "not by being more profitable at one size, but by decaying more slowly "
        "as size grows.",
        "",
        "| Configuration | " + " | ".join(money(n) for n in navs) + " |",
        "| :--- | " + " | ".join("---:" for _ in navs) + " |",
    ]
    for metric, fmt, name in (("net_ir", "{:+.4f}", "Net IR"),
                              ("net_arr", "{:+.2%}", "Net ARR"),
                              ("cost_bps", "{:.2f}", "Cost (bps/day)")):
        lines.append(f"| **{name}** |" + " |" * len(navs))
        for label, by_nav in results.items():
            cells = [fmt.format(by_nav[n][metric]) if n in by_nav else "—" for n in navs]
            lines.append(f"| &nbsp;&nbsp;{label} | " + " | ".join(cells) + " |")

    lines += ["", "## Where does it die?", ""]
    for label, by_nav in results.items():
        alive = [n for n in navs if n in by_nav and by_nav[n]["net_ir"] > 0]
        if not alive:
            lines.append(f"- **{label}**: net IR is negative at every size tested — "
                         "there is no capacity to speak of under this cost model.")
        else:
            biggest = max(alive)
            lines.append(f"- **{label}**: net IR stays positive up to **{money(biggest)}**"
                         + ("" if biggest == max(navs) else
                            f", and turns negative by {money(min(n for n in navs if n > biggest))}"))
    lines += ["", "A curve that falls steeply is a strategy whose edge is an artefact of "
              "assuming it trades in size it could not actually trade in."]

    text = "\n".join(lines)
    print()
    print(text)
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
