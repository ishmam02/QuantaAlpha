#!/usr/bin/env python
"""Re-measure max_weight now that the optimiser is known to be cornered.

Measured on the glm52 book: 32 of 34 positions sit EXACTLY at ``max_weight``
0.03, every weight decile is 0.0300, effective N = 33.6 = 1/0.03. The
"mean-variance" book is a 34-name equal-weight portfolio -- ``mu`` chooses WHICH
names, never how much. A cap sweep therefore does not tune a risk control; it
chooses the book's position count, and with it how much conviction the signal is
allowed to express.

An earlier sweep concluded 0.03 was optimal. That was measured with a different
combiner state and before the pinning was known, so it is re-run here rather
than inherited.

Reported per cap: positions held, how many sit at the cap, the transfer
coefficient (how much intended alpha survives construction), and net_ir/net_arr
after the full cost model. TC is the diagnostic the pinning predicts should
move: a looser cap lets the optimiser tilt, a tighter one forces more names.

Scored on the VALID window -- the same window admission uses. The holdout is
never touched by a parameter choice.

Usage::

    python scripts/qa_cap_sweep.py --library data/factorlib/all_factors_library_smoke_glm52_zoo.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--caps", default="0.01,0.02,0.03,0.05,0.08,0.15,0.30,1.0")
    ap.add_argument("--out", default="data/results/cap_sweep.json")
    a = ap.parse_args()

    proto_path = a.protocol or default_protocol_path()
    theta = load_protocol(proto_path)
    if a.zoo:
        exprs = list(replay_repository(a.zoo))
    else:
        payload = json.loads(Path(a.library).read_text())
        f = payload.get("factors", payload)
        items = f.values() if isinstance(f, dict) else f
        exprs = [e.get("factor_expression") or e.get("expression") for e in items]
        exprs = [e for e in exprs if e]

    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)
    panel = op._panel(p0, p1)
    print(f"protocol {theta.hash} | scored {win} | {len(exprs)} factors")
    print(f"  universe ~{panel.universe.sum(axis=1).median():.0f} names\n", flush=True)

    sig = {}
    for e in exprs:
        try:
            sig[e] = load_aligned_signal(e, panel)
        except Exception:
            pass
    if len(sig) < 2:
        print("need >= 2 factors")
        return 2

    caps = [float(x) for x in a.caps.split(",")]
    print(f"{'cap':>6} {'positions':>10} {'at cap':>7} {'TC':>8} {'net_IR':>9} "
          f"{'net_ARR':>9} {'turnover':>9}  secs")
    print("-" * 74)
    rows = []
    for cap in caps:
        t0 = time.time()
        # Θ is a FROZEN dataclass on purpose -- a protocol that can be mutated
        # in place is a protocol nothing can be reproduced from. So each cap is
        # a fresh protocol built from the YAML with one field overridden, which
        # also means each variant carries its own hash.
        import tempfile, yaml
        cfg = yaml.safe_load(Path(proto_path).read_text())
        cfg["portfolio"]["max_weight"] = cap
        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, fh)
        fh.close()
        th = load_protocol(fh.name)
        opc = EvaluationOperator(th)
        opc._panels = op._panels          # reuse the loaded panel
        try:
            res = opc.evaluate(sig, zoo_signals={}, zoo_metrics=[], report=False)
        except Exception as exc:
            print(f"{cap:>6.2f}   FAILED {type(exc).__name__}: {exc}")
            continue

        # position statistics, from the book this cap actually produced
        npos = atcap = float("nan")
        try:
            from quantaalpha.eval.portfolio import build_book
            from quantaalpha.eval import combiner as C
            pred, _ = C.fit_predict(sig, None, panel, th)
            close = pd.DataFrame(panel.close, index=pd.DatetimeIndex(panel.dates),
                                 columns=list(panel.instruments))
            w, _ = build_book(pred.loc[win[0]:win[1]], th,
                              universe=panel.universe.loc[win[0]:win[1]],
                              close=close.loc[win[0]:win[1]])
            npos = float((w > 1e-9).sum(axis=1).median())
            atcap = float((w >= cap - 1e-9).sum(axis=1).median())
        except Exception:
            pass

        r = {"cap": cap, "positions": npos, "at_cap": atcap,
             "tc": res.get("m_transfer_coefficient"),
             "net_ir": res.get("m_net_ir"), "net_arr": res.get("m_net_arr"),
             "turnover": res.get("m_turnover_book"),
             "rank_ic": res.get("m_rank_ic")}
        rows.append(r)

        def f(v, spec="{:+.4f}"):
            try:
                return spec.format(float(v))
            except (TypeError, ValueError):
                return "n/a"

        print(f"{cap:>6.2f} {npos:>10.0f} {atcap:>7.0f} {f(r['tc'], '{:+.3f}'):>8} "
              f"{f(r['net_ir'], '{:+.3f}'):>9} {f(r['net_arr'], '{:+.2%}'):>9} "
              f"{f(r['turnover'], '{:.4f}'):>9}  {time.time() - t0:.0f}", flush=True)

    ok = [r for r in rows if isinstance(r.get("net_ir"), (int, float))
          and r["net_ir"] == r["net_ir"]]
    if ok:
        best = max(ok, key=lambda r: r["net_ir"])
        cur = next((r for r in ok if abs(r["cap"] - float(theta.portfolio.max_weight)) < 1e-9), None)
        print("-" * 74)
        print(f"  best net_ir at cap {best['cap']:.2f}: {best['net_ir']:+.3f} "
              f"({best['positions']:.0f} positions, TC {best['tc']:+.3f})")
        if cur:
            print(f"  current cap {cur['cap']:.2f}          : {cur['net_ir']:+.3f} "
                  f"({cur['positions']:.0f} positions, TC {cur['tc']:+.3f})")
            if best["cap"] != cur["cap"]:
                print(f"  => loosening/tightening to {best['cap']:.2f} is worth "
                      f"{best['net_ir'] - cur['net_ir']:+.3f} net_ir here")
        print("\n  CAVEAT: one book, one window, in-sample to the valid split. A cap "
              "is a Θ\n  parameter -- confirm on a second factor set before changing it.")

    Path(a.out).write_text(json.dumps(
        {"window": list(win), "n_factors": len(sig), "rows": rows}, indent=2,
        default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
