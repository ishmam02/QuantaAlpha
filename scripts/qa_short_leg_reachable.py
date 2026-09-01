#!/usr/bin/env python
"""Gate 1 says the alpha is 62% short-leg. Can that leg actually be shorted?

A quantile measurement is dollar-neutral and cost-free by construction -- that
is what makes it the honest place to ask whether a signal sorts returns. It is
NOT a claim that the short leg is tradeable. In A-shares it very often is not:
shorting requires the name to be on the 融券 designated-securities list, which
skews hard toward large, liquid constituents, and borrow is charged daily
(``costs.beta_per_day`` already carries ~10%/yr for this, inert only because
``portfolio.signed`` is false).

So before "build long/short" follows from "the spread is in Q1", this asks what
the bottom decile is actually MADE of:

  * float market cap of Q1 vs Q10 -- from the derived `circ_mv` series, whose
    ORDERING was validated to 91% of attainable against the published CSI300
    weights (`qa_synth_benchmark.py`). Ordering is all this needs; the weight-
    level error that blocked the benchmark anchor does not matter here.
  * dollar volume of Q1 vs Q10 -- borrow availability tracks liquidity
  * how much of the short-leg return survives dropping the smallest names,
    which is the closest cheap proxy for a designated-securities restriction

If Q1 is the small illiquid tail, the measured short-leg alpha is largely
unreachable and gate 1's answer changes from "build long/short" to "the alpha
is in a place this market will not let a book go".

Usage::

    python scripts/qa_short_leg_reachable.py --library <zoo>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr, label_frame  # noqa: E402
from quantaalpha.eval.neutralize import residualize  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--drop-smallest", type=float, default=0.30,
                    help="fraction of the universe, by float cap, held to be "
                         "un-borrowable in the restricted arm")
    ap.add_argument("--out", default="data/results/short_leg_reachable.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
    payload = json.loads(Path(a.library).read_text())
    f = payload.get("factors", payload)
    items = f.values() if isinstance(f, dict) else f
    exprs = [e.get("factor_expression") or e.get("expression") for e in items]
    exprs = [e for e in exprs if e]

    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)
    panel = op._panel(p0, p1)
    lo, hi = str(win[0]), str(win[1])
    lab = label_frame(panel, theta).loc[lo:hi]
    print(f"protocol {theta.hash} | {win} | {len(exprs)} factors\n", flush=True)

    # float cap and dollar volume on the panel grid
    mc = pd.read_parquet("data/reference/market_cap.parquet")
    cap = (mc.pivot_table(index="date", columns="instrument", values="circ_mv",
                          aggfunc="last")
             .reindex(index=panel.dates, columns=panel.instruments).ffill())
    close = pd.DataFrame(panel.close, index=pd.DatetimeIndex(panel.dates),
                         columns=list(panel.instruments))
    vol = pd.DataFrame(panel.volume, index=pd.DatetimeIndex(panel.dates),
                       columns=list(panel.instruments))
    dv = (close * vol).reindex(index=panel.dates, columns=panel.instruments)
    cap_w, dv_w = cap.loc[lo:hi], dv.loc[lo:hi]
    print(f"  float cap coverage on the window: "
          f"{float(cap_w.notna().mean().mean()):.1%} of panel cells\n")

    # the un-borrowable mask: the smallest `drop` fraction by float cap, per date
    keep = cap_w.rank(axis=1, pct=True) > a.drop_smallest

    rows = []
    for i, e in enumerate(exprs, 1):
        try:
            s = residualize(load_aligned_signal(e, panel), panel, theta).loc[lo:hi]
        except Exception:
            continue
        ic = _cross_sectional_corr(s, lab, "spearman").dropna()
        if len(ic) < 100:
            continue
        s = s if float(ic.mean()) >= 0 else -s          # Q10 == "buy"
        q = np.ceil(s.rank(axis=1, pct=True) * 10).clip(1, 10)

        def leg(sel, frame):
            v = frame.where(sel).mean(axis=1).dropna()
            return float(v.mean()) if len(v) else np.nan

        mid = lab.mean(axis=1)
        short_full = (mid - lab.where(q == 1).mean(axis=1)).dropna()
        # the same short leg, but only among names large enough to borrow
        short_res = (mid - lab.where((q == 1) & keep).mean(axis=1)).dropna()
        rows.append({
            "expr": e,
            "cap_q1": leg(q == 1, np.log(cap_w)), "cap_q10": leg(q == 10, np.log(cap_w)),
            "dv_q1": leg(q == 1, np.log(dv_w + 1)), "dv_q10": leg(q == 10, np.log(dv_w + 1)),
            "short_full": float(short_full.mean()),
            "short_restricted": float(short_res.mean()),
            "borrowable_frac": float(((q == 1) & keep).sum(axis=1).sum()
                                     / max((q == 1).sum(axis=1).sum(), 1)),
        })
        print(f"  [{i:2d}] short {short_full.mean()*1e4:+6.1f} -> "
              f"{short_res.mean()*1e4:+6.1f}bp   "
              f"Q1 borrowable {rows[-1]['borrowable_frac']:.0%}", flush=True)

    if not rows:
        print("nothing measured")
        return 2
    d = pd.DataFrame(rows)
    surv = float(d.short_restricted.mean() / d.short_full.mean()) \
        if d.short_full.mean() else np.nan
    print("\n" + "=" * 70)
    print(f"SHORT-LEG REACHABILITY   n={len(d)} factors")
    print("=" * 70)
    print(f"  median log float cap   Q1 {d.cap_q1.median():.2f}   "
          f"Q10 {d.cap_q10.median():.2f}   "
          f"(Q1/Q10 cap ratio {np.exp(d.cap_q1.median()-d.cap_q10.median()):.2f}x)")
    print(f"  median log dollar vol  Q1 {d.dv_q1.median():.2f}   "
          f"Q10 {d.dv_q10.median():.2f}")
    print(f"  short leg, all names          {d.short_full.mean()*1e4:+7.2f} bp/day")
    print(f"  short leg, borrowable only    {d.short_restricted.mean()*1e4:+7.2f} bp/day")
    print(f"  Q1 names passing the size screen: {d.borrowable_frac.mean():.0%}")
    print(f"  fraction of short-leg alpha surviving: {surv:.0%}")
    print("-" * 70)
    if d.cap_q1.median() < d.cap_q10.median() - 0.35:
        print("  Q1 IS THE SMALL TAIL. The short leg sits in names materially")
        print("  smaller than the long leg, which is where borrow is thinnest.")
    else:
        print("  Q1 IS NOT A SIZE TAIL -- the bottom decile is comparable in")
        print("  size to the top, so borrow availability is not obviously the")
        print("  binding constraint.")
    if np.isfinite(surv) and surv > 0.70:
        print(f"  And {surv:.0%} of the short-leg return survives a size screen,")
        print("  so most of it lives in names large enough to plausibly borrow.")
    elif np.isfinite(surv):
        print(f"  Only {surv:.0%} survives a size screen -- much of the measured")
        print("  short alpha is in names a real book could not short.")

    Path(a.out).write_text(json.dumps(
        {"window": list(win), "drop_smallest": a.drop_smallest,
         "n": len(d), "survival": surv,
         "short_full": float(d.short_full.mean()),
         "short_restricted": float(d.short_restricted.mean()),
         "rows": rows}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
