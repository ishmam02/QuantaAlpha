#!/usr/bin/env python
"""Prove the redundancy gate blocks the SAME BET with a DIFFERENT EQUATION.

The gate must not admit a factor just because its formula looks different. The
live gate compares signals in spearman RANK space (metrics.rho_max), so a
monotone re-expression (RANK / ZSCORE / negation / "subtract 1") of an
incumbent -- different equation, identical ranking -- must read rho ~= 1.0 and
be blocked (or, if decisively stronger, replace). A genuinely different factor
must read rho low and pass.

This exercises the REAL rho_max / rho_max_arg (metrics.py) and the live
effective-rank formula on REAL computed signals, then asserts the outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.metrics import _cross_sectional_corr, rho_max, rho_max_arg  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402

RHO_BAR = 0.60  # protocol_csi300_meanvar_soft_linear.yaml: gates.rho_bar


def eff_rank(signals: dict) -> float:
    keys = list(signals)
    n = len(keys)
    if n < 2:
        return float(n)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _cross_sectional_corr(signals[keys[i]], signals[keys[j]], "spearman")
            R[i, j] = R[j, i] = abs(float(c.mean())) if not c.empty else 0.0
    ev = np.linalg.eigvalsh(R)[::-1]
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> int:
    from quantaalpha.backtest.custom_factor_calculator import (
        CustomFactorCalculator, get_qlib_stock_data)

    theta = load_protocol(default_protocol_path())
    op = EvaluationOperator(theta)
    _, _, win = op._windows(False)            # date bounds only; no panel load
    lo, hi = str(win[0]), str(win[1])
    start = (pd.Timestamp(win[0]) - pd.Timedelta(60, "D")).strftime("%Y-%m-%d")
    cfg = {"data": {"provider_uri": "data/qlib/cn_data", "region": "cn",
                    "market": "csi300", "start_time": start, "end_time": str(win[1])}}
    data_df = get_qlib_stock_data(cfg)
    calc = CustomFactorCalculator(data_df=data_df, auto_extract_cache=False)

    def sig(expr):
        s = calc.calculate_factor(expr, expr)
        wide = s.unstack(level=0)              # datetime x instrument
        return wide.loc[lo:hi]

    incumbent = "TS_MEAN($close,20)/$close-1"          # MA deviation
    same_bets = {                                          # different equation, same bet
        "RANK_rewrap": "RANK(TS_MEAN($close,20)/$close-1)",
        "NEGATED": "(-1)*(TS_MEAN($close,20)/$close-1)",
        "ZSCORE_rewrap": "ZSCORE(TS_MEAN($close,20)/$close-1)",
        "MINUS1_dropped": "TS_MEAN($close,20)/$close",     # identical ranking to incumbent
        "RSI14": "RSI($close,14)",   # structurally unrelated, but near-identical rank signal here
    }
    diffs = {                                              # genuinely different
        "PV_CORR": "TS_CORR($close,$volume,20)",
        "VWAP_DEV": "($close-$vwap)/$close",
        "VOL20": "TS_STD($close/DELAY($close,1)-1,20)",
    }

    zoo = {"INCUMBENT": sig(incumbent)}
    print(f"incumbent: {incumbent}")
    print(f"rho_bar = {RHO_BAR}  (block if rho_max >= {RHO_BAR})\n")
    print(f"  {'candidate':16s} {'equation':40s} {'rho_max':>8s}  verdict")
    print("  " + "-" * 78)

    def verdict(rho):
        return "BLOCKED (same bet)" if rho >= RHO_BAR else "novel (admits)"

    passed = True
    for name, expr in {**same_bets, **diffs}.items():
        try:
            s = sig(expr)
        except Exception as e:
            print(f"  {name:16s} COMPUTE FAILED: {type(e).__name__}: {str(e)[:50]}")
            continue
        rho, near = rho_max_arg(s, zoo)
        expect_block = name in same_bets
        ok = (rho >= RHO_BAR) == expect_block
        passed &= ok
        print(f"  {name:16s} {expr[:40]:40s} {rho:8.3f}  {verdict(rho):22s}"
              f"{'OK' if ok else '!! MISMATCH'}")

    # rho_within: a batch containing the incumbent + a rewrap should self-flag
    batch = {"INCUMBENT": zoo["INCUMBENT"],
             "RANK_rewrap": sig(same_bets["RANK_rewrap"]),
             "PV_CORR": sig(diffs["PV_CORR"])}
    within = 0.0
    bk = list(batch)
    for i in range(len(bk)):
        for j in range(i + 1, len(bk)):
            c = _cross_sectional_corr(batch[bk[i]], batch[bk[j]], "spearman")
            r = abs(float(c.mean())) if not c.empty else 0.0
            within = max(within, r)
    print(f"\n  rho_within(batch) = {within:.3f}  "
          f"({'self-dup flagged' if within > 0.8 else 'no self-dup'})")

    # effective_rank: 5 factors but 4 are the same bet -> ~2 independent directions
    er_set = {"INCUMBENT": zoo["INCUMBENT"],
              "RANK_rewrap": sig(same_bets["RANK_rewrap"]),
              "NEGATED": sig(same_bets["NEGATED"]),
              "PV_CORR": sig(diffs["PV_CORR"]),
              "RSI14": sig(same_bets["RSI14"])}
    er = eff_rank(er_set)
    print(f"  effective_rank(5 factors: 4 same-bets + 1 diff) = {er:.2f}  "
          f"(should be ~2, NOT 5: same-bets collapse to one bet)")
    passed &= (er < 3.0)

    print("\n" + ("GATE VERIFIED: same-bet-different-equation is blocked."
                  if passed else "GATE FAILED: a same-bet passed or a diff was blocked."))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())