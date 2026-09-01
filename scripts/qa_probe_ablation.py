"""One-shot probe: does Fix A make the ablation actually compute on the real panel?

Replicates controller._get_ablation_eval's closure exactly (the long-frame build
+ CustomFactorCalculator + eval_signal/score + ablate) and runs it on a real qlib
panel + a representative expression. Prints whether calculate_factor now returns
a real signal (vs the 30x "Factor computation failed [abl]" before the fix) and
whether the SegmentAblation carries non-NaN metrics + a non-empty summary.

NOT a mine: no evolution loop, no factor generation, no admission. One qlib load.
"""
import os
import sys

import numpy as np
import pandas as pd

# Match the smoke's protocol (the surviving single-arm protocol).
os.environ.setdefault("QA_PROTOCOL",
                      "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")

from quantaalpha.eval.data import align_signal
from quantaalpha.eval.metrics import (
    _cross_sectional_corr, _slice, label_frame, newey_west_t,
)
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import default_protocol_path, load_protocol
from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator
from quantaalpha.pipeline.evolution.segment_ablation import ablate

theta = load_protocol(os.environ.get("QA_PROTOCOL") or default_protocol_path())
op = EvaluationOperator(theta)
start, end, win = op._windows(False)
panel = op._panel(start, end)
label = label_frame(panel, theta)
print(f"[probe] panel loaded: dates={len(panel.dates)} instruments={len(panel.instruments)}")

# --- the Fix A long-frame build (verbatim from controller._get_ablation_eval) ---
_fields = (("$open", panel.open), ("$high", panel.high),
           ("$low", panel.low), ("$close", panel.close),
           ("$volume", panel.volume), ("$amount", panel.amount),
           ("$vwap", panel.vwap))
long_df = pd.concat({n: f.stack() for n, f in _fields}, axis=1)
long_df.index.names = ["datetime", "instrument"]
print(f"[probe] long_df: shape={long_df.shape} cols={list(long_df.columns)} "
      f"index.names={long_df.index.names} dtypes_ok={all(long_df[c].dtype.kind in 'fi' for n,c in [('x',c) for c in long_df.columns])}")

calc = CustomFactorCalculator(data_df=long_df, auto_extract_cache=False)

_nan = float("nan")
_empty = pd.Series(dtype=float)


def eval_signal(sub_expr):
    sig = calc.calculate_factor("abl", sub_expr)
    if sig is None or (hasattr(sig, "empty") and sig.empty):
        return None
    return align_signal(sig, panel)


def _rank_turnover(handle):
    if handle is None:
        return _nan
    s = _slice(handle, win)
    if s.empty:
        return _nan
    uni = _slice(panel.universe, win).reindex(
        index=s.index, columns=s.columns).fillna(False)
    ranks = s.where(uni).rank(axis=1, pct=True)
    d = np.abs(np.diff(ranks.values, axis=0))
    both = uni.values[1:] & uni.values[:-1]
    if not both.any():
        return _nan
    return float(np.nanmean(np.where(both, d, np.nan)))


def score(handle):
    if handle is None:
        return {"rank_ic": _nan, "t_nw": _nan, "ic_pos_frac": _nan,
                "monotonicity": _nan, "turnover_solo": _nan, "ric_series": _empty}
    ric = _cross_sectional_corr(_slice(handle, win), label, "spearman").dropna()
    if ric.empty:
        return {"rank_ic": _nan, "t_nw": _nan, "ic_pos_frac": _nan,
                "monotonicity": _nan, "turnover_solo": _nan, "ric_series": _empty}
    return {"rank_ic": float(ric.mean()), "t_nw": float(newey_west_t(ric)),
            "ic_pos_frac": float((ric > 0).mean()), "monotonicity": _nan,
            "turnover_solo": _rank_turnover(handle),
            "ric_series": ric}


# Representative expressions: the smoke's admitted factor + a windowed core.
EXPR = "TS_SUM($volume * SIGN($close / DELAY($close, 1)), 3)"
EXPR2 = "ZSCORE(TS_MEAN(($close-$low)/($high-$low+1e-12), 5))"

print("\n=== eval_signal probe (sub-expression rendering) ===")
for sub in ["$close", "TS_MEAN($close, 5)", "($close-$low)/($high-$low+1e-12)",
            "SIGN($close / DELAY($close, 1))", "$volume * SIGN($close / DELAY($close, 1))"]:
    h = eval_signal(sub)
    sc = score(h)
    print(f"  {sub[:48]:48} -> handle={'None' if h is None else f'{h.shape}'} "
          f"rank_ic={sc['rank_ic']:.4f} t_nw={sc['t_nw']:.3f} "
          f"turnover={sc['turnover_solo']:.3f}")

print("\n=== full ablate() ===")
for expr, sign in [(EXPR, "positive"), (EXPR2, "positive")]:
    abl = ablate(expr, sign, eval_signal=eval_signal, score=score)
    pp = abl.per_part
    n_real = sum(1 for v in pp.values()
                 if v is not None and not pd.isna(getattr(v, "rank_ic", _nan)))
    print(f"\n  EXPR: {expr}")
    print(f"  per_part entries: {len(pp)} (non-NaN rank_ic: {n_real})")
    for sig, v in list(pp.items())[:6]:
        d = v.as_dict() if hasattr(v, "as_dict") else dict(v)
        print(f"    {sig[:44]:44} rank_ic={d.get('rank_ic'):.4f} "
              f"t_nw={d.get('t_nw'):.2f} turnover={d.get('turnover_solo'):.3f} "
              f"role={d.get('role')} op={d.get('op')} win={d.get('window')}")
    ws = abl.window_sensitivity
    print(f"  window_sensitivity: {list(ws.keys())} "
          f"ic_neutral={[w.get('ic_neutral') for w in ws.values()]}")
    cs = abl.core_sign_stability
    print(f"  core_sign_stability: stable={cs.get('stable')} "
          f"ic_pos_frac={cs.get('ic_pos_frac')}")
    print(f"  summary ({len(abl.summary)} chars):\n    {abl.summary[:600]}")
print("\n[probe] DONE")