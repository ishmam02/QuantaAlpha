#!/usr/bin/env python
"""Measure the OHLCV information ceiling vs what the miner actually reaches.

Two breadths, same valid window, same effective-rank formula (``exp(-sum p log p)``
over the |Spearman| correlation spectrum) the live eval operator uses:

  * DIVERSE bench  -> the OHLCV ceiling in the miner's OWN operator space.
    20 Alpha158 seeds + ~40 hand-authored expressions deliberately spanning every
    operator family the construct prompt exposes (TS_CORR, REGBETA, REGRESI, RSI,
    MACD, COUNT, TS_RANK, TS_ZSCORE, SKEW, KURT, DECAYLINEAR, BB_*, EMA/WMA, ...)
    across price/volume/vwap/return and windows 5/10/20/60. This is the breadth
    the data+vocab PERMITS; it is NOT the miner's output. Computed fresh via the
    CustomFactorCalculator (get_qlib_stock_data -> D.features) on CSI300.
  * MINER library  -> the breadth the miner actually reaches (150 proposed, cached).
    The prior session asserted this was 34.8 (an uncited code comment). Measuring
    it tests that claim on its own terms.

Three estimators (never one): entropy-rank, 90%-variance PC count, participation
ratio. Noise null-control confirms estimators do not deflate uncorrelated signals.

Quality ceiling: |RankIC| of the diverse bench on the 1d label vs mined
median 0.0216 / best 0.0567 (same window).
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr, label_frame_at  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)

SEEDS = {
    "ROC0": "($close-$open)/$open", "ROC1": "$close/DELAY($close,1)-1",
    "ROC5": "($close-DELAY($close,5))/DELAY($close,5)",
    "ROC10": "($close-DELAY($close,10))/DELAY($close,10)",
    "ROC20": "($close-DELAY($close,20))/DELAY($close,20)",
    "VRATIO5": "$volume/TS_MEAN($volume,5)",
    "VRATIO10": "$volume/TS_MEAN($volume,10)",
    "VSTD5_RATIO": "TS_STD($volume,5)/TS_MEAN($volume,5)",
    "RANGE": "($high-$low)/$open",
    "VOLATILITY5": "TS_STD($close,5)/$close",
    "VOLATILITY10": "TS_STD($close,10)/$close",
    "RET_VOL5": "TS_STD($close/DELAY($close,1)-1,5)",
    "RSV5": "($close-TS_MIN($low,5))/(TS_MAX($high,5)-TS_MIN($low,5)+1e-12)",
    "RSV10": "($close-TS_MIN($low,10))/(TS_MAX($high,10)-TS_MIN($low,10)+1e-12)",
    "HIGH_RATIO5": "$close/TS_MAX($high,5)-1",
    "LOW_RATIO5": "$close/TS_MIN($low,5)-1",
    "SHADOW_RATIO": "($high-$close)/($close-$low+1e-12)",
    "BODY_RATIO": "($close-$open)/($high-$low+1e-12)",
    "MA_RATIO5_10": "TS_MEAN($close,5)/TS_MEAN($close,10)-1",
    "MA_RATIO10_20": "TS_MEAN($close,10)/TS_MEAN($close,20)-1",
}
HAND = {
    "REV1": "(-1)*($close/DELAY($close,1)-1)", "MOM60": "($close-DELAY($close,60))/DELAY($close,60)",
    "RET_MEAN5": "TS_MEAN($close/DELAY($close,1)-1,5)",
    "RET_MEAN20": "TS_MEAN($close/DELAY($close,1)-1,20)",
    "DECAY_RET20": "DECAYLINEAR($close/DELAY($close,1)-1,20)",
    "WMA_RET20": "WMA($close/DELAY($close,1)-1,20)",
    "EMA_RET20": "EMA($close/DELAY($close,1)-1,20)",
    "VOL20": "TS_STD($close/DELAY($close,1)-1,20)", "VOL60": "TS_STD($close/DELAY($close,1)-1,60)",
    "SKEW20": "TS_SKEW($close/DELAY($close,1)-1,20)", "KURT20": "TS_KURT($close/DELAY($close,1)-1,20)",
    "ZSCORE_RET20": "TS_ZSCORE($close/DELAY($close,1)-1,20)", "MAD_RET20": "TS_MAD($close/DELAY($close,1)-1,20)",
    "BOLL_Z": "($close-TS_MEAN($close,20))/(TS_STD($close,20)+1e-8)",
    "STOCH20": "($close-TS_MIN($close,20))/(TS_MAX($close,20)-TS_MIN($close,20)+1e-12)",
    "TSRANK_CLOSE20": "TS_RANK($close,20)", "BBDEV_UPPER": "BB_UPPER($close,20,2)-$close",
    "VRATIO20": "$volume/TS_MEAN($volume,20)-1", "VSTD20": "TS_STD($volume,20)/(TS_MEAN($volume,20)+1)",
    "PV_CORR20": "TS_CORR($close,$volume,20)",
    "AMIHUD20": "TS_MEAN(ABS($close/DELAY($close,1)-1),20)/($volume+1)",
    "SIGNED_VOL": "SIGN($close/DELAY($close,1)-1)*$volume", "LOG_VOL": "LOG($volume+1)",
    "VWAP_DEV": "($close-$vwap)/$close", "VWAP_DEV_Z": "($close-$vwap)/(TS_STD($close-$vwap,20)+1e-8)",
    "INTRADAY": "($close-$open)/$close", "OVERNIGHT": "($open-DELAY($close,1))/DELAY($close,1)",
    "HL_RANGE": "($high-$low)/$close",
    "REGBETA_PV": "REGBETA($close,$volume,20)", "REGRESI_PV": "REGRESI($close,$volume,20)",
    "UPDAY_FRAC": "COUNT($close/DELAY($close,1)-1>0,20)/20", "RSI14": "RSI($close,14)",
    "MACD": "MACD($close,12,26,9)",
    "RANK_CLOSE": "RANK($close)", "RANK_CLOSE_MA": "RANK($close)-RANK(TS_MEAN($close,20))",
    "RANK_VOL": "RANK($volume)", "ABSRET_SUM20": "TS_SUM(ABS($close/DELAY($close,1)-1),20)",
    "SIGN_RET": "SIGN($close/DELAY($close,1)-1)",
}


def eff_rank_estimators(R: np.ndarray) -> dict:
    n = R.shape[0]
    ev = np.linalg.eigvalsh(R)[::-1]
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return {"n": n, "entropy": float("nan"), "pc90": 0, "pr": float("nan")}
    p = ev / ev.sum()
    cum = np.cumsum(ev) / ev.sum()
    return {"n": n, "entropy": float(np.exp(-(p * np.log(p)).sum())),
            "pc90": int(np.searchsorted(cum, 0.90) + 1),
            "pr": float((ev.sum() ** 2) / (ev ** 2).sum())}


def corr_matrix(signals: dict) -> np.ndarray:
    keys = list(signals)
    n = len(keys)
    R = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        c = _cross_sectional_corr(signals[keys[i]], signals[keys[j]], "spearman")
        v = abs(float(c.mean())) if not c.empty else 0.0
        R[i, j] = R[j, i] = v
    return R


def compute_diverse(exprs, calc, panel, lo, hi) -> dict:
    out = {}
    for k, e in exprs.items():
        try:
            s = calc.calculate_factor(k, e)
            if s is None:
                continue
            wide = s.unstack(level=0)                       # instrument->cols, date->rows
            wide = wide.reindex(index=panel.dates, columns=panel.instruments)
            out[k] = wide.loc[lo:hi]
        except Exception:
            pass
    return out


def load_cached(exprs, panel, lo, hi) -> dict:
    out = {}
    for k, e in exprs.items():
        try:
            s = load_aligned_signal(e, panel)
            out[k] = s.loc[lo:hi]
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--mined-library",
                    default="data/_archive_pre_20260824/factorlib/all_factors_library_meanvar_fixed_s42.json")
    ap.add_argument("--live-zoo",
                    default="data/factorlib/all_factors_library_full_20260824_002254_zoo.json")
    ap.add_argument("--out", default="data/results/ohlcv_ceiling.json")
    ap.add_argument("--no-mined", action="store_true")
    a = ap.parse_args()

    from quantaalpha.backtest.custom_factor_calculator import (
        CustomFactorCalculator, get_qlib_stock_data)

    theta = load_protocol(a.protocol or default_protocol_path())
    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)
    panel = op._panel(p0, p1)
    lo, hi = str(win[0]), str(win[1])
    lab = label_frame_at(panel, theta, 1).loc[lo:hi]
    print(f"protocol {theta.hash} | valid window {win}\n", flush=True)

    # DIVERSE bench (computed)
    start = (pd.Timestamp(win[0]) - pd.Timedelta(200, "D")).strftime("%Y-%m-%d")
    cfg = {"data": {"provider_uri": "data/qlib/cn_data", "region": "cn",
                    "market": "csi300", "start_time": start, "end_time": str(win[1])}}
    data_df = get_qlib_stock_data(cfg)
    calc = CustomFactorCalculator(data_df=data_df, auto_extract_cache=False)
    bench = {**SEEDS, **HAND}
    print(f"  diverse bench: {len(bench)} expressions (computing...)", flush=True)
    bsig = compute_diverse(bench, calc, panel, lo, hi)
    print(f"  loadable: {len(bsig)}", flush=True)
    bics = []
    for s in bsig.values():
        c = _cross_sectional_corr(s, lab, "spearman").dropna()
        if len(c) >= 100:
            bics.append(abs(float(c.mean())))
    bics = sorted(bics, reverse=True)
    bench_er = eff_rank_estimators(corr_matrix(bsig)) if len(bsig) >= 2 else {"n": len(bsig)}

    # NOISE null
    rng = np.random.default_rng(42)
    dates = panel.dates[(panel.dates >= lo) & (panel.dates <= hi)]
    instr = panel.instruments
    n_noise = min(len(bsig), 50)
    noise = {f"N{i}": pd.DataFrame(rng.standard_normal((len(dates), len(instr))),
                                    index=list(dates), columns=list(instr)) for i in range(n_noise)}
    noise_er = eff_rank_estimators(corr_matrix(noise))

    # MINER (cached)
    mined_er = live_er = None
    if not a.no_mined:
        for tag, path in [("mined150", a.mined_library), ("live_zoo", a.live_zoo)]:
            if not Path(path).exists():
                continue
            payload = json.loads(Path(path).read_text())
            f = payload.get("factors", payload)
            items = list(f.values()) if isinstance(f, dict) else list(f)
            exprs = [e.get("factor_expression") or e.get("expression") for e in items if isinstance(e, dict)]
            exprs = [e for e in exprs if e]
            d = {f"{tag}_{i}": e for i, e in enumerate(exprs)}
            msig = load_cached(d, panel, lo, hi)
            print(f"  {tag}: {len(exprs)} exprs, {len(msig)} loadable", flush=True)
            er = eff_rank_estimators(corr_matrix(msig)) if len(msig) >= 2 else {"n": len(msig)}
            if tag == "mined150":
                mined_er = er
            else:
                live_er = er

    print("\n" + "=" * 66)
    print(f"  WINDOW {win}   (valid -- what the admission gate scores on)")
    print("=" * 66)
    print(f"  {'set':38s} {'n':>4s} {'entropy':>8s} {'pc90':>5s} {'pr':>6s}")
    print("  " + "-" * 62)
    def row(label, er):
        return (f"  {label:38s} {er.get('n',0):>4} "
                f"{er.get('entropy',float('nan')):>8.1f} {er.get('pc90',0):>5} "
                f"{er.get('pr',float('nan')):>6.1f}")
    print(row("DIVERSE bench (OHLCV ceiling, vocab)", bench_er))
    if mined_er:
        print(row("MINER 150-factor proposed library", mined_er))
    if live_er:
        print(row("MINER live zoo (37 admitted)", live_er))
    print(row("NOISE null (should ~= n)", noise_er))
    print("  " + "-" * 62)
    print(f"  prior-session asserted (uncited comment): 34.8")
    if bics:
        print(f"\n  QUALITY diverse |RankIC| 1d: median {np.median(bics):.4f} best {bics[0]:.4f} n={len(bics)}")
        print(f"  MINED (same window, prior) : median 0.0216 best 0.0567")
        print(f"  published / mined (median) : {np.median(bics)/0.0216:.2f}x")

    Path(a.out).write_text(json.dumps({
        "window": list(win), "protocol": theta.hash,
        "diverse_bench": bench_er, "noise_null": noise_er,
        "mined150": mined_er, "live_zoo": live_er,
        "quality_bench_median_ic": float(np.median(bics)) if bics else None,
        "quality_bench_best_ic": float(bics[0]) if bics else None,
    }, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())