#!/usr/bin/env python
"""Per-YEAR IC / Rank IC / ARR for backtest v2 runs (no re-fit).

backtest v2 (quantaalpha/backtest/run_backtest.py) reports only full-window
scalars in its metrics JSON, but Qlib's SigAnaRecord persists the per-DATE IC and
Rank IC series in the recorder (mlruns/<exp>/<run>/artifacts/sig_analysis/{ic,ric}.pkl)
and the runner writes the daily excess return to
data/results/backtest_v2_results/<name>_cumulative_excess.csv.

This groups both by calendar year: IC / Rank IC from the recorder pickles, ARR
from the daily excess series. No model is refit.

    python scripts/qa_report_bt2_yearly.py --out data/results/report_bt2_yearly.json
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BT2 = ROOT / "data/results/backtest_v2_results"
MLRUNS = ROOT / "mlruns"
TRADING_DAYS = 243

LIBS = {
    "main_zoo":  "all_factors_library_meanvar_20260828_194432_zoo",
    "main_full": "all_factors_library_meanvar_20260828_194432",
    "original":  "all_factors_library_original_20260831_012324",
}


def annualize(r: pd.Series):
    r = pd.Series(r).astype(float).dropna()
    if r.empty:
        return (np.nan,) * 4
    arr = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ir = float(arr / vol) if vol > 0 else np.nan
    curve = (1.0 + r).cumprod()
    mdd = float((curve / curve.cummax() - 1.0).min())
    return arr, vol, ir, mdd


def find_recorder(exp_name: str):
    """Newest mlruns run whose tags name this experiment, holding sig_analysis."""
    best = None
    for meta in MLRUNS.glob("*/meta.yaml"):
        try:
            txt = meta.read_text()
        except OSError:
            continue
        if f"name: {exp_name}" not in txt and f"name: '{exp_name}'" not in txt:
            continue
        for run in meta.parent.iterdir():
            sig = run / "artifacts" / "sig_analysis"
            if sig.is_dir() and (sig / "ic.pkl").exists():
                mt = (sig / "ic.pkl").stat().st_mtime
                if best is None or mt > best[0]:
                    best = (mt, sig)
    return best[1] if best else None


def load_series(p: Path):
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    s = pd.Series(obj).astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/report_bt2_yearly.json")
    a = ap.parse_args()

    out = {"test_window": None,
           "source": "backtest v2 (configs/backtest.yaml): LGBM + TopkDropout "
                     "topk50/n_drop5, flat fee, test 2022-01-01..2025-12-26",
           "libraries": {}}
    for label, stem in LIBS.items():
        mj = BT2 / f"{stem}_backtest_metrics.json"
        csv = BT2 / f"{stem}_cumulative_excess.csv"
        if not mj.exists():
            print(f"[{label}] MISSING {mj.name} -- run backtest v2 first")
            continue
        meta = json.loads(mj.read_text())
        exp_name = meta.get("experiment_name")
        # The recorder's ic.pkl spans the WHOLE dataset (train+valid+test). Only the
        # test window may be reported -- 2016-2021 are train/valid and reporting them
        # would break the no-leak guarantee.
        t_start, t_end = [x.strip() for x in
                          meta["config"]["test_range"].split("~")]
        rows = {}
        sig = find_recorder(exp_name) if exp_name else None
        ic = ric = None
        if sig:
            ic = load_series(sig / "ic.pkl").loc[t_start:t_end]
            ric = (load_series(sig / "ric.pkl").loc[t_start:t_end]
                   if (sig / "ric.pkl").exists() else None)
        exc = None
        if csv.exists():
            df = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
            exc = df["daily_excess_return"].astype(float)
        years = sorted(set(
            ([int(y) for y in ic.index.year.unique()] if ic is not None else []) +
            ([int(y) for y in exc.index.year.unique()] if exc is not None else [])))
        for yr in years:
            row = {}
            if ic is not None:
                s = ic[ic.index.year == yr]
                row["ic"] = float(s.mean()) if len(s) else np.nan
                row["icir"] = float(s.mean() / s.std()) if len(s) and s.std() > 0 else np.nan
            if ric is not None:
                s = ric[ric.index.year == yr]
                row["rank_ic"] = float(s.mean()) if len(s) else np.nan
                row["rank_icir"] = float(s.mean() / s.std()) if len(s) and s.std() > 0 else np.nan
            if exc is not None:
                r = exc[exc.index.year == yr]
                arr, _, ir_, mdd = annualize(r)
                row.update({"arr": arr, "ir": ir_, "mdd": mdd, "days": int(len(r))})
            rows[yr] = row
        full = {}
        if ic is not None:
            full["ic"] = float(ic.mean()); full["icir"] = float(ic.mean()/ic.std())
        if ric is not None:
            full["rank_ic"] = float(ric.mean()); full["rank_icir"] = float(ric.mean()/ric.std())
        if exc is not None:
            arr, _, ir_, mdd = annualize(exc)
            full.update({"arr": arr, "ir": ir_, "mdd": mdd, "days": int(len(exc))})
        rows["full"] = full
        out["test_window"] = [t_start, t_end]
        out["libraries"][label] = {"stem": stem, "n_factors": meta.get("num_factors"),
                                   "reported": meta.get("metrics"), "years": rows,
                                   "recorder": str(sig) if sig else None}
        print(f"[{label}] n={meta.get('num_factors')} full ic={full.get('ic')} "
              f"rank_ic={full.get('rank_ic')} arr={full.get('arr')}")
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
