#!/usr/bin/env python
"""Backtest the factors the live reseed test GENERATED, and compare to r0.

The live test (``qa_reseed_r0_live_test.py``) ran the reseed direction planner +
factor generator under two conditions (OFF = old digest, ON = Layer-1 fix) against
real glm-5.2:cloud, with the r0 (round-0, blind) trajectories as the only context.
It generated factors but did NOT backtest them -- so "are they better than r0?"
could not be answered. This scores them.

Faithfulness. The tearsheet the mine admits on is the SOLO neutralized rank IC +
its Newey-West t, computed per factor on the VALID window (2013-2015, the
``oos_window`` of ``_windows(False)`` -- protocol hash ``fbefcb65f408aee0``).
This script replicates that EXACT loop (``net_cost_runner._decide_standalone``
lines 835-901): ``align_signal`` -> ``residualize`` -> per-horizon
``_cross_sectional_corr`` -> ``newey_west_t`` -> the more conservative of
t_nw / t_overlap -> best |t|. No gates run here (the tearsheet is computed BEFORE
any gate and stored on every rejection path), so the IC measurement is clean
regardless of the mechanism / sign / marginal-er verdicts -- which is exactly the
quantity r0's pool tearsheets hold.

r0 baseline is read from the production pool (the 10 round-0 tearsheets), so the
comparison is on the same window + same neutralization the mine scored r0 on.

Run with the mine STOPPED -- this loads the price-volume panel and is memory-heavy
(the box is swap-pressed while the mine runs; a concurrent load can OOM the mine).

    python scripts/qa_reseed_factor_score.py
"""
from __future__ import annotations

import json
import os
import sys
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Throwaway ledger/workspace so NOTHING touches production paths. This script
# writes nothing to the library/ledger (it only reads the pool + computes IC),
# but EvaluationOperator / data imports may read these env vars.
os.environ.setdefault("EXPERIMENT_ID", "qa_reseed_score")
os.environ.setdefault("WORKSPACE_PATH", "/tmp/qa_reseed_score_ws")
os.environ.setdefault("PICKLE_CACHE_FOLDER_PATH_STR", "/tmp/qa_reseed_score_pickle")
os.environ.setdefault("QA_LEDGER", "/tmp/qa_reseed_score_ledger.jsonl")
# The marginal-er gate is a verdict gate, not an IC measurement; turn it off so
# nothing here can look like it depends on repo state (there is no repo).
os.environ["QA_MIN_MARGINAL_ER"] = "0"
for _d in ("/tmp/qa_reseed_score_ws", "/tmp/qa_reseed_score_pickle"):
    Path(_d).mkdir(parents=True, exist_ok=True)

PROD_POOL = REPO / "data" / "results" / "trajectory_pool_meanvar_20260825_123942.json"

# The 6 factors the live test generated (extracted verbatim from its log).
# 3 OFF-condition (old digest, no gate) + 3 ON-condition (Layer-1 fix).
GENERATED = [
    # ---- OFF (old digest, no failure memory / no gate) ----
    ("OFF", "OFF-1 skew-zscore-mean",
     "TS_MEAN(POW(($return - TS_MEAN($return, 20)) / (TS_STD($return, 20) + 1e-8), 3), 20)"),
    ("OFF", "OFF-2 rank-pctchange-close",
     "RANK(TS_PCTCHANGE($close, 5))"),
    ("OFF", "OFF-3 regbeta-close-60d",
     "REGBETA($close, SEQUENCE(60), 60) / (TS_STD($close, 60) + 1e-8)"),
    # ---- ON (failure memory + sign lesson + operator-novelty gate) ----
    ("ON",  "ON-1 ts-rank-return-60d",
     "TS_RANK($return, 60)"),
    ("ON",  "ON-2 skew-mean-over-std-cubed",
     "TS_MEAN(POW($return - TS_MEAN($return, 20), 3), 20) / (POW(TS_STD($return, 20), 3) + 1e-8)"),
    ("ON",  "ON-3 delta-delta-close-5d",
     "DELTA(DELTA($close, 5), 5) / (DELAY($close, 5) + 1e-8)"),
]


def load_r0_tearsheets():
    """The 10 round-0 pool tearsheets -- the r0 baseline the mine scored."""
    d = json.load(open(PROD_POOL))
    ts = d["trajectories"]
    items = list(ts.values()) if isinstance(ts, dict) else ts
    r0 = [t for t in items if t.get("round_idx") == 0 and t.get("phase") == "original"]
    out = []
    for t in sorted(r0, key=lambda x: x.get("direction_id", 0)):
        bm = t.get("backtest_metrics") or {}
        for fname, s in (bm.get("factor_tearsheets") or {}).items():
            out.append({
                "dir": t.get("direction_id"), "name": fname,
                "rank_ic_neutral": s.get("rank_ic_neutral"),
                "t_nw": s.get("t_nw"), "best_horizon": s.get("best_horizon"),
                "rank_ic": s.get("rank_ic"),
            })
    return out


def main() -> int:
    import argparse
    import pandas as pd
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol
    from quantaalpha.eval.operator import EvaluationOperator
    from quantaalpha.eval.data import align_signal
    from quantaalpha.eval.metrics import (
        _cross_sectional_corr, _slice, label_frame_at, newey_west_t,
    )
    from quantaalpha.factors.net_cost_runner import _label_horizons
    from quantaalpha.eval.neutralize import residualize
    from quantaalpha.backtest.custom_factor_calculator import (
        CustomFactorCalculator, get_qlib_stock_data)

    ap = argparse.ArgumentParser()
    ap.add_argument("--factors-json", default=None,
                    help="read factors from this JSON (qa_reseed_r0_live_test.py --dump); "
                         "else use the hardcoded 6-factor list")
    args = ap.parse_args()

    if args.factors_json:
        import json as _json
        loaded = _json.loads(Path(args.factors_json).read_text())
        factors_list = [(x["cond"], x["name"], x["expr"]) for x in loaded]
        print(f"loaded {len(factors_list)} factors from {args.factors_json}")
    else:
        factors_list = GENERATED
        print(f"using hardcoded {len(factors_list)}-factor list")

    # The mine (run.sh:235) exports QA_PROTOCOL to the soft_linear variant, which
    # is what r0 was scored under (ledger theta_hash=fbefcb65f408aee0). Load it
    # explicitly so the comparison is on the EXACT protocol r0's tearsheets used --
    # default_protocol_path() alone resolves to protocol_csi300.yaml (a different
    # hash) when QA_PROTOCOL is unset.
    PROTOCOL = "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml"
    theta = load_protocol(PROTOCOL)
    print(f"protocol {theta.hash}  ({PROTOCOL})")
    assert theta.hash == "fbefcb65f408aee0", (
        f"protocol hash moved ({theta.hash}); r0 tearsheets used fbefcb65f408aee0 -- "
        "comparison not apples-to-apples")

    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)            # valid window 2013-2015
    panel = op._panel(p0, p1)
    horizons = _label_horizons(theta)
    labels = {h: label_frame_at(panel, theta, h) for h in horizons}
    print(f"valid window {win}  horizons={horizons}  panel {panel.data.shape if hasattr(panel,'data') else '?'}")

    # Data for the factor calculator: warmup before the valid window.
    start = (pd.Timestamp(win[0]) - pd.Timedelta(200, "D")).strftime("%Y-%m-%d")
    provider = os.environ.get("QLIB_DATA_DIR", "data/qlib/cn_data")
    cfg = {"data": {"provider_uri": provider, "region": "cn", "market": "csi300",
                    "start_time": start, "end_time": str(win[1])}}
    print(f"loading price-volume data {start}..{win[1]} ...", flush=True)
    data_df = get_qlib_stock_data(cfg)
    calc = CustomFactorCalculator(data_df=data_df, auto_extract_cache=False)
    print(f"  data: {len(data_df)} rows", flush=True)

    neutralized_ok = True

    def score(expr):
        nonlocal neutralized_ok
        sig = calc.calculate_factor("f", expr)
        if sig is None or sig.empty:
            return {"rank_ic_neutral": float("nan"), "t_nw": float("nan"),
                    "best_horizon": None, "rank_ic": float("nan"), "n_obs": 0,
                    "error": "empty signal"}
        # The calculator returns a (instrument, datetime) MultiIndex Series -- the
        # inverse of what align_signal expects -- so unstack level=0 (instrument
        # -> columns, datetime -> rows) and reindex onto the panel grid, exactly as
        # qa_ohlcv_ceiling.compute_diverse does. The 2005-2012 rows have no data
        # (loaded only from 200 days before the valid window) -> NaN, sliced out.
        wide = sig.unstack(level=0)
        sig_raw = wide.reindex(index=panel.dates, columns=panel.instruments)
        sig = sig_raw
        try:
            sig = residualize(sig_raw, panel, theta)
        except Exception as exc:
            neutralized_ok = False
            sig = sig_raw
        sliced = _slice(sig, win)
        best = None
        for h, lab in labels.items():
            ic = _cross_sectional_corr(sliced, lab, "spearman").dropna()
            n = len(ic)
            if n < 100:
                continue
            mean, sd = float(ic.mean()), float(ic.std())
            if sd <= 0 or sd != sd:
                continue
            n_eff = max(n / float(h), 2.0)
            t_overlap = mean / (sd / (n_eff ** 0.5))
            t_nw = newey_west_t(ic)
            t = t_nw if (t_nw == t_nw and abs(t_nw) < abs(t_overlap)) else t_overlap
            pos = float((ic > 0).mean()) if mean >= 0 else float((ic < 0).mean())
            if best is None or abs(t) > abs(best[1]):
                best = (h, t, n, mean, pos, sd)
        if best is None:
            return {"rank_ic_neutral": float("nan"), "t_nw": float("nan"),
                    "best_horizon": None, "rank_ic": float("nan"), "n_obs": 0}
        h, t, n, ric, pos, sd = best
        # raw rank IC at the winning horizon (raw vs neutralized gap)
        try:
            raw_ic = _cross_sectional_corr(_slice(sig_raw, win), labels[h],
                                           "spearman").dropna()
            raw_rank_ic = float(raw_ic.mean()) if len(raw_ic) >= 100 else float("nan")
        except Exception:
            raw_rank_ic = float("nan")
        return {"rank_ic_neutral": ric, "t_nw": t, "best_horizon": h,
                "rank_ic": raw_rank_ic, "n_obs": n, "ic_pos_frac": pos}

    # Score the 6 generated factors.
    print("\nscored generated factors on the VALID window (solo neutralized IC):")
    results = []
    for cond, name, expr in factors_list:
        try:
            r = score(expr)
        except Exception as exc:
            r = {"rank_ic_neutral": float("nan"), "t_nw": float("nan"),
                 "best_horizon": None, "error": repr(exc)}
        r.update({"cond": cond, "name": name, "expr": expr})
        results.append(r)
        ric = r.get("rank_ic_neutral", float("nan"))
        t = r.get("t_nw", float("nan"))
        h = r.get("best_horizon")
        err = r.get("error", "")
        flag = "  <-- ERROR" if err else ""
        print(f"  [{cond}] {name:28s} ric_neut={ric:+.4f}  t_nw={t:+.2f}  h={h}  |t|={abs(t) if t==t else float('nan'):.2f}{flag}")
        if err:
            print(f"        err: {err}")

    # r0 baseline.
    r0 = load_r0_tearsheets()
    r0_t = [abs(s["t_nw"]) for s in r0 if s["t_nw"] == s["t_nw"]]
    r0_ric = [s["rank_ic_neutral"] for s in r0 if s["rank_ic_neutral"] == s["rank_ic_neutral"]]
    r0_clear3 = sum(1 for t in r0_t if t >= 3.0)

    def median(xs):
        xs = sorted(xs); m = len(xs)
        return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])

    print(f"\n{'='*78}\nr0 baseline (round-0 pool tearsheets, same valid window):")
    print(f"  n={len(r0)}  median |t_nw|={median(r0_t):.2f}  max |t_nw|={max(r0_t):.2f}  "
          f"clear |t|>=3.0: {r0_clear3}/{len(r0)}")
    print(f"  median rank_ic_neutral={median(r0_ric):+.4f}  "
          f"(neg share: {sum(1 for x in r0_ric if x<0)}/{len(r0_ric)})")
    print(f"  strongest r0: {max(r0, key=lambda s: abs(s['t_nw'] or 0))['name'][:40]}  "
          f"t={max(r0, key=lambda s: abs(s['t_nw'] or 0))['t_nw']:+.2f}")

    # Generated vs r0.
    print(f"\n{'='*78}\nGENERATED vs r0:")
    for cond in ("OFF", "ON"):
        grp = [r for r in results if r["cond"] == cond and r.get("t_nw") == r.get("t_nw")]
        ts = [abs(r["t_nw"]) for r in grp]
        rics = [r["rank_ic_neutral"] for r in grp if r["rank_ic_neutral"] == r["rank_ic_neutral"]]
        clear3 = sum(1 for t in ts if t >= 3.0)
        print(f"  {cond}: n={len(grp)}  median |t_nw|={median(ts) if ts else float('nan'):.2f}  "
              f"max |t_nw|={max(ts) if ts else float('nan'):.2f}  clear>=3.0: {clear3}/{len(grp)}  "
              f"median ric={median(rics) if rics else float('nan'):+.4f}")

    # Head-to-head: each generated factor vs the r0 median |t|.
    r0_med = median(r0_t)
    print(f"\n  (r0 median |t_nw| = {r0_med:.2f}; a generated factor BEATS r0 if |t| > {r0_med:.2f})")
    print(f"  {'cond':4s} {'name':28s} {'|t_nw|':>7s} {'beats r0 median?':>18s}")
    for r in results:
        t = r.get("t_nw", float("nan"))
        at = abs(t) if t == t else float("nan")
        beats = (at > r0_med) if at == at else False
        print(f"  {r['cond']:4s} {r['name']:28s} {at:7.2f} {'YES' if beats else 'no':>18s}")

    # ---- Conclusive non-parametric comparison (Mann-Whitney U + Cliff's delta). ----
    # Small n (one reseed round per arm): a difference in medians alone is not a
    # verdict. The two-sided Mann-Whitney U (normal approx, tie-corrected) tests
    # "same distribution"; Cliff's delta is the non-parametric effect size
    # (|d|<0.147 negligible, <0.33 small, <0.474 medium, else large). Pure python
    # so no scipy dependency; the mine box may not have scipy in the env.
    import math as _math

    def _mw_u_p(x, y):
        """Two-sided Mann-Whitney U p-value (normal approx, tie-corrected)."""
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            return float("nan"), float("nan")
        allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
        ranks, i = [], 0
        tie_corr = 0.0
        while i < len(allv):
            j = i
            while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
                j += 1
            r = (i + 1 + j + 1) / 2.0
            t = j - i + 1
            tie_corr += (t ** 3 - t)
            for k in range(i, j + 1):
                ranks.append((r, allv[k][1]))
            i = j + 1
        r_x = sum(r for r, g in ranks if g == 0)
        u1 = r_x - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        u = min(u1, u2)
        n = n1 + n2
        mu = n1 * n2 / 2.0
        sigma = _math.sqrt((n1 * n2 / 12.0) * ((n + 1) - tie_corr / (n * (n - 1))))
        if sigma <= 0:
            return u, 1.0
        z = (u - mu + 0.5) / sigma if u < mu else (u - mu - 0.5) / sigma
        p = 2.0 * (1.0 - 0.5 * (1.0 + _math.erf(abs(z) / _math.sqrt(2.0))))
        return u, min(max(p, 0.0), 1.0)

    def _cliffs_delta(x, y):
        """Cliff's delta: P(x>y) - P(x<y), in [-1, 1]."""
        if not x or not y:
            return float("nan")
        s = 0.0
        for a in x:
            for b in y:
                if a > b:
                    s += 1
                elif a < b:
                    s -= 1
        return s / (len(x) * len(y))

    def _delta_label(d):
        d = abs(d)
        if d != d:
            return "?"
        if d < 0.147:
            return "negligible"
        if d < 0.33:
            return "small"
        if d < 0.474:
            return "medium"
        return "large"

    grp_t = {}
    for cond in ("OFF", "ON"):
        grp_t[cond] = [abs(r["t_nw"]) for r in results
                       if r["cond"] == cond and r.get("t_nw") == r.get("t_nw")]
    pairs = [("OFF", "r0", grp_t.get("OFF", []), r0_t),
             ("ON", "r0", grp_t.get("ON", []), r0_t),
             ("ON", "OFF", grp_t.get("ON", []), grp_t.get("OFF", []))]
    print(f"\n{'='*78}\nNON-PARAMETRIC VERDICT (Mann-Whitney U two-sided + Cliff's delta on |t_nw|):")
    print(f"  {'pair':12s} {'n1':>3s} {'n2':>3s} {'med1':>6s} {'med2':>6s} "
          f"{'U':>7s} {'p':>8s} {'cliffs d':>9s} {'effect':>12s}")
    for a, b, xa, xb in pairs:
        u, p = _mw_u_p(xa, xb)
        d = _cliffs_delta(xa, xb)
        print(f"  {a+' vs '+b:12s} {len(xa):3d} {len(xb):3d} "
              f"{median(xa) if xa else float('nan'):6.2f} {median(xb) if xb else float('nan'):6.2f} "
              f"{u if u==u else float('nan'):7.1f} {p if p==p else float('nan'):8.3f} "
              f"{d if d==d else float('nan'):9.3f} {_delta_label(d):>12s}")
    print("  (d>0: left group has LARGER |t|; p<0.05 rejects 'same distribution')")

    print(f"\nneutralization: {'OK (rank_ic_neutral is neutralized, matches r0)' if neutralized_ok else 'FELL BACK TO RAW -- comparison compromised'}")
    if not neutralized_ok:
        print("  (the neutralize reference parquet was missing; re-run on a box where the mine has it)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())