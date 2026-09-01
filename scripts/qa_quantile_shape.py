#!/usr/bin/env python
"""Decision gate 1: is the alpha in the top decile or the bottom one?

`declarative-doodling-wadler.md` defers the long/short decision to exactly this
measurement rather than assuming it:

    Long/short constructor | Phase 1.2 quantile monotonicity | Build only if the
    alpha is one-sided -- Q1 underperforms but Q10 is unremarkable. If Q10
    carries it, long-only captures the alpha and shorting is unnecessary.

A long-only book can only express "hold this name" or "do not hold it". It has
no way to express "this name will fall". So if the return spread lives in the
BOTTOM decile -- the factor is good at finding losers -- a long-only book throws
that half away, and its measured transfer coefficient is bounded no matter how
good the signal or the optimiser gets.

Method, per factor, per date:

  * SIGN-ALIGN first. A negative-IC factor's "Q10" is the leg it says to SHORT.
    Without aligning, the two families cancel and the whole measurement is noise.
    Q10 always means "the decile this factor says to BUY".
  * neutralize against size / industry / beta before ranking (the research tier
    -- an unneutralized decile spread is mostly a size bet, which is the
    finding the whole plan rests on)
  * rank the cross-section into deciles, take each decile's mean forward return
  * MONOTONICITY = Spearman(decile index, decile mean return). A tails-only
    signal scores near zero here while still showing a wide Q10-Q1 spread, and
    separating those two cases is the point.

The decision statistic is `long_share`: of the total Q10-Q1 spread, how much is
earned above the cross-sectional median (reachable long-only) versus below it
(reachable only by shorting).

Every number carries a CI, and a null control (shuffled signal) must collapse.

Usage::

    python scripts/qa_quantile_shape.py --library data/factorlib/all_factors_library_smoke_glm52_zoo.json
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
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.metrics import _cross_sectional_corr, label_frame  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.ERROR)
RNG = np.random.default_rng(11)


def decile_profile(sig: pd.DataFrame, lab: pd.DataFrame, n_q: int = 10):
    """Mean forward return of each signal decile, plus the per-date spread series.

    Ranked per date, so this is dollar-neutral by construction: no benchmark, no
    optimiser, no cost model. TC ~ 1 here -- that is why the research tier is
    the honest place to ask whether the alpha exists at all.
    """
    s = sig.where(np.isfinite(sig))
    r = s.rank(axis=1, pct=True)
    q = np.ceil(r * n_q).clip(1, n_q)
    means = np.full(n_q, np.nan)
    for k in range(1, n_q + 1):
        m = lab.where(q == k)
        v = m.mean(axis=1).dropna()
        if len(v):
            means[k - 1] = float(v.mean())
    top = lab.where(q == n_q).mean(axis=1)
    bot = lab.where(q == 1).mean(axis=1)
    mid = lab.mean(axis=1)                      # the cross-sectional median leg
    spread = (top - bot).dropna()
    return means, spread, (top - mid).dropna(), (mid - bot).dropna()


def shuffle_within_row(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute each date's signal among the names that ACTUALLY HAVE one.

    Permuting the raw array moves NaNs around, which silently changes which
    names are in that day's cross-section -- so the "null" was measuring a
    different universe, not a scrambled signal. It came back at -9.2bp against
    a real spread of +2.2bp: a null larger than the signal is a broken null.
    """
    out = df.to_numpy(copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        ok = np.isfinite(row)
        if ok.sum() > 1:
            row[ok] = rng.permutation(row[ok])
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def boot_ci(x: np.ndarray, fn, n=400, alpha=0.05):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 5:
        return (float("nan"), float("nan"))
    vals = [fn(RNG.choice(x, size=len(x), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--raw", action="store_true",
                    help="skip neutralization (shows how much was a size bet)")
    ap.add_argument("--out", default="data/results/quantile_shape.json")
    a = ap.parse_args()

    theta = load_protocol(a.protocol or default_protocol_path())
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
    lab_all = label_frame(panel, theta)
    lo, hi = str(win[0]), str(win[1])
    lab = lab_all.loc[lo:hi]
    print(f"protocol {theta.hash} | scored {win} | {len(exprs)} factors")
    print(f"  neutralized: {not a.raw}\n", flush=True)

    resid = None
    if not a.raw:
        from quantaalpha.eval.neutralize import residualize
        resid = residualize

    rows, leg_series = [], []
    for i, e in enumerate(exprs, 1):
        try:
            s = load_aligned_signal(e, panel)
        except Exception:
            continue
        if resid is not None:
            try:
                s = resid(s, panel, theta)
            except Exception as exc:
                print(f"  [{i}] neutralize failed {type(exc).__name__}; skipped")
                continue
        s = s.loc[lo:hi]
        ic = _cross_sectional_corr(s, lab, "spearman").dropna()
        if len(ic) < 100:
            continue
        ic_m = float(ic.mean())
        # SIGN-ALIGN: Q10 must mean "the decile this factor says to buy".
        s_al = s if ic_m >= 0 else -s
        means, spread, long_leg, short_leg = decile_profile(s_al, lab)

        from scipy.stats import spearmanr
        mono = float(spearmanr(np.arange(len(means)), means,
                               nan_policy="omit").statistic)
        L, S = float(long_leg.mean()), float(short_leg.mean())
        tot = L + S
        rows.append({
            "expr": e, "ic": ic_m, "deciles": means.tolist(),
            "monotonicity": mono,
            "spread": float(spread.mean()), "spread_t": tstat(spread),
            "long_leg": L, "long_leg_t": tstat(long_leg),
            "short_leg": S, "short_leg_t": tstat(short_leg),
            "long_share": (L / tot) if tot else float("nan"),
        })
        leg_series.append((long_leg, short_leg, spread))
        print(f"  [{i:2d}] IC {ic_m:+.4f}  mono {mono:+.2f}  "
              f"spread {spread.mean()*1e4:+6.1f}bp (t {tstat(spread):+5.2f})  "
              f"long {L*1e4:+6.1f} / short {S*1e4:+6.1f}bp  "
              f"long_share {(L/tot if tot else float('nan')):5.0%}", flush=True)

    if not rows:
        print("no factors measured")
        return 2

    mo = np.array([r["monotonicity"] for r in rows], float)
    sp = np.array([r["spread"] for r in rows], float)

    # POOLED legs: average each leg across factors PER DATE, then treat the
    # resulting daily series as the population. A per-factor ratio L/(L+S) is
    # unbounded when S < 0 (this run produced 103% and 124%), so the ratio is
    # taken once, on the pooled means, not averaged over unstable per-factor
    # ratios.
    long_d = pd.concat([l for l, _, _ in leg_series], axis=1).mean(axis=1).dropna()
    short_d = pd.concat([s_ for _, s_, _ in leg_series], axis=1).mean(axis=1).dropna()
    sp_d = pd.concat([p for _, _, p in leg_series], axis=1).mean(axis=1).dropna()
    L, S = float(long_d.mean()), float(short_d.mean())
    long_share = L / (L + S) if (L + S) != 0 else float("nan")

    lo_L, hi_L = boot_ci(long_d.to_numpy(), np.mean)
    lo_S, hi_S = boot_ci(short_d.to_numpy(), np.mean)
    lo_mo, hi_mo = boot_ci(mo, np.mean)

    # NULL CONTROL, now scrambling the signal WITHIN each date's real universe.
    null_sp = float("nan")
    try:
        s0 = load_aligned_signal(rows[0]["expr"], panel)
        if resid is not None:
            s0 = resid(s0, panel, theta)
        s0 = s0.loc[lo:hi]
        _, nsp, _, _ = decile_profile(shuffle_within_row(s0, RNG), lab)
        null_sp = float(nsp.mean())
    except Exception as exc:
        print(f"  null control failed: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 74)
    print(f"QUANTILE SHAPE   n={len(rows)} factors, window {win}, "
          f"{len(sp_d)} trading days")
    print("=" * 74)
    print(f"  Q10-Q1 spread            {sp_d.mean()*1e4:+7.2f} bp/day  "
          f"(t {tstat(sp_d):+5.2f})")
    print(f"  NULL control (scrambled) {null_sp*1e4:+7.2f} bp/day")
    print(f"  LONG  leg (Q10 - median) {L*1e4:+7.2f} bp/day  "
          f"[{lo_L*1e4:+.2f}, {hi_L*1e4:+.2f}]  t {tstat(long_d):+5.2f}")
    print(f"  SHORT leg (median - Q1)  {S*1e4:+7.2f} bp/day  "
          f"[{lo_S*1e4:+.2f}, {hi_S*1e4:+.2f}]  t {tstat(short_d):+5.2f}")
    print(f"  long_share (pooled)      {long_share:7.1%}")
    print(f"  monotonicity             {mo.mean():+7.3f}  "
          f"[{lo_mo:+.3f}, {hi_mo:+.3f}]")
    print(f"  factors with a POSITIVE short leg: "
          f"{int(sum(1 for r in rows if r['short_leg'] > 0))}/{len(rows)}")
    print("-" * 74)
    short_matters = np.isfinite(lo_S) and lo_S > 0
    long_matters = np.isfinite(lo_L) and lo_L > 0
    if not long_matters and not short_matters:
        print("  NEITHER LEG IS DISTINGUISHABLE FROM ZERO. The quantile spread")
        print("  does not establish that these factors sort returns at all, so")
        print("  this cannot decide the long/short question -- it says the")
        print("  signal is the thing to fix first.")
    elif short_matters and not long_matters:
        print("  Q1-DRIVEN: the return is in the names the factors say to AVOID.")
        print("  A long-only book cannot express that leg, so its transfer")
        print("  coefficient is bounded by construction. Long/short is justified")
        print("  BY MEASUREMENT.")
    elif long_matters and not short_matters:
        print("  Q10-DRIVEN: the return is in the names the factors say to BUY,")
        print("  and the short leg is not distinguishable from zero. Long-only")
        print("  forfeits nothing measurable here -- the TC shortfall must be")
        print("  explained by the CONSTRUCTION, not by the missing short book.")
    else:
        print("  TWO-SIDED: both legs are individually significant. Long-only")
        print(f"  forfeits the {1-long_share:.0%} of the spread that sits below")
        print("  the median; whether that is worth a short book is a cost")
        print("  question, not a signal question.")

    Path(a.out).write_text(json.dumps(
        {"window": list(win), "neutralized": not a.raw, "n": len(rows),
         "n_days": len(sp_d), "protocol_hash": theta.hash,
         "long_share_pooled": long_share,
         "long_leg": L, "long_leg_ci": [lo_L, hi_L], "long_leg_t": tstat(long_d),
         "short_leg": S, "short_leg_ci": [lo_S, hi_S], "short_leg_t": tstat(short_d),
         "mean_monotonicity": float(mo.mean()), "monotonicity_ci": [lo_mo, hi_mo],
         "spread": float(sp_d.mean()), "spread_t": tstat(sp_d),
         "null_spread": null_sp, "rows": rows}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
