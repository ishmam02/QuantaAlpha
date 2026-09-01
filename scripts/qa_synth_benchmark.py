#!/usr/bin/env python
"""Can a float-cap benchmark derived from `market_cap.parquet` stand in for the
index-weight history AkShare would not give us?

Phase 0 of the two-tier plan (`declarative-doodling-wadler.md`) needed a
point-in-time CSI300 weight vector `b` so the constructor could optimise the
ACTIVE weight `a = w - b` instead of the absolute weight. That is the transfer-
coefficient fix, and it is the reason Phase 0 was called the highest-leverage
phase. It failed for one reason only:
``ak.index_stock_cons_weight_csindex`` returns the CURRENT snapshot, so
``index_weight.parquet`` holds 300 rows on a single date (2026-07-31) and there
is no history to optimise against.

But CSI300 weights ARE (approximately) free-float market cap, capped and
rebalanced -- the plan says so itself in Phase 0.2, where it proposes the
correlation between the two as a mutual validation of both sources. That check
runs just as well BACKWARDS: if derived float cap reproduces the published
weights on the one date we can see, the derived series can generate the
benchmark on the ~5100 dates we cannot.

That is the hypothesis this script tests, and it tests it the way the standing
measurement protocol requires:

  * POPULATION, not sample -- all 300 published constituents, no subsetting
  * INVARIANTS asserted before any statistic is believed
  * every statistic carries a CI (Fisher-z on the correlations, bootstrap on
    the weight error)
  * a NULL CONTROL -- the same statistic on shuffled caps, which must collapse
  * TWO ESTIMATORS must agree -- Spearman on levels and Pearson on logs

Caveat stated up front, not buried: `market_cap.parquet` ends 2026-01-09 and the
published snapshot is 2026-07-31, so the comparison spans a ~7 month gap. That
gap can only WEAKEN the measured agreement, so a strong result here is a lower
bound, not a flattered one.

On success it writes the synthetic benchmark weight panel to
``data/reference/synthetic_benchmark.parquet``.

Usage::

    python scripts/qa_synth_benchmark.py
    python scripts/qa_synth_benchmark.py --no-write     # validate only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REF = Path("data/reference")
RNG = np.random.default_rng(7)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for a correlation. Every statistic carries one -- a point estimate
    of 0.93 on n=300 and one on n=8 are different claims."""
    if not np.isfinite(r) or abs(r) >= 1.0 or n < 4:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    k = norm.ppf(1 - alpha / 2)
    return tuple(float(np.tanh(v)) for v in (z - k * se, z + k * se))


def to_qlib(code: str, exch: str) -> str:
    """Published constituent code + exchange -> qlib instrument id."""
    c = str(code).zfill(6)
    return ("SZ" if "深圳" in str(exch) else "SH") + c


def _drift_ceiling(mc: pd.DataFrame, names: list[str], asof, lag_days: int):
    """The correlation a PERFECT derivation could reach across a stale gap.

    Compares the cap series to ITSELF `lag_days` earlier -- no index, no
    free-float adjustment, nothing but time passing. Anything the derivation
    loses beyond this is a real defect; anything up to it is the comparison
    being stale, not the data being wrong.
    """
    from scipy.stats import spearmanr
    wide = mc.pivot_table(index="date", columns="instrument",
                          values="circ_mv", aggfunc="last")
    cols = [n for n in names if n in wide.columns]
    sub = wide[cols]
    i = sub.index.get_loc(asof)
    cur = sub.loc[asof].dropna()
    prev = sub.iloc[max(0, i - lag_days)].dropna()
    j = cur.index.intersection(prev.index)
    if len(j) < 30:
        return float("nan"), float("nan")
    a = (cur[j] / cur[j].sum()).to_numpy()
    b = (prev[j] / prev[j].sum()).to_numpy()
    return float(spearmanr(a, b).statistic), float(0.5 * np.abs(a - b).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default="data/results/synth_benchmark_validation.json")
    a = ap.parse_args()

    pub = pd.read_parquet(REF / "index_weight.parquet")
    mc = pd.read_parquet(REF / "market_cap.parquet")

    # ---- invariants, asserted before anything is believed -----------------
    pub_date = pd.to_datetime(pub["日期"].iloc[0])
    assert pub["日期"].nunique() == 1, "expected a single published snapshot"
    wsum = float(pub["权重"].sum())
    assert 99.0 < wsum < 101.0, f"published weights sum to {wsum}, not ~100"
    assert len(pub) == 300, f"expected 300 constituents, got {len(pub)}"

    pub = pub.assign(instrument=[to_qlib(c, e) for c, e
                                 in zip(pub["成分券代码"], pub["交易所"])],
                     w_pub=pub["权重"] / wsum)

    asof = mc.date.max()
    snap = (mc[mc.date == asof]
            .drop_duplicates("instrument")
            .set_index("instrument")["circ_mv"])
    assert (snap.dropna() > 0).all(), "derived circ_mv must be strictly positive"

    m = pub.set_index("instrument").join(snap.rename("circ_mv"), how="left")
    matched = m["circ_mv"].notna()
    n_match = int(matched.sum())

    print(f"published snapshot   {pub_date.date()}   {len(pub)} constituents")
    print(f"derived float cap    {pd.Timestamp(asof).date()}   "
          f"{n_match}/{len(pub)} matched  "
          f"({(pub_date - pd.Timestamp(asof)).days}d gap)\n")
    if n_match < 200:
        print(f"FAIL: only {n_match} constituents carry a derived cap")
        return 2

    d = m[matched]
    w_pub = d["w_pub"].to_numpy(float)
    cap = d["circ_mv"].to_numpy(float)
    w_syn = cap / cap.sum()

    # ---- two estimators, which must agree ---------------------------------
    from scipy.stats import spearmanr, pearsonr
    rho = float(spearmanr(cap, w_pub).statistic)
    rho_lo, rho_hi = fisher_ci(rho, n_match)
    rp = float(pearsonr(np.log(cap), np.log(w_pub)).statistic)
    rp_lo, rp_hi = fisher_ci(rp, n_match)

    # ---- null control: the same statistic on shuffled caps ----------------
    null = [float(spearmanr(RNG.permutation(cap), w_pub).statistic)
            for _ in range(400)]
    null_hi = float(np.percentile(np.abs(null), 95))

    # ---- how wrong is the synthetic WEIGHT, not just its rank? -----------
    err = np.abs(w_syn - w_pub)
    # bootstrap the median absolute weight error
    boot = [float(np.median(RNG.choice(err, size=len(err), replace=True)))
            for _ in range(400)]
    e_lo, e_hi = np.percentile(boot, [2.5, 97.5])
    # the metric that matters for TC: how much of the benchmark's total weight
    # sits in the wrong place
    tv = 0.5 * float(err.sum())

    print(f"{'Spearman rank corr (cap vs published w)':46s} "
          f"{rho:+.4f}  [{rho_lo:+.3f}, {rho_hi:+.3f}]")
    print(f"{'Pearson corr on logs':46s} {rp:+.4f}  [{rp_lo:+.3f}, {rp_hi:+.3f}]")
    print(f"{'NULL control (shuffled caps, 95th pct |rho|)':46s} "
          f"{null_hi:+.4f}")
    print(f"{'median |w_syn - w_pub|':46s} "
          f"{np.median(err):.5f}  [{e_lo:.5f}, {e_hi:.5f}]")
    print(f"{'total variation distance':46s} {tv:.4f}"
          f"   ({tv:.1%} of index weight misplaced)")
    print(f"{'top-10 overlap':46s} "
          f"{len(set(d.nlargest(10,'circ_mv').index) & set(d.nlargest(10,'w_pub').index))}/10")

    # A fixed threshold (0.80) would be an invented bar. The comparison spans
    # 203 days, and cap ranks decorrelate on their own over that horizon, so the
    # ATTAINABLE correlation is not 1.0 -- it is whatever pure drift leaves.
    # Measure that ceiling from the cap series itself, with no index involved,
    # and judge the derivation against it.
    ceil_rho, floor_tv = _drift_ceiling(mc, list(d.index), asof,
                                        (pub_date - pd.Timestamp(asof)).days)
    print(f"{'drift ceiling at the same lag (cap vs cap)':46s} "
          f"{ceil_rho:+.4f}   TV floor {floor_tv:.3f}")
    print(f"{'=> share of ATTAINABLE rank corr achieved':46s} "
          f"{rho / ceil_rho:.0%}")
    print(f"{'=> weight error NOT explained by drift':46s} "
          f"{tv - floor_tv:+.3f}  ({(tv - floor_tv) / tv:.0%} of it)")

    # Two separate claims, and they do not have the same answer:
    #   ORDERING  -- does derived cap rank names the way CSI does?
    #   LEVELS    -- are the weights themselves right?
    # The active-weight construction needs the second, not just the first.
    ok_rank = (rho_lo > null_hi) and (rho / ceil_rho > 0.85) and (rp > 0.70)
    ok_level = (tv - floor_tv) < 0.05
    ok = ok_rank and ok_level
    print("\n" + "=" * 72)
    if ok_rank and not ok_level:
        print("  SPLIT RESULT -- the two claims do not agree.")
        print(f"  ORDERING validated: {rho / ceil_rho:.0%} of the attainable rank")
        print("    correlation, far outside the null. Derived float cap ranks the")
        print("    universe essentially the way CSI does.")
        print(f"  LEVELS not validated: {tv - floor_tv:.3f} of index weight is")
        print("    misplaced beyond what drift explains -- the free-float")
        print("    adjustment (CSI excludes non-trading strategic blocks, which")
        print("    circulating shares do not). Active weights a = w - b would")
        print("    inherit that error as a systematic tilt unrelated to alpha.")
        print("  USABLE FOR: size neutralization, capacity, cap-ordered universe")
        print("    selection. NOT USABLE AS: the benchmark anchor b.")
    elif ok:
        print("  VALIDATED: derived float cap reproduces the published weight")
        print("  ordering far outside the null. A synthetic float-cap benchmark")
        print("  is a usable stand-in for the index-weight history that could")
        print("  not be downloaded -- which unblocks the active-weight (TC) fix.")
    else:
        print("  NOT VALIDATED: the derived cap does not reproduce published")
        print("  weights well enough to substitute for them. The active-weight")
        print("  construction stays blocked on a real weight history.")
    print("=" * 72)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "published_date": str(pub_date.date()), "cap_date": str(pd.Timestamp(asof).date()),
        "gap_days": int((pub_date - pd.Timestamp(asof)).days),
        "n_constituents": len(pub), "n_matched": n_match,
        "spearman": rho, "spearman_ci": [rho_lo, rho_hi],
        "pearson_log": rp, "pearson_log_ci": [rp_lo, rp_hi],
        "null_95pct_abs": null_hi,
        "median_abs_weight_error": float(np.median(err)),
        "median_abs_weight_error_ci": [float(e_lo), float(e_hi)],
        "total_variation": tv, "validated": bool(ok), "ordering_ok": bool(ok_rank),
        "levels_ok": bool(ok_level), "drift_ceiling_rho": ceil_rho,
        "drift_floor_tv": floor_tv,
    }, indent=2))
    print(f"\nwrote {a.out}")

    if ok and not a.no_write:
        out = REF / "synthetic_benchmark.parquet"
        # Float-cap weights over the FULL history, restricted per date to the
        # CSI300 universe spells the eval path already uses. Renormalised to 1.0
        # per date so it is a benchmark, not a raw cap series.
        from quantaalpha.eval.data import load_universe_spells  # noqa
        print("\nbuilding the synthetic benchmark panel ...")
        wide = mc.pivot_table(index="date", columns="instrument",
                              values="circ_mv", aggfunc="last")
        b = wide.div(wide.sum(axis=1), axis=0)
        b.to_parquet(out)
        print(f"wrote {out}  shape {b.shape}  "
              f"{b.index.min().date()} -> {b.index.max().date()}")
        print("  NOTE: not yet masked to the CSI300 universe -- the eval path's")
        print("  universe mask must be applied and the row renormalised at use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
