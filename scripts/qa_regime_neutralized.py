#!/usr/bin/env python
"""Decision gate 2: does the year-to-year swing survive neutralization?

`declarative-doodling-wadler.md` defers the regime-layer decision to this:

    Regime layer | Re-run per-fold net_ir after Phase 1 neutralization | A
    genuine regime dependence SURVIVES neutralization. The 2019/2020-negative /
    2021-positive pattern was measured to be the size bet (equal-weight vs
    cap-weight reproduces it with zero alpha), so it may vanish once size is
    removed.

The original 2019/2020/2021 folds no longer exist -- the split was re-cut to
train 2005-2012 / valid 2013-2015 / test 2016-2026 -- so the question is asked
in the form the new split supports: build the SAME book from RAW signals and
from NEUTRALIZED ones, and compare the dispersion of their yearly returns.

  * If neutralization shrinks the year-to-year swing, the swing was risk
    exposure the book was carrying unintentionally, and no regime machinery is
    warranted -- the plan's own conclusion, tested rather than assumed.
  * If the swing survives intact, the regime dependence is real, and the two
    arms will also agree year by year.

To keep it to two book evaluations rather than eighteen, the protocol is rebuilt
with the combiner fitting on an EARLY slice and scoring across a long one, and
the daily net-return series that ``evaluate`` already returns is sliced by year.
The fit window ends strictly before the scoring window begins, so no year is
scored by a combiner that saw it, and BOTH arms share the identical fit window
so only the neutralization differs.

Deliberately stays inside train+valid. The holdout decides nothing here.

Usage::

    python scripts/qa_regime_neutralized.py --library <zoo>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal  # noqa: E402
from quantaalpha.eval.ledger import replay_repository  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402
from quantaalpha.eval import combiner as combiner_mod  # noqa: E402

logging.basicConfig(level=logging.ERROR)
RNG = np.random.default_rng(31)


def year_stats(r: pd.Series) -> pd.DataFrame:
    """Annualised return and IR for each calendar year of a daily net series."""
    r = r.dropna()
    out = []
    for y, g in r.groupby(r.index.year):
        if len(g) < 120:
            continue
        out.append({"year": int(y), "n": len(g),
                    "arr": float((1 + g).prod() ** (252 / len(g)) - 1),
                    "ir": float(g.mean() / g.std() * np.sqrt(252))
                          if g.std() > 0 else np.nan})
    return pd.DataFrame(out).set_index("year")


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--library")
    src.add_argument("--zoo", metavar="LEDGER")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--fit-end", default="2008-12-31",
                    help="combiner fit ends here; scoring starts the next day")
    ap.add_argument("--score-end", default="2015-12-31")
    ap.add_argument("--out", default="data/results/regime_neutralized.json")
    a = ap.parse_args()

    proto_path = a.protocol or default_protocol_path()
    base = load_protocol(proto_path)
    if a.zoo:
        exprs = list(replay_repository(a.zoo))
    else:
        payload = json.loads(Path(a.library).read_text())
        f = payload.get("factors", payload)
        items = f.values() if isinstance(f, dict) else f
        exprs = [e.get("factor_expression") or e.get("expression") for e in items]
        exprs = [e for e in exprs if e]

    # One protocol, rebuilt from the YAML so the frozen dataclass is never
    # mutated: fit early, score long, holdout untouched.
    cfg = yaml.safe_load(Path(proto_path).read_text())
    train0 = cfg["splits"]["train"][0]
    cfg["splits"]["train"] = [train0, a.fit_end]
    score0 = (pd.Timestamp(a.fit_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cfg["splits"]["valid"] = [score0, a.score_end]
    cfg.setdefault("walk_forward", {})["enabled"] = False
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, fh); fh.close()
    theta = load_protocol(fh.name)

    op = EvaluationOperator(theta)
    p0, p1, win = op._windows(False)
    panel = op._panel(p0, p1)
    print(f"base protocol {base.hash}   variant {theta.hash}")
    print(f"  combiner FITS  {train0} .. {a.fit_end}")
    print(f"  book SCORED    {win[0]} .. {win[1]}   (holdout untouched)")
    print(f"  {len(exprs)} factors\n", flush=True)

    raw = {}
    for e in exprs:
        try:
            raw[e] = load_aligned_signal(e, panel)
        except Exception:
            pass
    if len(raw) < 2:
        print("need >= 2 loadable factors")
        return 2

    from quantaalpha.eval.neutralize import residualize
    print("neutralizing ...", flush=True)
    neu, t0 = {}, time.time()
    for e, s in raw.items():
        try:
            neu[e] = residualize(s, panel, theta)
        except Exception as exc:
            print(f"  skip ({type(exc).__name__}) {e[:50]}")
    print(f"  {len(neu)}/{len(raw)} neutralized in {time.time()-t0:.0f}s\n", flush=True)

    arms, series = {}, {}
    for name, sig in (("RAW", raw), ("NEUTRALIZED", neu)):
        if len(sig) < 2:
            continue
        # The combiner caches its prediction keyed by (zoo_hash of the
        # EXPRESSION keys, candidate expr, theta) -- NOT the signal values.
        # Raw and neutralized share the same expressions and protocol, so
        # without clearing, the NEUTRALIZED arm hits the RAW arm's cache entry
        # and returns a bit-identical book: the regime test would measure a
        # no-op (dispersion ratio exactly 1.00, corr +1.000). Clear between
        # arms so each is fit on its own signals.
        combiner_mod._PREDICTION_CACHE.clear()
        combiner_mod._ATTRIBUTION_CACHE.clear()
        t0 = time.time()
        res = op.evaluate(sig, zoo_signals={}, zoo_metrics=[], report=False)
        r = res.get("_net_return_series")
        if r is None or not isinstance(r, pd.Series):
            print(f"  {name}: no net-return series returned")
            continue
        series[name] = r
        arms[name] = {"net_ir": res.get("m_net_ir"), "net_arr": res.get("m_net_arr"),
                      "rank_ic": res.get("m_rank_ic"),
                      "tc": res.get("m_transfer_coefficient"),
                      "turnover": res.get("m_turnover_book")}
        print(f"  {name:12s} net_ir {arms[name]['net_ir']:+.3f}  "
              f"net_arr {arms[name]['net_arr']:+.2%}  "
              f"TC {arms[name]['tc']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    if len(series) < 2:
        print("\nneed both arms to compare")
        return 2

    ya, yb = year_stats(series["RAW"]), year_stats(series["NEUTRALIZED"])
    j = ya.index.intersection(yb.index)
    print("\n" + "=" * 68)
    print(f"PER-YEAR net return   ({len(j)} years, all inside train+valid)")
    print("=" * 68)
    print(f"  {'year':>6} {'RAW arr':>10} {'NEUT arr':>10} {'RAW ir':>8} {'NEUT ir':>8}")
    for y in j:
        print(f"  {y:>6} {ya.loc[y,'arr']:>+9.2%} {yb.loc[y,'arr']:>+9.2%} "
              f"{ya.loc[y,'ir']:>+8.2f} {yb.loc[y,'ir']:>+8.2f}")

    sa, sb = float(ya.loc[j, "arr"].std()), float(yb.loc[j, "arr"].std())
    # bootstrap the ratio of dispersions over years -- a point estimate of
    # "the swing shrank" means nothing on 7 observations without one.
    boot = []
    idx = np.arange(len(j))
    for _ in range(2000):
        k = RNG.choice(idx, size=len(idx), replace=True)
        va = float(np.std(ya.loc[j, "arr"].to_numpy()[k]))
        vb = float(np.std(yb.loc[j, "arr"].to_numpy()[k]))
        if va > 0:
            boot.append(vb / va)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    agree = int((np.sign(ya.loc[j, "arr"]) == np.sign(yb.loc[j, "arr"])).sum())
    corr = float(ya.loc[j, "arr"].corr(yb.loc[j, "arr"]))

    print("-" * 68)
    print(f"  year-to-year dispersion   RAW {sa:.2%}   NEUTRALIZED {sb:.2%}")
    print(f"  dispersion ratio (N/R)    {sb/sa:.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"  sign agreement            {agree}/{len(j)} years")
    print(f"  correlation of yearly returns  {corr:+.3f}")
    print("-" * 68)
    if np.isfinite(hi) and hi < 0.85:
        print("  THE SWING WAS RISK EXPOSURE. Neutralizing shrinks the")
        print("  year-to-year dispersion outside its CI. No regime layer is")
        print("  warranted -- the variation was something the book was carrying,")
        print("  not something the market was doing.")
    elif np.isfinite(lo) and lo > 1.15:
        print("  NEUTRALIZATION MAKES IT WORSE: the swing GROWS. The risk")
        print("  exposures were damping the book, not driving it.")
    else:
        print("  THE SWING SURVIVES. Dispersion is statistically unchanged and")
        print(f"  the two arms agree in {agree}/{len(j)} years, so the year-to-year")
        print("  variation is not an artifact of uncontrolled risk exposure.")
        print("  Whether it is tradeable regime structure is a separate question")
        print(f"  that {len(j)} observations cannot answer.")

    Path(a.out).write_text(json.dumps({
        "fit": [train0, a.fit_end], "score": list(win),
        "protocol_variant": theta.hash, "arms": arms,
        "years": [int(y) for y in j],
        "raw_arr": ya.loc[j, "arr"].tolist(), "neut_arr": yb.loc[j, "arr"].tolist(),
        "raw_ir": ya.loc[j, "ir"].tolist(), "neut_ir": yb.loc[j, "ir"].tolist(),
        "dispersion_raw": sa, "dispersion_neut": sb,
        "dispersion_ratio": sb / sa if sa else None,
        "dispersion_ratio_ci": [float(lo), float(hi)],
        "sign_agreement": agree, "n_years": len(j), "yearly_corr": corr,
    }, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
