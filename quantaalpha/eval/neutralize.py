"""Strip known risk exposures from a signal before it is scored.

This is the single biggest gap the audit found, and the direct remedy for the
measured dominant failure of this search: a **size bet worth ~10pp/yr that the
factors did not cause and could not fix**. Equal-weight versus cap-weighted
CSI300 reproduces the search's exact per-fold sign pattern (-9.8pp 2019,
-13.1pp 2020, +6.0pp 2021) with zero alpha involved, so a factor's raw IC cannot
be told apart from a repackaged size exposure.

The fix is standard practice and simple to state: at each date, regress the
signal cross-sectionally on the known risk factors and keep the **residual**.
What survives is the part of the signal that is not size, not industry, not
beta -- which is the only part that deserves to be called alpha.

Risk factors used here:

* **size** -- ``log(circ_mv)``, from ``data/reference/market_cap.parquet``.
  Circulating (free-float) market cap, not total: a large fraction of A-share
  total shares is restricted state and legal-person stock that never trades, so
  total market cap overstates investable size.
* **industry** -- CSRC top-level category (~19 buckets, the leading letter of
  the classification code). The ~100 sub-industries are deliberately NOT used:
  a 300-name cross-section cannot support 100 dummies without the regression
  becoming ill-conditioned and eating real signal.
* **beta** -- trailing regression of the name's return on the index, so stock
  selection is separated from an unintended market-timing bet.

``residualize_vs_library`` additionally regresses out the current library, so a
candidate is scored on its **incremental** contribution rather than on signal
the repository already holds.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)

_REF = Path("data/reference")
_BETA_WINDOW = 250          # ~1 trading year
_MIN_NAMES = 20             # a cross-section smaller than this cannot support the fit


@lru_cache(maxsize=4)
def _load_market_cap() -> pd.DataFrame:
    path = _REF / "market_cap.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run quantaalpha.data.reference_sync.fetch_market_cap()")
    return pd.read_parquet(path)


@lru_cache(maxsize=4)
def _load_industry() -> pd.Series:
    """instrument -> CSRC TOP-LEVEL industry letter.

    Known limitation, to be disclosed: baostock returns a current snapshot, not
    membership with in/out dates, so this applies today's classification to the
    whole history. Industry assignment changes rarely, so the bias is small --
    but it is not zero.
    """
    path = _REF / "industry.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run quantaalpha.data.reference_sync.fetch_industry()")
    df = pd.read_parquet(path)
    code = df["code"].str.replace(".", "", regex=False).str.upper()   # sh.600000 -> SH600000
    top = df["industry"].fillna("").str.slice(0, 1).replace("", "?")
    return pd.Series(top.to_numpy(), index=code.to_numpy()).groupby(level=0).first()


def size_frame(panel: PanelBundle) -> pd.DataFrame:
    """``log(circ_mv)``, aligned to the panel grid."""
    mc = _load_market_cap()
    wide = mc.pivot(index="date", columns="instrument", values="circ_mv")
    wide = wide.reindex(index=panel.dates, columns=panel.instruments).ffill()
    return np.log(wide.where(wide > 0))


def beta_frame(panel: PanelBundle, benchmark: pd.Series,
               window: int = _BETA_WINDOW) -> pd.DataFrame:
    """Trailing beta of each name against the benchmark.

    Uses only data strictly before the current date (``shift(1)`` on the rolling
    statistics), so a same-day return never informs its own beta.
    """
    ret = panel.close.pct_change()
    bm = benchmark.reindex(ret.index).astype(float)
    bm_c = bm - bm.rolling(window, min_periods=window // 4).mean()

    cov = ret.mul(bm_c, axis=0).rolling(window, min_periods=window // 4).mean()
    var = (bm_c ** 2).rolling(window, min_periods=window // 4).mean()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    return beta.shift(1)


def _industry_dummies(instruments: pd.Index) -> pd.DataFrame:
    ind = _load_industry().reindex(instruments)
    d = pd.get_dummies(ind.fillna("?"), prefix="ind", dtype=float)
    # Drop one level: with an intercept, a full dummy set is collinear.
    return d.iloc[:, 1:] if d.shape[1] > 1 else d


def winsorize(frame: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """Clip each date's cross-section to +/- ``n_mad`` median absolute deviations.

    The canonical factor pipeline is winsorize -> standardize -> neutralize, and
    this was the missing first step. It matters HERE specifically because
    neutralization is an ordinary least-squares fit, and OLS is not robust: one
    bad tick in a 300-name cross-section moves the fitted coefficients and
    therefore contaminates the residual for EVERY name on that date, not just
    the outlier's.

    MAD rather than standard deviations, because the dispersion estimate must
    not itself be moved by the outlier it is meant to catch. 5 MAD is roughly
    3.4 sigma for a normal cross-section -- loose enough to keep genuine tails,
    which on a reversal signal are where the information lives.
    """
    med = frame.median(axis=1)
    mad = (frame.sub(med, axis=0)).abs().median(axis=1)
    # A degenerate date (mad == 0) is left alone; clipping to a zero-width band
    # would erase the whole cross-section.
    lo = med - n_mad * mad.replace(0.0, np.nan)
    hi = med + n_mad * mad.replace(0.0, np.nan)
    return frame.clip(lower=lo, upper=hi, axis=0)


def residualize(signal_wide: pd.DataFrame, panel: PanelBundle, theta: Protocol,
                benchmark: pd.Series | None = None,
                extra: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Cross-sectional residual of ``signal_wide`` against the risk factors.

    One OLS per date on the names available that date. Dates with too few names
    are returned as NaN rather than fitted on a degenerate cross-section.
    """
    sig = signal_wide.reindex(index=panel.dates, columns=panel.instruments)
    sig = sig.where(panel.universe)
    # Winsorize BEFORE the fit: OLS is not outlier-robust, so one bad tick
    # contaminates the residual of every name on that date.
    n_mad = float(getattr(theta.admission, "winsor_mad", 5.0) or 5.0)
    sig = winsorize(sig, n_mad)

    # Only include factors that actually carry values. An all-NaN column makes
    # `isfinite(X).all(axis=1)` false for EVERY name, which silently skips every
    # date and returns an all-NaN residual -- the failure mode this hit first.
    factors: dict[str, pd.DataFrame] = {"size": size_frame(panel)}
    if benchmark is not None:
        factors["beta"] = beta_frame(panel, benchmark)
    factors.update(extra or {})
    factors = {k: f for k, f in factors.items() if f.notna().any().any()}
    # Winsorize the EXPOSURES on the same terms as the signal. Clipping y but
    # not X leaves the clipped tails sitting in the residual, and those tails
    # are exactly the size extremes -- a signal that IS size then survives
    # neutralization with most of its size correlation intact, which defeats
    # the whole point of neutralizing. Standard practice (Barra) winsorizes
    # exposures and signal alike before the cross-sectional regression.
    factors = {k: winsorize(f.reindex(index=sig.index, columns=sig.columns), n_mad)
               for k, f in factors.items()}

    dummies = _industry_dummies(panel.instruments)          # static, names x K
    dum = dummies.to_numpy(dtype=float)

    out = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)

    for dt in sig.index:
        y = sig.loc[dt].to_numpy(dtype=float)
        cols = [np.ones(y.size)]
        for f in factors.values():
            v = f.loc[dt].to_numpy(dtype=float) if dt in f.index else np.full(y.size, np.nan)
            cols.append(v)
        X = np.column_stack(cols + [dum])

        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < _MIN_NAMES:
            continue
        Xo, yo = X[ok], y[ok]
        # Drop constant columns (an industry absent that day, or a flat factor);
        # they are collinear with the intercept and make the solve singular.
        keep = np.r_[True, Xo[:, 1:].std(axis=0) > 1e-12]
        Xo = Xo[:, keep]
        try:
            coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        except np.linalg.LinAlgError:
            continue
        res = np.full(y.size, np.nan)
        res[ok] = yo - Xo @ coef
        out.loc[dt] = res

    return out


def residualize_vs_library(signal_wide: pd.DataFrame, zoo_signals: dict,
                           panel: PanelBundle) -> pd.DataFrame:
    """Residual of a candidate against the CURRENT LIBRARY, per date.

    Scores incremental alpha: a candidate that merely restates signal already
    held contributes nothing, and this is what makes that visible as a near-zero
    residual IC rather than as a merely-high correlation.
    """
    from quantaalpha.eval.data import align_signal

    if not zoo_signals:
        return signal_wide
    frames = {}
    for i, (expr, s) in enumerate(list(zoo_signals.items())[:40]):
        try:
            frames[f"lib{i}"] = align_signal(s, panel)
        except Exception:
            continue
    if not frames:
        return signal_wide

    sig = signal_wide.reindex(index=panel.dates, columns=panel.instruments)
    out = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)
    for dt in sig.index:
        y = sig.loc[dt].to_numpy(dtype=float)
        cols = [np.ones(y.size)]
        for f in frames.values():
            cols.append(f.loc[dt].to_numpy(dtype=float) if dt in f.index
                        else np.full(y.size, np.nan))
        X = np.column_stack(cols)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < _MIN_NAMES:
            continue
        Xo, yo = X[ok], y[ok]
        keep = np.r_[True, Xo[:, 1:].std(axis=0) > 1e-12]
        Xo = Xo[:, keep]
        try:
            coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        except np.linalg.LinAlgError:
            continue
        res = np.full(y.size, np.nan)
        res[ok] = yo - Xo @ coef
        out.loc[dt] = res
    return out


def exposure_report(signal_wide: pd.DataFrame, panel: PanelBundle,
                    benchmark: pd.Series | None = None) -> dict:
    """Average cross-sectional correlation of the signal with each risk factor.

    This is what the generator has never been shown. "Your factor was 0.71
    correlated with size" is a diagnosis it can act on; "your batch contributed
    -0.08 net_ir" is not.
    """
    sig = signal_wide.reindex(index=panel.dates, columns=panel.instruments)
    sig = sig.where(panel.universe)
    size = size_frame(panel)
    out: dict[str, float] = {}

    def _corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
        c = a.corrwith(b, axis=1, method="spearman").dropna()
        return float(c.mean()) if len(c) else float("nan")

    out["exposure_size"] = _corr(sig, size)
    if benchmark is not None:
        out["exposure_beta"] = _corr(sig, beta_frame(panel, benchmark))
    return out


__all__ = ["winsorize", "residualize", "residualize_vs_library", "exposure_report",
           "size_frame", "beta_frame"]
