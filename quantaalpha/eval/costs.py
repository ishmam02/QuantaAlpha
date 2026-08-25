"""Transaction costs (Eq. 7) — the term that makes the objective net-of-cost.

Four charges, all in units of NAV per period::

    c_t = κ₀·TO_t                       flat commission / exchange fee
        + κ₁·Σ σ_{i,t}·|Δw_{i,t}|       spread & slippage, scaled by volatility
        + κ₂·Σ φ(Δw_{i,t}; ADV_{i,t})   market impact, super-linear in size
        + Σ β_{i,t}·max(0, −w_{i,t})    borrow on short positions

``κ₀ = 0.0020`` is set to the existing Qlib round trip
(``open_cost 0.0005 + close_cost 0.0015``) precisely so that this engine
**nests** the published baseline's flat-fee exposure: the baseline is a
flat-fee reference, and κ₀ reproduces it exactly.

Every trailing statistic goes through :func:`_trailing`, which shifts before
rolling. That is the single place a look-ahead bug could enter, so it is the
single place the shift lives.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantaalpha.eval.execution import turnover
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)

# Below this, an ADV weight is treated as "no liquidity data" rather than
# "infinitely illiquid", which would otherwise make impact explode to inf.
_ADV_FLOOR = 1e-8


def _trailing(df: pd.DataFrame, window: int) -> "pd.core.window.rolling.Rolling":
    """Strictly trailing rolling window: period ``t`` sees data through ``t-1``.

    The explicit ``.shift(1)`` before ``.rolling`` is the whole point. Without
    it, today's volatility and today's volume would price today's trade — a
    look-ahead that inflates every capacity-related number and is invisible in
    the output.
    """
    return df.shift(1).rolling(window, min_periods=max(2, window // 2))


def trailing_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """σ_{i,t} — trailing daily-return standard deviation."""
    return _trailing(close.pct_change(fill_method=None), window).std()


def dollar_volume(panel, theta: Protocol) -> pd.DataFrame:
    """Daily traded value in CNY, recovered from Qlib's scaled ``$amount``.

    Qlib's cn_data stores ``$amount`` in THOUSANDS of yuan, already consistent
    with the *adjusted* ``$close`` and ``$volume`` it ships alongside::

        true_CNY = $amount · 1000

    **This previously divided by ``$factor`` as well, which double-counted the
    price adjustment and overstated ADV by 1/factor** — about 3.5x at the median
    factor (~0.29) and up to ~19x for heavily-adjusted names.

    The old docstring cited SH601398 (ICBC) on 2020-06-01 -- ``$amount``
    540,140.9, ``$factor`` 0.5412 ⟹ ¥9.98e8 -- as matching ICBC's ~¥1bn daily
    turnover. That reasoning is circular: it accepts the ``/factor`` form
    because the result lands near a remembered figure. An independent check on
    the same date, using **no** ``$amount`` at all, settles it::

        raw price  = $close / $factor = 2.793 / 0.5412 = 5.16 CNY
                     (ICBC genuinely traded ~5.0-5.3 CNY in June 2020)
        raw shares = $volume · $factor · 100          ($volume is in lots)
        notional   = 5.16 · raw shares = ¥5.42e8

    ``$amount · 1000`` = ¥5.40e8 reproduces that to 0.4%; ``$amount · 1000 /
    $factor`` = ¥9.98e8 is 1.85x too high. Cross-checked on 2021 large caps
    against known turnover (Kweichow Moutai traded roughly ¥3-5bn/day):
    SH600519 ¥72.9e8 plain vs ¥578e8 with ``/factor``; SH601318 ¥40.8e8 vs
    ¥770e8; SH600036 ¥30.0e8 vs ¥57.9e8.

    Consequence of the old form: ADV overstated ⇒ participation understated ⇒
    market impact understated. ``kappa2`` is the only capacity-aware cost term,
    so the backtest was optimistic about how much size the book could carry
    (real capacity is ~3.5x smaller than it implied). Factor *selection* was
    unaffected at ``nav`` = 1e8, where impact is ~0.001 bps either way.

    The scale lives in Θ (``costs.adv_scale``) so it is covered by the protocol
    hash rather than buried as a magic number.
    """
    return panel.amount * float(theta.costs.adv_scale)


def trailing_adv(panel, theta: Protocol) -> pd.DataFrame:
    """ADV_{i,t} — trailing mean daily traded value, in CNY."""
    return _trailing(dollar_volume(panel, theta), int(theta.costs.adv_window)).mean()


def impact(dw: pd.Series, adv_w: pd.Series, theta: Protocol,
           sigma: pd.Series | None = None) -> pd.Series:
    """φ — market impact, charged per unit of NAV.

    ``adv_w`` is ADV expressed as a *portfolio weight* (dollar ADV / NAV), so
    participation is ``p = |Δw| / adv_w``.

    With ``impact_exponent = p_exp`` (default 1.5)::

        φ = |Δw|^p_exp / adv_w^(p_exp - 1) = |Δw| · p^(p_exp - 1)

    κ₂ calibration (this is the derivation behind ``kappa2: 0.01`` in Θ):
    the impact charged **per unit of traded notional** is ``κ₂ · √p`` at the
    default exponent. Requiring 10 bps at 1% participation gives

        κ₂ · √0.01 = 0.0010  ⟹  κ₂ · 0.1 = 0.0010  ⟹  κ₂ = 0.01

    with ``NAV = 100_000_000`` (the ``account:`` field in all three Qlib
    configs). Every capacity conclusion in the write-up rests on this number,
    which is why Θ carries it explicitly and the plan calls for a sweep.
    """
    p_exp = float(theta.costs.impact_exponent)
    magnitude = dw.abs()

    liquidity = adv_w.reindex(magnitude.index)

    # A missing ADV means "no liquidity observation", NOT "zero liquidity".
    # Flooring a NaN would charge a normal-sized trade thousands of bps and let
    # a data gap masquerade as a capacity limit, so fall back to the date's
    # cross-sectional median instead.
    traded_missing = int((liquidity.isna() & (magnitude > 0)).sum())
    if liquidity.isna().any():
        fallback = liquidity.median()
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = _ADV_FLOOR
        liquidity = liquidity.fillna(fallback)
        if traded_missing:
            logger.debug(
                "impact: %d traded name(s) had no ADV; used cross-sectional median %.3e",
                traded_missing, float(fallback),
            )
    liquidity = liquidity.clip(lower=_ADV_FLOOR)

    charge = float(theta.costs.kappa2) * magnitude.pow(p_exp) / liquidity.pow(p_exp - 1.0)

    # Volatility scaling. Without it the model prices a 1%-of-ADV trade the same
    # in every name, which no standard impact model does. sigma is the trailing
    # daily return volatility already computed for the slippage term, so this
    # costs nothing extra to evaluate.
    if bool(getattr(theta.costs, "impact_vol_scaled", False)):
        if sigma is None:
            raise ValueError(
                "costs.impact_vol_scaled is on but no sigma was supplied; the "
                "caller must pass the per-name volatility rather than let the "
                "scaling silently drop out")
        vol = sigma.reindex(charge.index)
        # A missing volatility must not zero the charge -- that would make an
        # unmeasured name look free to trade. Fall back to the cross-sectional
        # median.
        vol = vol.fillna(vol.median() if vol.notna().any() else 0.02)
        charge = charge * vol
    return charge


def borrow_cost(w: pd.Series, theta: Protocol, offlist: pd.Series | None = None) -> float:
    """Σ β_{i,t}·max(0, −w_{i,t}) — the charge for holding shorts.

    Inert while ``portfolio.signed`` is false (no negative weights exist), so
    the borrow term is genuinely untested until the signed sensitivity pass
    runs. ``offlist`` marks names that cannot be borrowed at all; they carry
    ``beta_offlist`` (``inf`` by default), which makes an unshortable position
    infinitely expensive rather than silently free.
    """
    shorts = (-w).clip(lower=0.0)
    if not bool((shorts > 0).any()):
        return 0.0

    beta = pd.Series(float(theta.costs.beta_per_day), index=w.index)
    if offlist is not None:
        beta = beta.where(~offlist.reindex(w.index).fillna(False), float(theta.costs.beta_offlist))
    return float((beta * shorts).sum())


def cost(
    w: pd.Series,
    w_drift: pd.Series,
    sigma: pd.Series,
    adv: pd.Series,
    theta: Protocol,
    offlist: pd.Series | None = None,
) -> float:
    """c_t (Eq. 7) — total cost of moving from ``w_drift`` to ``w``.

    ``adv`` is in currency units; it is normalized by ``Θ.costs.nav`` here so
    that callers never have to remember which space it is in.
    """
    aligned_w, aligned_drift = w.align(w_drift, fill_value=0.0)
    dw = aligned_w - aligned_drift

    flat = float(theta.costs.kappa0) * turnover(aligned_w, aligned_drift)

    vol = sigma.reindex(dw.index).fillna(0.0)
    slippage = float(theta.costs.kappa1) * float((vol * dw.abs()).sum())

    adv_w = adv.reindex(dw.index) / float(theta.costs.nav)
    market_impact = float(impact(dw, adv_w, theta, sigma=vol).sum())

    return flat + slippage + market_impact + borrow_cost(aligned_w, theta, offlist)


def net_return(
    w: pd.DataFrame,
    y_tilde: pd.DataFrame,
    bench: pd.Series,
    c: pd.Series,
    delta: int = 0,
) -> pd.Series:
    """r_net (Eq. 8) — ``w·ỹ − r_bench − c``.

    The benchmark subtraction is what makes this an *excess* return, matching
    the published baseline's ``excess_return_with_cost`` reporting.

    **The two series must cover the same holding period, and they did not.**
    ``y_tilde[t]`` is the return earned between consecutive *fills*: with
    ``fill_rule="open_next"`` and ``δ=1`` that is ``open[t+2]/open[t+1] − 1``,
    which is essentially day ``t+1``'s move. ``load_benchmark`` returns
    ``close.pct_change()``, so ``bench[t]`` is day ``t``'s move. Subtracting
    them unshifted differences two series one day out of step, which does not
    hedge the market -- it adds a second, independent copy of it.

    Measured on the book over 2022-2025: ``corr(book, bench) = +0.025`` at
    the old alignment against ``+0.688`` at ``shift(-1)``, and the excess
    volatility was *higher* than either input (0.2795 annualised, against 0.2185
    for the book and 0.1798 for the benchmark) -- the signature of subtracting
    something uncorrelated. Aligning takes it to 0.1612. Qlib's own backtest of
    the same library reports 0.0804, so this closes most of a gap that had made
    our net IR and our geometrically-compounded net ARR look far worse than
    they are; the mean is barely affected, since the benchmark is near
    zero-mean at any lag.

    ``delta`` is the execution latency from Θ. Shifting by it is what keeps the
    alignment tied to the fill rule rather than to a constant someone tuned.
    """
    gross = (w * y_tilde.reindex(index=w.index, columns=w.columns)).sum(axis=1)
    aligned = bench.shift(-int(delta)) if delta else bench
    benchmark = aligned.reindex(gross.index).fillna(0.0)
    charges = c.reindex(gross.index).fillna(0.0)
    out = gross - benchmark - charges
    out.name = "r_net"
    return out


__all__ = [
    "borrow_cost",
    "cost",
    "impact",
    "net_return",
    "trailing_adv",
    "trailing_vol",
]
