"""Transaction costs (Eq. 7) — the term that makes the objective net-of-cost.

Four charges, all in units of NAV per period::

    c_t = κ₀·TO_t                       flat commission / exchange fee
        + κ₁·Σ σ_{i,t}·|Δw_{i,t}|       spread & slippage, scaled by volatility
        + κ₂·Σ φ(Δw_{i,t}; ADV_{i,t})   market impact, super-linear in size
        + Σ β_{i,t}·max(0, −w_{i,t})    borrow on short positions

``κ₀ = 0.0020`` is set to the existing Qlib round trip
(``open_cost 0.0005 + close_cost 0.0015``) precisely so that this engine
**nests** the published baseline's flat-fee exposure: the control arm is not a
zero-cost control, it is a flat-fee one, and κ₀ reproduces it exactly.

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

    Qlib's cn_data does not store turnover in yuan. Empirically, for every
    instrument and date::

        $amount / $volume = $close / 10

    with ``$volume`` in lots (100 shares) and ``$close`` the *adjusted* price,
    so that ``$amount = true_CNY · $factor / 1000`` and therefore

        true_CNY = $amount · 1000 / $factor

    Verified against SH601398 (ICBC) on 2020-06-01: ``$amount`` 540,140.9,
    ``$factor`` 0.5412 ⟹ ¥9.98e8, matching ICBC's ~¥1bn daily turnover. Taking
    ``$amount`` at face value instead understates ADV by three to four orders
    of magnitude, which silently turns every trade into a market-moving one —
    the impact term then dominates every other cost by ~50x.

    The scale lives in Θ (``costs.adv_scale``) so it is covered by the protocol
    hash rather than buried as a magic number.
    """
    return panel.amount * float(theta.costs.adv_scale) / panel.factor


def trailing_adv(panel, theta: Protocol) -> pd.DataFrame:
    """ADV_{i,t} — trailing mean daily traded value, in CNY."""
    return _trailing(dollar_volume(panel, theta), int(theta.costs.adv_window)).mean()


def impact(dw: pd.Series, adv_w: pd.Series, theta: Protocol) -> pd.Series:
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

    return float(theta.costs.kappa2) * magnitude.pow(p_exp) / liquidity.pow(p_exp - 1.0)


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
    market_impact = float(impact(dw, adv_w, theta).sum())

    return flat + slippage + market_impact + borrow_cost(aligned_w, theta, offlist)


def net_return(
    w: pd.DataFrame,
    y_tilde: pd.DataFrame,
    bench: pd.Series,
    c: pd.Series,
) -> pd.Series:
    """r_net (Eq. 8) — ``w·ỹ − r_bench − c``.

    The benchmark subtraction is what makes this an *excess* return, matching
    the published baseline's ``excess_return_with_cost`` reporting.
    """
    gross = (w * y_tilde.reindex(index=w.index, columns=w.columns)).sum(axis=1)
    benchmark = bench.reindex(gross.index).fillna(0.0)
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
