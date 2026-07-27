"""Execution model: the fill rule Φ, the latency δ, drift and turnover.

Implements Eq. 4-6 of ``problem_formulation.tex``.

This module is where the repository's **label/fill mismatch** is fixed. The
Qlib configs train on a close-to-close label
(``Ref($close,-2)/Ref($close,-1)-1``) but execute at ``deal_price: open``, so
the modelled return and the realized return come from different price series.
Here the realized return :func:`realized_return` is computed from the *fill*
series and nothing else; the close-to-close label survives only as the IC
target (§3.4 item 1), which is a prediction-quality statistic, not a P&L.

Pure numpy/pandas — the only QuantaAlpha import is the protocol.
"""

from __future__ import annotations

import logging

import pandas as pd

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)

_NAV_FLOOR = 1e-12


def fill_prices(panel: PanelBundle, theta: Protocol) -> pd.DataFrame:
    """Φ(P; t, δ) — the price at which an order decided on ``t`` is filled.

    With the default ``fill_rule="open_next"`` and ``δ=1``, a signal observed
    at the close of ``t`` is executed at the open of ``t+1``:
    ``P_fill[t] = open[t+1]``.

    Qlib's OHLC fields are **already split/dividend adjusted** — the raw traded
    price is recovered as ``$open / $factor``, not ``$open * $factor``. The
    adjusted series is therefore the continuous one, and is what returns must
    be computed from; re-multiplying by ``$factor`` would reintroduce exactly
    the split discontinuity the adjustment exists to remove. (Verified against
    SH600519 in 2020-06: ``$close`` 175.28, ``$factor`` 0.1235,
    ``$close/$factor`` 1419.5 — the real quoted price.)

    This ``shift(-1)`` is the **only** forward reference the engine makes, and
    it is a fill, not a feature: the decision at ``t`` still uses information
    through ``t`` alone.
    """
    if theta.execution.fill_rule != "open_next":
        raise NotImplementedError(f"unsupported fill_rule {theta.execution.fill_rule!r}")

    return panel.open.shift(-int(theta.execution.delta))


def realized_return(fill: pd.DataFrame) -> pd.DataFrame:
    """ỹ (Eq. 5) — the return actually earned between consecutive fills."""
    return fill.shift(-1) / fill - 1.0


def drift(w_prev: pd.Series, y_tilde: pd.Series, theta: Protocol | None = None) -> pd.Series:
    """w_drift (Eq. 6) — weights carried forward by one period of P&L.

    Computed from **notionals**, never by renormalizing weights directly::

        n = w_prev * (1 + ỹ)

    For the long-only book the post-P&L NAV is ``Σ|n|``, so
    ``w_drift = n / Σ|n|``.

    For the signed (dollar-neutral) book ``Σn`` is *not* the NAV — it is just
    the period P&L, and it approaches zero by construction. Dividing by it is
    the numerical trap this function exists to avoid. The capital base is
    instead ``1 + Σ(w_prev · ỹ)``: unit NAV grown by the book's return.
    """
    signed = bool(theta.portfolio.signed) if theta is not None else False
    positions = w_prev * (1.0 + y_tilde.reindex(w_prev.index).fillna(0.0))

    if signed:
        nav_post = 1.0 + float((w_prev * y_tilde.reindex(w_prev.index).fillna(0.0)).sum())
    else:
        nav_post = float(positions.abs().sum())

    if abs(nav_post) < _NAV_FLOOR:
        logger.warning("drift: post-P&L NAV %.3e below floor; carrying weights forward", nav_post)
        return w_prev.copy()

    return positions / nav_post


def turnover(w: pd.Series, w_drift: pd.Series) -> float:
    """TO_t (Eq. 6) — one-way turnover, in [0, 1].

    The 0.5 makes a full replacement of the book cost exactly 1.0 rather
    than 2.0, i.e. turnover is measured one-way.
    """
    aligned_w, aligned_drift = w.align(w_drift, fill_value=0.0)
    return 0.5 * float((aligned_w - aligned_drift).abs().sum())


__all__ = ["drift", "fill_prices", "realized_return", "turnover"]
