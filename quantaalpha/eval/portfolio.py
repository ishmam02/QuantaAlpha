"""The portfolio map ``g`` (Eq. 6) — top-k dropout.

``g`` is **stateful**: ``w_t = g(ŷ_t, w_drift_t)``. Today's book depends on
yesterday's drifted book, not on today's prediction alone. That is what makes
turnover a property of the strategy rather than of the signal, and it is why
this cannot be expressed as a per-date cross-sectional transform.

The construction is the published baseline's ``TopkDropoutStrategy``
(``topk: 50, n_drop: 5`` in all three Qlib configs), reimplemented natively so
that the whole path is inspectable and deterministic. One consequence worth
stating plainly: **book turnover is structurally capped at ``n_drop/topk``**
(0.10 here), so the ``τ_max = 0.30`` gate can never bind under this ``g``.
That is a property of the chosen construction, not a bug — capacity pressure
in this instantiation comes from κ₂ instead. ``turnover_solo`` is recorded
alongside precisely because it *does* discriminate between signals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantaalpha.eval.execution import drift
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)


def _select(
    scores: pd.Series,
    held: list[str],
    topk: int,
    n_drop: int,
) -> list[str]:
    """One rebalance step of top-k dropout.

    Names are dropped only when they have actually fallen out of the current
    top-k, and at most ``n_drop`` per period. A held name whose score has gone
    missing (delisted, or out of the index) is force-dropped regardless, since
    it is no longer tradeable.

    Dropping "at most" rather than "always" is what makes a constant
    prediction produce zero turnover, which is the behaviour the strategy
    should have: no new information, no trade.
    """
    ranked = scores.sort_values(ascending=False)
    available = list(ranked.index)
    if not available:
        return held

    target = available[: min(topk, len(available))]
    target_set = set(target)

    live_held = [name for name in held if name in scores.index]
    if len(live_held) < len(held):
        logger.debug("top-k dropout: %d held name(s) left the universe", len(held) - len(live_held))

    if not live_held:
        return target

    stale = [name for name in live_held if name not in target_set]
    n = min(n_drop, len(stale))

    if n:
        # Drop the worst-ranked of the names that have fallen out of the top-k.
        worst_first = sorted(stale, key=lambda name: scores[name])
        dropped = set(worst_first[:n])
        incoming = [name for name in target if name not in set(live_held)][:n]
        new_held = [name for name in live_held if name not in dropped] + incoming
    else:
        new_held = list(live_held)

    # Top up if names were force-dropped for leaving the universe.
    if len(new_held) < topk:
        held_set = set(new_held)
        new_held += [name for name in available if name not in held_set][: topk - len(new_held)]

    return new_held[:topk]


def topk_dropout(
    pred: pd.DataFrame,
    theta: Protocol,
    y_tilde: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    prev_w: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``g`` over a prediction panel.

    Returns ``(w, w_drift)`` as aligned ``(T × N)`` frames, so that
    ``0.5·|w − w_drift|`` summed across names is the per-date turnover and
    ``w − w_drift`` is the traded delta the cost model prices.

    ``y_tilde`` is optional: when omitted the book does not drift with P&L
    (``w_drift_t = w_{t-1}``), which is the right behaviour for inspecting the
    construction itself in isolation.
    """
    if theta.portfolio.construction != "topk_dropout":
        raise NotImplementedError(f"unsupported construction {theta.portfolio.construction!r}")

    topk = int(theta.portfolio.topk)
    n_drop = int(theta.portfolio.n_drop)
    signed = bool(theta.portfolio.signed)
    gross = float(theta.portfolio.gross_leverage)

    weights = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
    drifted = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)

    long_held: list[str] = []
    short_held: list[str] = []
    w_prev = prev_w.reindex(pred.columns).fillna(0.0) if prev_w is not None else None

    for date in pred.index:
        scores = pred.loc[date]
        if universe is not None:
            scores = scores.where(universe.loc[date].astype(bool))
        scores = scores.dropna()

        # --- carry the previous book forward through one period of P&L ---
        if w_prev is None:
            w_drift = pd.Series(0.0, index=pred.columns)
        elif y_tilde is None:
            w_drift = w_prev.copy()
        else:
            prior = y_tilde.shift(1).loc[date] if date in y_tilde.index else None
            w_drift = (
                drift(w_prev, prior, theta)
                if prior is not None
                else w_prev.copy()
            )

        long_held = _select(scores, long_held, topk, n_drop)
        if signed:
            short_held = _select(-scores, short_held, topk, n_drop)

        w = pd.Series(0.0, index=pred.columns)
        if signed:
            side = gross / 2.0
            if long_held:
                w.loc[long_held] = side / len(long_held)
            if short_held:
                w.loc[short_held] = -side / len(short_held)
        elif long_held:
            w.loc[long_held] = gross / len(long_held)

        exposure = float(w.abs().sum())
        if exposure > gross + 1e-9:
            raise AssertionError(f"gross leverage {exposure:.6f} exceeds {gross} on {date}")

        weights.loc[date] = w
        drifted.loc[date] = w_drift
        w_prev = w

    return weights, drifted


def solo_book(signal: pd.DataFrame, theta: Protocol, universe: pd.DataFrame | None = None):
    """The same construction driven by the candidate's own signal.

    Used only for ``turnover_solo``, the diagnostic that *does* vary across
    signals once the book-level turnover has been flattened by the structural
    ``n_drop/topk`` cap.
    """
    return topk_dropout(signal, theta, universe=universe)


__all__ = ["solo_book", "topk_dropout"]
