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
from quantaalpha.eval.tradability import TradeMask

logger = logging.getLogger(__name__)


def _swap_cost(theta: Protocol, w_each: float, sigma_i: float, sigma_j: float) -> float:
    """Modelled round-trip cost of replacing one holding with another.

    Both legs trade ``w_each`` of NAV, so the flat and volatility-scaled terms
    (Eq. 7) apply twice. Impact is deliberately omitted here: it depends on ADV
    at the traded name, which the selection step does not carry, and leaving it
    out makes this hurdle a *lower* bound on the true cost -- so the rule never
    talks itself into a trade by underestimating what it will pay.
    """
    c = theta.costs
    per_leg = lambda vol: w_each * (c.kappa0 + c.kappa1 * float(vol))
    return per_leg(sigma_i) + per_leg(sigma_j)


def _select(
    scores: pd.Series,
    held: list[str],
    topk: int,
    n_drop: int,
    *,
    theta: Protocol | None = None,
    can_buy: pd.Series | None = None,
    can_sell: pd.Series | None = None,
    sigma: pd.Series | None = None,
    locked_until: dict[str, int] | None = None,
    step: int = 0,
    pred_scale: float = 1.0,
) -> list[str]:
    """One rebalance step of top-k dropout.

    Names are dropped only when they have actually fallen out of the current
    top-k, and at most ``n_drop`` per period. A held name whose score has gone
    missing (delisted, or out of the index) is force-dropped regardless, since
    it is no longer tradeable.

    Dropping "at most" rather than "always" is what makes a constant
    prediction produce zero turnover, which is the behaviour the strategy
    should have: no new information, no trade.

    Three feasibility limits narrow the choices when they are supplied:
    ``can_buy``/``can_sell`` remove names locked at a price limit or suspended,
    and ``locked_until`` holds T+1 -- a name bought at step ``s`` cannot be sold
    before ``s+1``. A name that cannot be sold stays in the book even if its
    score has collapsed, which is the whole point: that is what actually
    happens.

    With ``theta.portfolio.cost_aware_dropout``, the number of swaps stops being
    a quota and becomes a decision. Candidates are considered best-first and a
    swap is made only while the predicted-return gain clears the modelled
    round-trip cost, so turnover responds to how much better the alternatives
    actually are.
    """
    ranked = scores.sort_values(ascending=False)
    available = list(ranked.index)
    if not available:
        return held

    buyable = (lambda name: bool(can_buy.get(name, True))) if can_buy is not None else (lambda name: True)
    sellable = (lambda name: bool(can_sell.get(name, True))) if can_sell is not None else (lambda name: True)

    def can_release(name: str) -> bool:
        if not sellable(name):
            return False
        if locked_until is not None and step < locked_until.get(name, -1):
            return False
        return True

    live_held = [name for name in held if name in scores.index]
    dropped_from_universe = [name for name in held if name not in scores.index]
    if dropped_from_universe:
        logger.debug("top-k dropout: %d held name(s) left the universe", len(dropped_from_universe))

    # Fresh book: take the best names we are actually allowed to buy.
    if not live_held:
        return [n for n in available if buyable(n)][:topk]

    target = [n for n in available[: min(topk, len(available))]]
    target_set = set(target)
    held_set = set(live_held)

    stale = [n for n in live_held if n not in target_set and can_release(n)]
    incoming_pool = [n for n in available if n not in held_set and buyable(n)]

    if not stale or not incoming_pool:
        new_held = list(live_held)
    elif theta is not None and theta.portfolio.cost_aware_dropout:
        # Worst holdings out, best candidates in -- but only while it pays.
        w_each = float(theta.portfolio.gross_leverage) / max(len(live_held), 1)
        hurdle = float(theta.portfolio.swap_hurdle)
        vol = (lambda n: float(sigma.get(n, 0.0)) if sigma is not None else 0.0)
        worst_first = sorted(stale, key=lambda n: scores[n])
        swaps: list[tuple[str, str]] = []
        for out_name, in_name in zip(worst_first, incoming_pool):
            if len(swaps) >= n_drop:
                break
            # pred_scale puts the score gap into expected-return units so it is
            # comparable with a cost; see execution.prediction_scale.
            gain = pred_scale * (float(scores[in_name]) - float(scores[out_name]))
            cost = _swap_cost(theta, w_each, vol(out_name), vol(in_name))
            if gain <= hurdle * cost:
                # Candidates are ordered best-first and holdings worst-first, so
                # once the best remaining swap fails to pay, no later one can.
                break
            swaps.append((out_name, in_name))
        if swaps:
            out_set = {o for o, _ in swaps}
            new_held = [n for n in live_held if n not in out_set] + [i for _, i in swaps]
        else:
            new_held = list(live_held)
    else:
        n = min(n_drop, len(stale))
        worst_first = sorted(stale, key=lambda name: scores[name])
        dropped = set(worst_first[:n])
        incoming = incoming_pool[:n]
        new_held = [name for name in live_held if name not in dropped] + incoming

    # Top up if names were force-dropped for leaving the universe.
    if len(new_held) < topk:
        held_set = set(new_held)
        new_held += [n for n in available if n not in held_set and buyable(n)][: topk - len(new_held)]

    return new_held[:topk]


def topk_dropout(
    pred: pd.DataFrame,
    theta: Protocol,
    y_tilde: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    prev_w: pd.Series | None = None,
    mask: "TradeMask | None" = None,
    sigma: pd.DataFrame | None = None,
    pred_scale: float = 1.0,
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

    # T+1: step index at which each holding first becomes sellable.
    enforce_t1 = bool(theta.constraints.enforce_t1 and theta.constraints.enabled)
    long_locked: dict[str, int] = {}
    short_locked: dict[str, int] = {}

    for step, date in enumerate(pred.index):
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

        row = lambda frame: (
            frame.loc[date] if frame is not None and date in frame.index else None
        )
        can_buy_row = row(mask.can_buy) if mask is not None else None
        can_sell_row = row(mask.can_sell) if mask is not None else None
        sigma_row = row(sigma)

        prev_long = set(long_held)
        long_held = _select(
            scores, long_held, topk, n_drop, theta=theta,
            can_buy=can_buy_row, can_sell=can_sell_row, sigma=sigma_row,
            locked_until=long_locked if enforce_t1 else None, step=step,
            pred_scale=pred_scale,
        )
        if enforce_t1:
            for name in long_held:
                if name not in prev_long:
                    long_locked[name] = step + 1        # sellable from the next step
            for name in list(long_locked):
                if name not in long_held:
                    long_locked.pop(name, None)

        if signed:
            prev_short = set(short_held)
            short_held = _select(
                -scores, short_held, topk, n_drop, theta=theta,
                # A short entry sells and a short exit buys, so the buy/sell
                # feasibility of the underlying is the other way round.
                can_buy=can_sell_row, can_sell=can_buy_row, sigma=sigma_row,
                locked_until=short_locked if enforce_t1 else None, step=step,
                pred_scale=pred_scale,
            )
            if enforce_t1:
                for name in short_held:
                    if name not in prev_short:
                        short_locked[name] = step + 1
                for name in list(short_locked):
                    if name not in short_held:
                        short_locked.pop(name, None)

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


__all__ = ["build_book", "mean_variance", "solo_book", "topk_dropout"]


def _mv_weights(
    mu: pd.Series,
    var: pd.Series,
    lam: float,
    max_weight: float,
) -> pd.Series:
    """argmax wᵀμ − (λ/2)wᵀΣw  s.t.  Σw = 1, 0 ≤ w ≤ max_weight, Σ diagonal.

    A diagonal risk model makes the problem separable, so it has a closed form
    given the budget multiplier ``η``::

        w_i(η) = clip((μ_i − η) / (λ σ_i²), 0, max_weight)

    ``Σw_i(η)`` is monotonically decreasing in ``η``, so bisection finds the η
    that spends exactly the budget. Exact to machine precision and fast enough
    to run per date, which a general QP over 300 names would not be.

    The diagonal model is a deliberate simplification and the honest limit of
    this implementation: it prices each name's own volatility but no
    correlation, so it cannot see that two names are the same bet. A factor
    risk model is the next refinement, and until then the diversity dimension
    of ``U`` is what carries redundancy control.
    """
    v = var.clip(lower=1e-8)
    lo = float(mu.min() - lam * v.max())
    hi = float(mu.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(0.0, index=mu.index)

    weights = lambda eta: ((mu - eta) / (lam * v)).clip(lower=0.0, upper=max_weight)
    if float(weights(lo).sum()) < 1.0:
        # Even at full risk appetite the box caps prevent full investment.
        return weights(lo)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(weights(mid).sum()) > 1.0:
            lo = mid
        else:
            hi = mid
    w = weights(0.5 * (lo + hi))
    total = float(w.sum())
    return w / total if total > 0 else w


def _apply_tradability(
    w: pd.Series,
    w_drift: pd.Series,
    can_buy: pd.Series | None,
    can_sell: pd.Series | None,
) -> pd.Series:
    """Clamp a target book to what can actually be traded, then re-invest.

    A name that cannot be bought may not rise above its drifted weight; one
    that cannot be sold may not fall below it. Clamping breaks the budget, so
    the residual is redistributed across the names that ARE free to trade --
    renormalising everything instead would silently push blocked names back
    through the very limits just imposed.
    """
    if can_buy is None and can_sell is None:
        return w
    out = w.copy()
    blocked = pd.Series(False, index=w.index)
    if can_buy is not None:
        no_buy = ~can_buy.reindex(w.index).fillna(True).astype(bool)
        out[no_buy] = np.minimum(out[no_buy], w_drift.reindex(w.index)[no_buy])
        blocked |= no_buy
    if can_sell is not None:
        no_sell = ~can_sell.reindex(w.index).fillna(True).astype(bool)
        out[no_sell] = np.maximum(out[no_sell], w_drift.reindex(w.index)[no_sell])
        blocked |= no_sell

    residual = 1.0 - float(out.sum())
    free = ~blocked
    if abs(residual) > 1e-12 and bool(free.any()):
        base = out[free]
        share = base / base.sum() if float(base.sum()) > 0 else pd.Series(
            1.0 / int(free.sum()), index=base.index)
        out[free] = (base + residual * share).clip(lower=0.0)
    return out


def mean_variance(
    pred: pd.DataFrame,
    theta: Protocol,
    y_tilde: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    prev_w: pd.Series | None = None,
    mask: "TradeMask | None" = None,
    sigma: pd.DataFrame | None = None,
    pred_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``g`` as a mean--variance optimiser with an explicit turnover budget.

    The member of the admissible family that top-k dropout cannot be: turnover
    is a *constraint the optimiser trades against*, not a by-product of
    ``n_drop/topk``. That is what makes it possible to ask whether a
    capacity-aware objective produces a book that survives its own costs --
    under top-k dropout the question is unanswerable because the book trades
    the same amount whatever it predicts.

    Expected returns are ``pred_scale · ŷ`` so that μ, the risk penalty and the
    turnover budget are all in the same units; see
    :func:`~quantaalpha.eval.execution.prediction_scale`.
    """
    port = theta.portfolio
    lam = float(port.risk_aversion)
    cap = float(port.turnover_cap)
    max_w = float(port.max_weight)
    if port.signed:
        raise NotImplementedError("mean_variance currently builds a long-only book")

    weights = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
    drifted = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
    w_prev = prev_w.reindex(pred.columns).fillna(0.0) if prev_w is not None else None

    for date in pred.index:
        scores = pred.loc[date]
        if universe is not None:
            scores = scores.where(universe.loc[date].astype(bool))
        scores = scores.dropna()

        if w_prev is None:
            w_drift = pd.Series(0.0, index=pred.columns)
        elif y_tilde is None:
            w_drift = w_prev.copy()
        else:
            prior = y_tilde.shift(1).loc[date] if date in y_tilde.index else None
            w_drift = drift(w_prev, prior, theta) if prior is not None else w_prev.copy()

        if scores.empty:
            weights.loc[date] = w_drift
            drifted.loc[date] = w_drift
            w_prev = w_drift
            continue

        mu = pred_scale * scores
        vol = (sigma.loc[date].reindex(scores.index)
               if sigma is not None and date in sigma.index
               else pd.Series(0.02, index=scores.index))
        target = _mv_weights(mu, vol.fillna(vol.median() if vol.notna().any() else 0.02) ** 2,
                             lam, max_w)

        # Spend at most the turnover budget: move partway from the drifted book
        # toward the target. Both are on the simplex, so the blend is feasible.
        w_full = target.reindex(pred.columns).fillna(0.0)
        step = w_full - w_drift
        move = 0.5 * float(step.abs().sum())
        if move > cap > 0:
            w_new = w_drift + step * (cap / move)
        else:
            w_new = w_full

        row = lambda f: (f.loc[date] if f is not None and date in f.index else None)
        w_new = _apply_tradability(
            w_new, w_drift,
            row(mask.can_buy) if mask is not None else None,
            row(mask.can_sell) if mask is not None else None,
        )

        exposure = float(w_new.abs().sum())
        if exposure > float(port.gross_leverage) + 1e-6:
            raise AssertionError(f"gross leverage {exposure:.6f} exceeds "
                                 f"{port.gross_leverage} on {date}")

        weights.loc[date] = w_new
        drifted.loc[date] = w_drift
        w_prev = w_new

    return weights, drifted


CONSTRUCTIONS = {"topk_dropout": topk_dropout, "mean_variance": mean_variance}


def build_book(pred: pd.DataFrame, theta: Protocol, **kwargs):
    """Dispatch to the construction named in Θ."""
    try:
        fn = CONSTRUCTIONS[theta.portfolio.construction]
    except KeyError:
        raise NotImplementedError(
            f"unsupported construction {theta.portfolio.construction!r}; "
            f"known: {sorted(CONSTRUCTIONS)}"
        ) from None
    return fn(pred, theta, **kwargs)
