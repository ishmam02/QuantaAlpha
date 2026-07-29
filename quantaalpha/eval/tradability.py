"""Which names can actually be traded on a given day (A-share constraints).

Three mechanics of the mainland market decide whether an order fills at all.
None of them is a cost -- they are hard feasibility limits, and omitting them
does not make a backtest slightly optimistic, it makes it wrong in a direction
that correlates with signal strength:

* **Daily price limits.** Main-board names are halted at ±10% from the previous
  close (±5% for ST, ±20% on STAR/ChiNext since 2020). A name locked at
  limit-up has no sellers, so a buy cannot fill; locked at limit-down it has no
  buyers, so a sell cannot fill. This is the single most dangerous omission for
  CSI 300 research: a factor that predicts limit-up moves scores beautifully on
  IC and is completely untradeable, and nothing in a cost model catches that.

* **Suspensions.** Halted names do not trade in either direction. A book that
  "sells" a suspended holding books a fill that could not have happened.

* **T+1 settlement.** Shares bought today cannot be sold until the next trading
  day. Verified to be **structurally non-binding here**: the book rebalances
  once daily, so a name entering at step ``s`` cannot be sold before step
  ``s+1``, which is already a session later. The check is retained for an
  intraday construction but enabling it at daily frequency changes no book.

Detection is deliberately conservative, and uses only fields the panel already
carries:

``high == low`` with positive volume means the name traded at exactly one price
all session -- the signature of a locked limit. Requiring the move to also
reach the limit threshold avoids flagging a genuinely flat illiquid name.
``volume <= 0`` (or missing) is a suspension.

Both masks are expressed at the **decision** date: entry ``t`` answers "if I
decide at ``t``, can the order fill?", which already accounts for the ``δ``-day
fill lag. Callers therefore index them by decision date and need no further
shifting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeMask:
    """Per-(date, name) feasibility, indexed by DECISION date.

    ``can_buy``/``can_sell`` are ``False`` where an order decided that day could
    not fill. ``suspended`` is kept separately because a suspended holding is
    not merely unsellable -- it also stops earning the tradeable return.
    """

    can_buy: pd.DataFrame
    can_sell: pd.DataFrame
    suspended: pd.DataFrame
    universe: pd.DataFrame

    @property
    def stats(self) -> dict[str, float]:
        """Share of blocked cells, measured **within the index universe**.

        Denominating by the full (date x name) grid instead would report a name
        that simply was not in the index yet as "suspended": on CSI 300 that
        read 52% against a true suspension rate of a few percent. The universe
        mask already excludes non-members from selection, so this only ever
        affected the diagnostic -- but a 52% figure in a log is the kind of
        thing that gets believed.
        """
        live = self.universe.astype(bool)
        total = float(live.sum().sum()) or 1.0
        blocked = lambda frame: 100.0 * float((frame & live).sum().sum()) / total
        return {
            "pct_no_buy": blocked(~self.can_buy),
            "pct_no_sell": blocked(~self.can_sell),
            "pct_suspended": blocked(self.suspended),
        }


def _locked_at_limit(panel: PanelBundle, pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(limit_up, limit_down) masks on the FILL date.

    A locked limit shows two signatures together: no intraday range at all
    (``high == low``) and a move that reaches the band. Requiring both is what
    keeps a flat, thinly traded name from being misread as limit-locked.
    """
    prev_close = panel.close.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        move = panel.close / prev_close - 1.0

    # A hair inside the band, since adjusted prices do not land exactly on it.
    threshold = pct * 0.98
    no_range = (panel.high <= panel.low) & (panel.volume > 0)

    limit_up = no_range & (move >= threshold)
    limit_down = no_range & (move <= -threshold)
    return limit_up.fillna(False), limit_down.fillna(False)


def trade_mask(panel: PanelBundle, theta: Protocol) -> TradeMask:
    """Build the feasibility masks for a panel, indexed by decision date.

    An order decided at ``t`` fills at ``t + δ``, so every fill-date condition
    is shifted back by ``δ`` to sit on the decision date. Rows at the end of the
    window have no fill date and are conservatively marked untradeable.
    """
    cons = theta.constraints
    dates, names = panel.dates, panel.instruments
    true_frame = pd.DataFrame(True, index=dates, columns=names)

    live = panel.universe.astype(bool)
    if not cons.enabled:
        return TradeMask(true_frame, true_frame.copy(),
                         pd.DataFrame(False, index=dates, columns=names), live)

    suspended = (panel.volume.fillna(0.0) <= 0) | panel.close.isna()

    can_buy_fill = pd.DataFrame(True, index=dates, columns=names)
    can_sell_fill = pd.DataFrame(True, index=dates, columns=names)

    if cons.enforce_price_limits:
        limit_up, limit_down = _locked_at_limit(panel, float(cons.price_limit_pct))
        # Locked limit-up: no sellers, so a buy cannot fill (and vice versa).
        can_buy_fill &= ~limit_up
        can_sell_fill &= ~limit_down

    if cons.enforce_suspension:
        can_buy_fill &= ~suspended
        can_sell_fill &= ~suspended

    # Shift fill-date feasibility back onto the decision date.
    delta = int(theta.execution.delta)
    if delta:
        # No fill date exists past the window edge; refuse rather than assume.
        # Build the bool frame explicitly -- shift() on a bool frame yields
        # object dtype, and .fillna on that is deprecated.
        can_buy = pd.DataFrame(
            can_buy_fill.shift(-delta).to_numpy() == True, index=dates, columns=names)
        can_sell = pd.DataFrame(
            can_sell_fill.shift(-delta).to_numpy() == True, index=dates, columns=names)
    else:
        can_buy, can_sell = can_buy_fill, can_sell_fill

    mask = TradeMask(can_buy, can_sell, suspended.fillna(True), live)
    stats = mask.stats
    logger.info(
        "trade constraints (within the index universe): %.2f%% cannot be bought, "
        "%.2f%% cannot be sold, %.2f%% suspended (price_limit=%s t1=%s)",
        stats["pct_no_buy"], stats["pct_no_sell"], stats["pct_suspended"],
        cons.enforce_price_limits, cons.enforce_t1,
    )
    return mask


__all__ = ["TradeMask", "trade_mask"]
