"""The seven quality dimensions (§3.4).

Metrics split cleanly by provenance, which is both a correctness property and
the performance story:

* **per-factor** — ``ic``, ``rank_ic``, ``icir``, ``rank_icir``,
  ``ic_pos_frac``, ``cx``, ``is_oos_gap``, the decay block, ``turnover_solo``.
  These depend on ``f`` alone and are invariant to the cost model, so they are
  cacheable by ``md5(expr)`` and identical across a zero-cost and a costed Θ.
* **repository** — ``rho_max``, which depends on ``(f, zoo)``.
* **strategy-level** — ``net_ir``, ``net_arr``, ``mdd``, ``turnover_book``,
  ``cost_bps``. These depend on ``(f, zoo)`` through the combiner refit, and
  are the only ones that cost a model fit.

**The evaluation window is read from Θ, never accepted as a caller argument.**
That is deliberate: a function that takes a window will eventually be handed
the test window by some future caller, and Property 3 would be violated
silently. The panel a caller supplies bounds what is available; Θ decides what
is *used*.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _to_wide(signal: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Accept either a ``(datetime, instrument)`` Series or a wide frame."""
    if isinstance(signal, pd.DataFrame) and not isinstance(signal.index, pd.MultiIndex):
        return signal
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]
    wide = signal.unstack(level=-1)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _slice(frame: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    start, end = window
    return frame.loc[str(start) : str(end)]


def _cross_sectional_corr(a: pd.DataFrame, b: pd.DataFrame, method: str) -> pd.Series:
    """Per-date cross-sectional correlation between two aligned wide frames.

    This is the shape ``runner.py:47-73`` already computed for the (dead)
    de-duplication path; reviving it here is what makes ``rho_max`` cheap.
    """
    b = b.reindex(index=a.index, columns=a.columns)
    out = {}
    for date in a.index:
        x, y = a.loc[date], b.loc[date]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 3:
            continue
        xs, ys = x[mask], y[mask]
        if method == "spearman":
            xs, ys = xs.rank(), ys.rank()
        if xs.std() == 0 or ys.std() == 0:
            continue
        out[date] = float(np.corrcoef(xs.to_numpy(), ys.to_numpy())[0, 1])
    return pd.Series(out, dtype=float).sort_index()


def label_frame(panel: PanelBundle, theta: Protocol) -> pd.DataFrame:
    """The close-to-close training label as a wide frame.

    Matches Θ's ``label_expr`` (``Ref($close,-2)/Ref($close,-1)-1``) evaluated
    on the panel's raw close, i.e. exactly what the baseline Qlib handler
    computes. Used for **IC only** — realized P&L comes from the fill series
    in ``execution.py``, which is the point of the label/fill split.
    """
    close = panel.close
    return close.shift(-2) / close.shift(-1) - 1.0


def cx(expr: str) -> int:
    """Structural complexity — symbol length, the scalar ``γ_cx`` bounds.

    ``calculate_symbol_length`` is what the repository's existing
    ``symbol_length_threshold`` already means, so reusing it keeps the gate
    comparable to the pre-existing complexity check.
    """
    from quantaalpha.factors.coder.factor_ast import calculate_symbol_length

    return int(calculate_symbol_length(expr))


def complexity_diagnostics(expr: str) -> dict[str, int]:
    """The other AST statistics, carried alongside ``cx`` as diagnostics."""
    from quantaalpha.factors.coder import factor_ast

    out: dict[str, int] = {}
    for key, fn in (
        ("cx_base_features", factor_ast.count_base_features),
        ("cx_free_args", factor_ast.count_free_args),
        ("cx_nodes", factor_ast.count_all_nodes),
    ):
        try:
            out[key] = int(fn(expr))
        except Exception as exc:  # a malformed expression must not kill an eval
            logger.debug("complexity diagnostic %s failed on %r: %s", key, expr[:40], exc)
            out[key] = -1
    return out


def rho_max(signal, zoo_signals: dict[str, object]) -> float:
    """ρ_max (Eq. 9) — redundancy against the repository.

    Maximum **absolute** time-averaged cross-sectional correlation against any
    incumbent. Absolute because a perfectly inverted copy of an existing factor
    carries no new information either.

    Defined as ``0.0`` for an empty zoo: the first factor is trivially novel.
    """
    if not zoo_signals:
        return 0.0
    candidate = _to_wide(signal)
    best = 0.0
    for expr, other in zoo_signals.items():
        corr = _cross_sectional_corr(candidate, _to_wide(other), "pearson")
        if corr.empty:
            continue
        best = max(best, abs(float(corr.mean())))
    return float(best)


# --------------------------------------------------------------------------
# per-factor metrics
# --------------------------------------------------------------------------
def _ic_block(signal: pd.DataFrame, label: pd.DataFrame, window: tuple[str, str]) -> dict:
    ic = _cross_sectional_corr(_slice(signal, window), label, "pearson")
    ric = _cross_sectional_corr(_slice(signal, window), label, "spearman")
    if ic.empty:
        return {"ic": np.nan, "rank_ic": np.nan, "icir": np.nan, "rank_icir": np.nan,
                "ic_pos_frac": np.nan, "_ic_series": ic}
    return {
        "ic": float(ic.mean()),
        "rank_ic": float(ric.mean()),
        "icir": float(ic.mean() / ic.std()) if ic.std() else np.nan,
        "rank_icir": float(ric.mean() / ric.std()) if ric.std() else np.nan,
        "ic_pos_frac": float((ic > 0).mean()),
        "_ic_series": ic,
    }


def _decay_block(ic_series: pd.Series, theta: Protocol) -> dict:
    """§3.4 item 7 — is the edge fading across the OOS window?

    Split OOS into ``K`` equal sub-periods, take the mean IC of each, and fit a
    line. A negative slope means the signal is decaying, which discounts the
    persistence-adjusted IC.
    """
    K = int(theta.decay.K)
    if len(ic_series) < K or K < 2:
        return {"decay_slope": np.nan, "persistence_ratio": np.nan, "ic_pers": np.nan}

    chunks = np.array_split(ic_series.to_numpy(), K)
    means = np.array([float(np.nanmean(c)) if len(c) else np.nan for c in chunks])
    if np.isnan(means).any():
        return {"decay_slope": np.nan, "persistence_ratio": np.nan, "ic_pers": np.nan}

    slope = float(np.polyfit(np.arange(K), means, 1)[0])
    mean_ic = float(np.nanmean(ic_series))
    # Fitted fractional change in IC across the whole OOS window.
    b_norm = float(np.clip(slope * (K - 1) / max(abs(mean_ic), 1e-8), -1.0, 1.0))
    ic_pers = mean_ic * (1.0 - float(theta.decay.lambda_d) * max(0.0, -b_norm))
    ratio = float(means[-1] / means[0]) if means[0] != 0 else np.nan
    return {"decay_slope": slope, "persistence_ratio": ratio, "ic_pers": float(ic_pers)}


def per_factor_metrics(
    signal,
    expr: str,
    panel: PanelBundle,
    theta: Protocol,
    *,
    oos: bool = True,
) -> dict:
    """Metrics that depend on the factor alone.

    Headline ``ic``/``rank_ic``/``icir``/``rank_icir`` are measured on the IS
    window from Θ; ``is_oos_gap`` contrasts them with the OOS proxy window.
    Both windows come from Θ, never from the caller.
    """
    wide = _to_wide(signal).reindex(index=panel.dates, columns=panel.instruments)
    wide = wide.where(panel.universe)
    label = label_frame(panel, theta)

    is_block = _ic_block(wide, label, theta.splits.is_window)
    metrics = {k: v for k, v in is_block.items() if not k.startswith("_")}

    oos_block = _ic_block(wide, label, theta.splits.oos_window) if oos else None
    if oos_block is not None and not np.isnan(oos_block["rank_ic"]):
        metrics["rank_ic_oos"] = oos_block["rank_ic"]
        metrics["is_oos_gap"] = float(metrics["rank_ic"] - oos_block["rank_ic"])
        metrics.update(_decay_block(oos_block["_ic_series"], theta))
    else:
        metrics["rank_ic_oos"] = np.nan
        metrics["is_oos_gap"] = np.nan
        metrics.update({"decay_slope": np.nan, "persistence_ratio": np.nan, "ic_pers": np.nan})

    metrics["cx"] = cx(expr)
    metrics.update(complexity_diagnostics(expr))
    return metrics


def solo_turnover(signal, panel: PanelBundle, theta: Protocol) -> float:
    """Turnover of a book built from the candidate's own signal.

    Recorded because ``turnover_book`` is structurally pinned at
    ``n_drop/topk`` under top-k dropout and therefore cannot discriminate
    between signals; this one can.
    """
    from quantaalpha.eval.portfolio import solo_book

    wide = _to_wide(signal).reindex(index=panel.dates, columns=panel.instruments)
    window = theta.splits.is_window
    sliced = _slice(wide, window)
    if sliced.empty:
        return float("nan")
    w, wd = solo_book(sliced, theta, universe=_slice(panel.universe, window))
    return float((0.5 * (w - wd).abs().sum(axis=1)).iloc[1:].mean())


# --------------------------------------------------------------------------
# strategy-level metrics
# --------------------------------------------------------------------------
def max_drawdown(r_net: pd.Series) -> float:
    """Maximum drawdown of the cumulative net-return path (negative)."""
    curve = (1.0 + r_net.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def strategy_metrics(
    w: pd.DataFrame,
    w_drift: pd.DataFrame,
    r_net: pd.Series,
    costs: pd.Series,
    theta: Protocol,
) -> dict:
    """Net-of-cost strategy performance of the book *containing* the candidate.

    These are marginal-contribution numbers, not stand-alone factor
    performance: the book is built from the combiner's composite prediction
    over ``zoo ∪ {f}``.
    """
    clean = r_net.dropna()
    periods = int(theta.periods_per_year)
    if clean.empty:
        return {"net_ir": np.nan, "net_arr": np.nan, "mdd": np.nan,
                "turnover_book": np.nan, "cost_bps": np.nan}

    std = float(clean.std())
    net_ir = float(clean.mean() / std * np.sqrt(periods)) if std else np.nan
    growth = float((1.0 + clean).prod())
    net_arr = float(np.sign(growth) * abs(growth) ** (periods / len(clean)) - 1.0) if growth > 0 else -1.0
    book_to = 0.5 * (w - w_drift).abs().sum(axis=1)

    return {
        "net_ir": net_ir,
        "net_arr": net_arr,
        "mdd": max_drawdown(clean),
        "turnover_book": float(book_to.iloc[1:].mean()) if len(book_to) > 1 else 0.0,
        "cost_bps": float(costs.reindex(clean.index).fillna(0.0).mean() * 1e4),
    }


__all__ = [
    "complexity_diagnostics",
    "cx",
    "label_frame",
    "max_drawdown",
    "per_factor_metrics",
    "rho_max",
    "solo_turnover",
    "strategy_metrics",
]
