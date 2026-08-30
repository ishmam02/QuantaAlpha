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

from typing import Sequence

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

    The pearson path is computed as whole-array sums rather than a Python loop
    over dates. This function was the single largest cost in an evaluation
    (~37% of it): ``rho_max`` calls it once per repository member, so the
    per-date pandas indexing was paid |zoo| times per evaluation and grew as
    the repository grew. The vectorised form measured 58-72x faster and matches
    the loop to 3e-16 with an identical index, skip rules included. The
    spearman branch still ranks row-by-row (ties need per-row handling) but
    masks and correlates in the same vectorised pass.
    """
    b = b.reindex(index=a.index, columns=a.columns)
    A = a.to_numpy(dtype=float)
    B = b.to_numpy(dtype=float)
    if A.size == 0:
        return pd.Series(dtype=float)
    mask = np.isfinite(A) & np.isfinite(B)
    n = mask.sum(axis=1)

    if method == "spearman":
        A = _rank_rows(A, mask)
        B = _rank_rows(B, mask)

    cnt = n.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Means over the pairwise-present entries only, then centred sums --
        # the same quantities np.corrcoef forms on the masked slice.
        ma = np.where(mask, A, 0.0).sum(axis=1) / cnt
        mb = np.where(mask, B, 0.0).sum(axis=1) / cnt
        Ac = np.where(mask, A - ma[:, None], 0.0)
        Bc = np.where(mask, B - mb[:, None], 0.0)
        saa = (Ac * Ac).sum(axis=1)
        sbb = (Bc * Bc).sum(axis=1)
        corr = (Ac * Bc).sum(axis=1) / np.sqrt(saa * sbb)

    # The loop skipped a date with <3 pairwise points or zero variance on
    # either side; zero centred sum-of-squares is exactly that zero-variance
    # test, so those dates are dropped from the index rather than carried as NaN.
    keep = (n >= 3) & (saa > 0) & (sbb > 0) & np.isfinite(corr)
    return pd.Series(corr[keep], index=a.index[keep], dtype=float).sort_index()


def _rank_rows(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row average ranks over the masked entries (pandas ``.rank()`` semantics).

    Ranks are 1-based with ties averaged, computed only over the entries the
    pairwise mask keeps -- matching ``x[mask].rank()`` in the loop this
    replaces.
    """
    # Vectorised via pandas' Cython ranker. The previous form was a Python loop
    # over dates (~1200) with an inner while-loop for tie runs, which made the
    # spearman path **43.65x slower than pearson** (718 ms vs 16.5 ms per pair).
    # That only became load-bearing when ``rho_max`` switched to spearman to
    # measure redundancy in the space the combiner actually fits in -- at which
    # point one rho_max call over a 9-factor zoo cost 6.5 SECONDS and grew
    # linearly with the repository.
    #
    # ``DataFrame.rank(axis=1)`` defaults to method="average" and leaves NaN as
    # NaN, which is exactly the 1-based average-rank-over-masked-entries
    # semantics implemented above; masked-out cells are set to NaN first so they
    # are excluded from the ranking rather than ranked as values.
    return (
        pd.DataFrame(np.where(mask, m, np.nan))
        .rank(axis=1)
        .to_numpy(dtype=float)
    )


_LABEL_FIELDS = {"$close": "close", "$open": "open", "$vwap": "vwap"}


def label_frame(panel: PanelBundle, theta: Protocol) -> pd.DataFrame:
    """Θ's training label as a wide frame, for IC.

    **Derived from ``Θ.execution.label_expr``, not hardcoded.** This used to
    return ``close.shift(-2)/close.shift(-1) - 1`` regardless of what the
    protocol said, which was correct only for as long as nobody changed the
    label. Changing it would then have moved the *training* target while
    leaving the *IC* target on close, so every IC in the system would have
    silently measured a quantity the model was no longer fitting.

    Only the ``Ref(P,-2)/Ref(P,-1)-1`` shape is understood, because that is the
    only shape the wide-panel form can express. Anything else raises rather
    than falling back to close and being quietly wrong about it.
    """
    # Delegates to the horizon-aware form so IC measures the SAME return the
    # combiner trains on and the book holds. Training on a 20-day target while
    # scoring IC on a 1-day one would report a quality the strategy never
    # realises -- the two must move together, which is why this is one call and
    # not two parallel implementations.
    h = int(getattr(theta.execution, "label_horizon", 1) or 1)
    expr = theta.execution.label_expr
    field = next((f for f in _LABEL_FIELDS if f in expr), None)
    if field is None or "Ref(" not in expr:
        raise ValueError(
            f"label_frame cannot express {expr!r}. It understands "
            f"Ref(P,-(1+h))/Ref(P,-1)-1 for P in {sorted(_LABEL_FIELDS)}; "
            "anything else must be evaluated through Qlib rather than on the panel.")
    return label_frame_at(panel, theta, h)


def label_frame_at(panel: PanelBundle, theta: Protocol, horizon: int) -> pd.DataFrame:
    """The training label generalised to an ``horizon``-day holding period.

    ``label_frame`` is this at ``horizon=1``: ``P[t+2]/P[t+1] - 1``, a one-day
    forward return measured from the fill. The generalisation keeps the FILL
    anchored at ``t+1`` and moves only the exit::

        horizon h  ->  P[t+1+h] / P[t+1] - 1

    so every horizon measures a return the book could actually have earned
    starting from the same entry, and h=1 reproduces ``label_frame`` exactly.

    **This does not change the book's holding period.** It is used to ask
    whether a factor is predictive at a longer horizon, for ADMISSION; the
    portfolio still rebalances daily. A factor that predicts 20-day returns is
    valuable precisely because it decays slowly and is therefore cheap to hold
    -- the cost model already rewards that through turnover.

    Why this matters: with a single horizon every mined factor competes to
    approximate the SAME target, which is a large part of why the search
    saturates at a handful of independent ideas. Signals predictive at 1 day
    and at 20 days are different economics, not variants of one.
    """
    h = max(1, int(horizon))
    expr = theta.execution.label_expr
    field = next((f for f in _LABEL_FIELDS if f in expr), None)
    if field is None or "Ref(" not in expr:
        raise ValueError(
            f"label_frame_at cannot express {expr!r}; it understands "
            f"Ref(P,-2)/Ref(P,-1)-1 for P in {sorted(_LABEL_FIELDS)}")
    price = getattr(panel, _LABEL_FIELDS[field])
    return price.shift(-(1 + h)) / price.shift(-1) - 1.0


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

    Measured with **spearman**, not pearson, because that is the space the
    combiner actually fits in: ``combiner.py`` applies ``_cs_rank_norm`` to every
    feature before use, so ``X`` and ``RANK(X)`` become the *same* column. Raw
    pearson does not see that. Measured on the live 9-factor zoo 2026-08-15,
    ``Up_Down_Volume_Ratio_10D`` vs ``Volume_Weighted_Return_Sign_10D``:
    pearson **0.628** (passes ``rho_bar`` 0.80) but rank-space **0.998** -- the
    gate admitted a pair the combiner then treated as one feature. Monotone
    re-expressions (RANK/ZSCORE/log of an existing factor) are exactly what the
    generator produces when asked for variants of one idea, so this was the
    common case, not a corner case.
    """
    return rho_max_arg(signal, zoo_signals)[0]


def rho_max_arg(signal, zoo_signals: dict[str, object]) -> tuple[float, str | None]:
    """``rho_max`` plus the identity of the incumbent that produced it.

    The scalar alone says "this duplicates something we hold" and throws away
    WHICH something. That is the difference between a rejection the generator
    can act on and one it cannot -- and it is also what a replacement decision
    needs: to know whether a near-duplicate is BETTER than the incumbent it
    resembles, you have to know which incumbent that is.
    """
    if not zoo_signals:
        return 0.0, None
    candidate = _to_wide(signal)
    best, best_expr = 0.0, None
    for expr, other in zoo_signals.items():
        corr = _cross_sectional_corr(candidate, _to_wide(other), "spearman")
        if corr.empty:
            continue
        r = abs(float(corr.mean()))
        if r > best:
            best, best_expr = r, expr
    return float(best), best_expr


def _abs_spearman(a, b) -> float:
    """Time-average absolute cross-sectional Spearman between two wide signals.

    Factored out of :func:`spearman_abs_matrix` so the marginal-er gate's
    incremental path (:func:`effective_rank_cached`) computes its extra-block
    entries with the SAME primitive as the full matrix. The cached and
    freshly-computed entries are then bit-identical, so caching the repo-repo
    block does not change the gate's verdict by a single eigenvalue.
    """
    c = _cross_sectional_corr(a, b, "spearman")
    return abs(float(c.mean())) if not c.empty else 0.0


def spearman_abs_matrix(signals: dict) -> "np.ndarray":
    """|Spearman| cross-sectional correlation matrix of a dict of wide signals.

    The same construction ``operator.evaluate`` uses for the book's effective
    rank: entry (i, j) is the time-average of the absolute per-date
    cross-sectional Spearman correlation between signal i and j. Expects
    wide (date x instrument) frames (what ``_to_wide`` returns).
    """
    keys = list(signals)
    n = len(keys)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = _abs_spearman(signals[keys[i]], signals[keys[j]])
            R[i, j] = R[j, i] = v
    return R


# Shared zoo x zoo |Spearman| block. ``operator.evaluate`` (the effective_rank
# METRIC) and ``runner._decide_standalone`` (the marginal-er GATE) compute the
# SAME block over the held zoo on the SAME panel -- both derive their signals
# from the repository in the SAME insertion order and align to ``op._panel`` of
# ``op._windows(False)`` -- so one shared build replaces two O(zoo**2) passes
# per batch (evaluate builds it first; _decide reuses it). ``effective_rank`` /
# ``effective_rank_cached`` are eigenvalue-based (permutation-invariant), and
# the block is built in the caller's dict-iteration order -- which both callers
# already used -- so a cached hit is bit-identical to a fresh build for both.
# Bounded to a couple of zoo states (the zoo changes one factor at a time).
_SPEARMAN_BLOCK_CACHE: dict[tuple, "np.ndarray"] = {}
_SPEARMAN_BLOCK_CACHE_MAX = 2


def spearman_block_cached(signals: dict, panel_key) -> "np.ndarray":
    """Shared cached zoo x zoo |Spearman| block, keyed by signal set + panel.

    ``signals`` is the held-zoo dict; the block is built in its iteration order
    and keyed by ``(tuple(exprs in that order), panel_key)``. Both callers pass
    the repository's insertion order on the same panel grid, so a block built
    by one is reused by the other. ``panel_key`` is whatever the caller uses to
    identify the alignment grid (a ``(start, end)`` panel span), so a re-split
    misses rather than returning a stale block.
    """
    key = (tuple(signals), panel_key)
    cached = _SPEARMAN_BLOCK_CACHE.get(key)
    if cached is not None:
        return cached
    R = spearman_abs_matrix(signals)
    if len(_SPEARMAN_BLOCK_CACHE) >= _SPEARMAN_BLOCK_CACHE_MAX:
        _SPEARMAN_BLOCK_CACHE.pop(next(iter(_SPEARMAN_BLOCK_CACHE)))
    _SPEARMAN_BLOCK_CACHE[key] = R
    return R


def clear_spearman_block_cache() -> None:
    """Drop the shared zoo x zoo block cache (tests / forced re-split)."""
    _SPEARMAN_BLOCK_CACHE.clear()


def effective_rank_cached(R_repo: "np.ndarray", repo_signals: dict,
                          extra_signals: dict) -> float:
    """``effective_rank`` of ``repo_signals + extra_signals`` reusing the
    precomputed repo-repo |Spearman| block ``R_repo``.

    The repo-repo block is invariant across a batch's candidates -- the
    repository changes only on a *replace*, which the caller handles by
    invalidating the cache -- so recomputing it per candidate made the
    marginal-er gate ``O(zoo**2)`` per batch (measured ~88s at zoo=14,
    heading to ~700s at zoo=40). Reusing the cached block leaves only the
    extra-involving entries (the kept batch-mates and the candidate), making
    the gate ``O(zoo)`` per candidate. ``R_repo``'s axes must be ordered as
    ``list(repo_signals)``. The result equals
    ``effective_rank(spearman_abs_matrix({**repo_signals, **extra_signals}))``
    -- eigenvalues are invariant to simultaneous row/column permutation, so
    the key ordering does not matter -- so the gate's verdict is unchanged.
    """
    rk = list(repo_signals)
    ek = list(extra_signals)
    n, m = len(rk), len(ek)
    R = np.eye(n + m)
    if n:
        R[:n, :n] = R_repo
    sigs = {**repo_signals, **extra_signals}
    order = rk + ek
    for i in range(n, n + m):
        si = sigs[order[i]]
        for j in range(i):
            v = _abs_spearman(si, sigs[order[j]])
            R[i, j] = R[j, i] = v
    return effective_rank(R)


def effective_rank(R: "np.ndarray") -> float:
    """Effective rank = exp(entropy) of the normalised eigenvalue spectrum.

    The standard participation-ratio form: 1.0 for one dominant direction, n
    for n uncorrelated signals. Matches the inline computation in
    ``operator.evaluate`` (operator.py:235-238) and the audit scripts, so a
    gate that uses it and the reported ``effective_rank`` metric are the same
    number. Eigenvalues <= 1e-12 are dropped to match.
    """
    ev = np.linalg.eigvalsh(R)[::-1]
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return float("nan")
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


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


def prediction_metrics(
    prediction: pd.DataFrame,
    panel: PanelBundle,
    theta: Protocol,
    eval_window: tuple[str, str],
) -> dict:
    """IC statistics of the COMBINED model prediction on the evaluation window.

    This is the quantity Qlib's ``SigAnaRecord`` reports: it computes IC and
    Rank IC of the combined model's output, never of an individual factor. So
    measuring the same thing here keeps the numbers comparable to Qlib.

    Also reports the IS→OOS gap, using the training window as IS.
    """
    label = label_frame(panel, theta)
    wide = _to_wide(prediction).reindex(index=panel.dates, columns=panel.instruments)
    wide = wide.where(panel.universe)

    out = {k: v for k, v in _ic_block(wide, label, eval_window).items()
           if not k.startswith("_")}
    ic_series = _ic_block(wide, label, eval_window)["_ic_series"]
    out.update(_decay_block(ic_series, theta))

    is_block = _ic_block(wide, label, theta.splits.is_window)
    out["rank_ic_is"] = is_block["rank_ic"]
    out["is_oos_gap"] = (
        is_block["rank_ic"] - out["rank_ic"]
        if not (np.isnan(is_block["rank_ic"]) or np.isnan(out["rank_ic"]))
        else np.nan
    )
    return out


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



# --------------------------------------------------------------------------
# Research tear sheet -- signal quality, with no portfolio anywhere
# --------------------------------------------------------------------------
# These answer the researcher's question ("does this signal carry alpha") rather
# than the trader's ("does this improve my book"). Deliberately optimizer-free:
# the whole reason selection moved here is that scoring candidates through a
# constrained long-only book measured a ~10pp/yr size exposure the factor did
# not cause and could not fix.

def quantile_metrics(signal, panel: PanelBundle, theta: Protocol,
                     window: tuple[str, str], n_q: int = 10,
                     horizon: int | None = None) -> dict:
    """Decile response of a signal -- monotonicity, spread, and a long/short leg.

    Built by cross-sectional RANK only: no optimizer, no cost model, no
    benchmark. The long/short leg is dollar-neutral by construction, which is
    what makes the transfer coefficient ~1 here and removes the benchmark, beta
    and long-only distortions from the measurement.

    ``monotonicity`` is the Spearman correlation between decile index and decile
    mean return. It is the number that distinguishes a usable signal from a
    tails-only one: a factor whose Q1 and Q10 differ sharply but whose middle is
    noise scores a wide ``q_spread`` and a monotonicity near zero, and it will
    die under any position cap -- the book holds ~34 of 300 names and cannot
    take a position that only exists in the extreme tail.
    """
    h = int(horizon if horizon is not None else getattr(theta.execution, "label_horizon", 1))
    wide = _to_wide(signal).reindex(index=panel.dates, columns=panel.instruments)
    wide = wide.where(panel.universe)
    label = label_frame_at(panel, theta, h)

    sig = _slice(wide, window)
    lab = label.reindex(index=sig.index, columns=sig.columns)

    # Per-date decile assignment on the signal's cross-sectional rank.
    ranks = sig.rank(axis=1, pct=True)
    per_q: list[pd.Series] = []
    for q in range(n_q):
        lo, hi = q / n_q, (q + 1) / n_q
        mask = (ranks > lo) & (ranks <= hi) if q else (ranks >= 0.0) & (ranks <= hi)
        per_q.append(lab.where(mask).mean(axis=1))

    q_means = [float(s.mean()) if s.notna().any() else float("nan") for s in per_q]
    if not np.isfinite(q_means).all():
        return {"monotonicity": np.nan, "q_spread": np.nan, "q_spread_t": np.nan,
                "ls_sharpe": np.nan, "ls_mdd": np.nan, "q_means": q_means}

    # Spearman of decile index vs decile mean return.
    idx = np.arange(n_q, dtype=float)
    mono = float(pd.Series(q_means).corr(pd.Series(idx), method="spearman"))

    # Dollar-neutral long/short: top decile minus bottom decile, daily.
    ls = (per_q[-1] - per_q[0]).dropna()
    spread = float(ls.mean())
    sd = float(ls.std())
    n = len(ls)
    # Overlapping labels share h-1 of their h days, so the effective sample is
    # n/h, not n -- the same correction the standalone admission path applies.
    n_eff = max(n / float(max(h, 1)), 2.0)
    t = spread / (sd / n_eff ** 0.5) if sd > 0 else float("nan")
    ppy = float(theta.periods_per_year)
    sharpe = float(spread / sd * (ppy / max(h, 1)) ** 0.5) if sd > 0 else float("nan")

    return {
        "monotonicity": mono,
        "q_spread": spread,
        "q_spread_t": float(t),
        "ls_sharpe": sharpe,
        "ls_mdd": max_drawdown(ls),
        "q_means": q_means,
        "q_n_obs": n,
    }


def ic_decay_curve(signal, panel: PanelBundle, theta: Protocol,
                   window: tuple[str, str],
                   horizons: Sequence[int] = (1, 2, 3, 5, 10, 20)) -> dict:
    """RankIC at each forecast horizon, and the horizon that maximises it.

    This is what should SET the holding period. The protocol currently fixes
    ``label_horizon: 1`` by assertion, never having measured whether the signal
    is a one-day or a ten-day effect -- and a factor predictive at 20 days is a
    different alpha from one predictive at 1 day, not a worse version of it.

    The reported ``t`` carries the overlapping-returns correction (``n_eff =
    n/h``). Without it long horizons win mechanically: consecutive observations
    of a horizon-h label share h-1 of their h days, so the naive t is inflated
    by roughly sqrt(h).
    """
    wide = _to_wide(signal).reindex(index=panel.dates, columns=panel.instruments)
    wide = wide.where(panel.universe)
    sig = _slice(wide, window)

    curve: dict[int, dict] = {}
    for h in horizons:
        lab = label_frame_at(panel, theta, int(h))
        ric = _cross_sectional_corr(sig, lab, "spearman").dropna()
        if len(ric) < 30 or not ric.std():
            curve[int(h)] = {"rank_ic": np.nan, "t": np.nan, "n": len(ric)}
            continue
        n_eff = max(len(ric) / float(h), 2.0)
        curve[int(h)] = {
            "rank_ic": float(ric.mean()),
            "t": float(ric.mean() / (ric.std() / n_eff ** 0.5)),
            "n": int(len(ric)),
        }

    usable = {h: v for h, v in curve.items() if v["rank_ic"] == v["rank_ic"]}
    best = max(usable, key=lambda h: abs(usable[h]["rank_ic"])) if usable else None
    return {
        "curve": curve,
        "best_horizon": best,
        "best_rank_ic": usable[best]["rank_ic"] if best else np.nan,
        "best_t": usable[best]["t"] if best else np.nan,
    }


def newey_west_t(series: pd.Series, lag: int | None = None) -> float:
    """t-statistic of a mean under HAC (Newey-West) standard errors.

    An IC series is autocorrelated -- from overlapping labels, and from the
    signal itself being persistent -- so the ordinary t overstates significance,
    often by 2-3x. ``lag`` defaults to the Newey-West rule of thumb
    ``floor(4 * (n/100)^(2/9))``.
    """
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 8:
        return float("nan")
    mu = x.mean()
    e = x - mu
    L = int(lag) if lag is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    L = max(0, min(L, n - 2))

    gamma0 = float(e @ e) / n
    var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)          # Bartlett kernel
        gk = float(e[k:] @ e[:-k]) / n
        var += 2.0 * w * gk
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))

__all__ = [
    "complexity_diagnostics",
    "cx",
    "effective_rank",
    "label_frame",
    "max_drawdown",
    "per_factor_metrics",
    "quantile_metrics",
    "ic_decay_curve",
    "newey_west_t",
    "rho_max",
    "rho_max_arg",
    "solo_turnover",
    "spearman_abs_matrix",
    "spearman_block_cached",
    "clear_spearman_block_cache",
    "strategy_metrics",
]
