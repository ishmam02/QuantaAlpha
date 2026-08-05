"""A k-factor statistical risk model, and the optimiser that uses it.

The first mean-variance implementation priced risk with a diagonal ``Sigma``.
That is separable and fast, and it cannot see that two names are the same bet:
with 300 correlated A-shares it will hold a dozen names that all express one
factor and report the risk of twelve independent ones. Its own docstring said
so. This supplies the structure that fixes it.

``Sigma = B diag(lambda) B' + diag(d)`` -- the form every commercial risk model
has (Barra, Axioma), estimated here by PCA on trailing returns rather than from
vendor fundamentals, because the panel carries prices and nothing else. Low rank
plus diagonal is what keeps it affordable: ``Sigma @ w`` costs ``O(nk)``, so a
2427-date backtest stays in seconds rather than minutes.

**Point-in-time.** Every estimate uses a window ending strictly before the date
it is used on, and the model is refit on a fixed calendar cadence rather than
whenever it would flatter the result. A risk model fitted on returns it is about
to trade through is a look-ahead, and it is the kind that flatters a backtest
without ever looking wrong.

**Idiosyncratic shrinkage is not optional.** 252 days of 300 names leaves the
smallest ``d_i`` badly under-estimated, and a mean-variance optimiser
preferentially loads exactly the names that look riskless. Shrinking toward the
cross-sectional mean is the cheap correction; without it the optimiser hunts
estimation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRisk:
    """``Sigma = B diag(f_var) B' + diag(d)`` on a fixed instrument order."""

    instruments: pd.Index
    B: np.ndarray        # (n, k) loadings
    f_var: np.ndarray    # (k,)   factor variances
    d: np.ndarray        # (n,)   idiosyncratic variances

    def quad(self, w: np.ndarray) -> float:
        """wᵀ Σ w, without ever forming Σ."""
        bw = self.B.T @ w
        return float(w @ (self.d * w) + bw @ (self.f_var * bw))

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Σ w, at O(nk)."""
        return self.d * w + self.B @ (self.f_var * (self.B.T @ w))

    def curvature(self) -> float:
        """An upper bound on the largest eigenvalue of Σ, for the step size."""
        return float(self.d.max() + (self.f_var * (self.B ** 2).sum(axis=0)).sum())


def fit_factor_risk(
    returns: pd.DataFrame,
    n_factors: int,
    idio_shrink: float,
) -> FactorRisk | None:
    """PCA a trailing return window into ``k`` factors plus idiosyncratic risk.

    ``returns`` must already be restricted to the estimation window and to
    instruments tradeable on the date this will be used for. Returns ``None``
    when the window is too thin to say anything, so the caller can fall back to
    the diagonal model rather than trust a rank-deficient fit.
    """
    R = returns.dropna(axis=1, how="all")
    if R.shape[0] < 20 or R.shape[1] < 2:
        return None
    R = R.fillna(0.0)
    X = R.to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)

    n_obs, n_inst = X.shape
    k = int(max(1, min(n_factors, n_inst - 1, n_obs - 1)))

    # Economise: the Gram matrix is (n_obs x n_obs) and n_obs << n_inst here,
    # so eigendecomposing it is far cheaper than the (n_inst x n_inst) sample
    # covariance, and gives the same leading factors.
    G = (X @ X.T) / max(n_obs - 1, 1)
    vals, vecs = np.linalg.eigh(G)
    idx = np.argsort(vals)[::-1][:k]
    vals, vecs = vals[idx], vecs[:, idx]
    vals = np.clip(vals, 1e-16, None)

    # Loadings from the time-series eigenvectors, normalised so that
    # B diag(f_var) B' reproduces the leading part of the sample covariance.
    B = (X.T @ vecs) / np.sqrt(vals * max(n_obs - 1, 1))
    f_var = vals.copy()

    total_var = X.var(axis=0, ddof=1)
    explained = (B ** 2) @ f_var
    d = np.clip(total_var - explained, 1e-10, None)

    if idio_shrink > 0:
        d = (1.0 - idio_shrink) * d + idio_shrink * float(d.mean())

    return FactorRisk(instruments=R.columns, B=B, f_var=f_var, d=d)


def project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto ``{w : Σw = 1, 0 ≤ w ≤ cap}``.

    ``w(t) = clip(v − t, 0, cap)`` has a sum that decreases monotonically in
    ``t``, so bisection finds the ``t`` that spends exactly the budget. If
    ``cap · n < 1`` the set is empty and the box is returned saturated -- the
    caller's leverage assertion is the right place for that to surface, not a
    silent renormalisation here.
    """
    n = v.size
    if n == 0:
        return v
    if cap * n <= 1.0:
        return np.full(n, cap)
    lo, hi = float(v.min() - 1.0), float(v.max())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, 0.0, cap).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi), 0.0, cap)


def mv_weights_factor(
    mu: pd.Series,
    risk: FactorRisk,
    lam: float,
    max_weight: float,
    w_start: pd.Series | None = None,
    iters: int = 120,
) -> pd.Series:
    """argmax μᵀw − (λ/2) wᵀΣw  s.t.  Σw = 1, 0 ≤ w ≤ max_weight.

    Projected gradient ascent. The objective is concave and the feasible set is
    convex and compact, so this converges to the global optimum; the step size
    is ``1/L`` with ``L`` an upper bound on the curvature of Σ, which is the
    standard guarantee for projected gradient on a smooth concave objective.

    A full QP would also work and is what a production optimiser uses; this
    stays in numpy because it has to run once per date for 2427 dates and three
    seeds, and the low-rank Σ makes each iteration O(nk).
    """
    order = mu.index
    B = pd.DataFrame(risk.B, index=risk.instruments)
    B = B.reindex(order).fillna(0.0).to_numpy()
    d = pd.Series(risk.d, index=risk.instruments).reindex(order)
    d = d.fillna(d.median() if d.notna().any() else 1e-4).to_numpy()
    local = FactorRisk(order, B, risk.f_var, d)

    m = mu.to_numpy(dtype=float)
    w = (w_start.reindex(order).fillna(0.0).to_numpy()
         if w_start is not None else np.full(order.size, 1.0 / max(order.size, 1)))
    w = project_capped_simplex(w, max_weight)

    L = max(lam * local.curvature(), 1e-12)
    step = 1.0 / L
    for _ in range(iters):
        g = m - lam * local.grad(w)
        w_next = project_capped_simplex(w + step * g, max_weight)
        if np.abs(w_next - w).sum() < 1e-9:
            w = w_next
            break
        w = w_next
    return pd.Series(w, index=order)


class RollingFactorRisk:
    """Refits ``FactorRisk`` on a calendar cadence, point-in-time.

    Holds one model at a time. ``for_date`` returns the most recent model whose
    estimation window closed strictly before that date, refitting only when the
    cadence says to, so a full backtest pays for ~1/refresh of the fits.
    """

    def __init__(self, close: pd.DataFrame, window: int, refresh: int,
                 n_factors: int, idio_shrink: float) -> None:
        self._rets = close.pct_change(fill_method=None)
        self._dates = close.index
        self._window = int(window)
        self._refresh = max(int(refresh), 1)
        self._k = int(n_factors)
        self._shrink = float(idio_shrink)
        self._cache: FactorRisk | None = None
        self._fitted_at: int | None = None

    def for_date(self, date, columns: pd.Index) -> FactorRisk | None:
        try:
            pos = self._dates.get_loc(date)
        except KeyError:
            return None
        if not isinstance(pos, int):
            return None
        if self._fitted_at is not None and pos - self._fitted_at < self._refresh:
            return self._cache
        lo = max(0, pos - self._window)
        if pos - lo < 20:
            return self._cache
        # `pos` excluded: the window must close before the date it prices.
        win = self._rets.iloc[lo:pos]
        win = win.loc[:, [c for c in columns if c in win.columns]]
        fitted = fit_factor_risk(win, self._k, self._shrink)
        if fitted is not None:
            self._cache, self._fitted_at = fitted, pos
        return self._cache


__all__ = ["FactorRisk", "RollingFactorRisk", "fit_factor_risk",
           "mv_weights_factor", "project_capped_simplex"]
