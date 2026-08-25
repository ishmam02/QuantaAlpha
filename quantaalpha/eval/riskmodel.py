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

    The bisection stops once the bracket is narrower than float64 can resolve
    instead of always running a fixed 100 steps. This is the hottest function
    in the whole evaluation -- ~200 calls per date from ``mv_weights_costed``,
    ~49k per book, five million ``np.clip`` calls per evaluation -- and the
    bracket halves every step, so it is below the ulp of the values involved
    after ~50 steps and the remaining ~50 move nothing. Measured over 300
    random cases: never more than 50 steps to converge, agreeing with the
    fixed-100 result to 6.4e-15 (machine precision), 2.1x faster.
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
        # Bracket resolved to within float64 precision: further halving is a
        # no-op on the returned value.
        if hi - lo <= 1e-14 * max(1.0, abs(hi)):
            break
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


def mv_weights_costed(
    mu: pd.Series,
    risk: FactorRisk | None,
    var: pd.Series | None,
    gamma: pd.Series,
    w_prev: pd.Series,
    lam: float,
    max_weight: float,
    adv_w: pd.Series | None = None,
    kappa2: float = 0.0,
    impact_exponent: float = 1.5,
    hurdle: float = 1.0,
    trade_penalty: float = 0.0,   # kappa: dimensionless, see the docstring
    iters: int = 200,
) -> pd.Series:
    """argmax μᵀw − (λ/2)wᵀΣw − Σᵢ cᵢ(|wᵢ − w_prev,ᵢ|)  s.t. Σw = 1, 0 ≤ w ≤ cap.

    Turnover as a *price* rather than a quota. The hard budget in
    :func:`~quantaalpha.eval.portfolio.mean_variance` scales every trade down by
    the same factor once the book moves too far, so the most valuable trade is
    cut as hard as the least valuable one -- it spends the budget without
    choosing what to spend it on. Here each name pays its own
    ``γᵢ = κ₀ + κ₁σᵢ`` to move, and a position changes only where the expected
    return justifies what moving it costs. Turnover stops being a parameter and
    becomes an outcome, which is the only regime in which a net-of-cost
    objective can express an advantage over a gross one.

    ``cᵢ`` is the whole of Eq. 7's per-name cost, not just its linear part::

        cᵢ(x) = γᵢ x + κ₂ x^e / advᵢ^(e−1),      x = |wᵢ − w_prev,ᵢ|

    with ``γᵢ = κ₀ + κ₁σᵢ``. Charging only ``γ`` was tried first and it
    over-traded badly: turnover came out at 0.1247 against 0.0194 under the hard
    cap, realised cost 4.50 bps/day, net ARR −2.30% against −0.82%. An optimiser
    given a lower bound on its costs will spend right up to that bound, so a
    "conservative" omission is not conservative at all -- it is an instruction
    to trade more than the evaluator will charge for.

    Including impact costs nothing structurally: it depends only on this name's
    own trade, so the objective stays separable, and ``x^1.5`` is convex, so it
    stays concave. The optimiser now prices exactly what ``costs.cost`` charges,
    which is the property that makes turnover an honest decision rather than an
    artefact of the gap between two cost models.

    ``trade_penalty`` is ``κ``, an inertia term ``(κ/2)·λ·Δwᵀ Σ Δw`` charged on
    top of the cash cost. It is what the hard turnover cap was supplying by
    accident: ``μ`` is a point estimate whose day-to-day movement is largely
    estimation error, and an optimiser that treats it as known chases that error
    -- Markowitz as an error-maximiser. A cash cost does not restrain that,
    because it is roughly constant per unit traded while the apparent edge keeps
    moving.

    **κ is dimensionless, and that is the point.** An absolute coefficient
    ``ν‖Δw‖²`` was tried first and did not survive the window it was tuned on:
    ``ν = 0.5`` gave turnover 0.0596 on the 2021 validation split and 0.1092 on
    2022-2025 with everything else identical, because ``μ``'s dispersion differs
    between windows and a fixed ``ν`` restrains a larger ``μ`` proportionally
    less. Measuring the inertia in the same units as the risk term removes the
    dependence: solving the unconstrained first-order condition gives

        w = 1/(1+κ) · w_markowitz + κ/(1+κ) · w_prev

    so ``κ`` sets the *fraction* of the way the book moves toward its target and
    that fraction is free of the scale of both ``μ`` and ``Σ``. This is the
    Gârleanu-Pedersen result -- with quadratic costs the optimal policy trades
    partway to the Markowitz portfolio rather than jumping to it.

    That identity is exact only *unconstrained*; verified to 1e-10 across a 100x
    range in ``μ`` and 5x in ``λ``, but the box and simplex constraints here mean
    it holds approximately rather than exactly. What matters is whether it
    transfers, and measured on the real backtest it does. Turnover under the
    same setting, validation (2021) against test (2022-2025):

        kappa    0     1     3     9    30       absolute nu = 0.5
        valid  .136  .131  .065  .045  .040            .060
        test   .145  .135  .081  .039  .030            .109
        ratio  1.06  1.03  1.25  0.89  0.74            1.83

    The absolute coefficient nearly doubled its turnover out of sample; ``κ``
    lands within a quarter either way. The scale dependence is gone.

    It is not, however, enough to make this construction preferable: at every
    ``κ`` the hard cap still returns more on test (-0.82% against -6.45% at the
    best ``κ``), and the gap is far larger than the turnover difference between
    them explains. Whatever remains is about *which* trades are chosen rather
    than how many, and the non-monotone ``κ=1`` row on validation suggests the
    subgradient solve is a suspect. ``cost_in_objective`` stays off by default.

    Solved by projected subgradient: the smooth part contributes
    ``μ − λΣw − κλΣΔw`` and the cash cost contributes ``−c'(|Δw|)·sign(Δw)``,
    and each iterate is projected back onto the capped simplex. The objective is
    concave and the feasible set convex and compact, so this converges; the step
    decays as ``1/√t`` because the subgradient of ``|·|`` does not vanish at the
    optimum and a fixed step would chatter around it. The step is scaled by
    ``(1+κ)`` so that raising the inertia does not silently slow convergence as
    well as the book.
    """
    order = mu.index
    m = mu.to_numpy(dtype=float)
    g = gamma.reindex(order).fillna(0.0).to_numpy(dtype=float)
    prev = w_prev.reindex(order).fillna(0.0).to_numpy(dtype=float)
    e = float(impact_exponent)
    if kappa2 > 0 and adv_w is not None:
        a = adv_w.reindex(order)
        a = a.fillna(a.median() if a.notna().any() else 1.0).clip(lower=1e-9).to_numpy()
        k2_scale = float(kappa2) / np.power(a, e - 1.0)
    else:
        k2_scale = np.zeros(order.size)

    h = float(hurdle)
    kappa = float(trade_penalty)

    def trade_cost(x):
        return h * (g * x + k2_scale * np.power(x, e))

    def trade_cost_deriv(x):
        return h * (g + e * k2_scale * np.power(np.maximum(x, 1e-12), e - 1.0))

    if risk is not None:
        B = pd.DataFrame(risk.B, index=risk.instruments).reindex(order).fillna(0.0).to_numpy()
        d = pd.Series(risk.d, index=risk.instruments).reindex(order)
        d = d.fillna(d.median() if d.notna().any() else 1e-4).to_numpy()
        local = FactorRisk(order, B, risk.f_var, d)
        sigma_w = local.grad
        curvature = local.curvature()
    else:
        v = (var.reindex(order).fillna(var.median() if var.notna().any() else 4e-4)
             .clip(lower=1e-8).to_numpy())
        sigma_w = lambda w: v * w  # noqa: E731 -- diagonal Sigma
        curvature = float(v.max())

    w = project_capped_simplex(prev.copy(), max_weight)
    L = max(lam * curvature, 1e-12)
    base = 1.0 / L
    best, best_obj = w.copy(), -np.inf
    for t in range(1, iters + 1):
        dw = w - prev
        # The inertia term is kappa*lam*Sigma*dw, NOT nu*dw: measured in the
        # same units as the risk term it sits beside, so its strength is
        # relative to the risk model rather than absolute.
        grad = (m - lam * sigma_w(w)
                - trade_cost_deriv(np.abs(dw)) * np.sign(dw)
                - kappa * lam * sigma_w(dw))
        w_next = project_capped_simplex(w + (base * (1.0 + kappa) / np.sqrt(t)) * grad,
                                        max_weight)
        # Early termination: the iterate warm-starts from w_prev and the
        # 1/sqrt(t) step decays, so under kappa-inertia it settles well before
        # the 200-iter cap. Break on L1 stall (the same test mv_weights_factor
        # uses) and keep the best-seen w -- the subgradient is non-monotone, so
        # the converged point is not necessarily the best objective. Cutting the
        # ~200x242-date kernel to its effective length is the dominant per-batch
        # saving. Run at least one full update before the test can fire.
        if t > 1 and np.abs(w_next - w).sum() < 1e-9:
            w = w_next
            break
        w = w_next
        dw = w - prev
        obj = float(m @ w - 0.5 * lam * (w @ sigma_w(w))
                    - trade_cost(np.abs(dw)).sum()
                    - 0.5 * kappa * lam * (dw @ sigma_w(dw)))
        if obj > best_obj:
            best_obj, best = obj, w.copy()
    # Subgradient iterates are not monotone, so return the best seen rather
    # than the last -- otherwise the answer depends on where the oscillation
    # happened to stop.
    return pd.Series(best, index=order)


def mv_weights_exact(
    mu: pd.Series,
    var: pd.Series,
    gamma: pd.Series,
    w_prev: pd.Series,
    lam: float,
    max_weight: float,
    k2_scale=None,
    impact_exponent: float = 1.5,
    hurdle: float = 1.0,
    trade_penalty: float = 0.0,
) -> pd.Series:
    """Exact solve of the costed mean-variance problem when Sigma is DIAGONAL.

        max  mu'w - (lam/2) w'Vw - sum_i c_i(|w_i - p_i|)
                  - (kappa/2) lam sum_i v_i (w_i - p_i)^2
        s.t. sum w = 1,  0 <= w_i <= cap

    with ``c_i(x) = hurdle * (gamma_i x + k2_i x^e)``.

    A diagonal ``V`` makes the objective SEPARABLE once the budget multiplier
    ``eta`` is fixed, so each name is an independent 1-D concave maximisation
    and the only coupling left is the scalar ``eta`` that spends the budget.
    ``sum_i w_i(eta)`` is non-increasing in ``eta``, so an outer bisection
    lands the budget exactly.

    This replaces the projected subgradient on this path. The subgradient does
    not converge here: measured against SLSQP on this exact shape it left
    4.4e-4 of objective unclaimed and placed ~15% of the book wrongly (mean L1
    0.148 on weights summing to 1). Separability makes that unnecessary.

    Per name, the objective has a kink at ``w_i = p_i`` because the cash cost is
    proportional to ``|w_i - p_i|``. The superdifferential there is
    ``[d_minus, d_plus]`` with::

        d_plus  = (mu_i - eta) - lam v_i p_i - hurdle*gamma_i
        d_minus = (mu_i - eta) - lam v_i p_i + hurdle*gamma_i

    so ``d_plus <= 0 <= d_minus`` means "do not trade this name" -- the edge
    does not cover the spread. That no-trade band is exactly what a cash cost
    should produce and is what the subgradient blurred.

    Off the kink the stationarity condition is solved in closed form when there
    is no impact term, and by bisection on the (monotone) derivative when there
    is -- ``x^e`` with ``e > 1`` is convex, so the derivative is strictly
    decreasing and bisection is exact to machine precision.
    """
    order = mu.index
    m = mu.to_numpy(dtype=float)
    v = var.reindex(order).fillna(var.median() if var.notna().any() else 4e-4)
    v = v.clip(lower=1e-8).to_numpy(dtype=float)
    g = hurdle * gamma.reindex(order).fillna(0.0).to_numpy(dtype=float)
    p = np.clip(w_prev.reindex(order).fillna(0.0).to_numpy(dtype=float), 0.0, max_weight)
    kap = float(trade_penalty)
    e = float(impact_exponent)
    k2 = (np.zeros(order.size) if k2_scale is None
          else hurdle * np.asarray(k2_scale, dtype=float))
    has_impact = bool(np.any(k2 > 0))

    # Curvature of the smooth part: lam*v from the risk term, kappa*lam*v from
    # the inertia. Both are per-name because V is diagonal.
    a = lam * v * (1.0 + kap)

    def deriv(w, eta):
        """d/dw of the per-name objective, off the kink."""
        d = (m - eta) - lam * v * w - kap * lam * v * (w - p)
        dx = w - p
        s_ = np.sign(dx)
        ax = np.abs(dx)
        d = d - s_ * g
        if has_impact:
            d = d - s_ * e * k2 * np.power(np.maximum(ax, 1e-16), e - 1.0)
        return d

    def solve(eta):
        base = (m - eta) - lam * v * p          # derivative at the kink, cost aside
        up = base > g                            # worth buying
        dn = base < -g                           # worth selling
        w = p.copy()                             # otherwise: no trade

        if not has_impact:
            # Closed form on each branch.
            w_up = ((m - eta) - g + kap * lam * v * p) / a
            w_dn = ((m - eta) + g + kap * lam * v * p) / a
            w = np.where(up, w_up, np.where(dn, w_dn, p))
        else:
            # Monotone derivative -> bisect each active branch on its own side.
            lo = np.where(up, p, 0.0)
            hi = np.where(up, max_weight, p)
            act = up | dn
            if act.any():
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    d = deriv(mid, eta)
                    # derivative decreasing in w: d>0 -> move right
                    lo = np.where(act & (d > 0), mid, lo)
                    hi = np.where(act & (d <= 0), mid, hi)
                w = np.where(act, 0.5 * (lo + hi), p)

        return np.clip(w, 0.0, max_weight)

    n = order.size
    if n == 0:
        return pd.Series(dtype=float)
    if max_weight * n <= 1.0:
        # The box cannot fund the budget; saturating it is the only feasible
        # point. The caller's leverage assertion is where that should surface.
        return pd.Series(np.full(n, max_weight), index=order)

    # Outer bisection on the budget multiplier. Bracket wide enough that the
    # low end over-spends and the high end under-spends.
    lo_e = float(np.min(m) - lam * np.max(v) * max_weight - np.max(g) - 1.0)
    hi_e = float(np.max(m) + np.max(g) + 1.0)
    if not (np.isfinite(lo_e) and np.isfinite(hi_e)):
        return pd.Series(np.full(n, 1.0 / n), index=order)
    if solve(lo_e).sum() < 1.0:
        return pd.Series(solve(lo_e), index=order)   # cannot reach full investment
    for _ in range(100):
        mid_e = 0.5 * (lo_e + hi_e)
        if solve(mid_e).sum() > 1.0:
            lo_e = mid_e
        else:
            hi_e = mid_e
        if hi_e - lo_e <= 1e-14 * max(1.0, abs(hi_e)):
            break
    w = solve(0.5 * (lo_e + hi_e))
    total = float(w.sum())
    if total > 0:
        w = w / total                                # exact budget
        w = np.minimum(w, max_weight)
        # Renormalising can breach the cap; push the residual onto uncapped names.
        for _ in range(50):
            resid = 1.0 - float(w.sum())
            if abs(resid) < 1e-12:
                break
            free = w < max_weight - 1e-15
            if not free.any():
                break
            w[free] = np.clip(w[free] + resid * (w[free] / w[free].sum()
                                                 if w[free].sum() > 0
                                                 else 1.0 / free.sum()),
                              0.0, max_weight)
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
           "mv_weights_costed", "mv_weights_exact", "mv_weights_factor",
           "project_capped_simplex"]
