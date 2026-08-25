"""Statistical defenses for a claim made after searching many strategies.

This search has mined 150+ factors across many configurations. At that trial
count an ordinary t-statistic carries almost no information: the best of N
random strategies has a high Sharpe by construction, and reporting it without
adjustment is the single most common reason a backtest fails review.

Three tools, each answering a question a reader will actually ask:

* ``deflated_sharpe_ratio`` -- "is this better than the best you would expect
  from N coin flips?" (Bailey & Lopez de Prado). Adjusts for the number of
  trials AND for non-normal returns, since skew and fat tails inflate a naive
  Sharpe.
* ``probability_of_backtest_overfitting`` -- "would the configuration you chose
  in-sample have been below median out-of-sample?" (CSCV). Needs several
  backtest paths, which combinatorial purged CV provides.
* ``alpha_vs_beta`` -- "is this alpha, or a known risk factor in disguise?"
  If the intercept is indistinguishable from zero once the factor loadings are
  fitted, there is no alpha. On this project that is the decisive test, because
  an uncontrolled size exposure worth ~10pp/yr was measured in the book.

None of these makes a result better. They state how much of it survives being
asked the obvious questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over the range that matters here, and avoids a scipy
    dependency in a module that may run inside a worker.
    """
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) * (p - 0.5)
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass
class DSRResult:
    sharpe: float           # observed, per-observation
    threshold: float        # expected max Sharpe under the null, given n_trials
    dsr: float              # probability the true Sharpe exceeds zero
    n_trials: int
    n_obs: int

    def clears(self, level: float = 0.95) -> bool:
        return self.dsr == self.dsr and self.dsr >= level


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """The Sharpe you should EXPECT from the best of ``n_trials`` null strategies.

    This is the number a raw Sharpe has to beat before it means anything. It
    grows with the number of things tried and with how dispersed their Sharpes
    are, which is why "we tested 150 factors" makes a given Sharpe less
    impressive, not more.
    """
    if n_trials < 2 or not (trial_sharpe_std > 0):
        return 0.0
    n = float(n_trials)
    return float(trial_sharpe_std * ((1 - _EULER) * _norm_ppf(1 - 1 / n)
                                     + _EULER * _norm_ppf(1 - 1 / (n * math.e))))


def deflated_sharpe_ratio(returns, n_trials: int,
                          trial_sharpe_std: float | None = None) -> DSRResult:
    """Bailey & Lopez de Prado's deflated Sharpe ratio.

    ``returns`` is the strategy's per-period excess return series. ``n_trials``
    is how many strategies/configurations were searched to find it -- be honest
    here; understating it is the whole failure mode this corrects.

    Returns a probability. 0.95 is the conventional bar.
    """
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8 or not (r.std(ddof=1) > 0):
        return DSRResult(float("nan"), float("nan"), float("nan"), n_trials, n)

    sr = float(r.mean() / r.std(ddof=1))
    z = (r - r.mean()) / r.std(ddof=1)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())          # non-excess

    if trial_sharpe_std is None:
        # Without the dispersion of the trials themselves, the conventional
        # fallback is the standard error of a Sharpe under the null.
        trial_sharpe_std = 1.0 / math.sqrt(n)
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)

    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return DSRResult(sr, sr0, float("nan"), n_trials, n)
    stat = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return DSRResult(sr, sr0, float(_norm_cdf(stat)), n_trials, n)


def probability_of_backtest_overfitting(returns_matrix, n_splits: int = 8) -> float:
    """PBO by combinatorially symmetric cross-validation.

    ``returns_matrix`` is (observations x strategies) -- one column per
    configuration tried. The matrix is cut into ``n_splits`` blocks; every
    balanced split gives an in-sample half and an out-of-sample half. Pick the
    best strategy in-sample, then see where it ranks out-of-sample. PBO is how
    often that rank lands below the median.

    A PBO near 0.5 means the in-sample winner is a coin flip out of sample --
    the selection procedure itself is overfitting, whatever any single backtest
    shows.
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    n_splits = max(2, n_splits - (n_splits % 2))
    blocks = np.array_split(np.arange(M.shape[0]), n_splits)

    worse = 0
    total = 0
    for combo in combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_m, oos_m = M[is_idx], M[oos_idx]

        with np.errstate(invalid="ignore", divide="ignore"):
            is_sr = is_m.mean(axis=0) / is_m.std(axis=0, ddof=1)
            oos_sr = oos_m.mean(axis=0) / oos_m.std(axis=0, ddof=1)
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue
        best = int(np.nanargmax(is_sr))
        # Rank of the in-sample winner among all strategies, out of sample.
        rank = float((oos_sr < oos_sr[best]).sum()) / (len(oos_sr) - 1)
        total += 1
        if rank < 0.5:
            worse += 1
    return float(worse / total) if total else float("nan")


@dataclass
class AlphaBetaResult:
    alpha: float            # intercept, per period
    alpha_t: float
    betas: dict[str, float]
    r_squared: float
    n_obs: int

    def has_alpha(self, t_bar: float = 2.0) -> bool:
        return self.alpha_t == self.alpha_t and abs(self.alpha_t) >= t_bar


def alpha_vs_beta(strategy_returns, factor_returns: dict) -> AlphaBetaResult:
    """Regress strategy returns on known factor returns; report the intercept.

    If the intercept is indistinguishable from zero once the loadings are
    fitted, the strategy is factor beta wearing alpha's name. This is the test
    that decides whether a result on this project is the alpha or the size
    exposure -- and the 2022-2025 window carries roughly a +1.7%/yr tailwind for
    the current construction, so the question WILL be asked.

    The t-statistic on the intercept uses ordinary standard errors; for a
    strongly autocorrelated residual, deflate it with ``metrics.newey_west_t``.
    """
    y = np.asarray(list(strategy_returns), dtype=float)
    names = list(factor_returns)
    cols = [np.asarray(list(factor_returns[k]), dtype=float) for k in names]
    n = min([y.size] + [c.size for c in cols]) if cols else y.size
    y = y[:n]
    X = np.column_stack([np.ones(n)] + [c[:n] for c in cols])

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if y.size <= X.shape[1] + 1:
        return AlphaBetaResult(float("nan"), float("nan"), {}, float("nan"), int(y.size))

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = y.size - X.shape[1]
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se0 = float(math.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        se0 = float("nan")

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return AlphaBetaResult(
        alpha=float(coef[0]),
        alpha_t=float(coef[0] / se0) if se0 and se0 == se0 else float("nan"),
        betas={k: float(v) for k, v in zip(names, coef[1:])},
        r_squared=r2,
        n_obs=int(y.size),
    )


__all__ = ["deflated_sharpe_ratio", "expected_max_sharpe",
           "probability_of_backtest_overfitting", "alpha_vs_beta",
           "DSRResult", "AlphaBetaResult"]
