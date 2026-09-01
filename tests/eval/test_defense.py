"""A claim made after searching 150 strategies must survive the obvious questions.

S1  the best of N pure-noise strategies must NOT clear the deflated Sharpe
S2  a genuinely strong strategy DOES clear it
S3  the bar rises with the number of trials -- searching more makes a given
    Sharpe less impressive, not more
S4  PBO on pure noise is near 0.5 (the in-sample winner is a coin flip)
S5  a pure factor bet shows NO alpha once the factor is regressed out
S6  real alpha on top of a factor bet survives the same regression
"""
import numpy as np
from quantaalpha.eval.defense import (
    deflated_sharpe_ratio, expected_max_sharpe,
    probability_of_backtest_overfitting, alpha_vs_beta,
)

rng = np.random.default_rng(7)
T, N = 750, 150

# --- S1: best of 150 nulls must not clear ------------------------------------
null = rng.normal(0, 0.01, (T, N))
sr = null.mean(axis=0) / null.std(axis=0, ddof=1)
best = int(np.argmax(sr))
res = deflated_sharpe_ratio(null[:, best], n_trials=N,
                            trial_sharpe_std=float(sr.std(ddof=1)))
assert not res.clears(0.95), (
    f"S1: the luckiest of {N} pure-noise strategies cleared DSR "
    f"(sharpe {res.sharpe:.4f} vs threshold {res.threshold:.4f}, dsr {res.dsr:.3f})")
print(f"S1 PASS  best of {N} nulls: sharpe {res.sharpe:+.4f} vs null-max threshold "
      f"{res.threshold:+.4f} -> DSR {res.dsr:.3f}, does not clear")

# Naive framing would have called it significant:
naive_t = res.sharpe * np.sqrt(T)
print(f"         (its naive t-stat was {naive_t:+.2f} -- 'significant' by the usual bar)")

# --- S2: a real edge clears ---------------------------------------------------
# NOTE: 0.0008/0.01 (~1.27 annualised) does NOT clear, and should not --
# the best-of-150 threshold is ~0.0975 per period, i.e. ~1.55 annualised. That
# is the honest arithmetic of searching 150 strategies, not a defect. A genuine
# edge has to beat the luckiest null, so the fixture must actually be strong.
strong = rng.normal(0.0020, 0.01, T)          # ~3.17 annualised Sharpe
res2 = deflated_sharpe_ratio(strong, n_trials=N, trial_sharpe_std=1/np.sqrt(T))
assert res2.clears(0.95), f"S2: a genuine edge failed DSR (dsr {res2.dsr:.3f})"
print(f"S2 PASS  a genuine edge clears: DSR {res2.dsr:.3f}")

# --- S3: more trials -> higher bar --------------------------------------------
bars = [expected_max_sharpe(n, 1/np.sqrt(T)) for n in (1, 10, 150, 5000)]
assert bars == sorted(bars) and bars[-1] > bars[1], f"S3: bar must rise with trials: {bars}"
print("S3 PASS  threshold rises with trials: " +
      ", ".join(f"N={n}:{b:.4f}" for n, b in zip((1, 10, 150, 5000), bars)))

# --- S4: PBO on noise ~ 0.5 ---------------------------------------------------
pbo = probability_of_backtest_overfitting(rng.normal(0, 0.01, (T, 20)), n_splits=8)
assert 0.25 <= pbo <= 0.75, f"S4: PBO on pure noise should be near 0.5, got {pbo:.3f}"
print(f"S4 PASS  PBO on pure noise = {pbo:.3f} (in-sample winner is a coin flip)")

# --- S5: a pure factor bet has NO alpha ---------------------------------------
size_factor = rng.normal(0.0004, 0.012, T)
pure_bet = 1.3 * size_factor + rng.normal(0, 0.0005, T)     # levered factor, no skill
ab = alpha_vs_beta(pure_bet, {"size": size_factor})
assert not ab.has_alpha(2.0), (
    f"S5: a pure factor bet must show no alpha; got alpha {ab.alpha:+.6f} "
    f"t={ab.alpha_t:+.2f}")
assert abs(ab.betas["size"] - 1.3) < 0.05, f"S5: beta not recovered: {ab.betas}"
print(f"S5 PASS  pure size bet: beta {ab.betas['size']:.2f}, alpha t={ab.alpha_t:+.2f} "
      f"(no alpha), R^2 {ab.r_squared:.3f}")

# --- S6: real alpha on top of the same bet survives ---------------------------
with_alpha = pure_bet + 0.0009                              # constant skill
ab2 = alpha_vs_beta(with_alpha, {"size": size_factor})
assert ab2.has_alpha(2.0), f"S6: genuine alpha was not detected (t={ab2.alpha_t:+.2f})"
assert abs(ab2.alpha - 0.0009) < 3e-4, f"S6: alpha not recovered: {ab2.alpha:+.6f}"
print(f"S6 PASS  same bet + real alpha: alpha {ab2.alpha:+.5f} t={ab2.alpha_t:+.1f} (detected)")

print("\nALL PASS")
