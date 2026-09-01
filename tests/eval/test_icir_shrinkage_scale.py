"""The Ledoit-Wolf numerator must be the ESTIMATION ERROR of ICIR, not the
daily IC variance -- otherwise shrinkage always saturates and the ICIR fit
never happens.

    ICIR_hat = mean_IC / sd_IC, from T dates
    Var(mean_IC) = sd_IC**2 / T   =>   Var(ICIR_hat) ~ 1/T

The numerator was ``sum(ic_std**2)``: a sum of DAILY IC variances, T times too
large. Measured 2026-08-24 on two independent factor sets, valid window
(T=727):

    15-factor book: delta 1.000, every weight -1/N, |IC| 0.0554
    37-factor book: delta 1.000, every weight -1/N, |IC| 0.0593

The combiner computed ICIRs spanning [-0.36, -0.10] and then discarded all of
that differentiation. "ICIR combiner" was an equal-weight combiner in every run
to date, and the composite scored BELOW its own best single factor (0.98x).

After the fix: delta 0.240 / 0.234, |IC| 0.0612 / 0.0629, and the composite
crosses to 1.08x its best constituent.

K1  realistic inputs no longer saturate
K2  weights actually differentiate (the fit happens)
K3  delta responds to T -- less data shrinks harder
K4  delta responds to dispersion -- spread-out ICIRs shrink less
K5  degenerate inputs fall back to FULL shrinkage, never a wild weight
K6  an explicit shrink value still overrides "auto"
K7  the old formula is what saturated (documents the defect)
"""
import numpy as np

from quantaalpha.eval.combiner import _icir_weights

RNG = np.random.default_rng(11)


def ic_matrix(T=727, n=15, mean_ic=-0.03, sd_ic=0.17, spread=0.01, seed=3):
    """A realistic per-date IC matrix: daily ICs are noisy (sd ~ 0.17) around
    small means (~0.03), which is the regime the live books sit in."""
    r = np.random.default_rng(seed)
    means = mean_ic + r.normal(0, spread, n)
    return r.normal(means, sd_ic, size=(T, n))


# ---------------------------------------------------------------------------
# K1 -- realistic inputs must NOT saturate.
# ---------------------------------------------------------------------------
ic = ic_matrix()
w, d = _icir_weights(ic, "auto")
assert d < 0.99, f"K1: shrinkage still saturates on realistic input (delta={d:.4f})"
assert d > 0.0, f"K1: some shrinkage is expected on noisy ICs (delta={d:.4f})"
print(f"K1 PASS  realistic input gives delta={d:.3f}, not 1.0")

# ---------------------------------------------------------------------------
# K2 -- the fit actually happens: weights differ from one another.
# ---------------------------------------------------------------------------
spread_w = float(w.max() - w.min())
assert spread_w > 1e-6, (
    f"K2: all weights identical ({w[0]:.6f}) -- the ICIR fit was discarded")
print(f"K2 PASS  weights differentiate (spread {spread_w:.4f}, not 0)")

# ---------------------------------------------------------------------------
# K3 -- less data => noisier ICIR => shrink harder. This is the property the
# old formula could not express, because its numerator ignored T entirely.
# ---------------------------------------------------------------------------
d_short = _icir_weights(ic_matrix(T=60), "auto")[1]
d_long = _icir_weights(ic_matrix(T=2000), "auto")[1]
assert d_short > d_long, (
    f"K3: delta must fall as T grows; got T=60 -> {d_short:.3f}, "
    f"T=2000 -> {d_long:.3f}")
print(f"K3 PASS  delta falls with more data: T=60 {d_short:.3f} > T=2000 {d_long:.3f}")

# ---------------------------------------------------------------------------
# K4 -- widely dispersed ICIRs carry more signal to keep => shrink less.
# ---------------------------------------------------------------------------
d_tight = _icir_weights(ic_matrix(spread=0.002), "auto")[1]
d_wide = _icir_weights(ic_matrix(spread=0.05), "auto")[1]
assert d_wide < d_tight, (
    f"K4: dispersed ICIRs should shrink less; tight {d_tight:.3f}, "
    f"wide {d_wide:.3f}")
print(f"K4 PASS  delta falls with dispersion: tight {d_tight:.3f} > wide {d_wide:.3f}")

# ---------------------------------------------------------------------------
# K5 -- degenerate inputs must fall back to FULL shrinkage. With too little
# data there is no ICIR estimate to trust, and equal weight is the safe answer.
# ---------------------------------------------------------------------------
assert _icir_weights(np.zeros((0, 3)), "auto")[1] == 1.0, "K5: empty -> full shrink"
assert _icir_weights(RNG.normal(size=(1, 3)), "auto")[1] == 1.0, (
    "K5: a single date carries no ICIR estimate -> full shrink")
w_nan, _ = _icir_weights(np.full((10, 3), np.nan), "auto")
assert np.all(np.isfinite(w_nan)), "K5: all-NaN input must not produce NaN weights"
print("K5 PASS  empty / single-date / all-NaN degrade to full shrinkage")

# ---------------------------------------------------------------------------
# K6 -- an explicit shrink value still wins over "auto".
# ---------------------------------------------------------------------------
for explicit in (0.0, 0.5, 1.0):
    assert _icir_weights(ic, explicit)[1] == explicit, (
        f"K6: explicit shrink={explicit} was overridden")
print("K6 PASS  an explicit shrink value overrides the auto estimate")

# ---------------------------------------------------------------------------
# K7 -- document the defect: the OLD numerator saturates on this same input.
# Without this, a future edit could reinstate it and K1 alone would not say why.
# ---------------------------------------------------------------------------
m = np.nanmean(ic, axis=0)
s = np.nanstd(ic, axis=0, ddof=1)
icir = m / (s + 1e-8)
disp = float(np.nansum((icir - np.nanmean(icir)) ** 2))
old_delta = min(float(np.nansum(s ** 2)) / (disp + 1e-12), 1.0)
new_delta = min(ic.shape[1] / ic.shape[0] / (disp + 1e-12), 1.0)
assert old_delta == 1.0, "K7: the old formula is expected to saturate here"
assert new_delta < 1.0, "K7: the corrected formula must not"
ratio = float(np.nansum(s ** 2)) / (ic.shape[1] / ic.shape[0])
print(f"K7 PASS  old numerator is {ratio:.0f}x too large (T={ic.shape[0]}): "
      f"delta 1.000 -> {new_delta:.3f}")

print("\nALL PASS")
