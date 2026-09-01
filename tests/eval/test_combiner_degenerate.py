"""A factor must not be silenced by a date on which it has no variance.

Measured on the live design matrix: extending `train` to 2005 while the cached
signals start 2008-12-29 left ~776 dates filled with 0.0 -- a constant column.
`_per_date_ic` divided by that zero variance and returned +/-inf on 131 dates.
`notna()` counts inf as present and `np.nanmean` skips NaN but NOT inf, so every
one of 19 mined factors reported a NaN mean IC whose true value was -0.00917,
and a NaN weight makes `feat_mat @ weights` NaN for EVERY row.

G1  a constant column yields NaN, never +/-inf
G2  the surviving dates still produce the correct IC
G3  one degenerate date cannot NaN a factor's weight
G4  a wholly unusable column contributes nothing instead of poisoning the rest
"""
import numpy as np, pandas as pd
from quantaalpha.eval.combiner import _per_date_ic, _icir_weights

rng = np.random.default_rng(0)
D, N = 40, 60
dates = pd.date_range("2020-01-01", periods=D)
idx = pd.MultiIndex.from_product([dates, [f"s{i}" for i in range(N)]],
                                 names=["datetime", "instrument"])

y = pd.Series(rng.normal(0, 1, D * N), index=idx)
good = y * 0.3 + rng.normal(0, 1, D * N)          # a real signal
x = pd.DataFrame({"good": good.to_numpy(), "flat": good.to_numpy()}, index=idx)
# make "flat" constant on the first 10 dates -- exactly the fillna(0.0) case
mask = x.index.get_level_values("datetime") < dates[10]
x.loc[mask, "flat"] = 0.0

ic = _per_date_ic(x, y)
flat = ic["flat"].to_numpy(float)

# G1
assert not np.isinf(flat).any(), (
    f"G1: degenerate dates produced {int(np.isinf(flat).sum())} infinities; "
    "they must be NaN -- inf survives notna() and poisons nanmean")
assert np.isnan(flat[:10]).all(), "G1: the constant dates must be NaN"
print("G1 PASS  a zero-variance date yields NaN, not +/-inf")

# G2
assert np.isfinite(flat[10:]).all(), "G2: the usable dates must survive"
assert abs(np.nanmean(flat[10:]) - np.nanmean(ic["good"].to_numpy(float)[10:])) < 1e-9, \
    "G2: on the dates where the columns agree, their IC must agree"
print(f"G2 PASS  surviving dates keep their IC (mean {np.nanmean(flat[10:]):+.4f})")

# G3: the weight must be finite and reflect the usable dates
w, _delta = _icir_weights(ic.to_numpy(float), 0.5)
assert np.isfinite(w).all(), f"G3: weights must be finite, got {w}"
assert abs(w[1]) > 1e-6, "G3: a factor with 30 usable dates must get a real weight"
print(f"G3 PASS  weights finite despite degenerate dates: {np.round(w, 4)}")

# G4: a column that is unusable EVERYWHERE must not take out the others
x2 = x.copy(); x2["dead"] = 0.0
ic2 = _per_date_ic(x2, y)
w2, _ = _icir_weights(ic2.to_numpy(float), 0.5)
assert np.isfinite(w2).all(), f"G4: one dead column NaN'd every weight: {w2}"
assert abs(w2[0]) > 1e-6, "G4: the good factor must still carry weight"
pred = np.nan_to_num(x2.to_numpy(float)) @ w2
assert np.isfinite(pred).all(), "G4: the composite must not be NaN"
print(f"G4 PASS  a wholly dead column contributes nothing; composite stays finite")

print("\nALL PASS")
