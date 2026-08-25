"""ICIR + shrinkage combiner (Ding-Martin Redux) -- the math, not the pipeline.

Exercises ``_icir_weights`` and ``_fit_predict_icir`` on synthetic data so the
Part 1 logic is validated without standing up Qlib / a real panel. The
dispatch (``model == "icir"``) and LightGBM byte-identity are end-to-end
properties covered by the 1-worker smoke replay, which runs the full pipeline.
"""
import numpy as np
import pandas as pd
from dataclasses import replace

from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.combiner import _icir_weights, _fit_predict_icir


# --- 1. _icir_weights: IC/σ_IC, shrunk toward 1/N -------------------------
# A known per-date IC matrix: feat 0 stable-positive, feat 1 same mean but
# volatile, feat 2 negative. ICIR must put σ_IC into the weights (Ding-Martin).
rng = np.random.default_rng(0)
n_dates, n_feats = 80, 3
ic_arr = np.column_stack([
    rng.normal(0.05, 0.02, n_dates),   # mean .05, σ .02 -> ICIR ~ 2.5
    rng.normal(0.05, 0.10, n_dates),   # mean .05, σ .10 -> ICIR ~ 0.5 (same mean, 5x σ)
    rng.normal(-0.03, 0.02, n_dates),  # mean -.03       -> ICIR ~ -1.5
])
ic_mean = np.nanmean(ic_arr, axis=0)
ic_std = np.nanstd(ic_arr, axis=0, ddof=1)
icir_expect = ic_mean / (ic_std + 1e-8)
print(f"mean IC  : {np.round(ic_mean, 4)}")
print(f"σ_IC     : {np.round(ic_std, 4)}")
print(f"ICIR     : {np.round(icir_expect, 4)}")

def _icir_of(col):
    return float(np.nanmean(col) / np.nanstd(col, ddof=1))


# shrinkage=0 -> raw ICIR (no shrink): the Ding-Martin weights.
w0, d0 = _icir_weights(ic_arr, 0.0)
assert d0 == 0.0, "shrinkage=0 must report delta=0"
assert np.allclose(w0, icir_expect, atol=1e-6), "shrinkage=0 must return raw ICIR"
assert w0[0] > w0[1], "same mean IC, higher σ_IC -> lower ICIR weight"
print(f"\nshrink=0 weights: {np.round(w0, 4)}  (raw ICIR)")
print("OK  IC/σ_IC weights: stable factor outranks the volatile one at equal mean IC")

# shrinkage=1 -> equal weight (the high-σ_IC limit / null model).
w1, d1 = _icir_weights(ic_arr, 1.0)
assert d1 == 1.0
# The target is sign(icir)/N, NOT a positive 1/N. A uniform positive target
# flips the sign of every factor with |icir| < d/(1-d)/N, and mined factors run
# icir ~0.01-0.05 -- so on a reversal-dominated market the old target inverted
# nearly every factor's contribution (measured: combiner rank IC -0.0063 vs
# +0.0319 for a sign-aligned equal weight). This test asserted the pre-fix
# behaviour; the sign-preserving target is the intended one.
signs = np.sign([_icir_of(ic_arr[:, j]) for j in range(n_feats)])
assert np.allclose(w1, signs / n_feats), \
    f"shrinkage=1 must equal-weight WITH SIGN: got {w1}, want {signs / n_feats}"
print(f"shrink=1 weights: {np.round(w1, 4)}  (sign-preserving equal)")
print("OK  full shrinkage collapses to sign(icir)/N, keeping each factor's direction")

# auto -> Ledoit-Wolf δ in [0,1], weights between raw ICIR and equal.
wa, da = _icir_weights(ic_arr, "auto")
assert 0.0 <= da <= 1.0, "auto delta must be a valid intensity"
equal = signs / n_feats          # the sign-preserving target, as above
expected_auto = (1.0 - da) * icir_expect + da * equal
assert np.allclose(wa, expected_auto, atol=1e-6), "auto must be (1-δ)·ICIR + δ·equal"
# Each weight lies between its raw-ICIR and equal-weight endpoints.
for i in range(n_feats):
    lo, hi = sorted((icir_expect[i], equal[i]))
    assert lo - 1e-9 <= wa[i] <= hi + 1e-9, f"auto weight {i} not between endpoints"
print(f"\nauto delta={da:.3f} weights: {np.round(wa, 4)}")
print("OK  auto shrinkage is a convex blend of ICIR and equal weight")


# --- 2. σ_IC actually enters the weights ----------------------------------
# The whole point of ICIR vs mean-IC: at equal mean IC, the volatile factor
# gets a strictly smaller weight for any δ < 1.
assert w0[0] > w0[1] > w0[2], "ICIR must order stable+ > volatile+ > negative"
print("\nOK  weight ordering stable+ > volatile+ > negative holds at δ=0")


# --- 3. bootstrap over dates gives honest seed variance (se > 0) ----------
# Build a synthetic panel: 4 factors with the same structure as above, a label
# that loads on the latent signal, and a PanelBundle with only the fields the
# combiner touches (dates/instruments/universe, all derived from close).
rng = np.random.default_rng(1)
dates = pd.date_range("2018-01-01", periods=60, freq="B")
inst = [f"S{i}" for i in range(40)]
idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
signal = rng.normal(size=(len(dates), len(inst)))
feat = np.stack([
    signal + 0.3 * rng.normal(size=signal.shape),     # F0: stable +
    signal + 3.0 * rng.normal(size=signal.shape),     # F1: volatile + (same mean)
    rng.normal(size=signal.shape),                    # F2: noise
    -signal + 0.3 * rng.normal(size=signal.shape),    # F3: negative
], axis=-1)                                           # (T, N, 4)
features = pd.DataFrame(
    feat.reshape(len(dates) * len(inst), n_feats := 4), index=idx,
    columns=[f"F{i}" for i in range(4)],
)
label_arr = signal + 0.5 * rng.normal(size=signal.shape)
y = pd.Series(label_arr.ravel(), index=idx, name="LABEL0")

close = pd.DataFrame(100.0, index=dates, columns=inst)
universe = pd.DataFrame(True, index=dates, columns=inst)
panel = PanelBundle(close=close, open=close, high=close, low=close, volume=close,
                    amount=close, vwap=close, factor=close, universe=universe)

th = load_protocol(default_protocol_path())
def icir_theta(seeds, shrink):
    return replace(th, combiner=replace(
        th.combiner, model="icir", seeds=seeds, params={"shrinkage": shrink}))

def seed_preds(shrink):
    out = []
    for s in [42, 1, 7, 13, 29]:
        t = icir_theta([s], shrink)
        out.append(_fit_predict_icir(features, features, y, t, panel)[0].to_numpy())
    return out

# δ=0: raw ICIR -> bootstrap resamples the dates, so the 5 seeds differ.
p_raw = seed_preds(0.0)
se_raw = float(np.std([p.mean() for p in p_raw], ddof=1))
print(f"\nδ=0 per-seed mean pred: {[round(float(p.mean()), 6) for p in p_raw]}  se={se_raw:.2e}")
assert not np.allclose(p_raw[0], p_raw[1]), "δ=0: different seeds must give different predictions"
assert se_raw > 0.0, "δ=0: bootstrap must produce positive seed variance"
print("OK  raw ICIR: 5 bootstrap seeds give se > 0 (honest σ_IC)")

# δ=1: the target is sign(icir)/N, and that SIGN is estimated from the same
# bootstrap-resampled IC array. So full shrinkage no longer produces a constant
# weight vector: a factor whose ICIR sits near zero can flip direction between
# seeds, and the prediction moves with it.
#
# This is a repair, not a regression. Under the old positive 1/N target the
# weights were seed-independent, the 5-seed admission gate measured se=0, and
# the variance test was INERT -- it could not distinguish a stable factor from
# one whose direction was a coin flip. Now that instability is visible, which is
# the entire purpose of testing on 5 seeds.
p_eq = seed_preds(1.0)
means = [float(p.mean()) for p in p_eq]
se_eq = float(np.std(means, ddof=1))
print(f"δ=1 per-seed mean pred: {[round(m, 6) for m in means]}  se={se_eq:.2e}")
assert se_eq > 0.0, "δ=1: an ESTIMATED sign target must still carry seed variance"
# The spread is discrete, not continuous: seeds cluster into a small number of
# groups, one per sign assignment. Continuous jitter would mean something else.
groups = len({round(m, 9) for m in means})
assert 1 < groups <= len(means), f"δ=1: expected sign-flip clustering, saw {groups} distinct means"
print(f"OK  full shrinkage: {groups} distinct seed outcomes -- the sign of a "
      "near-zero-ICIR factor flips between bootstrap draws, and the gate now sees it")

# The returned frame is wide T×N, aligned to the panel, masked by universe.
pred = _fit_predict_icir(features, features, y, icir_theta([42], "auto"), panel)[0]
assert pred.shape == (len(dates), len(inst)), f"wide shape {pred.shape}"
assert (pred.index == panel.dates).all() and (pred.columns == panel.instruments).all()
assert pred.notna().any().any(), "prediction must not be entirely NaN"
print(f"\nOK  output is wide {pred.shape}, aligned to panel.dates/instruments")

print("\nALL OK  ICIR + shrinkage combiner math and bootstrap variance verified")


# --- 4. Grinold α = IC_c · σ · s (sign-guaranteed, per-name vol-scaled) ----
from quantaalpha.eval.execution import grinold_alpha

rng = np.random.default_rng(2)
gdates = pd.date_range("2018-01-01", periods=80, freq="B")
ginst = [f"S{i}" for i in range(40)]
sig = rng.normal(size=(len(gdates), len(ginst)))
gsigma = pd.DataFrame(rng.uniform(0.01, 0.05, size=sig.shape), index=gdates, columns=ginst)


def xs_corr(a, b):
    """Mean per-date cross-sectional Pearson corr of two wide T×N frames."""
    vals = []
    for d in a.index:
        av, bv = a.loc[d].to_numpy(), b.loc[d].to_numpy()
        m = np.isfinite(av) & np.isfinite(bv)
        if m.sum() >= 5:
            av, bv = av[m] - av[m].mean(), bv[m] - bv[m].mean()
            den = np.sqrt((av**2).sum() * (bv**2).sum())
            if den > 0:
                vals.append((av * bv).sum() / den)
    return float(np.mean(vals)) if vals else float("nan")


def zscore_per_date(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0)


y = pd.DataFrame(sig + 0.5 * rng.normal(size=sig.shape), index=gdates, columns=ginst)
win = (str(gdates[0].date()), str(gdates[-1].date()))

# positive case: pred correlates with y -> IC_c > 0 -> α aligns with pred and y.
pred_pos = pd.DataFrame(sig + 0.2 * rng.normal(size=sig.shape), index=gdates, columns=ginst)
alpha_pos = grinold_alpha(pred_pos, y, gsigma, win)
ic_c_pos = xs_corr(zscore_per_date(pred_pos), y)
print(f"\nGrinold pos: IC_c~{ic_c_pos:.3f}  corr(α,pred)={xs_corr(alpha_pos, pred_pos):+.3f}  corr(α,y)={xs_corr(alpha_pos, y):+.3f}")
assert ic_c_pos > 0, "positive case must have IC_c > 0"
assert xs_corr(alpha_pos, pred_pos) > 0, "IC_c>0 -> α aligns with pred"
assert xs_corr(alpha_pos, y) > 0, "α must predict y positively"

# negative case: pred ANTI-correlates with y -> IC_c < 0, but α FLIPS and still
# predicts y. This is the property the empirical β lacks: prediction_scale
# falls back to 1.0 when β≤0 (flattening the signal), Grinold flips it.
pred_neg = pd.DataFrame(-sig + 0.2 * rng.normal(size=sig.shape), index=gdates, columns=ginst)
alpha_neg = grinold_alpha(pred_neg, y, gsigma, win)
ic_c_neg = xs_corr(zscore_per_date(pred_neg), y)
print(f"Grinold neg: IC_c~{ic_c_neg:.3f}  corr(α,pred)={xs_corr(alpha_neg, pred_neg):+.3f}  corr(α,y)={xs_corr(alpha_neg, y):+.3f}")
assert ic_c_neg < 0, "negative case must have IC_c < 0"
assert xs_corr(alpha_neg, pred_neg) < 0, "IC_c<0 -> α flips vs pred"
assert xs_corr(alpha_neg, y) > 0, "sign-guaranteed: α still predicts y when pred inversely relates to y"
print("OK  Grinold α is sign-guaranteed (a negative IC_c flips the book the right way)")

# exact reconstruction: α = IC_c · σ · s. The ~1e-3 "matches β·μ / preserves
# λ=25" claim is a REAL-data property (there IC_c ≈ 0.03 -> |α| ≈ 6e-4); this
# synthetic signal is intentionally strong (IC_c ≈ 0.87) so |α| is larger.
# The load-bearing invariant -- α = IC_c·σ·s -- is what the allclose asserts.
s_pos = zscore_per_date(pred_pos)
expected = ic_c_pos * gsigma * s_pos
assert np.allclose(alpha_pos.to_numpy(), expected.to_numpy(), equal_nan=True, atol=1e-12), \
    "α must equal IC_c·σ·s exactly"
mag = float(np.nanmean(np.abs(alpha_pos.to_numpy())))
assert mag > 0 and mag < abs(ic_c_pos) * float(gsigma.to_numpy().max()) * 5, \
    "α scale must track IC_c·σ (non-degenerate, not exploding)"
print(f"OK  α = IC_c·σ·s exactly; |α|~{mag:.2e} (scale tracks IC_c·σ; λ=25 preserved at real IC_c≈0.03)")

# masking: α is NaN wherever the prediction is NaN (universe respected).
pred_nan = pred_pos.copy()
pred_nan.iloc[:5, 0] = np.nan
alpha_nan = grinold_alpha(pred_nan, y, gsigma, win)
assert alpha_nan.iloc[:5, 0].isna().all(), "α must be NaN where pred is NaN"
print("OK  α masked to the prediction's universe")


# --- 5. Transfer coefficient corr(α, w*) ∈ [-1, 1] ------------------------
from quantaalpha.eval.operator import _transfer_coefficient

a = pd.DataFrame(rng.normal(size=(20, 30)),
                 index=pd.date_range("2022-01-03", periods=20, freq="B"),
                 columns=[f"S{i}" for i in range(30)])
tc_p, n_p = _transfer_coefficient(a, a.copy())            # w = α  -> TC = 1
tc_n, _ = _transfer_coefficient(a, -a)                    # w = -α -> TC = -1
tc_0, _ = _transfer_coefficient(
    a, pd.DataFrame(rng.normal(size=a.shape), index=a.index, columns=a.columns))
print(f"\nTC perfect={tc_p:.3f}  anti={tc_n:.3f}  noise~{tc_0:.3f}  (n={n_p})")
assert abs(tc_p - 1.0) < 1e-9, "w=α -> TC=1"
assert abs(tc_n + 1.0) < 1e-9, "w=-α -> TC=-1"
assert -1.0 <= tc_0 <= 1.0, "TC must lie in [-1, 1]"
assert abs(tc_0) < 0.5, "uncorrelated w -> TC near 0"
print("OK  transfer coefficient: 1 for w=α, -1 for w=-α, ~0 for noise")

print("\nALL OK  Grinold α and transfer coefficient verified")