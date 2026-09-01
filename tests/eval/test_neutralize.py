"""Neutralization must remove size, and must not remove alpha.

N1  a signal that IS size neutralizes to ~zero IC vs size
N2  a signal orthogonal to size survives essentially unchanged
N3  exposure_report names the exposure -- the readout the generator has never had
N4  the library residual kills a duplicate but keeps a genuinely new signal
"""
import numpy as np, pandas as pd

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.neutralize import (residualize, residualize_vs_library,
                                         exposure_report, size_frame)
from quantaalpha.eval.protocol import load_protocol

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")

mc = pd.read_parquet("data/reference/market_cap.parquet")
# A window where plenty of names are live, using REAL market caps.
mc = mc[(mc["date"] >= "2020-01-01") & (mc["date"] <= "2020-12-31")]
counts = mc.groupby("instrument").size()
insts = pd.Index(sorted(counts[counts > 200].index))[:180]
assert len(insts) >= 60, f"only {len(insts)} usable instruments in the partial sync"
dates = pd.DatetimeIndex(sorted(mc["date"].unique()))
print(f"fixture: {len(insts)} instruments x {len(dates)} dates of REAL market cap")

rng = np.random.default_rng(0)
px = pd.DataFrame((1 + rng.normal(0, 0.01, (len(dates), len(insts)))).cumprod(axis=0),
                  index=dates, columns=insts)
ones = pd.DataFrame(1.0, index=dates, columns=insts)
panel = PanelBundle(open=px, high=px, low=px, close=px, volume=ones, amount=ones,
                    vwap=px, factor=ones,
                    universe=pd.DataFrame(True, index=dates, columns=insts))

SZ = size_frame(panel)
assert SZ.notna().sum().sum() > 0, "no market cap resolved onto the panel grid"


def mean_xs_corr(a, b):
    c = a.corrwith(b, axis=1, method="spearman").dropna()
    return float(c.mean()) if len(c) else float("nan")


# --- N1: a pure size signal must not survive ---------------------------------
pure_size = SZ.copy()
before = mean_xs_corr(pure_size, SZ)
after = mean_xs_corr(residualize(pure_size, panel, TH), SZ)
assert abs(before) > 0.95, f"N1 fixture: pure size should correlate ~1, got {before:.3f}"
assert abs(after) < 0.10, (
    f"N1: a signal that IS size must neutralize away. corr vs size "
    f"{before:+.3f} -> {after:+.3f}")
print(f"N1 PASS  pure-size signal: corr vs size {before:+.3f} -> {after:+.3f}")

# --- N2: an orthogonal signal must survive -----------------------------------
noise = pd.DataFrame(rng.normal(0, 1, (len(dates), len(insts))),
                     index=dates, columns=insts)
res_noise = residualize(noise, panel, TH)
kept = mean_xs_corr(res_noise, noise)
assert kept > 0.80, (
    f"N2: a signal orthogonal to the risk factors must survive; only {kept:.3f} "
    f"of it did — neutralization is eating real signal")
print(f"N2 PASS  orthogonal signal retained: corr(residual, original) = {kept:+.3f}")

# --- N3: the exposure is NAMED ----------------------------------------------
rep = exposure_report(pure_size, panel)
assert abs(rep["exposure_size"]) > 0.9, f"N3: size exposure not detected: {rep}"
rep2 = exposure_report(noise, panel)
assert abs(rep2["exposure_size"]) < 0.2, f"N3: false size exposure on noise: {rep2}"
print(f"N3 PASS  exposure named: size signal {rep['exposure_size']:+.3f}, "
      f"noise {rep2['exposure_size']:+.3f}")

# --- N4: incremental alpha vs the library ------------------------------------
lib = {"held": noise}
dup = residualize_vs_library(noise * 1.7 + 0.3, lib, panel)     # affine duplicate
new = residualize_vs_library(
    pd.DataFrame(rng.normal(0, 1, (len(dates), len(insts))), index=dates, columns=insts),
    lib, panel)
dup_left = float(np.nanstd(dup.to_numpy()))
new_left = float(np.nanstd(new.to_numpy()))
assert dup_left < 0.05 * new_left, (
    f"N4: an affine duplicate of a held factor must residualize to ~nothing "
    f"(residual sd {dup_left:.4f} vs {new_left:.4f} for a genuinely new signal)")
print(f"N4 PASS  duplicate residual sd {dup_left:.5f} vs new signal {new_left:.3f}")

print("\nALL PASS")
