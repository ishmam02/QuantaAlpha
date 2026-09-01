"""The research tear sheet must separate signals a scalar cannot.

T1  a monotone signal scores monotonicity ~1
T2  a TAILS-ONLY signal scores monotonicity ~0 while still showing a wide
    Q10-Q1 spread -- that separation is the whole point, because a tails-only
    factor dies under any position cap and a spread-only view cannot see it
T3  ic_decay_curve on a known h-day signal peaks at h
T4  Newey-West t is materially SMALLER than the naive t on autocorrelated data
"""
import numpy as np, pandas as pd
from dataclasses import replace

from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.metrics import quantile_metrics, ic_decay_curve, newey_west_t
from quantaalpha.eval.protocol import load_protocol

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")
rng = np.random.default_rng(0)

D, N = 400, 200
dates = pd.bdate_range("2019-01-01", periods=D)
insts = pd.Index([f"S{i:03d}" for i in range(N)])
WIN = (str(dates[20].date()), str(dates[-25].date()))


def make_panel(close):
    ones = pd.DataFrame(1.0, index=dates, columns=insts)
    return PanelBundle(open=close, high=close, low=close, close=close,
                       volume=ones, amount=ones, vwap=close, factor=ones,
                       universe=pd.DataFrame(True, index=dates, columns=insts))


def build(effect, horizon=1):
    """Prices whose h-day forward return is driven by `effect` (dates x insts)."""
    # label_frame_at: P[t+1+h]/P[t+1]-1. Put the effect into the step at t+1.
    step = pd.DataFrame(rng.normal(0, 0.01, (D, N)), index=dates, columns=insts)
    step = step + effect.reindex(index=dates, columns=insts).fillna(0.0)
    close = (1.0 + step).cumprod()
    return make_panel(close)


sig = pd.DataFrame(rng.normal(0, 1, (D, N)), index=dates, columns=insts)
rank = sig.rank(axis=1, pct=True)

# --- T1: a clean monotone response -------------------------------------------
lin = (rank - 0.5) * 0.02                      # return rises linearly with rank
eff = lin.shift(2).fillna(0.0)                 # step[u] = f(sig[u-2]) => label[t] = f(sig[t])
q = quantile_metrics(sig, build(eff), TH, WIN, n_q=10)
assert q["monotonicity"] > 0.9, f"T1: monotone signal scored {q['monotonicity']:.3f}"
assert q["q_spread"] > 0, "T1: spread must be positive"
print(f"T1 PASS  monotone: monotonicity={q['monotonicity']:+.3f} spread={q['q_spread']:+.5f}")

# --- T2: tails only ----------------------------------------------------------
# Only the extreme deciles move; the middle 80% is pure noise.
tails = pd.DataFrame(0.0, index=dates, columns=insts)
tails = tails.mask(rank > 0.9, 0.010).mask(rank < 0.1, -0.010)
qt = quantile_metrics(sig, build(tails.shift(2).fillna(0.0)), TH, WIN, n_q=10)
assert qt["q_spread"] > 0.01, f"T2: tails spread should be wide, got {qt['q_spread']:.5f}"
assert abs(qt["monotonicity"]) < 0.75 < abs(q["monotonicity"]), (
    f"T2: tails-only must score materially LOWER monotonicity than the monotone "
    f"case ({qt['monotonicity']:.3f} vs {q['monotonicity']:.3f}); a spread-only "
    f"view cannot tell them apart, which is why monotonicity is gated on")
print(f"T2 PASS  tails-only: monotonicity={qt['monotonicity']:+.3f} "
      f"spread={qt['q_spread']:+.5f}  <- wide spread, weaker monotonicity")

# --- T3: the decay curve finds the true horizon ------------------------------
# A signal that predicts the 5-day return, not the 1-day return.
H = 5
bump = (rank - 0.5) * 0.05
# Accumulate the effect into the daily STEPS, then cumulate into a price.
# Multiplying an already-cumulated price instead perturbs two consecutive
# returns per factor and destroys the horizon structure being tested.
step5 = pd.DataFrame(rng.normal(0, 0.006, (D, N)), index=dates, columns=insts)
for k in range(H):                              # spread the effect over h days
    step5 = step5 + (bump / H).shift(2 + k).fillna(0.0)
close5 = (1.0 + step5).cumprod()
dc = ic_decay_curve(sig, make_panel(close5), TH, WIN, horizons=(1, 2, 3, 5, 10, 20))
curve_s = ", ".join(f"h{h}={v['rank_ic']:+.4f}" for h, v in dc["curve"].items())
assert dc["best_horizon"] == H, f"T3: expected peak at h={H}, got {dc['best_horizon']}; {curve_s}"
print(f"T3 PASS  decay curve peaks at h={dc['best_horizon']} "
      f"(rank_ic {dc['best_rank_ic']:+.4f}); "
      f"h=1 was {dc['curve'][1]['rank_ic']:+.4f}")

# --- T4: HAC deflates an autocorrelated t ------------------------------------
e = rng.normal(0, 1, 600)
ar = np.zeros(600)
for i in range(1, 600):
    ar[i] = 0.7 * ar[i - 1] + e[i]
ar = pd.Series(ar + 0.15)
naive = float(ar.mean() / (ar.std() / np.sqrt(len(ar))))
nw = newey_west_t(ar)
assert abs(nw) < abs(naive) * 0.8, (
    f"T4: HAC t ({nw:.2f}) must be materially below naive t ({naive:.2f})")
print(f"T4 PASS  naive t={naive:.2f} -> Newey-West t={nw:.2f} "
      f"({100*(1-abs(nw)/abs(naive)):.0f}% deflation)")

print("\nALL PASS")
