"""The mean-variance member: turnover is a budget the optimiser trades against."""
import numpy as np, pandas as pd
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.portfolio import build_book, _mv_weights
from quantaalpha.eval.execution import turnover as to_fn

th = load_protocol(default_protocol_path())
MV = lambda **kw: replace(th, portfolio=replace(
    th.portfolio, construction="mean_variance", signed=False, **kw),
    constraints=replace(th.constraints, enabled=False))

rng = np.random.default_rng(3)
dates = pd.date_range("2022-01-03", periods=60, freq="B")
names = [f"S{i}" for i in range(60)]
pred = pd.DataFrame(rng.normal(size=(len(dates), len(names))), index=dates, columns=names)
sigma = pd.DataFrame(rng.uniform(0.01, 0.05, size=(len(dates), len(names))),
                     index=dates, columns=names)
BETA = 4e-4

def book(theta):
    return build_book(pred, theta, sigma=sigma, pred_scale=BETA)

# --- 1. the QP respects its own constraints -------------------------------
mu = pd.Series(rng.normal(size=40), index=names[:40])
var = pd.Series(rng.uniform(1e-4, 3e-3, size=40), index=names[:40])
w = _mv_weights(mu, var, lam=10.0, max_weight=0.05)
print(f"QP: sum={w.sum():.6f} min={w.min():.6f} max={w.max():.6f}")
assert abs(w.sum() - 1.0) < 1e-6, "budget not spent exactly"
assert w.min() >= -1e-12, "negative weight in a long-only book"
assert w.max() <= 0.05 + 1e-9, "position cap breached"

# --- 2. the turnover budget actually binds --------------------------------
print(f"\n{'turnover_cap':>13}  {'realised':>9}")
prev = None
for cap in (0.02, 0.05, 0.20, 0.50):
    w, d = book(MV(turnover_cap=cap, risk_aversion=10.0, max_weight=0.05))
    real = float(np.mean([to_fn(w.loc[t], d.loc[t]) for t in dates[1:]]))
    print(f"{cap:>13.2f}  {real:>9.4f}")
    assert real <= cap + 1e-6, f"turnover {real:.4f} exceeded its budget {cap}"
    if prev is not None:
        assert real >= prev - 1e-9, "a looser budget must not trade less"
    prev = real
assert prev > 0.02, "the budget never binds — it is not a real constraint"
print("OK  turnover is bounded by the budget and rises with it")

# --- 3. risk aversion moves the book toward minimum variance ---------------
# With a fully-invested long-only budget and a diagonal risk model the optimum
# is w_i ∝ (mu_i - eta)/sigma_i^2 with eta set by the budget. As lambda -> inf
# that tends to the inverse-variance (minimum-variance) portfolio; as
# lambda -> 0 it concentrates in the highest-mu names. Measuring "max weight"
# hides this whenever the position cap binds, so measure the effective number
# of names, 1/sum(w^2), and the correlation with 1/sigma^2.
eff_n = lambda w: 1.0 / float((w ** 2).sum())
print(f"\n{'risk_aversion':>13}  {'effective N':>11}  {'corr with 1/var':>16}")
prev_eff = None
for lam in (0.5, 10.0, 1000.0):
    w, _ = book(MV(risk_aversion=lam, turnover_cap=1.0, max_weight=1.0))
    last = w.iloc[-1]
    inv_var = 1.0 / (sigma.iloc[-1] ** 2)
    corr = float(np.corrcoef(last.to_numpy(), inv_var.to_numpy())[0, 1])
    print(f"{lam:>13.1f}  {eff_n(last):>11.2f}  {corr:>16.4f}")
    if prev_eff is not None:
        assert eff_n(last) >= prev_eff - 1e-6, "more risk aversion must not concentrate more"
    prev_eff = eff_n(last)
    last_corr = corr
assert last_corr > 0.95, "at high risk aversion the book should be ~inverse-variance"
print("OK  risk aversion diversifies the book toward minimum variance")

# --- 4. tradability is respected ------------------------------------------
from quantaalpha.eval.tradability import TradeMask
can_buy = pd.DataFrame(True, index=dates, columns=names)
can_sell = pd.DataFrame(True, index=dates, columns=names)
can_buy.iloc[:, :10] = False                      # first 10 names unbuyable throughout
m = TradeMask(can_buy, can_sell, pd.DataFrame(False, index=dates, columns=names),
              pd.DataFrame(True, index=dates, columns=names))
w, d = build_book(pred, MV(turnover_cap=0.5), sigma=sigma, pred_scale=BETA, mask=m)
grew = ((w.iloc[:, :10].to_numpy() - d.iloc[:, :10].to_numpy()) > 1e-9).sum()
print(f"\nunbuyable names that nonetheless grew: {grew} (want 0)")
assert grew == 0, "an unbuyable name increased its weight"
assert abs(float(w.iloc[-1].sum()) - 1.0) < 1e-6, "budget lost after clamping"
print("OK  blocked names never grow, and the book stays fully invested")
