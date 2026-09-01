"""mv_weights_exact must actually solve the problem it claims to.

It replaces a projected subgradient on the diagonal-Sigma path, so it is
checked against a general-purpose NLP solver on the same objective, plus the
structural properties the closed form is supposed to have.

E1  feasible: sum(w)=1, 0<=w<=cap
E2  matches SLSQP on the full costed objective
E3  beats the subgradient it replaces (objective value)
E4  the no-trade band exists: a large spread means the book does not move
E5  degenerates to the plain MV solution when all costs are zero
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

from quantaalpha.eval.riskmodel import (mv_weights_exact, mv_weights_costed,
                                        project_capped_simplex)
from quantaalpha.eval.portfolio import _mv_weights

rng = np.random.default_rng(7)
N, CAP, LAM = 60, 0.05, 25.0

def objective(w, m, v, g, p, lam, kap, k2, e):
    dx = np.abs(w - p)
    return float(m @ w - 0.5*lam*np.sum(v*w*w) - np.sum(g*dx) - np.sum(k2*dx**e)
                 - 0.5*kap*lam*np.sum(v*(w-p)**2))

def slsqp(m, v, g, p, lam, cap, kap, k2, e):
    f = lambda w: -objective(w, m, v, g, p, lam, kap, k2, e)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    r = minimize(f, np.full(len(m), 1.0/len(m)), method="SLSQP",
                 bounds=[(0.0, cap)]*len(m), constraints=cons,
                 options={"maxiter": 800, "ftol": 1e-14})
    return r.x

fails = 0
for trial in range(6):
    m = rng.normal(0, 6e-4, N)                 # Grinold-alpha scale
    v = (rng.uniform(0.01, 0.035, N))**2       # daily variances
    g = 0.0020 + 0.03*np.sqrt(v)               # kappa0 + kappa1*sigma
    p = project_capped_simplex(rng.dirichlet(np.ones(N)), CAP)
    kap, e = 5.0, 1.5
    k2 = np.zeros(N)

    idx = pd.Index([f"s{i}" for i in range(N)])
    S = lambda a: pd.Series(a, index=idx)
    w = mv_weights_exact(S(m), S(v), S(g), S(p), LAM, CAP,
                         k2_scale=None, impact_exponent=e,
                         hurdle=1.0, trade_penalty=kap).to_numpy()

    # E1 feasibility
    assert abs(w.sum() - 1.0) < 1e-9, f"E1: budget {w.sum():.12f}"
    assert w.min() >= -1e-12 and w.max() <= CAP + 1e-9, "E1: box violated"

    # E2 optimality vs SLSQP
    ref = slsqp(m, v, g, p, LAM, CAP, kap, k2, e)
    o_ex, o_ref = (objective(x, m, v, g, p, LAM, kap, k2, e) for x in (w, ref))
    if o_ex < o_ref - 1e-9:
        print(f"  trial {trial}: exact {o_ex:.10f} < slsqp {o_ref:.10f}")
        fails += 1

    # E3 vs the subgradient it replaces
    sub = mv_weights_costed(S(m), None, S(v), S(g), S(p), LAM, CAP,
                            kappa2=0.0, impact_exponent=e, hurdle=1.0,
                            trade_penalty=kap).to_numpy()
    o_sub = objective(sub, m, v, g, p, LAM, kap, k2, e)
    assert o_ex >= o_sub - 1e-12, f"E3: exact {o_ex} worse than subgradient {o_sub}"

assert fails == 0, f"E2: {fails}/6 trials below SLSQP"
print("E1-E3 PASS  feasible, matches SLSQP, and >= the subgradient it replaces")

# E4 no-trade band: make the spread enormous -> nothing should move.
m = rng.normal(0, 6e-4, N); v = (rng.uniform(0.01, 0.03, N))**2
p = project_capped_simplex(rng.dirichlet(np.ones(N)), CAP)
idx = pd.Index([f"s{i}" for i in range(N)]); S = lambda a: pd.Series(a, index=idx)
w_big = mv_weights_exact(S(m), S(v), S(np.full(N, 1.0)), S(p), LAM, CAP,
                         trade_penalty=5.0).to_numpy()
assert np.abs(w_big - p).sum() < 1e-6, (
    f"E4: a 100% spread must freeze the book; it moved {np.abs(w_big-p).sum():.4f}")
print("E4 PASS  no-trade band exists (a cash cost produces inaction, as it must)")

# E5 zero costs -> the plain MV solution
w0 = mv_weights_exact(S(m), S(v), S(np.zeros(N)), S(p), LAM, CAP,
                      trade_penalty=0.0).to_numpy()
ref0 = _mv_weights(S(m), S(v), LAM, CAP).to_numpy()
assert np.abs(w0 - ref0).sum() < 1e-6, \
    f"E5: zero-cost solution differs from _mv_weights by L1 {np.abs(w0-ref0).sum():.2e}"
print("E5 PASS  reduces to the plain mean-variance solution at zero cost")
print("\nALL PASS")
