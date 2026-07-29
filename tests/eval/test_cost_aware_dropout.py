import numpy as np, pandas as pd
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.portfolio import topk_dropout
from quantaalpha.eval.execution import turnover as to_fn

th = load_protocol(default_protocol_path())
th = replace(th, portfolio=replace(th.portfolio, topk=10, n_drop=3),
                 constraints=replace(th.constraints, enabled=False))
dates = pd.date_range("2022-01-03", periods=80, freq="B")
names = [f"S{i}" for i in range(40)]
sigma = pd.DataFrame(0.02, index=dates, columns=names)
BETA = 4e-4      # beta * sd(score); measured 0.0085 * 0.0498 on real predictions

def ar1(phi, seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=len(names)); rows=[]
    for _ in dates:
        x = phi*x + np.sqrt(1-phi**2)*rng.normal(size=len(names)); rows.append(x.copy())
    return pd.DataFrame(rows, index=dates, columns=names)

def bt(pred, theta, beta=BETA):
    w, d = topk_dropout(pred, theta, sigma=sigma, pred_scale=beta)
    return float(np.mean([to_fn(w.loc[t], d.loc[t]) for t in pred.index[1:]]))

noisy, persistent = ar1(0.0, 2), ar1(0.95, 1)
K = lambda t,k0,k1: replace(t, costs=replace(t.costs, kappa0=k0, kappa1=k1))
fixed = replace(th, portfolio=replace(th.portfolio, cost_aware_dropout=False))
ca    = replace(th, portfolio=replace(th.portfolio, cost_aware_dropout=True, swap_hurdle=1.0))

print("does the cost model influence how much we trade?\n")
print(f"{'construction':<24} {'kappa0=0':>9} {'kappa0=.002':>12} {'kappa0=.01':>11}  responds?")
row = lambda name, t: (name, bt(noisy,K(t,0.0,0.0)), bt(noisy,K(t,0.002,0.03)), bt(noisy,K(t,0.01,0.30)))
for name, a, b, c in (row("fixed n_drop (current)", fixed), row("cost-aware dropout", ca)):
    print(f"{name:<24} {a:>9.4f} {b:>12.4f} {c:>11.4f}  {'yes' if c<a-1e-9 else 'NO'}")

f = row("", fixed); c = row("", ca)
assert abs(f[1]-f[3]) < 1e-12, "fixed n_drop must be cost-insensitive"
assert c[3] < c[1], "cost-aware turnover must fall as cost rises"
print("\nOK  cost now changes the book; it provably did not before")

print("\ndoes it respond to signal persistence?")
for lab, sig in (("noisy (phi=0.00)", noisy), ("persistent (phi=0.95)", persistent)):
    print(f"  {lab:<24} turnover {bt(sig, K(ca,0.002,0.03)):.4f}")
assert bt(persistent,K(ca,0.002,0.03)) < bt(noisy,K(ca,0.002,0.03))
print("OK  persistent signals trade less than noisy ones")
