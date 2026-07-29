"""T+1 cannot bind at daily rebalance frequency — proved, not assumed.

A name entering the book at step s has its buy filled at s+delta; the earliest
sale decision is step s+1, filling at s+1+delta. Those are always a session
apart, whatever delta is. So the constraint is satisfied by the rebalance
frequency itself and enabling it changes no book.

This test exists so the claim is checked rather than argued. If a future
construction rebalances more than once a day, it will start failing, which is
exactly when the machinery becomes load-bearing.
"""
import numpy as np, pandas as pd
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.portfolio import topk_dropout

th = load_protocol(default_protocol_path())
th = replace(th, portfolio=replace(th.portfolio, topk=5, n_drop=5, cost_aware_dropout=False),
                 constraints=replace(th.constraints, enforce_price_limits=False,
                                     enforce_suspension=False))
rng = np.random.default_rng(7)
dates = pd.date_range("2022-01-03", periods=40, freq="B")
names = [f"S{i}" for i in range(20)]

# Adversarial: a signal that completely reshuffles every day, so the book wants
# to sell everything it just bought at every single step.
pred = pd.DataFrame(rng.normal(size=(len(dates), len(names))), index=dates, columns=names)

books = {}
for flag in (True, False):
    t = replace(th, constraints=replace(th.constraints, enforce_t1=flag))
    w, _ = topk_dropout(pred, t)
    books[flag] = w

diff = float((books[True] - books[False]).abs().to_numpy().max())
print(f"max |w(T+1 on) - w(T+1 off)| over {len(dates)} days = {diff:.2e}")
assert diff == 0.0, "T+1 changed the book — it is no longer structurally inert"

for delta in (0, 1, 2):
    t = replace(th, execution=replace(th.execution, delta=delta),
                    constraints=replace(th.constraints, enforce_t1=True))
    w, _ = topk_dropout(pred, t)
    assert float((w - books[False]).abs().to_numpy().max()) == 0.0, \
        f"T+1 binds at delta={delta}"
print("OK  T+1 is inert at delta = 0, 1 and 2, even against a signal that")
print("    reshuffles the entire book every single day")
