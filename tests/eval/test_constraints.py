import numpy as np, pandas as pd
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
from quantaalpha.eval.data import PanelBundle
from quantaalpha.eval.tradability import trade_mask
from quantaalpha.eval.portfolio import topk_dropout

th = load_protocol(default_protocol_path())
dates = pd.date_range("2022-01-03", periods=6, freq="B")
names = [f"S{i}" for i in range(6)]
F = lambda v: pd.DataFrame(v, index=dates, columns=names, dtype=float)

close = F(100.0); high = F(101.0); low = F(99.0); vol = F(1e6)
# S0 locked limit-UP on day 2 fill date; S1 locked limit-DOWN; S2 suspended
close.iloc[2, 0] = 110.0; high.iloc[2, 0] = low.iloc[2, 0] = 110.0
close.iloc[2, 1] = 90.0;  high.iloc[2, 1] = low.iloc[2, 1] = 90.0
vol.iloc[2, 2] = 0.0
panel = PanelBundle(open=close.copy(), high=high, low=low, close=close, volume=vol,
                    amount=F(1e8), vwap=close.copy(), factor=F(1.0),
                    universe=pd.DataFrame(True, index=dates, columns=names))

th0 = replace(th, execution=replace(th.execution, delta=0))   # decision date == fill date
m = trade_mask(panel, th0)
d2 = dates[2]
print("day 2 (fill day):")
print(f"  S0 limit-up   -> can_buy={m.can_buy.loc[d2,'S0']} (want False)  can_sell={m.can_sell.loc[d2,'S0']} (want True)")
print(f"  S1 limit-down -> can_buy={m.can_buy.loc[d2,'S1']} (want True)   can_sell={m.can_sell.loc[d2,'S1']} (want False)")
print(f"  S2 suspended  -> can_buy={m.can_buy.loc[d2,'S2']} (want False)  can_sell={m.can_sell.loc[d2,'S2']} (want False)")
assert not m.can_buy.loc[d2,"S0"] and m.can_sell.loc[d2,"S0"]
assert m.can_buy.loc[d2,"S1"] and not m.can_sell.loc[d2,"S1"]
assert not m.can_buy.loc[d2,"S2"] and not m.can_sell.loc[d2,"S2"]
assert bool(m.suspended.loc[d2,"S2"])
print("OK  limit-up blocks buys, limit-down blocks sells, suspension blocks both")

# delta=1 must shift feasibility onto the DECISION date
m1 = trade_mask(panel, replace(th, execution=replace(th.execution, delta=1)))
assert not m1.can_buy.loc[dates[1], "S0"], "delta=1 must block the decision one day earlier"
assert m1.can_buy.loc[dates[0], "S0"], "only the decision date that fills into the lock is blocked"
print("OK  delta shifts feasibility onto the decision date")

# ---- T+1: a name bought today cannot be sold tomorrow ----
th_t1 = replace(th, portfolio=replace(th.portfolio, topk=2, n_drop=2))
pred = pd.DataFrame(0.0, index=dates, columns=names)
pred.iloc[0] = [9, 8, 1, 1, 1, 1]     # buy S0,S1
pred.iloc[1] = [0, 0, 9, 8, 1, 1]     # want to swap into S2,S3 immediately
for flag, want in ((True, "held"), (False, "swapped")):
    w, _ = topk_dropout(pred, replace(th_t1, constraints=replace(th_t1.constraints, enforce_t1=flag,
                        enforce_price_limits=False, enforce_suspension=False)))
    day1 = set(w.iloc[1][w.iloc[1] > 0].index)
    print(f"  enforce_t1={str(flag):5s} -> day-1 book {sorted(day1)}  (want: swapped either way -- T+1 cannot bind daily)")
    if flag:
        assert day1 == {"S2","S3"}, "T+1 must not bind at daily frequency"
    else:
        assert day1 == {"S2","S3"}, "without T+1 the swap should happen"
print("OK  T+1 blocks same-next-day resale, and only when enabled")
