# 13 — NAV that compounds with the book

**Depends on:** 01. **Blocks:** 14.

## Why

`Costs.nav` is a hardcoded constant, `1e8` CNY. It normalises dollar ADV into
weight space, so it is what makes market impact and capacity mean anything:

```
adv_w = adv / theta.costs.nav          # costs.py
impact = kappa2 * |dw|^1.5 / adv_w^...
```

A fund that doubles should find trading harder, and one that halves should find it
easier. With a frozen NAV neither happens, so the "cost-aware" objective is only
cost-aware at one wealth level — the one nobody is at after the first year.

User decision: **simulated, compounding from ¥100M.**

## Goal

NAV tracks the strategy's own equity curve through the backtest, so impact and
capacity scale with the fund, while every existing result stays reproducible.

## Approach

1. **`Costs` gains `nav_mode: str = "fixed"`** with values `fixed` | `compounding`,
   and `nav_initial: float = 1e8`. Default `fixed` reproduces today exactly.
2. In `compounding` mode the cost model reads NAV **as of the trade date** from
   the realised equity curve — `nav_t = nav_initial * cumprod(1 + r_net)` up to
   `t-1`.

   **The lag is the whole correctness question.** NAV at `t` must use returns
   strictly **before** `t`, or the cost of today's trade depends on today's
   return and the backtest is circular. Use `t-1`, and test it.
3. `costs.charges(...)` takes NAV per date instead of a scalar. Keep the scalar
   path intact for `fixed`.
4. **`theta.hash` moves.** `nav_mode`/`nav_initial` are `Protocol` fields and
   `asdict(self)` covers them, so adding them changes the hash for every protocol —
   which invalidates the `net_cost_runner` disk cache (a clean miss, so no stale
   reads) and orphans `data/results/ic_breakeven_{hash}.json`, silently degrading
   the soft economic gate to plain `|t_nw|`. **`scripts/qa_ic_ladder.py` must be
   re-run for the new hash** as part of this task, not after it.

## Validation

* `nav_mode: fixed` reproduces a stored full-cost result **bit-for-bit**.
* Circularity test: NAV on date `t` is unchanged when the return on `t` is
  perturbed. This is the test that matters most.
* A book that doubles shows measurably higher impact cost per unit turnover in the
  second half than the first, on the same trades.
* A degenerate path (NAV → 0) is handled: floor NAV and log, rather than dividing
  by zero and producing `inf` costs that silently blank the book.
* `qa_ic_ladder.py` re-run and the new `ic_breakeven_{hash}.json` present.

## Risks

* At ¥100M the impact term is ~0.001 bps — effectively inert either way. **So this
  change will likely move nothing at the current capital base**, and its value is
  that the loop stays correct as the base changes. Say that plainly in the report
  rather than presenting compounding NAV as an improvement it is not.
* Every published CSI300 number shifts if the default is ever flipped to
  `compounding`. Keep `fixed` as the default until the replay is complete, so the
  ablation is not confounded by a cost-model change.
