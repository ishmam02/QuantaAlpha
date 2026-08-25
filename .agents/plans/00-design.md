# Design: an evaluation the generator can learn from

**Status:** design doc. Executable steps live in `01-` … `09-`.
**Written:** 2026-08-18, from measurements taken the same day.

---

## The problem, stated as measurements

Every number below was measured on this repo, not assumed.

1. **The selection criterion is the portfolio.** `E_Θ` scores a candidate by building a
   long-only, 3%-capped, mean-variance book against a cap-weighted benchmark and
   measuring `net_ir`. Admission compares that book with and without the batch. The
   researcher-side question ("does this signal contain alpha") and the trader-side
   question ("does this improve my book") are the same number.

2. **That number is dominated by an uncontrolled size bet.** Equal-weight CSI300
   members minus the cap-weighted index reproduces the search's exact per-fold sign
   pattern with **zero alpha involved**: −9.81pp (2019), −13.05pp (2020), +6.04pp
   (2021). Of 2000 random 34-name books, only **9.4% / 13.9%** beat the index in
   2019 / 2020 versus **77.1%** in 2021. In 2020 the *median* CSI300 stock returned
   −19.45% while the index returned +27.21%.

3. **Consequence: the criterion prefers worse signals.** Same factors, same window:
   LightGBM produced **+76% rank IC** and **+110% ICIR** versus the ICIR combiner and
   delivered **less** `net_ir` (+0.341 vs +0.444 at |zoo|=150). A gate on `net_ir`
   rejects the better signal.

4. **The transfer coefficient is the binding constraint, and it cannot be fixed
   in the current construction.** `IR ≈ IC · √BR · TC`. Long-only, 3% cap, ~34 names
   from 300, benchmarked to a cap-weighted index. Fixing TC means optimising active
   weight `w − b`, which needs index weights — and qlib `cn_data` has **no market-cap
   or shares-outstanding field**. Verified. Optimising IC while TC is pinned is
   optimising the numerator of a product.

5. **Reported excess return is inflated by +4.25%/yr.** The book prices with adjusted
   closes (dividends reinvested); the benchmark is SH000300 price return. Measured
   dividend contribution: +5.26 / +4.46 / +3.03 pp for 2019 / 2020 / 2021.

6. **The generator learns from all of the above.** Feedback, diagnosis and refinement
   routing are all derived from the verdict, which is derived from `net_ir`. Whatever
   is wrong with the criterion is what the search is trained on.

---

## The split this design enforces

Two roles, two criteria, two cadences. Collapsing them is the root defect.

| | Researcher | Trader |
|---|---|---|
| Question | Does this signal carry alpha? | Does this book make money? |
| Book | Dollar-neutral long/short, unconstrained | Long-only, capped, benchmark-relative |
| TC | ≈ 1 (no constraint to destroy it) | structurally capped; unfixable without index weights |
| Metrics | neutralized RankIC, ICIR, positive-ratio, IC decay, quantile monotonicity, turnover, capacity | net return, IR, drawdown, capacity, cost attribution |
| Cadence | every candidate | periodically, on the selected library |
| Consumes | one signal | the library |

**The research claim is the long/short book.** Not a dodge: it is the standard research
formulation (Alpha101 / WorldQuant lineage), it is what the literature is comparable
against, and it is the only formulation in which TC is not the binding constraint given
this dataset. The long-only book is retained as a *reporting* artefact, not as the gate.

---

## The chain a factor must survive

Each link is a separate plan. A factor that fails a link is rejected **with the name of
the link**, which is the feedback the generator has never had.

```
raw expression
  → coverage + pathology            (exists)
  → winsorize / standardize          (02)
  → neutralize: size, industry, library residual   (02)
  → RankIC ≠ 0 on the neutralized signal           (04)
  → positive-ratio ≥ 0.50                           (04)
  → survives the horizon it is actually traded at   (04, IC decay curve)
  → quantile-monotone, not tails-only               (04)
  → implementable: turnover, capacity, concentration (04)
  → adds to the library at |ρ| < 0.60               (05)
  → survives full costs incl. borrow                (06)
  → survives multiple-testing correction            (08)
```

## What the generator must be told

Today the LLM receives fold-averaged scalars and a verdict string. It is never told
*which* link broke, *which* regime broke it, or *what was tried before that broke the
same way*. Plan `07` makes the rejection reason structured and the memory three-level
(archetype / generation / cycle), storing failed assumptions rather than only successes.

## Non-goals

- Chasing a headline number against papers that report gross, un-neutralized returns.
  The gap between a gross number and a full-cost number **is a result**; report both
  conventions (plan `09`) and explain the gap.
- Long-only benchmark-relative optimisation. Blocked by missing index weights. Revisit
  only if a market-cap source is added.

## Order and dependencies

```
01 sample + splits ──┬── 02 neutralization ── 04 tear sheet ── 05 selection ── 07 feedback
                     └── 03 long/short book ──┘                     │
                                    06 full cost model ─────────────┤
                                                  08 statistics ────┴── 09 holdout
```

`01` first: everything downstream measures on the sample. `09` last and once.
