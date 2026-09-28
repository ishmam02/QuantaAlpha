# 09 — The market-state report (computed, leak-free, descriptive)

**Depends on:** 02. **Blocks:** 08 (regime matching), 10, 11.

## Why

The direction agent must decide what to mine "based on the current market up to
this point". The user's decision: **computed market state only — no news corpus.**

Two reasons this is the stronger choice, not merely the cheaper one:

1. **It is leak-free by construction.** There is no text corpus to be silently
   revised, backfilled or timestamped wrong. Every number is computed from the
   panel at dates ≤ T by the same code that prices the book.
2. **It is genuine discovery.** The 2023 rotation the paper describes *is* a rise
   in cross-sectional dispersion plus an inversion of the size-decile spread. The
   agent reads those as numbers with no label attached. Nobody tells it "the
   market rotated to small caps" — which is precisely the constraint the user set.

## Goal

A dated, reproducible, purely descriptive statistics block over data ≤ T, usable
by the direction agent (`10`), the decay agent (`11`), and regime matching (`08`).

## Approach

**`quantaalpha/eval/market_state.py`** — `compute(asof, panel, theta) -> MarketState`.

Every series is computed to `asof` inclusive and reported as a **level plus a
trend** (current value, 1-year change, and percentile within the available
history), because a level alone cannot express "this is changing":

* cross-sectional return **dispersion** (daily σ across names) — the direct
  measure of how much alpha is available at all;
* **breadth** — fraction of names beating the cap-weighted index;
* **size-decile spread** — top vs bottom decile by market cap (`market_cap.parquet`);
* realized **volatility** level and its term structure (20d vs 120d);
* **return autocorrelation** at 1/5/20 days — whether short-horizon returns
  continue or revert, *measured* rather than assumed (the momentum-vs-reversal
  prior this repo explicitly forbids asserting);
* **turnover** level and trend;
* **overnight-gap share** of total return;
* **industry dispersion** (`industry.parquet`);
* mean pairwise **correlation** — a crowding proxy;
* **the book's own state:** the tier census, and per-archetype IC now vs at
  admission (from `07`).

**Rendering rules — load-bearing:**

* numbers and dates only, no interpretation;
* **no market name, no index name, no ticker, no event, no sector name in prose** —
  industry dispersion is reported as a statistic, not as "technology led";
* no imperative and no remedy; closes with the house formula
  ("what this implies is yours to determine");
* inherits the **existing** `_market_context()` disclosure clamp: nothing after the
  clamp date is computed *or* named, and the clamp **fails closed** (on any error,
  disclose nothing rather than risk a wider window).

**Determinism:** `compute` is a pure function of `(asof, panel, theta)`. Cache to
`data/results/market_state/{asof}.json` and assert equality on recompute.

## Validation

* **Leakage test (mandatory):** `compute(T)` is bit-identical whether the panel is
  loaded to `T` or to `T + 2 years`. Any dependence on data after `asof` is a leak
  and this test is the guard.
* **Discovery test:** `compute("2023-12-31")` shows a materially different size
  spread and dispersion from `compute("2019-12-31")` — the regime shift is visible
  in the statistics. If it is not, the chosen statistics cannot support the claim
  and must be revised before `10` is built on them.
* Golden-file test on the rendered block: no 4-digit year in prose, no index name,
  no imperative verb.
* Every field is finite or explicitly `null`; a `NaN` rendered into a prompt as
  "nan" is a silent failure.

## Risks

* **Statistic choice is a design prior.** These nine are chosen to be regime-
  descriptive and market-agnostic; document *why each one*, and resist adding one
  because it would have flagged 2023 — that would be hindsight smuggled in as
  feature engineering.
* `market_cap.parquet` was measured to be unreliable for reconstructing index
  weights (that reconstruction failed validation). It is fine for a *decile
  spread*, which needs only a ranking — state that distinction explicitly so
  nobody later reuses it for weights.
