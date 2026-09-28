# 04 — Soft decay as feature-subset bagging

**Depends on:** 02. **Blocks:** 05, 14.

## Why

The specified soft-decay action — "weight capped at 50% of model weight" — cannot
be implemented literally on the production combiner:

* gradient-boosted trees are **invariant to monotone feature scaling**, so halving
  a column's values changes nothing;
* the LightGBM path produces **no per-factor weights at all** (`library.py` states
  this in `_per_factor_metrics`: attribution exists only for the ICIR combiner).

The ensemble needed to express a fractional weight **already exists**:
`combiner.fit_predict` fits one model **per seed and averages the predictions**
(`combiner.py:734-750`, over `theta.combiner.seeds`).

## Goal

A factor's column is included in only ⌈p·K⌉ of the K seed models, giving a true
fractional influence: `p = 1.0` healthy, `0.5` soft, `0.0` hard — reproducing
`decay.WEIGHT_MULTIPLIER` in the only currency a tree ensemble has.

## Approach

**`quantaalpha/eval/combiner.py`**, LightGBM branch only:

1. Accept an optional `inclusion: dict[str, float] | None` (factor expression or
   `factor_id` → p). `None` ⇒ **byte-identical to today**, which is the
   compatibility requirement: every existing result must reproduce exactly.
2. Inside the existing per-seed loop, derive the column mask deterministically:

   ```
   include = rank_of(seed, factor_id) < ceil(p * K)
   ```

   where `rank_of` is a stable hash of `(factor_id, seed)` — so the *same* factor
   is dropped from the *same* subset of seeds on every run. Reproducibility is
   non-negotiable here; `random` would make the book unreproducible.
3. Drop the masked columns from that seed's `x_train` **and** from the matrix it
   predicts on. Base features are never masked.
4. Extend the cache key: `_PREDICTION_CACHE` is keyed `(zoo_hash, cand_id,
   theta.hash)` and must also carry a hash of `inclusion`, or a decayed book will
   silently serve a stale full-weight prediction. This is the highest-risk detail
   in the task.
5. Record the realised per-factor inclusion count per refit for the ledger.

**By-product worth keeping:** the spread of predictions across seed models is a
free per-date confidence signal. Persist it; do not wire it into anything yet.

## Validation

* `inclusion=None` reproduces a stored prediction **bit-for-bit** on a fixed
  window. Assert exact equality, not a tolerance.
* `p = 0.0` for every factor gives a prediction identical to fitting with those
  columns absent.
* `p = 0.5`, K = 5 ⇒ each affected factor appears in exactly **3** models
  (`⌈0.5·5⌉ = 3`). Pin the rounding convention in the test and state it in the
  docstring: ⌈⌉ means soft decay retains a bare majority of the ensemble rather
  than a minority. With an odd K there is no exact half, and rounding **up** is
  the conservative choice — a decaying factor is demoted, not silently evicted.
* Same `(factor_id, seed)` ⇒ same mask across processes.
* Cache test: fit with `p=1.0`, then `p=0.5`, and confirm the second is **not**
  served from cache.

## Risks

* **Cache collision is the dangerous failure** — it would present a full-weight
  book as a decayed one, silently. The test above is mandatory.
* K is currently small (5 seeds), so p is quantised to fifths. Fine for
  `{0, 0.5, 1}`; if finer tiers are ever wanted, K must rise and cost scales
  linearly.
