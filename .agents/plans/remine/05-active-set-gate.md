# 05 — The active-set gate: capped, orthogonal, tier-aware

**Depends on:** 02, 04. **Blocks:** 14.

## Why

The library grows without bound by design (every factor ever mined survives). The
**active set** — what the combiner actually gets as features — cannot. At ~150
factors per mine, five re-mines is ~750 features on ~300 names/day, where
LightGBM fits noise and the refit and signal-cache costs grow linearly forever.

The user's requirement is a *gated* cap: **"a good lib where no two bets are the
same."** Every primitive for that already exists and two of them are switched off.

## Goal

A single, auditable selection run per refit that takes the unbounded library to a
capped, mutually-distinct, tier-respecting active set — and records why each
factor is in or out.

## Approach

**`quantaalpha/eval/active_set.py`** (new). `select(asof, library, states, theta)`
runs a gate chain, reusing existing code at every step rather than scoring anew:

1. **Tier filter** — `hard_decay` excluded from scoring entirely; `soft_decay`
   passed through with `p = 0.5` for `04`; `baseline_missing` excluded (never
   defaulted).
2. **Pathology filter** — the existing `admission.check_pathology`: coverage floor,
   zero variance. Also rejects the type-invalid expressions that currently become
   silent NaN columns (a cross-sectional reducer feeding a per-instrument TS
   operator — `MEAN`/`STD`/`MEDIAN` collapse the instrument level and `TS_*` then
   raises `KeyError: 'instrument'`; 7 of 150 in one library).
3. **Orthogonality — the "no two bets the same" gate.** **Set `gates.rho_bar`**,
   which currently defaults to `None` so the duplicate check in
   `check_pathology` can never fire. Enforce `rho_max ≤ rho_bar` against the
   *accepted-so-far* set, greedily, highest-contribution first. Reuse the existing
   across-batch Spearman cache; do not recompute a full pairwise matrix per refit.
   Cross-mine near-duplicates are the *expected* failure once several mines'
   output coexists, and this is the gate that catches them.
4. **Capacity + replacement** — `admission.capacity` (default **200**): once full a
   candidate enters only by displacing the incumbent it beats, which is already
   `admission.decide`'s replacement branch.
5. **Re-measured contribution** — `admission.should_evict` against the set *as it
   now stands*. This repo's own finding: "a factor that was additive when the zoo
   held twelve members can be redundant at a hundred and fifty."

**Output:** `data/results/active_set/{asof}.json` — the selected `factor_id`s with
inclusion `p`, plus **every rejection with its gate and its number**. "What is
trading today and why" must be answerable months later.

## Validation

* Cap is respected exactly; no member violates `rho_bar` against any other member.
* Turning the orthogonality gate on **reduces** the set on a real library —
  report before/after counts and the median pairwise ρ both ways. If ρ does not
  fall, the gate is not doing anything and something is mis-wired.
* A hard-decayed factor never appears; a soft-decayed one appears with `p = 0.5`.
* Determinism: same inputs ⇒ same set, same order.
* Selecting from a single-mine library reproduces approximately today's book, so
  the change is attributable to the gates rather than to a new scorer.

## Risks

* **This will reject factors that are currently trading.** That is the intent, but
  it changes library composition and therefore every downstream number — report it
  as its own effect, separate from the decay loop's.
* `rho_bar` is a new free parameter. Sweep it in `12`; do not tune it against
  final-test.
* Greedy selection is order-dependent. Fix the order (contribution-descending,
  ties by `factor_id`) and state it.
