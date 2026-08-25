# Plan: across-batch cache for the two O(zoo²) Spearman builds

## Context (measured, 2026-08-25)

The live glm-5.2 mine (`EXPERIMENT_ID=meanvar_20260825_123942`, frozen protocol
`fbefcb65f408aee0`, `admission.mode: standalone`, `combiner.seeds: [42]`,
`n_factors: 1` per batch) slows as the zoo grows:

| zoo | eval-only median gap (min/decision) |
|---|---|
| 14 | 3.2 |
| 19 | 6.0 |
| 20 | 5.8 |
| 21 | 6.8 |

Eval-only slope ≈ **22 s / zoo-unit** (intra-batch gaps, LLM-gen excluded).
Projection: **~14 min/decision at zoo 40, ~21 min at zoo 60.** Left alone this
throttles the run as it fills.

**The creep is O(zoo²), not O(zoo).** With only zoo 14→19 you cannot separate
the two ((19/14)² = 1.84× predicts 3.6→6.6 min, inside noise of the observed
6.9), but the *pair-count* delta is unambiguous: the quadratic Spearman builds
add ~165 pairs going 14→19 vs ~10 for the linear terms — a 16× ratio that the
wall-clock tracks. The "linear" appearance is a two-point artifact.

### The two residual O(zoo²) builds (both confirmed in current code)

1. **`operator.py:223-243` — the `m_effective_rank` METRIC.** A double
   `for _i/_j` loop calling `_cross_sectional_corr(..., "spearman")` over every
   `(zoo ∪ cand)` pair → full `(zoo+1)²` abs-Spearman matrix + `eigvalsh`.
   **No cache. Not gated by `skip_book`. Runs every batch.** Added in the same
   uncommitted diff as the marginal-er gate cache but given no cache of its own.
   At zoo 20: ~441 spearman pairs/batch.

2. **`net_cost_runner.py:1192` — the marginal-er GATE's `spearman_abs_matrix(_repo_sigs)`.**
   O(zoo²) pairs. The `_mer_repo_cache` (added this session) caches the repo-repo
   block **across candidates within one `_decide_standalone` call** — but with
   `n_factors: 1` there is only one candidate, so the cache is built and used
   once then discarded: **it is rebuilt every batch.** At zoo 20: ~400 pairs.
   (The cache still helps within the call — `er_held` and `er_all` share `R_repo`
   instead of building it twice — which is the 282s→190s drop measured at zoo 14.
   It does NOT help across batches.)

Together ~860 spearman pairs/batch at zoo 20 — the dominant share of the
~5.7 min/decision. The combiner itself is only O(zoo) (linear feature width,
~0.55 s/factor per the 2026-08-14 profile) and is NOT the lever here.

### What is NOT on the hot path (do not optimize for these)

- The 5-seed `_decide_marginal` loop (`net_cost_runner.py:1241-1360`) runs only
  under `mode: marginal_contribution` (the *other* protocol, hash
  `9fc612172cc6e776`). The live mine is `standalone`; `delta_per_seed == []` in
  every ledger record. **Not executed.**
- The LOO eviction `_prune_by_contribution` (`:1645-1739`) fires every
  `evict_every: 20` batches, fork-parallelized. **Off the per-batch path.**
  Deferred (see Open items).

## Design

**Lift both O(zoo²) builds onto instance-level caches keyed by `zoo_hash`,
invalidated when the zoo changes.** This is the exact pattern
`operator._baselines` (`operator.py:519-543`, keyed `(zoo_hash, eval_window,
report)`) already uses and is NOT cleared per batch. On a **reject batch** (zoo
unchanged — ~60% of decisions in the current ledger: 30 rejects / 50 records)
both builds become an O(1) cache hit plus an O(zoo) candidate-vs-zoo tail. On an
**admit/replace/evict batch** (~40%) the zoo composition changes → `zoo_hash`
changes → one rebuild, then cached. Net: the O(zoo²) term fires on ~40% of
batches instead of 100%; the average per-decision growth slope drops to ~40% of
current and absolute time roughly halves at current zoo.

The key insight that makes this safe + cheap: **the zoo×zoo |Spearman| block is
a pure function of the (zoo signals, panel/eval-window).** `zoo_hash` (sha256 of
sorted expression md5s) + the eval-window is an exact key — same expressions on
the same panel ⇒ identical aligned signals ⇒ identical Spearman ⇒ identical
eigenvalues ⇒ identical `effective_rank`. The expensive-to-compute artifact (the
R matrix, `zoo²` floats = ~12 KB at zoo 40) is trivial to store; we do NOT need
to cache the aligned signal frames (the ~115 MB memory hazard the combiner
wholesale-clear at `:687` was built to avoid).

`effective_rank_cached(R_repo, repo_signals, extra_signals)` already exists in
`metrics.py` and does exactly the "reuse the zoo block, compute only the
extra-involving entries" math — it's what the gate uses today. The metric site
just needs to use the same helper against a persisted zoo-block cache.

### Task 1 — Lift the gate's `_mer_repo_cache` to `self`, keyed by `zoo_hash`

**File:** `quantaalpha/factors/net_cost_runner.py`

- Replace the per-call local `_mer_repo_cache = None` (~line 817) with an
  instance attribute initialized in `__init__`: `self._mer_repo_cache:
  tuple[str, np.ndarray, dict] | None = None` holding `(zoo_hash, R_repo,
  repo_sigs)`.
- In the gate block (~1192): before building, compute the current `zoo_hash`
  (the runner already has the zoo signals / a zoo hash available; reuse the same
  `zoo_hash` the operator uses, or compute via `combiner.zoo_hash`). If
  `self._mer_repo_cache is not None and self._mer_repo_cache[0] == zoo_hash`:
  reuse `(R_repo, repo_sigs)` from the cache. Else build
  `spearman_abs_matrix(repo_sigs)` and store `(zoo_hash, R_repo, repo_sigs)`.
- **Invalidation:** the existing replace `pop` (~1149) and any
  admit/evict that changes the zoo already change `zoo_hash`, so a stale cache is
  naturally missed on the next batch. Add an explicit
  `self._mer_repo_cache = None` at the admit and evict sites for clarity/defensiveness
  (same place `_mer_repo_cache = None` is set on replace today).

**Validation (manual):** run a short smoke; confirm (a) gate verdicts on a fixed
  batch are bit-identical to pre-change (the cached path returns the same R_repo
  the local path built — `effective_rank_cached` is already documented
  bit-identical to the uncached path, `metrics.py:314-318`); (b) on a reject
  batch, `spearman_abs_matrix` is no longer re-run (add a temporary print /
  counter and observe it fires only when `zoo_hash` changes); (c) `m_effective_rank`
  and `marginal_er` in the ledger are unchanged.

### Task 2 — Cache the operator's `m_effective_rank` zoo-block; refactor to `effective_rank_cached`

**File:** `quantaalpha/eval/operator.py` (`:223-243`)

- Add an instance cache `self._er_zoo_cache: tuple[str, str, np.ndarray, list] | None
  = None` holding `(zoo_hash, eval_window_key, R_zoo, zoo_keys)` — the zoo×zoo
  abs-Spearman block, keyed by `(zoo_hash, eval_window)`.
- Replace the inline double loop with: if the cached `(zoo_hash, eval_window)`
  matches, reuse `R_zoo` + `zoo_keys`; else build the zoo×zoo block once (the
  same `_abs_spearman` / `_cross_sectional_corr("spearman")` semantics the inline
  block uses at `:231-233`, which equal `metrics._abs_spearman:280`). Then add
  only the candidate-vs-zoo column/row (O(zoo)) and the candidate self-term, form
  the full `(zoo+1)²` matrix, `eigvalsh` → `effective_rank`. Store the zoo block.
- Keep the `try/except: pass` guard (never lose a batch to a metric).
- **Invalidation:** keyed by `zoo_hash` + `eval_window`; both change naturally on
  admit/replace/evict and on walk-forward window rolls, so no explicit
  invalidation needed beyond the key check. (Mirror `_baselines`'s keying.)

**Validation (manual):** on a fixed zoo, `m_effective_rank` /
  `m_rank_density` / `m_book_n_factors` bit-identical before vs after (run
  `op.evaluate` on the same candidates+zoo with the old inline block vs the
  cached helper in one process; assert `np.allclose(..., atol=0, rtol=0)` on the
  metric, or to the documented `9.4e-13` tolerance floor if float order differs).
  Confirm the zoo×zoo block is built once and reused across reject batches (temp
  counter).

### Task 3 — (optional, after 1+2) Share the zoo×zoo block between metric and gate

The operator metric (Task 2) and the gate (Task 1) compute the **same** zoo×zoo
abs-Spearman block on the **same** zoo signals on the **same** panel. After both
have their own cache, deduplicate: compute the block once per `(zoo_hash,
eval_window)` and hand the same `R_zoo` to both. Saves the second of the two
quadratic builds entirely on cache-miss batches.

**Risk:** the two sites currently live in different objects (`EvaluationOperator`
vs `NetCostRunner`), so sharing needs either a shared cache object passed between
them or lifting the block into a small module-level keyed cache (bounded to 1
entry — the current zoo). Lowest-risk version: a 1-entry module-level
`_ZOO_SPEARMAN_CACHE` in `metrics.py` keyed `(zoo_hash, eval_window)` with the
block, used by both `effective_rank_cached` callers.

**Validation (manual):** both `m_effective_rank` and `marginal_er` unchanged;
  on a cache-miss batch exactly ONE zoo×zoo build runs (not two).

### Task 4 — (optional, smaller) Cache `align_signal` per `(signal_hash(expr), panel-key)`

**File:** `quantaalpha/eval/data.py` (`align_signal`/`_align`) and
`operator.py:163`. Suspect #3 — O(zoo) T×N reindex per evaluate, uncached. An
operator-level dict keyed `(signal_hash(expr), panel.dates[0], panel.dates[-1])`
removes it. **Memory-bounded:** cap the dict to the current zoo size + a small
margin, or key-evict on zoo change. Lower priority (the O(zoo) term is ~10 pairs
of the ~860 at zoo 20).

## Behavior preservation (the hard constraints)

- **Frozen protocol hash `fbefcb65f408aee0` stays byte-identical.** The caches
  are runtime behavior keyed on `zoo_hash`/`eval_window`/`signal_hash` — not
  `Theta` fields. `theta.hash = sha256(canonical asdict(theta))` is untouched.
  (Same reason `QA_MIN_MARGINAL_ER` being env-driven left the hash unchanged.)
- **Gate verdicts bit-identical.** `spearman_abs_matrix` and `effective_rank_cached`
  are pure functions of the zoo signals; `zoo_hash` is an exact key (same
  expressions ⇒ same aligned signals ⇒ same Spearman). The cached `R_repo` is the
  same matrix the local path builds. `marginal_er = er_all - er_held` unchanged.
  The 2026-08-14 measurement already documents `effective_rank_cached` is
  bit-identical to the uncached path (eigenvalues permutation-invariant).
- **`m_effective_rank` metric bit-identical** (or within the 9.4e-13 float floor
  if the candidate-column computation reorders arithmetic — assert and confirm;
  prefer restructuring so the zoo-block + cand-column math is identical to the
  current inline loop, making it bit-identical, not just close).
- **Memory:** cache the **R matrices** (zoo² floats, ~12 KB at zoo 40) and
  **keys** — NOT the aligned signal frames. Do not remove the
  `_clear_combiner_cache()` at `net_cost_runner.py:687` (it bounds the ~1500-wide-frame combiner cache, a separate hazard). The new caches are tiny by comparison.

## Heritage / safety (separate commit, do first)

**The entire optimized eval stack is uncommitted — 0 git commits.** Confirmed:
`git log -S` for `_evict_worker_eval`, `_rank_rows`, `reduceat`,
`effective_rank_cached` all return 0 commits (only `_decide_marginal` was
committed, in 826a214). The working-tree diffs: net_cost_runner +1321, metrics
+320, combiner +229, operator +253, riskmodel +155, portfolio +87 lines
uncommitted. The live mine runs them only because the install is editable and
the files predate the process start — a `git checkout` or lost working tree
collapses mine performance AND removes the marginal-er gate.

**Action:** commit the current optimized eval stack as-is (it's verified:
1.78–4.7× on the 2026-08-14 opts, G1-G6 + regression green on the marginal-er
fix, frozen hash unchanged) BEFORE the new caching work, so the new plan starts
from a committed baseline. This is not a speed lever (the mine already runs it)
but a safety imperative.

## Open / deferred

- **LOO eviction** (`_prune_by_contribution`, every 20 batches): each of the
  |zoo| re-pricings rebuilds the full zoo×zoo matrix via `op.evaluate` → the
  metric site. Once Task 2's operator-level zoo-block cache is parameterized by
  the *reduced* `zoo_hash` of each held-out subset, the eviction round drops
  O(zoo³)→O(zoo²). Medium-high risk (the `m_net_ir` verdict `should_evict` reads
  must stay bit-identical across the reduced zoos). Separate plan.
- **Combiner cache wholesale-clear (`:687`):** replacing `clear()` with a
  bounded LRU keyed `(zoo_hash, cand_hash, theta.hash)` would let the zoo-baseline
  prediction survive a reject batch — but the book is priced only 1/25 batches
  (`book_eval_every: 25`), so the benefit is small. Not worth the memory-cap work
  now.
- **Reducing `test_seeds` / `combiner.seeds`:** not on the standalone hot path;
  moot for this mine.

## Verification (whole-plan)

1. `python tests/eval/test_marginal_er_gate.py` — G1–G6 still pass (gate
   behavior unchanged).
2. Frozen-hash check: `python -c "from quantaalpha.eval.protocol import
   load_protocol as l; print(l('quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml').hash)"`
   → still `fbefcb65f408aee0`.
3. Bit-identity: `op.evaluate` on a fixed zoo+candidates, old inline vs cached
   helper in one process → `m_effective_rank`, `marginal_er`, gate verdicts
   bit-identical (assert `atol=0` or document the float floor).
4. Live cadence: restart the mine pinned to the same `EXPERIMENT_ID` (zoo
   rehydrates), confirm per-decision eval-only gap drops on reject batches and
   that the slope vs zoo flattens (re-measure the zoo 14→21 gap table; expect the
   reject-batch median to fall toward the O(zoo) combiner floor while
   admit/replace batches stay comparable).