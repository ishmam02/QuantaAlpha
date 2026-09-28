# 12 — Dynamic split selection per re-mine (+ parameter sweeps)

**Depends on:** 01. **Blocks:** 14.

## Why

A single fixed train/valid block can sit entirely inside one regime, so the model
is selected on a market that no longer exists by the time it trades. The existing
`Splits.walk_forward_folds` already handles this correctly — expanding windows,
purge and embargo scaled by the label horizon — but the *span* is fixed by hand in
a YAML and never revisited.

Expanding versus rolling has a real argument on each side, and this repo has
already written down the expanding one ("dropping early history would confound
'later regime' with 'less data'"). The counter-argument is the whole motivation for
this plan: an 18-year expanding window dilutes a regime shift to one year in
eighteen. Per the user's decision, the mine **fits both and picks by validation**
rather than assuming.

## Goal

Each re-mine chooses its own training span empirically, records the choice, and
accumulates an answer to the expanding-vs-rolling question across the replay.

## Approach

**`quantaalpha/eval/split_selector.py`** — `choose(asof, theta, candidates)`:

1. Build candidate spans: **expanding** from the earliest data, and **rolling** at
   N ∈ {8, 10, 12} years.
2. For each, derive walk-forward folds with the **existing**
   `Splits.walk_forward_folds(n, horizon)` — do not reimplement purge/embargo.
3. Fit the combiner per fold and score on each fold's validation window.
4. Select the span with the best fold-averaged score; **record every candidate's
   score**, not just the winner.
5. Return a `Splits` for the mine to use.

**Hard guardrails:**

* `final_test` is never read, never scored, never used in selection. This is the
  property that keeps the replay honest.
* The disclosure clamp in `_market_context()` derives from `theta` and so
  recomputes against the *chosen* folds automatically — add a test proving it
  moves when the span moves, because a stale clamp would disclose the selection
  window.
* `valid` is **inert on the LightGBM path**: `lgb.train` is called with no
  `valid_sets`, so `early_stopping_round` never fires and all rounds always run.
  Selection here therefore uses an explicitly scored validation window, not
  LightGBM's internal early stopping, which does not exist. State this in the
  docstring or someone will "fix" it later.

## Parameter sweeps (same task, because they share the harness)

Run once on the **validation** windows only, never on `final_test`:

* **refit cadence** — monthly (default) vs quarterly vs annual, reported **net of
  cost**, because a fresher model trades more and monthly is a default rather than
  a finding;
* **active-set cap** — 100 / 200 / 300 (`05`);
* **`rho_bar`** — the orthogonality threshold (`05`);
* **parent budget split** — 50/30/20 vs 70/20/10 vs 40/40/20 (`08`).

Output one table per parameter with the metric and its spread across folds. Where a
sweep cannot resolve a difference, **say so** and keep the default rather than
picking the point estimate's winner — this repo has already been burned by
single-point estimates that flipped across seeds.

## Validation

* Selection never touches `final_test` — assert by instrumenting the window
  accessor during a selection run.
* A synthetic series with a planted regime break at a known date selects the
  rolling span; a stationary synthetic series selects expanding. If it cannot
  distinguish those two, the selector is not working.
* The clamp test above.
* Determinism for a fixed `asof`.

## Risks

* Fitting several spans multiplies combiner fits per mine. Fits are the cheap part
  relative to generation, but confirm against `01`'s envelope before launching.
* Selecting a span on validation is itself a selection step and adds to the trial
  count. Feed it into `n_tests` so the DSR bar accounts for it; quietly omitting it
  would understate the multiple-testing burden.
