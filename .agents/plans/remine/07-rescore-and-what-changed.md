# 07 — Re-score prior trajectories on new data + the What-Changed digest

**Depends on:** 06. **Blocks:** 08, 10.

## Why

The user's requirement: *"When a new mining is done all the old trajectories after
being refreshed and tested on the newer data will be used as feedback to guide the
new mining so what stopped working is clear to the generation loop."*

This is also the **second channel by which the system discovers a regime shift on
its own**. It will observe that an entire archetype died, with nobody naming the
cause — which is exactly the constraint that the 2023 narrative must not be fed.

## Goal

Before each re-mine: every prior trajectory re-scored on data up to the new T, and
a digest of *what changed* handed to the generator as **measurement only**.

## Approach

**Re-score** — `scripts/qa_rescore_trajectories.py`:

* For every factor in the global store with a resolvable signal, recompute its
  solo metrics on the **new** window: rank IC, `t_nw`, `ic_pos_frac`, `ic_crash`,
  `ic_rally`, turnover, coverage.
* Reuse the existing scoring path (`_cross_sectional_corr`, `newey_west_t`) — a
  second IC definition is how the 3× btv2/Θ gap happened.
* Write `data/results/rescore/{asof}.json`: for each `factor_id`, `then` (at
  admission), `now`, and the delta.
* **Coverage gate:** refuse to emit if <90% of expected signals resolve.

**What-Changed digest** — `quantaalpha/pipeline/evolution/what_changed.py`:

Groups factors into **archetypes** (operator class × horizon bucket, from
`operator_coverage.py` / `factor_ast.py`) and reports per archetype: n, median IC
then, median IC now, the delta, and how many crossed into soft/hard decay.

Rendered into the direction prompt as a block that follows the **existing** house
style of `_build_reseed_digest`, `_what_worked_well_block` and
`_regime_conditional_block`:

* states counts and numbers;
* **names no market, no year, no event, and no remedy**;
* closes with "What this implies for the next direction is yours to determine."

This is a hard rule in this repo (`qa-prompts-diagnose-never-prescribe`,
`qa-no-market-specific-priors-in-prompts`) and the single easiest place in the
whole plan to violate it. A line like "momentum factors stopped working in 2023"
hands the system the answer and invalidates the entire discovery claim. The
permitted form is: "archetype TS_MEAN/20d: n=14, median rank IC 0.041 → 0.008."

**Suppression floor:** an archetype with n < 6 is not reported, matching
`_regime_conditional_block`'s existing floor — a lesson must never be asserted
from a handful.

## Validation

* Re-scoring a factor over its *original* window reproduces its stored metrics to
  ~1e-9. If not, the two scoring paths have diverged and that is the bug.
* A golden-file test on the rendered digest asserts the absence of: any 4-digit
  year, any index or ticker name, and any imperative verb ("use", "avoid",
  "prefer", "focus on", "switch to").
* On the real library the digest reproduces the known 2023 collapse **as numbers**
  — a measurable IC drop in some archetypes — without the string "2023" appearing.
* Archetypes with n < 6 are absent.

## Risks

* **Prompt-prior leakage is the main risk** and it is subtle. The golden test is
  the guard; extend its banned-token list whenever the digest changes.
* Re-scoring the whole library at every re-mine is expensive and grows with the
  library. Cache by `(factor_id, window)` and only recompute the new tail.
