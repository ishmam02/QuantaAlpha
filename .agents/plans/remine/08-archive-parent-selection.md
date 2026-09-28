# 08 — Archive-based parent selection: exploit / explore / resurrect

**Depends on:** 06, 07. **Blocks:** 14.

## Why

Current selection is `parent_selection_strategy: best` — pure exploitation over a
pool that is destroyed at the end of every run. Two measured consequences:

* the search has hit **operator monoculture twice** (TS_MEAN), requiring manual
  de-priming both times;
* **crossover is worse than mutation** (median |rank_ic| 0.0067 vs 0.0114) yet
  received the best parents.

The user's requirement is an explicit exploration/exploitation balance, and one
more thing the current design cannot express at all: **re-using a dead factor when
its regime returns.**

## Goal

Parent selection from a behavioural **archive** spanning all mines, on a logged
budget split, with resurrection as a first-class source.

## Approach

**Archive** — `quantaalpha/pipeline/evolution/archive.py`. Bin every library factor
on descriptors that are already computable:

* **operator class** — `factors/operator_coverage.py`, `coder/factor_ast.py`;
* **horizon bucket** — lookback parsed from the expression (short/mid/long);
* **regime profile** — the sign pattern of `ic_crash` / `ic_rally`, **already
  computed** per factor in the tearsheets;
* **decay tier** and age.

A cell holds its occupants, its best-ever contribution, and its **trial count** —
the last is what makes the bandit below estimable.

**Three pools, fixed budget, default 50 / 30 / 20:**

1. **EXPLOIT (50%)** — best by EB-shrunk marginal contribution among *currently
   healthy* factors. `controller._shrunk_fitness` / `_rank_by_fitness` already do
   exactly this; reuse them unchanged.
2. **EXPLORE (30%)** — occupants of **sparsely populated** cells, regardless of
   fitness. This is the structural fix for monoculture: novelty is selected for
   directly rather than patched by de-priming a prompt after the fact.
3. **RESURRECT (20%)** — **hard-decayed** factors whose regime profile matches the
   *current* market state (`09`). `decay.due_for_retest` already supplies the
   candidate list. This implements "regimes change" as a mechanism instead of a
   comment.

**Selection within a pool is UCB over archive CELLS, not over individual factors.**
The reason is this repo's own measurement: individual fitness is noisy enough that
**3 of 4 verdicts flip across combiner seeds**, so a per-factor bandit is
estimating noise and will chase seed luck. Cell-level yield pools many trials and
is actually estimable. Score a cell by mean admitted-contribution with a
`sqrt(log N / n_cell)` exploration bonus; pick the factor within the cell by
shrunk fitness.

**Config:** `EvolutionConfig` gains `parent_pool_budget: dict[str, float]` and
`archive_descriptors: list[str]`. `parent_selection_strategy: "archive"` selects
the new path; every existing value keeps its current behaviour **byte-identically**,
so the arm is switchable and the old results stay reproducible.

**Logging:** the realised split, the cells drawn from, and the resurrection
decisions are written per mine — so the budget split becomes a measured parameter
rather than a guess.

## Validation

* Budgets are respected within rounding; a pool that cannot be filled degrades to
  the next pool **and logs loudly** (an empty RESURRECT pool is normal early on).
* Archive occupancy on the real library is reported: cell count, occupancy
  histogram, and the Gini of occupancy. If one cell holds most factors, that *is*
  the monoculture, quantified for the first time.
* `parent_selection_strategy: "best"` reproduces today's selection exactly.
* A synthetic test: a factor whose regime profile matches the current state is
  preferentially resurrected over one whose profile does not.

## Risks

* Descriptor choice determines what "novelty" means. Bad bins make the explore
  pool meaningless. Report occupancy before trusting it.
* Resurrection could re-admit a factor that is simply dead. It re-enters as a
  *parent candidate*, not into the book — admission and the active-set gate still
  apply, and that separation must not be blurred.
* UCB needs trial counts; early on every cell is under-sampled and selection is
  near-random. That is correct behaviour, not a bug — say so in the log.
