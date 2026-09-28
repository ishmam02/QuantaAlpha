# 10 — Direction-selector agent + temporal-leakage controls

**Depends on:** 07, 09. **Blocks:** 14.

## Why

Re-mining with the *same* human-typed direction cannot produce new factors — the
direction is the search region, and an unchanged region re-explores exhausted
space. So each re-mine needs its own directions, derived from the market as it
stands at T and from what has stopped working.

That immediately raises the question the user asked: **can temporal leakage be
removed completely while still using the latest model?** No. The published position
is blunt about it — a model with a 2025 cutoff has already seen how the 2010-2020
period unfolded, and that bias "lives inside the model's weights and is invisible
to data-pipeline audits." Prompt-level cutoffs do not retract pretraining.

What *is* achievable, and what this task builds, is a bound on what leakage can do
plus a measurement of whether it happened.

## Goal

An agent that proposes initial directions from the `09` market-state report and the
`07` What-Changed digest, under enforced specificity limits, with a leakage audit
attached to every invocation.

## The four layers (design doc §5, restated as build steps)

**L1 — Context hygiene.** Nothing dated after T enters context. `planning._market_context()`
already does this well: it discloses only the train window, clamps to the day
before the **earliest fold's** validation day, fails **closed**, and carries an
explicit knowledge-cutoff block banning post-T events, post-T factor literature and
hindsight phrasing. Extend it to carry the market-state report; change nothing
about the clamp. This is the formal *temporal non-interference* property — the
output must not depend on any input after T — and `09`'s panel-window test is its
data-side proof.

**L2 — Output audit.** A validator rejects a direction naming a dated event, a
ticker, an index, an index level, a 4-digit year, a named anomaly, or a sector by
name. Mechanical, cheap, catches the obvious cases. Rejection triggers a
regeneration with the violation quoted back, up to N attempts, then **fails loudly
rather than substituting a canned direction** — the failure mode `_no_directions`
already exists to prevent (canned fallbacks are a human prior wearing the system's
credit).

**L3 — Structural containment.** This is the argument that makes the whole approach
defensible. A direction selects a **search region**; every factor is then selected
by point-in-time measurement on data ≤ T. Leakage can therefore only *misallocate
search budget* — it cannot manufacture a result — **provided the direction cannot
encode an answer.** So directions are constrained to name **mechanisms and
measurement targets** and nothing else. Enforce a specificity budget: a direction
may name operator families, horizons and economic mechanisms; it may not name
instruments, dates, events, or a numeric threshold to aim at.

### L4 — Measurement

**The strongest control is the IC decay curve, and it has already run.**

If the generator had exploited post-T knowledge, the resulting factors would hold
up in the leaked period. Measured on the existing library (mined under a context
clamped to ~2012), per-year Rank IC is:

```
2017   2018   2019   2020   2021   2022   2023   2024   2025
.1104  .0898  .0587  .0543  .0504  .0524  .0160  .0288  .0469
```

IC decays **monotonically with distance from the fit window** and bottoms out in
**2023** — the single most-documented regime break in recent A-share history, and
therefore exactly where a leaking model's knowledge is densest. A library built on
hindsight does not fail hardest precisely there. **This is direct evidence that
parametric leakage bought the existing system nothing.**

There is a structural reason it could not. **Admission is itself an anti-leakage
mechanism:** factors are selected on data ≤ T, so a factor whose edge exists only
in the future scores badly on the past and is rejected. The search mechanically
filters out the kind of factor leakage would produce.

That narrows the residual risk sharply. Leakage can only survive the gate if the
leaked knowledge points at a factor that **also** worked before T — in which case
the factor is genuinely good on point-in-time data and the harm is not financial.
What remains is a **credit-assignment** problem: the system would look more capable
than it is, having been handed an idea rather than reasoning to it. Worth
detecting; not a threat to the returns.

**So the decay-shape test is the primary diagnostic**, run per arm in `14`:

* **Compare the IC decay *shape* between arms.** If arm B (agent directions) shows
  materially **flatter** decay than arm A (fixed direction) — its factors
  mysteriously holding up in later years — that is the leakage signature. If both
  decay alike, leakage bought nothing, whatever the placebo says.
* Report the decay slope and the year-of-trough per arm alongside the headline.

**Why the other three controls still run.** The evidence above is about the *old*
configuration, where the model received little more than a window range. This
design hands it substantially more period-identifying information — nine regime
statistics plus a What-Changed digest naming which archetypes died. That is a far
richer fingerprint, so the finding must be **re-established, not assumed to carry
over.** The three controls below bound the channel; the decay curve measures the
outcome:

1. **Date-relabel placebo (decisive).** Same market-state report, different date
   label. Compute the similarity of the returned direction sets. High similarity ⇒
   the agent is keying on the **state**; materially different ⇒ it is keying on the
   **date**, i.e. parametric recall. This is a black-box behavioural audit for
   parametric hindsight and it needs no model internals.
2. **State-shuffle control.** A market-state report from a *different* period,
   labelled T. Directions should track the state, not the label — the mirror of
   test 1, and it catches an agent that ignores the report entirely.
3. **The arm ablation** (`14`). If the agent's advantage concentrates on famous,
   memorable regimes and vanishes elsewhere, that pattern *is* leakage showing up
   as a result.

## Approach

**`quantaalpha/pipeline/direction_agent.py`** —
`propose(asof, market_state, what_changed, n) -> list[str]`.

* Reuses the existing prompt/parse/retry scaffolding of
  `planning.generate_informed_directions` (JSON extraction, attempt loop,
  loud-empty-on-failure). Do not write a second LLM client path.
* Prompt sections: market-state block (`09`), What-Changed block (`07`), archive
  occupancy (`08` — which regions are already crowded), the knowledge-cutoff block
  (L1), and the specificity contract (L3).
* Showing the agent the **archive of elites** as few-shot context is the
  established quality-diversity pattern and is what makes "propose something
  *different*" concrete rather than rhetorical.
* Prompts **diagnose, never prescribe** — a hard rule in this repo. The prompt
  states measurements and closes "what this implies is yours to determine". It
  never says which mechanism to pursue.

**`quantaalpha/pipeline/leakage_audit.py`** — the three L4 controls, writing
`data/results/leakage_audit/{asof}.json`.

## Validation

* L2 golden tests: directions containing "2023", "CSI 300", "COVID", "600519",
  "momentum crashed in", "above 0.05" are each rejected with the violated rule
  named.
* An LLM outage yields **zero** directions and a loud error — never a canned set.
* Placebo harness runs end-to-end and emits a similarity number; assert the number
  exists and is in [0,1]. **Its value is a finding, not a pass condition** — record
  it, do not gate on it, and never tune the prompt until it looks good, which would
  be optimising the audit rather than the system.
* Determinism at temperature 0 for a fixed `(asof, report)`.

## Reporting rule

The honest claim is **"L1-L3 enforced; decay shape matched between arms; placebo
similarity = X"**. The decay-shape comparison is the load-bearing half — it is an
outcome measurement, not a proxy — and on the existing library it already passes.
The claim "no temporal leakage" is not available to us and must not appear in any
report.

## Risks

* **The placebo could come back bad** — i.e. the agent keys on the date. That is a
  real possible outcome and the plan must be willing to report it. The mitigation
  if so: strip the date from the prompt entirely and pass only the statistics, then
  re-run the audit.
* A heavier mitigation exists in the literature (logit adjustment against a pair of
  small models fine-tuned on information to forget vs retain). Out of scope here —
  note it as the escalation path if L4 fails badly.
* The gate's own scoring window is a separate, pre-existing leak: `_windows(False)`
  scores on `search_oos` = valid, and `net_cost_feedback` renders those metrics to
  the LLM, so that window is soft-selected on. Fix: the re-mine's validation window
  must sit strictly before the new `final_test`, and the window must never be
  reported as clean OOS.

---

## What the existing direction machinery already gives you

Measured 2026-09-28, before writing anything. The agent is not a new pipeline —
it is a new **supplier of one string**, and knowing which string matters is most of
the design.

**How directions are made today.**

* `run.sh` takes one human-typed direction (default: *"cross-sectional equity
  factors from daily price and volume"*). `planning.generate_parallel_directions`
  expands it into `num_directions` = 10 parallel directions via the LLM, with the
  clamped `_market_context()` leading the prompt.
* Within a mine, `_reseed_if_stale` fires on staleness (`reseed_after_stale_rounds`
  = 2, `growth_floor` = 1) or on schedule (`reseed_interval` = 4), builds a digest
  (`_build_reseed_digest`), calls `generate_informed_directions`, **appends**
  `num_directions` more (never replaces — working parents stay), marks saturated
  directions, and resets the phase to ORIGINAL. `generate_informed_directions`
  takes `initial_direction` too, so the seed string frames the reseed prompt as
  well as the initial expansion.

**What that produced in `meanvar_20260828_194432`** — 40 directions over 10 rounds:
10 initial + 3 reseeds × 10. The budget split is lopsided:

| | directions | trajectories | share |
|---|---|---|---|
| initial | 10 | 124 | **77%** |
| reseeded | 30 | 37 | 23% |

Twenty-three of the thirty reseeded directions got **exactly one** trajectory.

**Three consequences that shape this task.**

1. **The seed direction is the highest-leverage input in a mine.** It frames 77% of
   the search. Replacing it with the agent's output is therefore not a marginal
   change to a corner of the pipeline — it re-aims most of the budget, which is
   exactly why the A/B contrast is worth the compute, and why the L3 specificity
   budget (a direction names a mechanism, never an answer) is load-bearing rather
   than decorative.
2. **A direction seeded late is mostly wasted.** The cap is `max_rounds` = 15 (the
   measured run finished 10). A direction appended at round 8 gets one pass and
   dies with the run. In a loop that repeats every year, so the *output* of a
   reseed has to persist across mines even though its *trigger* is within-run.
3. **`direction_id` is a per-run list index, and the text is not persisted
   anywhere.** Not in the trajectory pool (only `direction_id`), not in the ledger,
   not in the library JSON. `grep` for the run.sh default finds it in smoke logs
   only. Two mines' direction `3` are unrelated.

**Required before the replay can run.**

* **Record the direction.** `{direction_id, text, source: original|reseed_N,
  round_added}` into the trajectory pool and the ledger. Without it arm A cannot be
  told which direction to hold fixed — its one input is unrecoverable — and no
  cross-mine direction memory is possible.
* **Content-address it.** Key cross-mine direction memory on `md5(text)`, not the
  index, so a direction re-proposed in a later mine is recognised as the same one.
* **Pin the search dynamics across arms.** A and B must share `num_directions`,
  `reseed_interval`, `reseed_after_stale_rounds`, `growth_floor`, `max_rounds` and
  both prompt files. Otherwise `B − A` is not one argument, and the within-run
  reseed — not the agent — is what differs.
* **Filter novelty against the library, not the zoo.** `_filter_novel_directions`
  currently compares against the operators exercised in the *current* zoo; in a
  loop it must compare against the whole library's coverage, or each re-mine
  re-enters the monoculture the gate exists to prevent.
* **Persist direction outcomes.** `_direction_status` (attempts, admissions,
  `last_admit_round`, saturated) is in-memory and dies with the run, so every
  re-mine re-derives the same archetypes and re-exhausts them. Persisting it into
  the `06` global store lets a saturated archetype stay retired across mines and a
  productive one be re-seeded directly — which is the direction-level form of the
  "what stopped working" memory in `07`.
