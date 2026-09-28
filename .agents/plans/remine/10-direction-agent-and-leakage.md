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

**L4 — Measurement.** Three controls, run at every re-mine point and reported:

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

The honest claim is **"L1+L2+L3 enforced; L4 placebo similarity = X"**. The claim
"no temporal leakage" is not available to us and must not appear in any report.

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
