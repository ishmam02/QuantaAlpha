# Design: closing the research loop — decay-driven dynamic re-mining

**Status:** design doc. Executable steps live in `01-` … `14-`.
**Written:** 2026-09-28, from measurements taken 2026-09-26/27 on the first
genuinely leak-free CSI300 window.

---

## 1. Context — why this change

The system mines **once**, from a direction a human typed, and trades the result
forever. Every measurement below says that cannot survive contact with a live
market.

All figures: `protocol_csi300_lightgbm_topk_decay.yaml`, fit 2005-01-01…2016-12-31
(purged to 2016-12-09), scored 2017-01-01…2025-12-31, LightGBM + topk_dropout,
full κ cost model, benchmark `SH000300TR` (official cap-weighted CSI300 **total
return**). CRR = cumulative excess return.

| configuration | Rank IC | ARR | **CRR** | TC |
|---|---|---|---|---|
| main_full (147 factors) | .0565 | −5.96% | **−41.32%** | 0.37 |
| original (147) | .0488 | −9.78% | **−59.07%** | 0.33 |
| null model (no factors at all) | — | −14.33% | −73.87% | — |
| **main_full + annual combiner refit (2021-25)** | — | +0.55% | **+2.68%** | 0.42 |

Four measured facts drive the whole design:

1. **Signal decays on a ~3-year clock.** Rank IC runs .1104 (2017) → .0898 →
   .0587 → .0543 (2020), then plateaus ~.045. Distance from the fit is the
   dominant term.
2. **Refitting the combiner alone recovers ~20%** — and it is the only
   configuration of nine tested with positive CRR (+2.68%). It improves **5 of 5
   years, +4.57pp/yr mean**, on the *same* factors. **It is currently not done at
   all.**
3. **Refreshing the factors is worth ~2× more: ~+90%.** Fresh factors score
   .1001 (2017-18) and .1136 (2016) against .0526 for stale-factors/fresh-combiner
   and .0439 for both-stale.
   **⚠ Carries a known confound.** These are *first-year-after-fit* numbers, and
   §5 shows that window is the one place parametric leakage could still operate —
   hindsight can select, from the factors that work before T, those it knows
   persist longest. Arm E in `14` is the control. **Until it clears, treat +90% as
   an upper bound**, and note that this figure is what carries re-mined+refit to
   gross IR 0.68 in point 4.
4. **Only the combination clears viability.** Implied gross `IR = TC·IC·√N`:
   both stale **0.30**, refit-only **0.36**, re-mined + refit **0.68** — the first
   configuration above the ~0.5 bar.

So mining cadence is the primary knob and combiner refit the secondary one, and
**both belong inside one closed loop**. That loop does not exist today.

### 1.1 The regime claim must be discovered, never fed

The paper attributes the 2023 collapse to an A-share rotation from large-cap
"core assets" to small-cap/thematic names. On clean data 2023 **is** anomalous
(main_full Rank IC .0160; the zoo's IC goes negative) — but **~half of it is
combiner staleness**: an annual refit turns 2023 from −7.17% into −1.99% and
lifts Rank IC 86%.

Either way, **no part of that narrative may enter a prompt.** This is a hard
constraint already enforced elsewhere in the repo (`qa-no-market-specific-priors-in-prompts`,
`qa-prompts-diagnose-never-prescribe`) and it is preserved here by construction:
the direction agent reads **computed statistics only**. The size-decile spread
inverting and cross-sectional dispersion rising *are* the 2023 rotation, stated
as numbers with no label. Discovery, not instruction.

### 1.2 Scope

**In:** decay monitoring, dynamic re-mining, combiner refit, cross-mine memory,
parent/trajectory selection, the direction-selector agent, dynamic splits,
NAV compounding, and the replay + ablation that validates all of it.

**Out:** shorting (user decision, 2026-09-28 — deferred to the construction
phase), news corpora, live broker integration.

---

## 2. What already exists — reuse, do not rebuild

This is the single most important section for whoever implements it. **Most of
this system is already written and simply not wired up.**

| Capability needed | Exists? | Where | Gap |
|---|---|---|---|
| Decay tiers: healthy/soft/hard, 63d window, 70%/50%, 30/60 sustained days, quarterly re-test, **UUID retained** | **Yes — complete + tested** | `quantaalpha/eval/decay.py` (225 ln), `tests/eval/test_decay_tiers.py` | **Zero production callers** |
| Per-date rank-IC series (the monitor's input) | Yes | `ric_series` in `controller._get_ablation_eval` (~ln 760-790), `evolution/segment_ablation.py` | Computed for ablation; never persisted as a live series |
| **Ensemble of models per refit** (what feature-subset bagging needs) | **Yes** | `combiner.fit_predict` fits one model **per seed and averages** (`combiner.py:734-750`) | Needs per-seed *column* masking |
| Capped active set with replacement semantics | Yes | `admission.decide` capacity branch (`admission.py:293-306`) | `capacity` currently uncapped (`QA_MAX_LIBRARY=0`) |
| Re-measure incumbents against the zoo *as it now stands*, evict | Yes | `admission.should_evict`; runner `net_cost_runner.py:1719,1767` | Wired; needs tier-awareness |
| Orthogonality — "no two bets the same" | **Partial** | `check_pathology` `rho_max > rho_bar`; `scoring.RANKED_DIMENSIONS["diversity"]` | **`gates.rho_bar` defaults to `None` → the duplicate gate is OFF** |
| Multiple-testing correction as the library grows | Yes | `defense.deflated_sharpe_ratio`, `_fdr_bar` (BH) with `n_tests` accumulating from the ledger | Wired |
| Temporal-leakage clamp on prompts | **Yes, strong** | `planning._market_context()` — strict knowledge-cutoff block + clamp to the day before the **earliest fold's** validation day, fails **closed** | Extend to the market-state report; add an audit |
| Outcome-informed direction regeneration | Yes | `planning.generate_informed_directions` + `controller._build_reseed_digest` | **Within-run only** |
| Regime-conditional reporting (ic_crash / ic_rally) | Yes | `controller._regime_conditional_block` (~ln 2941) | Feeds reseed only |
| Trajectory pool, lineage, EB-shrunk parent ranking | Yes | `evolution/trajectory.py`, `controller._shrunk_fitness` / `_rank_by_fitness` | `fresh_start=True` → **dies at run end** |
| Walk-forward folds, purge + embargo | Yes | `Splits.walk_forward_folds` (`protocol.py:116-163`) | Off in report mode *by design* |
| Append-only trial ledger | Yes | `eval/ledger.py` | Wired |
| Factor UUIDs | Yes | `library.py` `factor_id = md5(...)` | Per-run files, not one global store |
| Borrow cost for shorts | Yes | `costs.borrow_cost`, `beta_per_day`, `beta_offlist` | **Out of scope this change** |

### 2.1 What does NOT exist — no substrate at all

The table above lists only primitives that already exist, so reading it alone
overstates how much is done. **Everything below has to be written from nothing.**
In particular, the agent layer — the genuinely novel part of this design — has no
substrate whatsoever.

| Component | Plan | Substrate today |
|---|---|---|
| **Decay agent** — recommendation schema, authority boundary, regime-exemption with validation + expiry + cap, failure semantics | `11` | **None.** No advisory layer of any kind exists anywhere in the repo. |
| **Scheduler** — cadences, the three re-mine triggers, decision log, inter-refit gap, re-mine-in-progress lock | `03` | **None.** Nothing schedules anything; a mine is launched by hand from `run.sh`. |
| **Market-state report** — the nine regime statistics | `09` | **None.** No statistic in `09` is computed anywhere. |
| **Leakage audit** — date-relabel placebo, state-shuffle control | `10` | **None.** |
| **Direction-selector agent** | `10` | **Partial:** reuses `planning`'s prompt/parse/retry scaffolding. The agent, the specificity validator and the archive-as-context prompt are new. |
| **Live IC service + DecayStateStore** | `02` | **Thin:** `ric_series` exists as a computation *inside* the ablation path. No service, no persistence, no incremental update, no baseline tracking. |
| **Cross-mine persistence** | `06` | **Thin:** `TrajectoryPool` can load, but `fresh_start=True` and there is no global store. |
| **Re-score + What-Changed digest** | `07` | **None.** |
| **Archive / MAP-Elites parent selection** | `08` | **None.** Descriptors are computable; no archive, cells, UCB, budget split or resurrection path exists. |
| **Split selector** (expanding vs rolling, chosen by validation) | `12` | **Thin:** folds exist; choosing *between* candidate spans does not. |
| **NAV compounding** | `13` | **None.** `nav` is a constant. |
| **Replay harness + ablation** | `14` | **None.** |

**Net, stated honestly.** Of the 14 tasks, exactly **one** (`04`, bagged soft
decay) is mostly wiring against existing infrastructure; `02`, `05` and `12` are
part-wiring; **the remaining ten are new construction.** What the existing code
buys is real but narrow: the **deterministic tier arithmetic is done** (`decay.py`
is complete and tested), the per-seed ensemble makes fractional weighting cheap,
and admission/eviction/orthogonality already supply the active-set machinery. That
is the *floor*, not the system. The decay agent, the direction agent, the
market-state report, the scheduler and the replay — the parts that make this a
closed loop rather than a set of primitives — do not exist in any form.

---

## 3. Architecture

```
                      ┌──────────────── DETERMINISTIC (always on, hard floor) ──┐
  daily ──▶ Live IC service ──▶ decay.classify_book ──▶ DecayStateStore
                                        │                      │
                                        │ tier transition      │ scheduled floors
                                        ▼                      ▼
                      ┌─────────────────────────────────────────────────┐
                      │  Scheduler: refit monthly · re-mine annually    │
                      │  quarterly re-test of hard-decayed UUIDs        │
                      └─────────────────────────────────────────────────┘
                                        ▲
                      ┌─────────────────┴──── ADVISORY (may fail safely) ──┐
  weekly + on-event ──▶ Decay agent ── accelerate only, + bounded exemption
                                        │
                                        ▼
                 ┌──── refit ────┐              ┌──── re-mine ────┐
                 │ active-set    │              │ market state    │
                 │ gate (capped, │              │      ▼          │
                 │ orthogonal)   │              │ Direction agent │
                 │      ▼        │              │      ▼          │
                 │ bagged fit    │              │ archive parent  │
                 └───────────────┘              │ selection       │
                                                │      ▼          │
                                                │ mine ──▶ library│
                                                └─────────────────┘
```

**The invariant that makes this safe:** the deterministic layer runs on a fixed
schedule and never consults the agent. If the agent errors, times out, returns
garbage or is switched off, the loop still refits monthly, still re-mines
annually, still retires hard-decayed factors. The agent can only make the system
act **sooner**, plus one bounded exception (§6).

---

## 4. Cadence — decided

The user asked for a decision. Each cadence is justified by what it can actually
observe or change, not by intuition.

| Job | Cadence | Why this and not faster/slower |
|---|---|---|
| **Live IC update + tier classification** | **Daily** | Cheap (one cross-sectional corr per factor per day, then a rolling mean). The tiers need *sustained* 30/60-day breaches, so daily is the natural resolution of the rule itself. |
| **Decay agent** | **Weekly + immediately on any tier transition** | The deterministic rule cannot fire before a 30-day sustained breach, so a *daily* agent adds cost without new information. Weekly gives 4-8 observations inside a soft-decay window — enough to act early — and the event trigger removes the latency objection entirely. ~52 scheduled calls/yr. |
| **Combiner refit** | **Monthly floor**, plus on any tier transition or agent request | An *annual* refit already buys +4.57pp/yr and is the only positive-CRR configuration. A fit is seconds-to-minutes. There is no compute reason to be slower than monthly. **Caveat: a fresher model trades more** — refit cadence is a swept parameter in `12-`, monthly is the default, not a finding. |
| **Full re-mine** | **Annual hard floor**, earlier on trigger | A mine is the expensive operation (~150 factors of LLM work). Annual is the *minimum* defensible cadence, not a conservative one: our own Rank IC runs .1104 → .054 over three years, and the published estimate for a medium-frequency factor's half-life is now ~**18 months** (against 5-7 years pre-AI). Triggers that fire it earlier are below. |
| **Re-test of hard-decayed UUIDs** | **Quarterly** | Already specified by `DecayRule.retest_days = 91` and implemented by `decay.due_for_retest`. Regimes return; a retired factor is re-tested, never deleted. |

### Deterministic re-mine triggers (any one fires; all are hard floors)

1. **≥ 33% of the active set** in `hard_decay`, or
2. **book-level** rolling 63d Rank IC < 50% of its validation baseline for 60
   sustained days (same rule as a factor, applied to the composite), or
3. **12 months** since the last mine completed.

The agent may fire a re-mine earlier. It may never postpone one.

---

## 5. Temporal leakage — what is actually achievable

The user asked whether leakage can be *completely* removed while still using the
latest model. **It cannot, and any design claiming otherwise is lying.** The
model's weights already contain the future; no prompt retracts that. What is
achievable is a bound on what leakage can *do*, plus a measurement of whether it
happened. Four layers, weakest to strongest:

**L1 — Context hygiene (bounds what is *stated*).** Nothing dated after T enters
the context. Already implemented well by `_market_context()`: it discloses only
the train window, clamps to the day before the *earliest fold's* validation day,
fails **closed** (returns `""` rather than risk a wider disclosure), and carries
an explicit knowledge-cutoff block forbidding post-T events, post-T factor
literature, and hindsight phrasing. The market-state report inherits this path.

**L2 — Output audit (detects, does not prevent).** A validator rejects a
direction that names a dated event, a ticker, an index level, a year, or a named
anomaly. Cheap, mechanical, catches the obvious cases.

**L3 — Structural containment (the strongest *argument*).** A direction selects a
**search region**; every factor is then selected by point-in-time measurement on
data ≤ T. Leakage can therefore only *misallocate search budget* — it cannot
manufacture a result — **provided the direction is not specific enough to encode
an answer.** So directions are constrained to name *mechanisms and measurement
targets*, never instruments, dates or events. This constraint is what makes the
whole approach defensible, and `09-` enforces it.

**L4 — Empirical measurement (the only actual evidence).**

**The primary control is the IC decay curve, and on the existing library it has
already passed.** Per-year Rank IC runs .1104 → .0898 → .0587 → .0543 → .0504 →
.0524 → **.0160 (2023)** → .0288 → .0469: monotone decay with distance from the
fit window, with the trough at the most-documented regime break in the sample. A
library built on hindsight would not fail hardest exactly where the hindsight is
densest. **Parametric leakage bought the existing system nothing.**

There is a structural reason: **admission is itself an anti-leakage mechanism**,
because factors are selected on data ≤ T, so an edge that exists only in the future
cannot pass. The residual risk is therefore narrow — leakage survives only when it
points at a factor that *also* worked before T, which is a **credit-assignment**
problem (the system looks more capable than it is) rather than a performance-
inflation one.

**The decay curve does NOT clear the immediate post-fit window, and that is the
exposure that matters.** Admission rejects a factor whose edge exists only in the
future — but *within* the pool that works before T, hindsight can select the ones
the model knows persist longest. That inflates **year 1** and converges to the
honest baseline as even the leaked picks decay, reproducing the observed curve
exactly. The 2023 trough does not rule it out either: by 2023 a leaked pick is dead
too. Worse, **the A/B ablation is blind to it** — both arms author factors with the
same LLM, so this leakage is common mode and differences out.

This lands on the project's core justification: the ~+90% factor-freshness figure
(fresh .1001 / .1136 vs .0526 stale) is a **first-year** measurement, and an
annually re-mining loop lives on exactly that number.

Hence **arm E** in `14`: Alpha158 — a fixed, published, pre-LLM factor set — run
through the identical pipeline at every T. No LLM touches generation, so its
first-year lift is pure fit-distance. Compare `year-1 IC / steady-state IC` as a
**ratio**; a materially larger ratio for the mined arms is the leakage signature at
the layer A/B cannot reach. **Until that control clears, the deployable expectation
is steady-state IC, not first-year.**

This is about the *old* configuration, where the model got little beyond a window
range. This design supplies far more period-identifying information, so it must be
re-established per arm in `14`: **if the agent arm shows flatter IC decay than the
fixed-direction arm, that is the leakage signature.** Three further controls bound
the input channel:

* **Date-relabel placebo (decisive, cheap).** Run the agent on the *same*
  market-state report labelled with a different date. If the directions change
  materially, the model is keying on the date label — i.e. parametric recall —
  not on the state. Report a similarity score per re-mine point.
* **State-shuffle control.** Feed a market-state report drawn from a *different*
  period but labelled T. Directions should track the **state**, not the label.
* **The ablation the user already specified** (fixed direction vs agent). If the
  agent's advantage concentrates on famous, memorable regimes and vanishes
  elsewhere, that is leakage showing up as a result.

**What gets reported:** not "leak-free", but "L1+L2+L3 enforced; L4 placebo
similarity = X". That is an honest claim; "no temporal leakage" is not.

---

## 6. Agent authority — the boundary

Per the user's decision (**accelerate + bounded exemption**):

**Always permitted (accelerate):** trigger a refit early; trigger a re-mine
early; demote a factor earlier than the sustained-breach rule would.

**The one delay permitted — bounded regime exemption.** The agent may exempt a
factor from a **soft** demotion when it judges the underperformance explained by
an adverse regime. Every exemption must:

* cite a **validated regime gate** — a named, computed market-state condition
  currently true, with the factor's `ic_crash`/`ic_rally` profile as evidence;
* carry an **expiry of one quarter**, after which it lapses unless re-granted on
  *fresh* evidence;
* be **recorded** with its justification in the decay ledger;
* **never** apply to hard decay, and never to more than a configured fraction
  (default 25%) of the active set at once.

**Never permitted:** delaying hard decay, postponing a scheduled refit or
re-mine, deleting a factor, editing the deterministic thresholds.

Failure semantics: agent error / timeout / malformed output ⇒ log, emit no
recommendation, deterministic layer proceeds unchanged. **An agent outage is a
non-event.**

---

## 7. The library, the active set, and "no two bets the same"

**The library is append-only and unbounded.** Every factor ever mined — admitted
or rejected — is retained under its `factor_id`, with its full trajectory,
metrics, verdict, and decay history. Nothing is ever deleted. (`library.py`
already mints `md5` factor ids; `05-` promotes per-run files to one global store.)

**The active set is capped and gated per refit** (user decision). Selection runs
as a gate chain, reusing existing machinery rather than inventing a new scorer:

1. **Tier filter** — `hard_decay` excluded from scoring (`decay.py`); `soft_decay`
   admitted at half weight via bagging (§8).
2. **Orthogonality — the "no two bets the same" requirement.** Turn the existing
   duplicate gate **on**: set `gates.rho_bar` (currently `None`, so it never
   fires) and enforce `rho_max ≤ rho_bar` against the *active set*, greedily,
   highest-contribution first. Cross-mine near-duplicates are the expected
   failure mode once the library holds several mines' output, and this is exactly
   the gate that catches them.
3. **Capacity + replacement** — `admission.capacity`: once full, a candidate
   enters only by displacing the incumbent it beats (`admission.decide`).
4. **Re-measured contribution** — `should_evict` against the set as it now
   stands, because "additive at twelve members can be redundant at a hundred and
   fifty" is already this repo's own finding.

Default cap: **200**, swept in `12-`. Every selection decision is written to the
ledger so "what is trading today, and why" is answerable after the fact.

**Multiple testing grows with the library and is already handled** — `_fdr_bar`
accumulates `n_tests` from the ledger and `deflated_sharpe_ratio` deflates for
trial count. As re-mines accumulate this bar *correctly* tightens; the plan does
not weaken it, and `13-` reports DSR against the true cross-mine trial count.

---

## 8. Soft decay under LightGBM — feature-subset bagging

The specified action ("weight capped at 50% of model weight") is **not
implementable literally**: gradient-boosted trees are invariant to monotone
feature scaling, and the LightGBM path produces no per-factor weights at all
(`library.py` says so explicitly).

**Resolution (user decision): feature-subset bagging** — and the infrastructure
already exists. `combiner.fit_predict` already fits **one model per seed and
averages the predictions** (`combiner.py:734-750`, `theta.combiner.seeds`). Soft
decay becomes: a factor's column is **included in only ⌈p·K⌉ of the K seed
models**, deterministically chosen by `(factor_id, seed)` so it stays
reproducible. `p = 1.0` healthy, `p = 0.5` soft, `p = 0.0` hard — which
reproduces `decay.WEIGHT_MULTIPLIER` exactly, in the only currency a tree
ensemble has.

Two properties worth having: it degrades gracefully (a factor fades rather than
snapping out), and the cross-model prediction spread becomes a free confidence
signal for the book.

---

## 9. Parent and trajectory selection — exploration vs exploitation

**Today:** `parent_selection_strategy: best` — pure exploitation over a pool that
`fresh_start=True` destroys at the end of every run. Crossover is measured
*worse* than mutation (median |rank_ic| 0.0067 vs 0.0114).

**New: an archive with an explicit, logged budget split.** The library is binned
into a behavioural archive on descriptors that are already computable:

* operator class (`factors/operator_coverage.py`, `coder/factor_ast.py`)
* lookback-horizon bucket
* regime profile — the sign pattern of `ic_crash` / `ic_rally`, **already
  computed** per factor
* decay tier and age

Parents are drawn from **three pools on a fixed budget**, default **50/30/20**:

1. **EXPLOIT (50%)** — best by EB-shrunk marginal contribution among *currently
   healthy* factors. `_shrunk_fitness` / `_rank_by_fitness` already do this.
2. **EXPLORE (30%)** — occupants of **sparsely populated archive cells**,
   regardless of fitness. This is what stops the operator monoculture this repo
   has hit twice.
3. **RESURRECT (20%)** — hard-decayed factors whose **regime profile matches the
   current market state**. This is the user's "regimes change" requirement, and
   `decay.due_for_retest` already supplies the candidates.

**Selection *within* a pool is UCB over archive cells, not over individual
factors.** Rationale, from this repo's own measurement: individual fitness is so
noisy that 3 of 4 verdicts flip across combiner seeds — so per-factor bandit
statistics are estimating noise. Cell-level yield pools many trials and is
actually estimable. The budget split is configuration, logged per mine, so the
split itself becomes a measurable parameter rather than a guess.

---

## 10. Cross-mine memory — "what stopped working"

Three pieces, in order:

1. **Persist.** `TrajectoryPool(fresh_start=False)` against a **global** store
   keyed by `trajectory_id`, spanning mines — hypothesis, expressions, lineage,
   metrics, verdict, refine actions.
2. **Re-score on the new data.** Before each re-mine, every prior trajectory is
   re-evaluated on data up to the new T. This is the user's "refreshed and tested
   on the newer data" and it is what produces the honest answer to *what stopped
   working*.
3. **Digest, as measurement only.** A **What-Changed** block: per archetype, IC
   then vs IC now, with the delta and the count. Extends the existing
   `_build_reseed_digest` / `_regime_conditional_block` pattern and obeys the same
   hard rule — **state the measurement, never the remedy, never a market name.**

This is the second channel by which the system discovers a regime shift on its
own: it observes that an entire archetype died, with no one telling it why.

---

## 11. Dynamic splits per mine

Per the user's decision (**fit both, pick by validation**), each re-mine:

1. proposes two candidate training spans — **expanding** from 2005, and
   **rolling** (last N years, N swept);
2. builds walk-forward folds for each via the existing
   `Splits.walk_forward_folds` (purge + embargo already correct);
3. selects the span that wins on walk-forward validation;
4. **records which won**, so "expanding vs rolling" is answered by accumulated
   data across the replay instead of assumed once.

Guardrails: `final_test` is never touched; the disclosure clamp in
`_market_context` recomputes against the *chosen* folds (it already derives from
`theta`, so this follows automatically — with a test to prove it).

---

## 12. Validation — historical replay, two arms

Per the user's decision (**historical replay first**):

* **Arm A (control):** the same fixed initial direction at every re-mine point.
* **Arm B (treatment):** the direction-selector agent.

Identical in every other respect: same trigger points, same seeds, same protocol,
same cost model, same cap. Primary metric **stitched CRR** across the replay;
secondary Rank IC, IR, TC, turnover, and the arm-vs-arm per-year win count.
Plus the three leakage controls from §5.

**Compute is the main risk and it is not yet measured.** Annual triggers across
2017-2025 = 9 points × 2 arms = **18 full mines**. `01-` measures the per-mine
wall-clock and LLM cost *first* and reports before anything is launched; if the
envelope does not fit, the fallback is a reduced trigger set (2016/2019/2022 →
6 mines) chosen **before** seeing any result, not after.

---

## 13. Order and dependencies

```
01 compute envelope (gates everything)
     │
     ├── 02 live IC + DecayStateStore ── 03 decay monitor + scheduler ── 11 decay agent
     │            └── 04 bagged soft decay ── 05 active-set gate (cap + rho_bar)
     │
     ├── 06 global stores ── 07 re-score + What-Changed ── 08 archive parents
     │
     ├── 09 market-state report ── 10 direction agent + leakage controls
     │
     ├── 12 dynamic splits + parameter sweeps
     └── 13 NAV compounding
                             all ──▶ 14 replay + ablation + report
```

`01` first and blocking. `14` last and once — and within it, **arm C first and
alone**: if the harness cannot reproduce the known static numbers, nothing else
counts.

---

## 14. Risks

* **Compute.** 18 mines may not fit. Mitigated by measuring first (`01-`) and
  pre-committing to a reduced trigger set.
* **Refit cadence raises turnover.** Monthly refit is a default, not a finding;
  `12-` sweeps it and reports net-of-cost.
* **Turning `rho_bar` on will reject factors that are currently admitted.** That
  is the point, but it changes library composition — report the before/after.
* **Enabling the admission bar starves the search.** `protocol.py:588-604`
  records that a blocking bar rejected 141 of 150 factors. This plan keeps
  `blocking: false` for *generation* and applies selection at the **active-set
  gate** instead — selection moves to where the book is built, not to where ideas
  are born.
* **The gate's scoring window leaks into generation.** `_windows(False)` scores on
  `search_oos` = valid (2013-2015) and `net_cost_feedback` renders those metrics
  to the LLM. That window is therefore soft-selected on and must not be reported
  as clean OOS. Pre-existing; documented here, fixed in `10-`.
* **The agent becomes load-bearing by accident.** Guarded by §6 and by an
  explicit "agent disabled" run in the ablation.
* **Multiple testing.** Trial count grows with every mine; the DSR bar correctly
  rises. Do not quietly reset `n_tests`.

---

## 15. Prior work this design leans on

Checked 2026-09-28. Listed because three of the design's load-bearing choices are
not original and should not be presented as such.

**Alpha decay is expected, quantified, and faster than it used to be.**
McLean & Pontiff: ~26% out-of-sample decline attributable to data mining, rising
to ~58% post-publication as the signal is arbitraged — roughly half of a published
anomaly's alpha disappears. Recent work models the crowding dynamic explicitly and
puts a medium-frequency factor's half-life near 18 months in the AI era. This is
the external justification for the annual re-mine floor and for treating decay as
the normal state rather than a failure.
· [Not All Factors Crowd Equally](https://arxiv.org/html/2512.11913v1)
· [AI-Driven Alpha Decay](https://arxiv.org/html/2605.23905v1)
· [Factor Decay](https://microalphas.com/factor-decay/)

**Temporal leakage cannot be eliminated, only bounded and audited.** The relevant
formalism is **look-ahead-freedom as temporal non-interference** — the output must
not depend on inputs after T — which is exactly what `09`'s panel-window test
asserts on the data side. For the *parametric* half, which lives in the weights and
is invisible to pipeline audits, the established instrument is a **black-box
behavioural audit** (HindsightBench): re-run the same task under a different date
label and measure whether the answer moves. That is the date-relabel placebo in §5,
and it is why the reportable claim is "L1-L3 enforced, placebo similarity = X"
rather than "leak-free". A heavier mitigation exists (inference-time logit
adjustment against paired forget/retain models) and is the escalation path if the
audit fails badly.
· [Look-Ahead-Freedom as Temporal Non-Interference](https://arxiv.org/html/2607.04958v1)
· [HindsightBench](https://arxiv.org/pdf/2607.18867)
· [Temporal Leakage in LLM Backtesting](https://arxiv.org/html/2608.02985)
· [Mitigating Look-Ahead Bias in Financial Backtesting with LLMs](https://arxiv.org/html/2605.24564)
· [A Fast and Effective Solution to Look-ahead Bias in LLMs](https://arxiv.org/html/2512.06607v1)

**The archive is MAP-Elites.** Partition a behaviour space into cells and keep the
best solution per cell; this is what buys exploration *and* exploitation at once,
instead of the pure-exploitation `best` strategy that has produced operator
monoculture here twice. The LLM-specific variants also establish the pattern of
showing the archive of elites back to the generator as few-shot context, which `10`
adopts.
· [MAP-Elites](https://www.emergentmind.com/topics/map-elites-algorithm)
· [LLMs as In-context AI Generators for Quality-Diversity](https://arxiv.org/html/2404.15794v1)
· [Diverse Prompts: Illuminating the Prompt Space with MAP-Elites](https://arxiv.org/pdf/2504.14367)
