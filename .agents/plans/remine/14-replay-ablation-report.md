# 14 — Historical replay, the nine-arm experiment, and the report

**Depends on:** everything. **Last, and once.**

## Why

Every claim in this design is a hypothesis until the loop is run against history.
The replay is the only way to answer the question that started this: **does a
dynamically re-mined, continuously refit system beat the static one out of
sample?** — and it is the only setting in which the direction-agent ablation the
user asked for can be measured at all.

## Design

**Arms** (identical in every other respect — same trigger points, same seeds, same
protocol, same cost model, same caps):

| arm | directions | schedule | re-mines | what it isolates |
|---|---|---|---|---|
| **C — static** | — | none | no | the number to beat: CRR **−41.32%** |
| **D — refit only** | — | calendar | no | already measured: CRR **+2.68%** |
| **A** | fixed | deterministic floor | yes | **re-mining itself** (vs D) |
| **B** | `10` agent | deterministic floor | yes | **the direction agent** (vs A) |
| **G** | fixed | **+ decay agent** | yes | **the decay agent** (vs A) |
| **H** | `10` agent | **+ decay agent** | yes | **the full system**; both agents together |
| **H⁻** | `10` agent | + agent, **accelerate-only** | yes | **the regime-exemption mechanism** (vs H). Conditional: run only if H beats G |
| **E — non-LLM control** | **Alpha158** (fixed, published, pre-LLM) | calendar | no | leakage at the **generation** layer |
| **F — non-LLM control** | **random DSL expressions** | calendar | no | same, without Alpha158's survivorship |

**A, B, G, H are a 2×2 factorial** over {fixed, agent} directions × {deterministic,
+agent} scheduling. Contrasts of interest:

* `A − D` — is re-mining worth it at all?
* `B − A` — the direction agent, under deterministic scheduling
* `G − A` — the decay agent, under a fixed direction
* `H − G`, `H − B` — each agent in the presence of the other
* `(H − G) − (B − A)` — the **interaction**: do the two agents compound or conflict?
* `H⁻ − H` — what the bounded regime exemption is worth (it is the riskiest
  mechanism in the design, and the only one that can *delay* a demotion)

C and D are already measured, so they cost nothing. E and F need **no mining at
all** — only the existing pipeline run on a fixed or mechanically generated factor
set.

**Why the decay-agent arms need their own mines.** The agent changes *when* a
re-mine fires, so an accelerated trigger is a different mine. Mitigation: cache
mines keyed on `(direction_source, T)`, so G shares A's mine (and H shares B's)
wherever trigger dates coincide. Only the divergences cost extra — estimate
**22-26 mines total** rather than 36.

**Why arm F exists as well as E.** Alpha158 is a *survivor* set: published because
it worked. It may therefore show unusually flat decay for reasons unrelated to
leakage, which would make the mined arms look worse by comparison. Randomly
composed DSL expressions through the same gate have no survivorship and no LLM, so
F is the cleaner control; E is the cheaper and more interpretable one. Run both and
report both ratios — if they disagree, the disagreement is itself the finding.

### Why the non-LLM control arms exist

**The A/B ablation cannot detect generation-level parametric leakage**, because
both arms write factors with the same LLM: the leakage is common mode and
differences out. A/B tests the *direction agent*; it is blind to leakage inherent
in using an LLM to author factors at all.

The concern is specific. Admission selects on data ≤ T, so a factor whose edge
exists only in the future is rejected — but within the pool that works before T,
hindsight can pick the ones the model knows **persist longest**. That inflates the
**first-year-after-fit** IC and converges to the honest baseline as even the leaked
picks decay. It reproduces the observed curve exactly, so the decay shape does not
discriminate it, and the 2023 trough does not either (by 2023 a leaked pick is dead
too).

This matters more than it first appears: the entire justification for re-mining is
a **first-year** measurement — fresh .1001 (2017-18) / .1136 (2016) against .0526
stale, the ~+90% figure behind gross IR 0.68. An annually re-mining loop *lives on*
first-year performance. If that lift is partly hindsight, the payoff is overstated
and will not appear live.

**The test:** compare `year-1 IC / steady-state IC` — a **ratio**, because Alpha158
is weaker in level — between each mined arm and arms E/F. Matching ratios ⇒ the lift
is honest fit-distance. A materially larger ratio for the mined arms ⇒ the leakage
signature, isolated at the layer A/B cannot reach.

**Trigger points:** annual, 2017-2025 (9 points × 4 re-mining arms, reduced to an
estimated **22-26 mines** by the `(direction_source, T)` mine cache above), or the
reduced set (2016 / 2019 / 2022) if `01`'s envelope says so. **The choice
is made before any result is seen**, and recorded.

**Walk-forward honesty:** at each trigger point T, everything — market state,
direction, mining, admission, active-set gate, combiner fit — uses data ≤ T only.
The following year is scored and never looked at again. Returns are stitched into
one continuous series per arm, which is what an operator running this loop would
actually have earned.

## Metrics

Primary: **stitched CRR** (cumulative excess over `SH000300TR`, net of κ costs).
Secondary, per year and per arm: ARR, IR, MDD, Rank IC, IC, ICIR, TC, turnover,
cost in bps, active-set size, factors admitted, tier census.

Report the benchmark's own return alongside, because this project has already
established that **excess is beta-driven** — corr(excess, benchmark) **−0.73**
against corr(excess, Rank IC) **+0.11** — so an arm that "wins" in a falling market
has not necessarily produced alpha. The size decomposition runs on the winner.

**Statistical honesty:** DSR against the **true cross-mine trial count** (it grows
with every mine, and the bar correctly rises), plus the BH/FDR bar already wired
into `_fdr_bar`. Do not reset `n_tests` between mines.

**Leakage:** two tests at two different layers, because no single one covers both.

1. **Agent layer — decay shape, arm B vs arm A.** If arm B decays materially
   flatter, post-T knowledge entering through the *direction* is the only thing
   that would produce it. The bar is "arm B matches arm A's shape", not "arm B
   decays at all", since honest fit-distance decay is the shared baseline.
2. **Generation layer — first-year ratio, mined arms vs arms E and F.** The test
   above is **blind** to leakage common to both LLM arms. Compare
   `year-1 IC / steady-state IC` against the two non-LLM controls. This is the one
   that bears on whether re-mining's measured payoff is real.

Report the decay slope, the year-of-trough, and the year-1 ratio per arm beside
the headline. The three `10` controls run alongside and bound the input channel.

**Reporting rule — first year is quoted separately, always.** Never fold
first-year-after-fit IC into an average and never headline it. State it beside the
steady-state figure with the control-arm ratio attached. **Until arms E and F clear year 1, the
deployable expectation is the steady-state IC (~.045-.05), not the first-year
(~.11)** — and any forward projection of the re-mining payoff uses the
steady-state number.

The claim is "L1-L3 enforced; decay shape matched between arms; year-1 ratio
matches the non-LLM controls; placebo similarity = X" — never "no leakage".

## Harness

`scripts/qa_remine_replay.py --arm {A,B,C,D,E,F,G,H,Hminus} --triggers ... --resume`

* Drives the **same** `quantaalpha/loop/scheduler.py` used in live mode — if replay
  and live take different code paths, the replay proves nothing about live.
* **Resumable at trigger-point granularity**, and persists the **daily return
  series** per arm per year. (`qa_walkforward_refit.py` stores only summary stats,
  and that omission has already blocked two separate questions this month.)
* Memory: the box is 16 GB with swap already heavily used, and a run has been
  jetsam-killed at `n_jobs: 4`. Run arms sequentially, `n_jobs: 1`, and record peak
  RSS per mine.

## Validation

* Arm C reproduces the known static numbers (Rank IC .0565, ARR −5.96%, CRR
  −41.32%). **If it does not, the harness is wrong and nothing downstream counts.**
  This is the first thing to run and the gate on everything else.
* Arm D reproduces the known refit numbers (CRR +2.68%, 5/5 years improved).
* No arm's year-T book uses any data after T — asserted by instrumenting the panel
  loader during a replay, not by inspection.
* Re-running a completed trigger point is byte-identical.

## The report

Written **after** the numbers exist, from what they actually say — never templated
ahead of the run. It must state:

* stitched CRR per arm, with the per-year table and the benchmark beside it;
* total and annualised return, and Sharpe both annualised and total;
* whether re-mining beat refit-only, and whether the agent beat the fixed
  direction — **including if either answer is no**;
* the leakage audit numbers;
* how much of any improvement is beta rather than alpha (the decomposition);
* what was *not* tested: shorting, construction changes, live news, real capital.

If the dynamic system does **not** beat the static one, that is the result and it
gets reported as the result. The measured prior is not encouraging in isolation —
long-only against a cap-weighted index has a structural beta shortfall no amount of
IC fixes, and that is the construction problem queued behind this work.

## Risks

* **Compute is the binding constraint.** 18 mines is the largest commitment this
  project has made; `01` gates it and the reduced set is the pre-committed fallback.
* **A failed arm C invalidates everything.** Run it first, alone.
* Partial completion is likely. Resumability and per-trigger persistence are
  requirements, not conveniences.
