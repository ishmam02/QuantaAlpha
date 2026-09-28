# 14 — Historical replay, the two-arm ablation, and the report

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

| arm | directions | purpose |
|---|---|---|
| **A — control** | the same fixed initial direction at every re-mine | isolates re-mining itself |
| **B — treatment** | the `10` direction-selector agent | isolates the agent's contribution |
| **C — static baseline** | no re-mine, no refit — the 2016-fit book | the number to beat: CRR **−41.32%** |
| **D — refit only** | refit, never re-mine | already measured: CRR **+2.68%** |

C and D are already measured, so they cost nothing to include and they are what
makes A and B interpretable.

**Trigger points:** annual, 2017-2025 (9 points × 2 new arms = **18 mines**), or the
reduced set (2016 / 2019 / 2022 → 6 mines) if `01`'s envelope says so. **The choice
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

**Leakage:** the primary test is the **IC decay shape, compared between arms**. If
arm B (agent directions) decays materially flatter than arm A (fixed direction) —
its factors holding up in later years — that is the leakage signature, because
post-T knowledge is the only thing that would produce it. Report the decay slope
and the year-of-trough per arm beside the headline. On the existing library this
test already passes (Rank IC falls monotonically from .1104 to a .0160 trough in
2023, the most-documented break in the sample), so the bar is "arm B matches arm
A's shape", not "arm B decays at all". The three `10` controls run alongside and
bound the channel. The claim is "L1-L3 enforced; decay shape matched; placebo
similarity = X" — never "no leakage".

## Harness

`scripts/qa_remine_replay.py --arm {A,B,C,D} --triggers ... --resume`

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
