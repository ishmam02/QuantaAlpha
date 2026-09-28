# 01 — Measure the compute envelope (BLOCKING)

**Blocks:** everything. Nothing else in this plan is launched until this reports.

## Why

The validation design is a historical replay: 9 annual trigger points × 2 ablation
arms = **18 full mines**. Nobody currently knows what one mine costs in wall-clock
or LLM spend. Committing 18 of them on an unmeasured estimate is how a month
disappears. The fallback (a reduced 3-point trigger set → 6 mines) has to be
chosen **before** any result is visible, so this measurement must land first.

## Goal

One artifact stating, from the existing run history:

* wall-clock per mine, per round, and per hypothesis;
* LLM calls and token spend per mine;
* peak RSS (the box is 16 GB and `rep/orig150` was already jetsam-killed once at
  `n_jobs: 4`);
* disk growth per mine (`log/` was measured at ~1.5 GB/batch and is never
  auto-cleaned; one 111 GB run directory had to be deleted this month);
* signal-cache growth: ~256 MB per factor pickle × ~150 factors per mine.

## Approach

`scripts/qa_remine_envelope.py`:

1. Walk completed run directories under `log/` and the `data/factorlib/*.json`
   stamps to recover each historical mine's start/end and factor count.
2. Parse the run logs for per-round timings and the admission/round counters.
3. Derive per-mine medians and a 9-point / 3-point replay projection for both
   arms, including disk and RAM headroom against what the box actually has.
4. Print a go / no-go against free disk and a stated RAM ceiling.

## Deliverable

`data/results/remine_envelope.json` + a printed table. Then **stop and report**:
the trigger-set decision (9 points vs 3) is the user's, made on these numbers.

## Validation

* Numbers reconcile against at least two independent historical mines.
* The projection is arithmetic on measured medians, with the medians shown — no
  extrapolation from a single run.
* Explicitly states free disk now and required disk for the full replay.

## Risks

* Old run directories may have been deleted (one was, deliberately). If fewer
  than two complete mines survive, say so and instrument the *next* mine rather
  than guessing from one sample.
