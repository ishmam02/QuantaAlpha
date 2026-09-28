# 03 — Deterministic decay monitor + scheduler (the hard floor)

**Depends on:** 02. **Blocks:** 11, 14.

## Why

This is the layer that must keep working when the agent does not. It consults no
LLM, makes no judgement calls, and runs on a fixed schedule. If the agent is
disabled, erroring or absent, the loop still refits monthly, still re-mines
annually, still retires hard-decayed factors, still re-tests them quarterly.

## Goal

One deterministic driver with no optional dependencies, producing an auditable
decision record per run.

## Approach

**`quantaalpha/loop/scheduler.py`** (new package `quantaalpha/loop/`)

`run(asof, dry_run=False)` executes, in order:

1. `ic_service.update(asof)`.
2. `decay_state.classify_all(asof)` → tier transitions appended to the event log.
3. **Quarterly re-test** — `decay.due_for_retest(states, asof)` (exists) re-scores
   those UUIDs on current data and, if a factor now clears its baseline, returns
   it to `healthy` with the resurrection recorded. Hard decay is never deletion.
4. **Refit decision.** Fire when *any* holds: last refit ≥ 1 month ago; any tier
   transition since the last refit; an agent request is pending (`11`).
5. **Re-mine decision.** Fire when *any* holds:
   * ≥ 33% of the active set in `hard_decay`;
   * book-level rolling 63d Rank IC < 50% of its validation baseline for 60
     sustained days — reuse `decay.classify` on the **composite** prediction's IC
     series, so the book is tiered by exactly the same rule as a factor;
   * ≥ 12 months since the last completed mine.
6. Write `data/results/decay/decisions.jsonl` — one row per invocation carrying
   `asof`, every trigger's evaluated state, the action taken, and the tier census.
   **Always written, including when the action is "do nothing"**: a scheduler that
   only logs when it acts cannot be audited for what it failed to do.

**Thresholds** live in a `DecayPolicy` dataclass in `quantaalpha/loop/policy.py`,
defaults matching `DecayRule` plus the three re-mine triggers above. **Not** a
`Protocol` field — `theta.hash` must not move, and `asdict(Protocol)` covers every
field, so adding one silently invalidates every cached artifact and orphans
`ic_breakeven_{hash}.json`.

**Driver:** a CLI (`quantaalpha decay-tick --asof`) so it can be cron'd in live
mode and called directly by the replay harness in `14`. Same code path in both —
if replay and live diverge, the replay proves nothing.

## Validation

* **Agent-absent test:** with the agent module importable-but-disabled and with it
  deleted outright, a full simulated year still produces 12 refits and 1 re-mine.
  This is the load-bearing test of the whole design.
* Trigger unit tests: each of the three re-mine conditions fires alone, and the
  12-month floor fires even when the book looks healthy.
* `dry_run=True` writes a decision row and mutates nothing else.
* Replaying the same date range twice yields identical decisions.

## Risks

* **Double-firing.** Refit-on-transition plus monthly could refit repeatedly in a
  volatile week. Add a minimum inter-refit gap (default 5 trading days) and log
  every suppression.
* A re-mine is long-running; the scheduler must record "re-mine in progress" and
  refuse to start a second, or two mines will fight over the signal cache and the
  16 GB of RAM.
