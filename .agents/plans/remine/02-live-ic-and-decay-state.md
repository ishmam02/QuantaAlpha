# 02 — Live per-factor IC series + DecayStateStore

**Depends on:** 01. **Blocks:** 03, 04, 05, 07.

## Why

`quantaalpha/eval/decay.py` is complete and tested and has **zero production
callers**. It needs two things that do not exist: a per-factor daily IC series to
classify, and somewhere to persist the resulting state across days and across
process restarts.

The computation itself already exists — `controller._get_ablation_eval` produces
`ric_series`, a per-date rank-IC `pd.Series`, via `_cross_sectional_corr(...,
"spearman")`. It is computed for sub-tree ablation and thrown away.

## Goal

1. A **live IC service** that, for every factor in the library, maintains a daily
   rank-IC series against the traded label, incrementally.
2. A **DecayStateStore** persisting one `DecayState` per `factor_id`, with history.
3. The **baseline IC** each factor is judged against, recorded at admission.

## Approach

**`quantaalpha/eval/ic_service.py`** (new)

* `update(asof)` — computes per-date rank IC for every active `factor_id` from
  the last update through `asof`, appending to a Parquet store partitioned by
  year: `data/results/decay/ic_daily.parquet` (`date, factor_id, rank_ic, n`).
* Reuses the existing signal path — `load_aligned_signal` → `_cross_sectional_corr`
  — and the same label as the protocol (`label_frame`). Do **not** re-implement
  either; a second IC definition is how the btv2/Θ 3× IC gap happened.
* Incremental and idempotent: re-running for a date already present is a no-op.
* **Hard coverage floor.** If fewer than 90% of expected factors resolve a
  signal, raise. A thin book silently prices as the null model — that failure
  mode has already produced published-then-retracted numbers in this project.

**`quantaalpha/eval/decay_state.py`** (new)

* `DecayStateStore` over `data/results/decay/state.json`: load, upsert, history.
* `classify_all(asof, rule)` → calls the **existing** `decay.classify_book`, no
  re-implementation of the tier logic.
* Records every tier transition with its date to
  `data/results/decay/transitions.jsonl` — the event stream `03` and `11` consume.

**Baseline IC.** `baseline_ic` is the factor's **validation-window IC at
admission** (`decay.py` is explicit that comparing to a live mean would move the
goalposts with the decay and could never fire). Read it from the ledger row that
admitted the factor; where a legacy factor has none, mark it `baseline_missing`
and **exclude it from tiering** rather than defaulting — a default here silently
retires or protects factors.

## Validation

* `pytest tests/eval/test_decay_tiers.py` still passes untouched.
* New test: a synthetic factor whose IC steps from 0.10 to 0.04 tiers to
  `soft_decay` on exactly the 30th sustained day and `hard_decay` on the 60th,
  and `soft_start` ≠ `soft_trigger` (the touch-vs-sustained distinction
  `decay.py` warns about).
* Idempotency: `update(asof)` twice leaves the Parquet byte-identical.
* Spot-check one factor's live IC against `qa_alpha_decay_deepdive.py` for the
  same window — they must agree to ~1e-9.

## Risks

* **Cost.** Per-factor daily IC across a growing library is the recurring compute
  here. Incremental updates keep it to one new date per day; a full rebuild is
  the expensive path and must be explicit, not accidental.
* `factor_cache_path` is `md5(expression)` only and records nothing about which
  `daily_pv.h5` produced it — the cross-market collision that has already bitten
  twice. Keep the CSI300 cache dir pinned.
