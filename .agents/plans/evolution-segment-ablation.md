# Per-segment expression ablation — implementation record

Status: **IMPLEMENTED + TESTED 2026-08-23** (uncommitted). Design doc at
`~/.claude/plans/gentle-conjuring-wave.md`. This file is the numbered
executable-plan record (per the `.agents/plans/` convention) plus the manual
validation that remains.

## What was built

### B0 — sign survival (DONE, prior session)
Persist `expected_ic_sign` on `StrategyTrajectory` at `controller.py:1352`
(where the `AlphaAgentHypothesis` is still in hand) so frozen expression-refine
children carry the parent's direction → the falsifiability gate stops rejecting
them `no_mechanism`. Mirrors: `parent_prefix["expected_ic_sign"]` (refine.py) +
table-path `directive.parent_expected_ic_sign` (diagnosis.py common tail).
Tests: `tests/evolution/test_sign_survives_refine.py` S1–S6 PASS.

### A — diagnosis blindness (DONE)
Three defects fixed: (A) casing `rank_ic`↔`RankIC` via `_GLOSS_ALIASES` +
`_get`; (B) 14 per-factor tearsheet scalars + `dsr` flattened in
`_extract_net_cost_metrics` (controller.py) with candidate selection
(expr-match → strongest-t_nw → sole); (C) `_METRIC_GLOSS` extended with U,
e_*, delta_*, cost_bps, RankICIR, weakest_dimensions, factor_attribution
(dict→rows renderer). Tests: `tests/evolution/test_diagnosis_blindness.py`
C1–C6 + P1–P5 PASS (27 lines where ~2).

### B — per-segment expression ablation (DONE)

- **B1 `segment_ablation.py`** (NEW, pure — no qlib import): `ablate(expr,
  predicted_sign, *, eval_signal, score) -> SegmentAblation`. Structural strip
  (`_strip_chain`, stops at signal-core), window sweep (IC-neutrality),
  core sign-stability (ic_pos_frac + date-half signs). `_render_summary` is
  measurement-only, ends "How to fix that is yours to determine."
- **B2 controller closure** (`controller._get_ablation_eval`): env-gated
  `QA_ABLATION_DIAGNOSIS=1` (default **OFF** → `None` → byte-identical
  diagnosis). Lazy `CustomFactorCalculator` + `EvaluationOperator` panel cached
  once; `eval_signal`/`score` closures wired to the metrics.py primitives
  (`_cross_sectional_corr(_slice(s, win), label, "spearman")` + `newey_west_t`
  + `solo_turnover`). Wired into the `build_task_extras` call at the
  `_build_mutation_task` site ONLY; the two classification `diagnose_parent`
  callers intentionally NOT (they only need `is_refinement()`).
- **B3 `RefinementDirective.ablation_summary`**: new field + `to_dict`/
  `from_dict` (survives the T5 lineage round-trip).
- **B4 LLM path** (`llm_diagnosis.py`): `_build_prompt` inserts a dedicated
  `## Per-part solo measurement` block; `_to_directive` carries
  `ablation_summary`; `llm_diagnose` threads `ablation`/`ablation_summary` to
  both the LLM call and the table fallback.
- **B5 table routing** (`diagnosis.py`): `_ablation_part_note` /
  `_window_ic_neutral` / `_core_health` helpers; `_build_target(...,
  ablation=None)`. **Decay branch IC-neutral backstop** (Q2 window-trap): if
  the slowest temporal window is IC-neutral, do NOT hand the model a window to
  move — point at the core. **Cost branch** keeps the window target + appends
  the core's measured health. **Signal branch** appends core health. `diagnose`
  signature gains `*, ablation=None, ablation_summary=""`; common tail sets
  `directive.ablation_summary`.
- **B6 tests**: `tests/evolution/test_segment_ablation.py` (A1 3-part split +
  IC-neutral window + unstable core + measurement-only summary; A2 AST
  round-trip) and `tests/evolution/test_diagnosis_ablation.py` (A3 decay→core
  backstop + turnover→window + none→structural; A4 directive carries
  ablation_summary through `diagnose_parent` + dict round-trip, and `""` with
  no eval). ALL PASS.

## Hard rules preserved (verified by tests)
- Prompts diagnose, never prescribe: `summary` ends "how to fix that is yours
  to determine"; `mechanism_hint` is a located measurement. (A1e, test_operator)
- No market-specific priors: no index names, no hardcoded IC, no
  continuation/reversal prior. (A1e)
- Q2 defenses: IC + cost measured separately per part; route on the broken
  part; IC-neutral-window backstop; perturbation deltas never reach a prompt
  (only `summary` + `per_part` scalars leave the module). (A1, A3)

## Manual validation remaining (requires a live run / qlib load)
1. **Flag-OFF regression** — PROVEN by A4b + the env-gate: with
   `QA_ABLATION_DIAGNOSIS` unset, `_get_ablation_eval()` returns `None` →
   `diagnose_parent(ablation_eval=None)` → `abl, summary = None, ""` → the
   directive is byte-identical to the pre-ablation path.
2. **Flag-ON real-panel smoke (TODO in a relaunch):** `QA_ABLATION_DIAGNOSIS=1`
   on a short mine → confirm (a) a refine child is no longer rejected
   `no_mechanism` for an empty sign (B0), (b) the directive carries a non-empty
   `ablation_summary`, (c) on the 1817 close-location factor the window is
   IC-neutral + the core sign-unstable + the `mechanism_hint` points at the core
   not the window. This needs the qlib panel (~163 MB) and is the one check not
   covered by the hermetic mock tests.

## Increment-2 hardening (out of scope, noted)
neutralization in solo metrics; `min(t_nw, t_overlap)` + `n_eff=n/h` overlap
correction; cache ablation by `md5(expr)` across a batch; multi-factor parents
(per-factor ablation rather than `factors[0]`).