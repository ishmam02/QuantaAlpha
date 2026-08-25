# Marginal effective_rank gate + ICIR/stability surfacing

Implemented 2026-08-24 (uncommitted). The frozen protocol hash is unchanged
(`fbefcb65f408aee0`); the gate is OFF by default and the admission path is
byte-identical when disabled.

## Why
- `rho_max` (pairwise, `rho_bar=0.60`) blocks a candidate duplicating ONE
  incumbent but lets through a candidate that is <rho_bar to each of several
  held factors individually and so adds few independent directions. The de-
  prime + operator-coverage run targets the ~2-direction breadth gap (miner
  12.2 vs OHLCV ceiling 14.2, measured on the production valid window
  2013-2015); without this gate, new diverse factors get diluted by redundant
  admissions. See `qa-34-ceiling-unmeasured-generation-not-exonerated`.
- ICIR (the industry stability metric, weak 0.20 / good 0.30-0.50) was
  computed in `metrics.py` but dropped before the per-factor tearsheet and
  absent from the feedback gloss, so the generator never saw it or its bar.

## Sufficiency verdict (measured)
The gate is **necessary, not sufficient, and not the profit lever**:
1. it filters but does not generate — breadth growth needs the de-prime +
   operator-coverage work to actually produce under-used-operator factors;
2. breadth is a ~8% composite-IC lever (√(14.2/12.2)); the binding profit
   constraint is the combiner (capture 27% → ~45% to clear break-even 0.0872);
3. too-tight δ shrinks admission — calibrate from the marginal-er distribution.

## What changed
- `quantaalpha/eval/metrics.py`: added `effective_rank(R)` and
  `spearman_abs_matrix(signals)` (lifts the inline `operator.py:223-241`
  formula; threshold `> 1e-12` to match). Exported in `__all__`.
- `quantaalpha/factors/net_cost_runner.py` `_decide_standalone`:
  - keeps `sd` in the best-horizon tuple and stores `rank_icir = mean/sd`
    (neutralized, the stability of the scored alpha) and `rank_ic` (raw, on
    `sig_raw` at the best horizon — makes the existing dead gloss entry live).
  - reads `QA_MIN_MARGINAL_ER` (default `0.0` = off; env-driven so the frozen
    protocol hash does not move — a Θ field would, since hash=sha256(asdict))).
  - on the NOVEL path only (before `kept.append`, after the rho_max/replace
    block; the replace path is a like-for-like swap by construction and is NOT
    gated): when the bar is truthy, `marginal = er(held+cand) - er(held)`; if
    `marginal < bar`, `notes.append(... redundant at the margin)` + `continue`.
    Default-off ⇒ the frozen path computes nothing and is byte-identical.
- `quantaalpha/factors/net_cost_feedback.py`: `_RAW_METRICS` now renders
  `rank_icir` (with bars) and `ls_mdd`; `_METRIC_HELP` adds both (measurement
  + bars, "yours to determine" — diagnose-never-prescribe). `rank_ic` already
  listed; now live.
- `tests/eval/test_marginal_er_gate.py` (new): helper math (G1), decision
  mirror (G2), source-ordering/env-default (G3), gloss surfacing (G4),
  economic_gap-not-regressed (G5).

## Verification
- Frozen hash: `load_protocol(...).hash == fbefcb65f408aee0` (unchanged).
- Tests: `test_marginal_er_gate` G1-G5; `test_rho_within_gate` W1-W8;
  `test_rho_max_replacement` R1-R7; `test_economic_bar` E1-E9;
  `test_mechanism_gate` M1-M6; `test_library_cap` C1-C7; `test_mechanism_channel`
  C1-C4; full `tests/eval/` suite green — no regressions.
- Gate-on smoke (NOT run — mining is gated on user approval): when ready,
  `QA_MIN_MARGINAL_ER=0.1` on a short mine; confirm rejection reasons include
  "redundant at the margin" and the zoo's `effective_rank` grows per admit.

## Calibration
`QA_MIN_MARGINAL_ER` in independent-direction units (zoo-size-independent).
Start at `0.1` (reject factors adding < 0.1 directions). A real-factor
calibration run prints the marginal-er of diverse-bench factors vs zoo-clones
on the held 33-factor zoo — set δ just below the diverse cluster and above the
clone cluster. Do NOT set δ ≥ 0.5 (shrinks admission). Log `marginal_er` per
candidate to tune.

## Out of scope / noted gaps
- Per-factor OOS/decay (`rank_ic_oos`, `is_oos_gap`) is intentionally NOT
  in-loop (reading the holdout test window every batch leaks); decay is
  surfaced via the segment ablation `core_sign_stability` (within valid).
- `operator.py`'s inline `effective_rank` block can be refactored to call the
  new helper (optional follow-up; left untouched to keep the frozen eval path
  byte-identical).