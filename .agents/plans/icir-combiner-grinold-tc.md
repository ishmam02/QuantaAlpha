# ICIR+shrinkage combiner, Grinold α, TC diagnostic, seed-in-generation

## Context

The replay showed soft-penalty + plain IC-weighting admits **9/125** (vs 26 under hard-cap LightGBM). The gap is the Ding-Martin (2017) Redux bound biting: the bootstrap-over-dates honestly measures σ_IC, which is ~5× LightGBM's seed-se, so positive-but-volatile factors fail `t>1`. Worse, **3 of LightGBM's 9 admissions (batches 29, 35, 40) are resolvably NEGATIVE** under the linear combiner — the 26 is inflated by LightGBM's `se≈0` auto-admit (`admission.py:193-194`).

The literature-prescribed fix (Ding-Martin Redux; the user's validated methodology): **ICIR weighting** (puts σ_IC into the weights, suppressing volatile factors at the source) + **shrinkage to equal weight** (James-Stein/DeMiguel — equal-weight is the high-σ_IC limit). Plus the three low-risk changes that make the linear path honest: **Grinold structural α** (sign-guaranteed, per-name vol; replaces the empirical β that falls back to 1.0 when ≤0), **scale_split: valid** (kills in-sample β inflation), and a **transfer-coefficient diagnostic** (Clarke-de Silva-Thorley — long-only + 3% cap + κ=5 are exactly the constraints that cut TC). Finally, feed the **Alpha158(20) seed library** to the generator as orthogonal-signal context (common-mode, per user decision 2026-08-12).

**Untouched:** the frozen Θ — `protocol.py` dataclass defaults, `protocol_csi300.yaml`, hash `f4f03cd5e5013329`, `test_seeds [42,1,7,13,29]`. **No new dataclass fields** (would rehash the frozen protocol via `asdict`); shrinkage lives in the existing `combiner.params` dict. Eval-objective changes are gated on `model: icir` → LightGBM/default path byte-identical. Seed-in-generation is common-mode (user-authorized relaxation of "control byte-for-byte unchanged from baseline"; the arms still differ only in the objective).

## User-confirmed decisions (2026-08-12)
- **Shrinkage:** Ledoit-Wolf data-adaptive δ toward equal weight, overridable via `combiner.params.shrinkage` (default `auto`; a float = fixed intensity).
- **Scaling:** Grinold structural α = IC_c × σ_i × s_i, `pred_scale=1.0`, IC_c estimated on valid.
- **max_weight:** keep 0.03; measure via TC; loosen only on a validation sweep if TC<0.4 or the cap binds.
- **Seeds:** Alpha158(20) into direction-generation, **both arms**, round-0 + reseed.

## Part 1 — ICIR + shrinkage combiner (replaces IC)
**File:** `quantaalpha/eval/combiner.py`
- Keep `_per_date_ic`. In `_fit_predict_ic`, replace the mean-IC weight line:
  - `ic_mean = np.nanmean(ic_arr[samp], axis=0)`; `ic_std = np.nanstd(ic_arr[samp], axis=0, ddof=1)`
  - `icir = ic_mean / (ic_std + 1e-8)`
  - Read `shrink = theta.combiner.params.get("shrinkage", "auto")`:
    - `auto`: Ledoit-Wolf δ = `clip( sum(ic_std²) / sum((icir - mean(icir))²), 0, 1 )` per bootstrap sample.
    - float f: δ = f.
  - `equal = np.full(n_feats, 1.0/n_feats)`; `weights = (1-δ)*icir + δ*equal`; `np.nan_to_num`.
  - `preds.append(feat_mat @ weights)`. Returns the **raw composite score** — Grinold α applied in the operator.
- **Dispatch:** `if model in ("ic","ic_weighted","linear"):` → `if model == "icir":`. Route to renamed `_fit_predict_icir`. Remove `ic` aliases.
- **Hardening:** move `import lightgbm as lgb` into the LightGBM branch (just-in-time) so the ICIR path never imports LightGBM → fork-after-OpenMP safety.

## Part 2 — Grinold structural α (operator, gated on `model: icir`)
**File:** `quantaalpha/eval/operator.py`, `_book`
- Gate the scaling block: for `model == "icir"`, `prediction = _grinold_alpha(prediction, y_tilde, sigma, scale_window)` (alpha in return units), `beta = 1.0`; else (LightGBM) byte-identical `beta = prediction_scale(...)`.
- New helper `_grinold_alpha(prediction, y_tilde, sigma, window)`:
  - z-score `prediction` cross-sectionally per date → `s`.
  - `IC_c = mean per-date corr(s, y_tilde)` over `window` (valid). Sign-preserving.
  - `alpha = IC_c * sigma * s` (elementwise; `sigma` = wide T×N trailing vol at operator.py:309).
  - Return wide T×N (~0.001 scale, matches empirical β×pred → λ=25 preserved).
- `build_book(..., pred_scale=1.0, ...)` for icir.

## Part 3 — Transfer-coefficient diagnostic (operator, gated on `model: icir`)
**File:** `quantaalpha/eval/operator.py`, after `build_book` returns.
- `tc = mean over dates of corr(window_pred.loc[d], w.loc[d])`; log it; add `"transfer_coefficient": mean_tc` to the metrics dict `_book` returns. LightGBM path: not computed (gated).

## Part 4 — Protocol YAMLs
- `protocol_csi300_meanvar_soft_linear.yaml`: `combiner.model: ic` → `icir`; add `portfolio.scale_split: valid`; add `shrinkage: auto` under `combiner.params`; update header docstring.
- `protocol_csi300_meanvar_soft_ridge.yaml`: byte-identical duplicate — apply the same three edits so it doesn't break when `ic` is removed; note it's a duplicate to differentiate later.
- **Do NOT touch** `protocol_csi300_meanvar_soft.yaml` (LightGBM) or `protocol_csi300.yaml` (frozen).

## Part 5 — Seed-in-generation (common-mode, direction-level)
**Files:** `planning.py`, `planning_prompts.yaml`, `informed_planning_prompts.yaml`
- Build `seed_context` from `SEED_POOL` (seed_pool.py:43-64): 20 names + expressions, framed as "canonical Alpha158(20) signals — propose NEW directions/factors ORTHOGONAL to these; do NOT copy them."
- Inject `{seed_context}` into both prompt YAMLs; wire into `generate_parallel_directions` and `generate_informed_directions`. Common-mode both arms (no env gate). Double literal braces for `str.format`.
- Seeds stay `admitted:False`; no change to library.py/ledger/admission.py.

## Part 6 — Configs
- Add `planning.seed_in_generation: true` to `configs/experiment.yaml` + `configs/experiment_paper.yaml`; read in `factor_mining.py`, pass to `generate_parallel_directions`. `run.sh` unchanged.

## MV compatibility — confirmed, no change
`build_book`/`mv_weights_costed` consume `mu = pred_scale × scores`; combiner-agnostic. Grinold α with `pred_scale=1.0` → `mu = α` (~0.001, matches empirical β×pred). λ=25 ~preserved (verify in smoke). No `riskmodel.py`/`portfolio.py` change.

## Implementation order (per-task manual validation)
1. Part 1 (combiner) → import + unit test.
2. Part 4 (YAMLs) → load_protocol; hash guards (linear/ridge ≠ 0919959dbe69945b; frozen = f4f03cd5e5013329).
3. Part 2 (Grinold α) → unit test; LightGBM else branch byte-identical.
4. Part 3 (TC) → TC logged for icir, absent for lightgbm, TC∈[-1,1].
5. Part 5 (seeds) → formatted prompt contains 20 seed names; both arms; novelty gate intact.
6. Part 6 (config key) → read; default true.
7. Smoke (1 worker, 2 batches): `qa_replay_soft_penalty_mp.py --protocol .../soft_linear.yaml --workers 1`.
8. Full replay only on user go-ahead.

## Constraints honored
- Frozen Θ untouched; no new dataclass fields (shrinkage in `params`).
- Eval-objective changes gated on `model:icir` → LightGBM/default byte-identical.
- Seeds never enter zoo/ledger (`admitted:False`; library.py/ledger/admission.py unchanged).
- Seed-in-generation common-mode (user-authorized 2026-08-12); arms differ only in the objective.
- No mining restart without explicit go-ahead. Fork only (no spawn). Branch off `main` before implementing.