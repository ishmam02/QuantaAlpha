# Implementation Plan — Net-of-Cost, Capacity-Aware Objective for QuantaAlpha

## Context

QuantaAlpha currently implements the **frictionless objective** that `problem_formulation.tex` argues against: `f* = argmax L(f(X), y) − λR(f)`, where `L` is RankIC. Concretely, `StrategyTrajectory.get_primary_metric()` returns `backtest_metrics["RankIC"]` and `is_successful()` is `RankIC > 0` — a single correlation scalar drives every selection decision in the evolutionary search.

The new formulation replaces this with

> `f* = argmax_{f ∈ F_Θ} U(m(f))`, where `m(f) = E_Θ(f)`

evaluated by a **frozen, deterministic engine** `E_Θ` that prices transaction cost *and* execution latency inside the objective, scores factors on a 7-dimensional quality vector, ranks them relative to the evolving factor repository, and enforces hard feasibility gates. The paper is explicit that **the factor-generation process is held fixed** — we change only *what is optimized* and *how it is measured*.

Three deliverables:
1. Point the LLM at **Ollama Cloud / `kimi-k2.5:cloud`**.
2. Re-run the paper's CSI 300 experiment as a **control arm** (same model, same protocol, old objective).
3. Implement the new objective and run the **treatment arm** for a like-for-like comparison.

### Decisions taken

| Question | Decision |
| :--- | :--- |
| Venue / construction | CSI 300, signed dollar-neutral **long-short**, high/infinite β for hard-to-borrow names |
| Control arm | **Re-run** the unmodified pipeline with kimi-k2.5; paper numbers are a sanity reference only |
| Execution engine | **Standalone vectorized** engine; Qlib retained only for data loading + LightGBM |
| Utility `U` | Weighted repository-relative scalarization `U = Σ_j ω_j·e_j`, `e_j = 1 − R(f, m_j, zoo)` |

---

## Findings from the current codebase

These shape the plan and are worth knowing before touching anything.

**Good news — two properties already hold structurally:**

- **Point-in-time integrity is already correct.** The in-loop backtest (`quantaalpha/factors/factor_template/conf_combined_factors.yaml`) trains 2016→2019, validates 2020, and runs port-analysis on **2021 only**. The final standalone backtest (`configs/backtest.yaml`) reports on **2022-01-01→2025-12-26**. The test window is genuinely untouched during search. **Preserve this invariant — do not widen the in-loop window.**
- **The plugin wiring makes this tractable.** `quantaalpha/pipeline/settings.py` holds dotted class-path strings resolved by `core.utils.import_class`; `AlphaAgentLoop` has no hardcoded imports of the runner/summarizer. The new objective can be swapped in via `QLIB_FACTOR_RUNNER` / `QLIB_FACTOR_SUMMARIZER` **without forking the loop** — which is exactly how we honour "generation process held fixed."

**Problems the new formulation forces us to fix:**

- **Label/fill mismatch.** The label is `Ref($close,-2)/Ref($close,-1)-1` (close-to-close) while `deal_price: open`. Eq. 4–5 require the realized return to be computed from *fill* prices consistently. Today the model is trained on one price series and executed on another.
- **Redundancy is AST-subtree based** (`RedundancyChecker` in `factors/regulator/consistency_checker.py`), but Eq. 9 requires `ρ_max` = max absolute cross-sectional **correlation** against incumbents. Different mechanism entirely.
- **No cost model beyond a flat rate.** `open_cost: 0.0005` / `close_cost: 0.0015` is roughly κ₀ only. No slippage (κ₁), no market impact (κ₂), no borrow (β), no shorting at all.
- **The generator is literally shown cost-free performance.** `factors/feedback.py::process_results` hardcodes the metrics the LLM sees (lines 42–47, duplicated at 88–93):

  ```
  1day.excess_return_without_cost.max_drawdown
  1day.excess_return_without_cost.information_ratio
  1day.excess_return_without_cost.annualized_return
  IC
  ```

  Qlib computes the `excess_return_with_cost.*` variants too — they are simply discarded before reaching the prompt. This is §3.2's frictionless/tradeable gap reduced to four string literals, and it is the most direct lever in the whole codebase: the feedback that shapes the next hypothesis never mentions cost. Replacing this selection with net-of-cost dimension scores is a large part of the contribution's actual mechanism, not an incidental edit. Note the list appears **twice** — refactor to one constant rather than editing both.
- **Config drift:** `learning_rate` is 0.05 in the in-loop template but 0.1 in `configs/backtest.yaml`. Must be unified and frozen into Θ.

**Assets we can reuse rather than build:**

- `factors/runner.py:58` `deduplicate_new_factors()` / `calculate_information_coefficient()` already compute per-datetime cross-sectional correlations between factor columns — **exactly the shape ρ_max needs**. Currently dead code behind `if False:` at line 138.
- Per-factor signal panels are already persisted and retrievable: `FactorLibraryManager._sync_h5_to_md5_cache` writes `data/results/factor_cache/{md5(expression)}.pkl` for every factor. This is what makes ρ_max against the whole repository cheap.
- `ComplexityChecker` already computes symbol length / base features / free-args ratio — reuse directly as `cx(f)`.
- `factor_entry["backtest_results"]` is a plain dict, and `_extract_backtest_results` passes dicts through unchanged. The metric vector can be stored with **no schema migration**.

**Blocking environment gap:** nothing is installed or downloaded. Active Python is **3.12.4** (project targets 3.10); `qlib` and `rdagent` are both absent; there is no `.env`, no `~/.qlib/qlib_data/cn_data`, no `git_ignore_folder/`, and **no test infrastructure anywhere in the repo**.

---

## Phase 0 — Environment bring-up

Prerequisite for everything. Roughly a day, mostly download time.

1. `conda create -n quantaalpha python=3.10 && conda activate quantaalpha` — 3.12 will not work; `numpy<2.0` and `pyqlib` both constrain this.
2. `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . && pip install -r requirements.txt`
3. Download from HuggingFace `QuantaAlpha/qlib_csi300`: `cn_data.zip`, `daily_pv.h5`, `daily_pv_debug.h5`. Place per README §3 (note the debug file must be **renamed** to `daily_pv.h5` inside the debug folder).
4. `cp configs/.env.example .env`, fill paths.
5. Add dev tooling not currently present: `pytest`, `scipy` (explicit rather than via sklearn). Put in a new `requirements/dev.txt`.
6. Verify: `quantaalpha health_check` and a `--dry-run` standalone backtest on `alpha158_20` to prove the Qlib data path works before any LLM spend.

---

## Phase 1 — Ollama Cloud / kimi-k2.5

`LLMSettings` (`quantaalpha/llm/config.py`) has **no `env_prefix`**, so fields map directly from uppercased names. No code change is needed for the happy path.

Add to `.env`:

```bash
OPENAI_API_KEY=<ollama-cloud-key>
OPENAI_BASE_URL=https://ollama.com/v1
CHAT_MODEL=kimi-k2.5:cloud
REASONING_MODEL=kimi-k2.5:cloud
CHAT_SEED=42            # llm/config.py defaults this to None — set it for reproducibility
CHAT_TEMPERATURE=0.0    # default 0.5; pin to 0 so the two arms differ only in objective
CHAT_STREAM=False       # default True; non-streaming is safer for OpenAI-compat providers
CHAT_MAX_TOKENS=8000    # default 3000 is tight for this repo's large prompts
```

**Confirm by smoke test, not assumption** — I could not reach the Ollama docs to verify these (web access was blocked during planning):

- **Exact base URL.** `https://ollama.com/v1` is the expected OpenAI-compatible path; verify with a raw `curl` before running the pipeline.
- **`response_format={"type":"json_object"}` support.** `APIBackend` sets JSON mode explicitly. If Ollama Cloud rejects it, the request fails before `robust_json_parse`'s fallbacks can help. **Mitigation:** add a `json_mode_supported: bool = True` field to `LLMSettings` and guard the `response_format` kwarg in `llm/client.py`; when false, rely on `robust_json_parse` (it already handles fenced blocks, balanced braces, and LaTeX-escape repair).
- **Embeddings.** `embedding_model` defaults to `""`. Determine whether the default coder path (`QlibFactorParser`, template-first) ever calls embeddings, or only CoSTEER's RAG knowledge base. If required, point `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` at a separate provider — do **not** assume Ollama Cloud serves embeddings for this model.
- **Context window.** `factors/prompts/prompts.yaml` is 34 KB and `proposal.py` already has an `is_input_length_error` handler, so this repo has hit context limits before.

Deliverable: a `scripts/smoke_llm.py` that issues one plain completion and one JSON-mode completion and prints which succeeded.

---

## Phase 2 — Control arm (baseline reproduction)

Run the **unmodified** pipeline on CSI 300 with kimi-k2.5, producing our own baseline rather than relying on the paper's GPT-5.2 numbers.

- Unify the two configs first: set `learning_rate` consistently and freeze the LGBM params + seeds into what will become Θ.
- `EXPERIMENT_ID=baseline_kimi ./run.sh "<paper's initial direction>"` — `run.sh` already isolates `WORKSPACE_PATH` and `PICKLE_CACHE_FOLDER_PATH_STR` per experiment ID, which we need so the two arms cannot share pickle cache.
- Then the standalone backtest on 2022–2025 with `--factor-source combined`.
- Record IC / RankIC / ARR / IR / MDD. Compare against the paper's 0.0472 / 0.0459 / 4.68% / 0.6453 / 11.80% as a **sanity reference**, expecting divergence from the model swap.

Preserve the resulting factor library JSON — the treatment arm must be scored against the same repository semantics, and the baseline's factors get **re-scored under the new `E_Θ`** for the head-to-head (§5).

---

## Phase 3 — The evaluation engine: `quantaalpha/eval/`

New self-contained package implementing `E_Θ` as a pure function. No dependency on the rest of QuantaAlpha except the AST complexity helper — this keeps it unit-testable and makes Property 2 (determinism) enforceable.

```
quantaalpha/eval/
├── protocol.py     # frozen Θ dataclass + content hash
├── execution.py    # Φ, δ, ỹ, portfolio map g, drift, turnover
├── costs.py        # κ₀ κ₁ κ₂ β, impact φ  →  c_t
├── metrics.py      # the 7 quality dimensions
├── gates.py        # F_Θ feasibility (Eq. 9)
├── scoring.py      # R(f,m,zoo), e_j, U (Eq. 10–12)
├── operator.py     # EvaluationOperator — the E_Θ entry point
├── ledger.py       # append-only trial record
└── protocol_csi300.yaml
```

### Key design points

**Protocol (Eq. 13).** `@dataclass(frozen=True)` holding splits, gates, cost coefficients, δ/Φ, and the portfolio map config. `Θ.hash` = SHA-256 of its canonical JSON, logged with **every** evaluation. Loaded once from YAML at startup. This is the mechanical enforcement of Property 1 (Immutability): the generator can read the gates — it needs to, to propose feasible factors — but any mutation changes the hash and invalidates the run.

Note how this interacts with session resume: `LoopBase.run()` (`utils/workflow.py:140`) pickles the **entire loop object** after every step. A Θ held on the runner is therefore captured in each snapshot — good, because a resumed run provably re-uses the Θ it started with. The flip side is that editing `protocol_csi300.yaml` mid-run has **no effect** on a resumed session. That is the correct semantics under Property 1, but it will surprise someone debugging; the operator should log `Θ.hash` at every evaluation so a stale-protocol resume is visible rather than silent.

**Execution (Eq. 4–5).** Fill rule `Φ(P; t, δ)`; default δ=1, Φ = open(t+1), which matches the existing `deal_price: open` and so keeps the baseline comparable. Realized return `ỹ_{i,t} = P_fill_{i,t+1}/P_fill_{i,t} − 1`. **This is where the label/fill mismatch gets fixed** — returns come from the fill series, not from close-to-close.

**Portfolio map g (Eq. 6).** Rank predictions cross-sectionally, centre, normalize to `Σ|w| = 1`: long the top, short the bottom, dollar-neutral by construction. Critical implementation note: **track notional positions, not normalized weights**, when applying drift — for a dollar-neutral book the normalizing denominator `Σ w(1+ỹ)` approaches zero and normalized-weight drift is numerically unstable. Compute drifted weights from drifted notionals over the post-drift NAV.

**Costs (Eq. 7).** Four terms: `κ₀·TO_t` + `κ₁·Σσ_{i,t}|Δw|` + `κ₂·Σφ(Δw; ADV)` + `Σβ_{i,t}·max(0,−w)`, with `φ ∝ |Δw|^{3/2}/√ADV`. `σ_{i,t}` (trailing realized vol) and `ADV_{i,t}` (trailing dollar volume) must be computed **strictly trailing** — this is where a look-ahead bug would silently enter, so both go through a shared `_trailing()` helper with an explicit shift.

**Metrics (§3.4).** The seven dimensions. Reuse `ComplexityChecker` for `cx(f)`. For `ρ_max`, revive the correlation machinery from `factors/runner.py` and load incumbent signals from the existing `factor_cache/{md5}.pkl` store — defined as per-date cross-sectional correlation, time-averaged, then max over incumbents.

**The IS→OOS problem.** Dimension 6 needs `Δ_{IS→OOS}` and dimension 7 needs an OOS trend — but the test split must never be touched during search. Resolution:

| | During search | Final report |
| :--- | :--- | :--- |
| IS | train (2016–2019) | train (2016–2020) |
| OOS proxy | valid (2020) / in-loop test (2021) | test (2022–2025) |

Only the final report uses 2022–2025. This must be enforced in code by making the window a **protocol field the operator reads**, not a caller argument — otherwise it will eventually be passed the test window by accident.

**Scoring (Eq. 10–12).** `R(f,m,zoo) = (1/|zoo|)·Σ 1[m(f) < m(f')]`, `e_j = 1 − R`. Dimensions where lower is better (turnover, ρ_max, complexity, IS→OOS degradation) are **sign-flipped before ranking**. `U = Σ ω_j e_j`. Over-fitting risk is scored structurally, not by repository rank, per §3.4.

### Default protocol for CSI 300

Starting values — all are protocol fields, and the sensitive ones get a sensitivity sweep.

| Param | Value | Rationale |
| :--- | :--- | :--- |
| δ, Φ | 1, open(t+1) | Matches existing `deal_price: open`; keeps baseline comparable |
| κ₀ | 0.0020 | Equals the existing round-trip `open_cost + close_cost`, so the baseline is nested |
| κ₁ | 0.10 | Slippage ≈ 10% of daily vol per unit traded weight |
| κ₂ | calibrate so 1% of ADV ≈ 10 bps | The capacity knob — **most sensitive parameter** |
| β | ~10%/yr ≈ 0.0004/day, ∞ off-list | A-share securities lending is restricted; ∞ encodes unshortable names |
| γ_ic, γ_ir | 0.02, 0.20 | RankIC / RankICIR floors on train+valid |
| τ_max | 0.30 | One-way daily turnover cap |
| ρ̄ | 0.70 | Redundancy ceiling vs repository |
| γ_cx | 200 | Matches existing `symbol_length_threshold` in `configs/experiment.yaml` |
| ω | effectiveness-weighted | Fixed in Θ; report a sweep |

---

## Phase 4 — Integration

Every seam uses the existing plugin mechanism. **No edits to the loop, the DSL, CoSTEER, or the mutation/crossover operators** — that is the "generation held fixed" constraint.

| Seam | Change | Mechanism |
| :--- | :--- | :--- |
| Evaluation | New `factors/net_cost_runner.py::NetCostFactorRunner` subclassing `QlibFactorRunner`; calls `E_Θ` and sets `exp.result` to a dict containing `m(f)`, `e_j`, `U`, and `Θ.hash` | `QLIB_FACTOR_RUNNER` env var |
| Feedback | New summarizer surfacing per-dimension scores so the LLM gets **dimension-targeted** feedback instead of one scalar | `QLIB_FACTOR_SUMMARIZER` env var |
| Fitness | `pipeline/evolution/trajectory.py:90-97` — `get_primary_metric()` returns `U`, `is_successful()` becomes feasibility + `U > 0`. Keep RankIC in `backtest_metrics` so both arms stay reportable | Small direct edit |
| Repository | `factors/library.py` — store the metric vector in `backtest_results` (dict passthrough, no migration) | No schema change |
| Gates | Enforce `F_Θ` post-computation, pre-backtest. Keep the AST redundancy check as a **cheap pre-screen**; `ρ_max` is authoritative | `gates.py`, called from the runner |

Three hazards to handle explicitly:

- **The `QLIB_FACTOR_` prefix is shared.** `AlphaAgentFactorBasePropSetting`, `FactorBasePropSetting`, and `FactorBackTestBasePropSetting` all declare `env_prefix="QLIB_FACTOR_"` (`pipeline/settings.py:50,63,76`). Setting `QLIB_FACTOR_RUNNER` therefore swaps the runner in **all three** simultaneously — convenient here, since we want the new engine everywhere, but it means the env var cannot give the mining loop and the backtest loop different runners. If the arms ever need to diverge, edit `settings.py` directly or introduce a distinct prefix.

- **Pickle cache collision.** `@cache_with_pickle(CachedRunner.get_cache_key, ...)` keys on task info strings only — it does **not** include the objective, so the treatment arm would silently serve baseline-cached results. Mitigate by including `Θ.hash` in the cache key *and* using per-arm `PICKLE_CACHE_FOLDER_PATH_STR` (which `run.sh` already does per `EXPERIMENT_ID`).
- **Determinism leaks.** `TrajectoryPool.select_parents_for_crossover` calls `random.shuffle` unseeded, and `MutationOperator._generate_fallback_hypothesis` uses `random.choice`. Property 2 binds `E_Θ`, not the generator — but for a clean A/B both arms should be seeded identically. Add a single seeding entry point.

---

## Phase 5 — Comparison

The head-to-head must score **both factor sets under the same `E_Θ`** — otherwise it compares evaluation engines rather than objectives.

| Arm | Search objective | Final scoring |
| :--- | :--- | :--- |
| A — baseline | RankIC | `E_Θ` |
| B — treatment | `U(m(f))` | `E_Θ` |

Report side by side on the 2022–2025 test window: IC, RankIC, ICIR, **net IR**, **net ARR**, MDD, mean turnover, ρ_max, complexity, and IS→OOS degradation. Plus the paper's published numbers as a third reference column.

The expected and interesting result is that Arm A wins on raw IC/RankIC while Arm B wins on net IR/ARR after costs — that is precisely the gap §3.2 argues exists. **If Arm B also wins on raw RankIC, be suspicious of a bug**, not pleased.

Every evaluation appends `(factor_id, Θ.hash, m(f), timestamp)` to the ledger, which is the reproducible trial record the paper's roadmap needs for the future deflated-Sharpe / multiple-testing work.

---

## Verification

There is **no test infrastructure in this repo today**, so this is greenfield. Add `pytest` and `tests/eval/`.

**Unit invariants — the ones that actually catch bugs:**

1. **Nesting:** with `κ₀=κ₁=κ₂=β=0` and `δ=0`, `r_net` must equal the gross frictionless return to floating-point tolerance. This proves the new engine reduces to the old objective and is the single most valuable test.
2. **Dollar neutrality:** `Σw_t ≈ 0` and `Σ|w_t| ≤ 1` on every date, including after drift.
3. **Turnover bounds:** `TO_t ∈ [0, 1]`; a held-constant portfolio gives `TO = 0`.
4. **Cost monotonicity:** `c_t` is non-decreasing in each κ; impact is super-linear — doubling `Δw` more than doubles the impact term.
5. **Determinism:** `E_Θ(f)` called twice on identical input returns bit-identical output; a mutated Θ produces a different hash.
6. **No look-ahead:** truncating the price panel after date `T` must not change any `m(f)` computed on `≤ T`. This is the strongest available proxy for Property 3 and worth the effort to write.
7. **Scoring:** a factor that beats every incumbent on a dimension scores `e_j = 1`; the worst scores `0`.

**End-to-end:** a 1-loop, 1-factor mining run on `daily_pv_debug.h5` with a stub LLM, asserting that the trial ledger gains one row carrying the expected Θ hash. This proves the plugin wiring without LLM spend.

---

## Risks

| Risk | Handling |
| :--- | :--- |
| Ollama Cloud rejects `response_format` json_object | Guarded `json_mode_supported` flag + existing `robust_json_parse` fallbacks. Verify in Phase 1 before any long run. |
| Shorting on A-shares is economically strained | β encodes it honestly — this is what the formulation intends. Report the long-only arm alongside so the claim doesn't rest on an infeasible book. |
| κ₂ dominates results | It is the capacity knob and the most sensitive parameter; report a sensitivity sweep rather than a single point estimate. |
| Baseline diverges sharply from paper numbers | Expected — different LLM. This is why we re-run the control rather than cite. Only the A-vs-B delta is claimed. |
| Compute/token cost | `docs/user_guide.md` estimates ~500K tokens / 2–4 h for 3 directions × 5 rounds × 5 factors — and we need two full arms. |
| Qlib on Python 3.10 with `numpy<2` | Pinned in `requirements.txt`; do not upgrade opportunistically. |

## Out of scope

Per the paper's own scoping: the falsification critic (deflated Sharpe, multiple-testing correction) and the decay-monitored deployment library. The trial ledger in Phase 3 is the substrate they will need, and is built now so that work is not blocked later.
