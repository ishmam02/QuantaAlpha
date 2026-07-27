# Feature: Net-of-Cost, Capacity-Aware Objective for QuantaAlpha

**Plan version 2** — revised 2026-07-27 against the updated `problem_formulation.tex`.
Every `file:line` reference below was **re-verified against the working tree on 2026-07-27**,
including the uncommitted macOS bring-up changes. Line numbers moved since v1 (notably
`factor_mining.py`, which was reformatted, and `factors/runner.py`, which gained ~13 lines).

Read `problem_formulation.tex` and `woolly-orbiting-pie.md` before implementing.
`Get_Started.md` is the codebase map.

---

## 0. WHAT CHANGED IN THE FORMULATION (and what it costs us)

The `.tex` update is not cosmetic. Six changes have direct architectural consequences.
This table is the single most important thing in this document — v1 of this plan is wrong
in three places because of it.

| # | Change in `problem_formulation.tex` | Consequence for the implementation |
| :-- | :--- | :--- |
| 1 | **The portfolio is built from the combiner's composite prediction `ŷ_t`, not from the candidate's own signal.** `M` (LightGBM) is **refit at every strategy-level evaluation** on the repository as it stands: `θ = A(F_zoo, D_tr)`. | `E_Θ` must **run a LightGBM fit inside every evaluation**. v1's design (portfolio straight from the factor signal) is wrong. New module `eval/combiner.py`. Strategy-level metrics are properties of the *book containing f* — i.e. **marginal contribution**, not stand-alone performance. |
| 2 | **`A` — the fitting procedure, "algorithm, hyperparameters, and random seed" — is now an element of `Θ`.** | The in-loop LGBM config has **no seed at all** (`conf_combined_factors.yaml:82-93`) and a **different `learning_rate`** from the standalone path (0.05 vs 0.1). That is a live Property 1 + 2 violation. Fixing it is now Task 1, a prerequisite, not tidy-up. |
| 3 | **`g` is stateful — `w_t = g(ŷ_t, w_drift_t)` — and restricted to a named family:** (i) **top-k dropout** (the published baseline's construction; long-only, excess over benchmark; signed variant shorts bottom-k), or (ii) mean–variance with a turnover cap. `(k, n_drop)` recorded in `Θ`. | v1 chose a rank-based **dollar-neutral long–short** book. The formulation now names top-k dropout as the comparability-preserving default, and the repo already runs exactly that (`topk: 50, n_drop: 5` in **all three** Qlib configs). Switch the default; keep signed long–short as a `Θ`-selectable sensitivity arm. |
| 4 | **`m(f) = E_Θ(f; F_zoo)`** — determinism is w.r.t. the **pair** `(f, zoo)`. "Recording the repository state alongside each trial" is what makes a trial reproducible. | `EvaluationOperator.evaluate()` takes `zoo` explicitly; the ledger must record a **`zoo_hash`** alongside `theta_hash`. Metrics split cleanly into **per-factor** (depend on `f` alone) and **strategy-level** (depend on `(f, zoo)`). |
| 5 | **Net excess return now subtracts the benchmark:** `r_net_t = w_tᵀ ỹ_t − r_bench_t − c_t`. | The engine must load the **SH000300** daily return series. v1 had no benchmark term. |
| 6 | **Prior work is explicitly "not entirely cost-blind"** — flat exchange commissions *are* deducted inside the Qlib backtest, so the baseline has incidental (flat-fee-only) cost exposure. | Sharpens the A/B claim. Arm A is **not** a zero-cost control; it is a flat-fee, fixed-low-turnover control. The head-to-head narrative must say so, and `κ₀ = 0.0020` is chosen precisely so the new engine **nests** that flat fee. |

Two smaller clarifications, also load-bearing:

- **Admissibility and scoring are explicitly separate steps.** Gates use *fixed absolute
  floors* ("a property of the market and the existing book, not a bar that should drift with
  search progress"); scoring uses *repository-relative rank*. Do not let a threshold leak from
  one into the other.
- **No `λR(f)` penalty exists in the objective, by design** — novelty is the `ρ_max` gate plus
  the diversity dimension; complexity is the `cx` gate plus the over-fitting dimension. Adding
  a penalty term would double-count. Do not reintroduce one.

---

## Feature Description

QuantaAlpha optimizes a **frictionless** objective: `f* = argmax L(f(X), y) − λR(f)` with
`L` = RankIC. `StrategyTrajectory.get_primary_metric()` returns `backtest_metrics.get("RankIC")`
(`trajectory.py:90-92`); `is_successful()` is `RankIC > 0` (`trajectory.py:94-97`); and the
feedback the LLM sees is four `excess_return_without_cost.*` string literals
(`feedback.py:42-47`, duplicated at `88-93`).

We replace this with `f* = argmax_{f ∈ F_Θ} U(m(f))`, `m(f) = E_Θ(f; zoo)`, evaluated by a
frozen, deterministic engine that prices transaction cost **and** execution latency inside the
objective, refits the combiner on the current repository, builds the baseline's top-k dropout
book from the composite prediction, scores the factor on a 7-dimensional quality vector ranked
against the evolving repository, and enforces hard feasibility gates.

**The factor-generation process is held fixed.** No edits to the loop, the DSL, CoSTEER, or the
mutation/crossover prompt bodies. We change only *what is optimized* and *how it is measured*.

Delivered as a controlled A/B: **Arm A** (control, RankIC objective) vs **Arm B** (treatment,
`U` objective), both finally scored under the same `E_Θ`.

## Problem Statement

Maximizing RankIC rewards signals concentrated in small/illiquid names with high turnover —
exactly the signals whose edge is consumed by spread, slippage, impact, and latency. The
current pipeline (a) shows the generator flat-fee-only performance, (b) models no slippage /
impact / borrow, (c) trains on close-to-close labels but fills at the open, (d) uses AST-subtree
redundancy where the formulation requires cross-sectional correlation `ρ_max`, and (e) refits
its combiner with an **unseeded** LightGBM whose hyperparameters differ between the in-loop and
standalone paths — so even the existing numbers are not reproducible in the sense Property 2
demands.

## Solution Statement

A self-contained `quantaalpha/eval/` package implementing `E_Θ` as a pure function of
`(f, zoo, Θ)`, wired into the loop through the existing plugin mechanism
(`QLIB_FACTOR_RUNNER` / `QLIB_FACTOR_SUMMARIZER`), with the trajectory's primary metric made
arm-selectable, the metric vector stored in the factor library with no schema migration, and an
append-only ledger recording `(factor_id, theta_hash, zoo_hash, m(f), timestamp)`.

## Feature Metadata

- **Type**: Enhancement (new objective + evaluation engine; generation process held fixed)
- **Complexity**: High
- **Systems affected**: `quantaalpha/eval/` (new), `quantaalpha/factors/` (runner, feedback),
  `pipeline/evolution/{trajectory,controller}.py`, `pipeline/factor_mining.py`,
  `factors/factor_template/*.yaml`, `configs/`, `run.sh`, `cli.py`
- **Paper**: arXiv:2602.07085 §3 (`problem_formulation.tex`)

---

## 1. VERIFIED ENVIRONMENT INVENTORY (2026-07-27)

The environment is **already set up**. Do not re-run bring-up. Verified state:

| Item | Verified value |
| :--- | :--- |
| Python | 3.10.20 @ `/opt/anaconda3/envs/quantaalpha/bin/python`, env `quantaalpha` |
| Core deps | `numpy 1.26.4`, `pandas 2.3.3`, `scipy 1.15.3`, `qlib 0.9.7`, `lightgbm 4.7.0`, `tables 3.10.1`, `rdagent` present |
| **Absent** | `pytest`, `statsmodels`, `h5py` — **none are needed**: validation is manual, `tables` covers HDF5, and slope fitting uses `numpy.polyfit` |
| Qlib data | `data/qlib/cn_data`, symlinked to `~/.qlib/qlib_data/cn_data` (absolute target — the `run.sh` fix). Per-instrument features present: `open, high, low, close, volume, amount, vwap, factor, change, adjclose` |
| Mining panel | `git_ignore_folder/factor_implementation_source_data/daily_pv.h5` — cols `[$open,$close,$high,$low,$volume,$factor,$return]`, 14,215,449 rows, 5,982 instruments, 2008-12-29 → 2026-01-09. Debug copy: 2018–2019, 100 instruments, 48,700 rows |
| LLM | local Ollama `http://127.0.0.1:11434/v1`, `CHAT_MODEL=minimax-m3:cloud`, embeddings `nomic-embed-text`. `CHAT_TEMPERATURE=0.7`, **no `CHAT_SEED`** |
| Last full run | `log/2026-07-22_22-03-57-444608/` — 6 trajectories (4 with metrics), 12 factors → `data/factorlib/all_factors_library.json` |
| Last standalone backtest | `data/results/backtest_v2_results/` — IC 0.04943, RankIC 0.04790, ARR 3.79%, IR 0.4705, MDD −13.54% on 2022-01-01 → 2025-12-26 |

> **`$vwap` and `$amount` exist in Qlib but not in `daily_pv.h5`.** The evaluation engine
> therefore sources prices, ADV, and the benchmark **from Qlib**, not from the mining panel.
> That also keeps `E_Θ` point-in-time-auditable independently of the mining data.

> **Two prerequisites before any run:** commit the 11 uncommitted bring-up fixes (they are
> load-bearing — the Darwin symlink branch, the absolute `factor.py` paths, the qlib symlink,
> `MLFLOW_ALLOW_FILE_STORE`), and set `CHAT_TEMPERATURE=0.0` + `CHAT_SEED=42` in `.env` so the
> two arms differ only in objective.

---

## 2. ARCHITECTURE DECISION: vectorized `E_Θ` + Qlib comparability anchor

The formulation now describes, almost exactly, what the repo's in-loop `qrun` path already
does: refit LightGBM on the accumulated factor set, build a top-50/drop-5 book, measure excess
return over SH000300. So there are two candidate implementations.

| | **A. Re-price the Qlib book** | **B. Standalone vectorized engine** ✅ |
| :--- | :--- | :--- |
| How | Run `qrun` as today, extract the realized position path from the Qlib recorder, add `κ₁, κ₂, β` on top | Refit LightGBM ourselves under frozen `A`, apply our own top-k dropout `g`, price Eq. 4–8 natively |
| Pro | Uses the baseline construction literally | Pure function; deterministic; no double-counted fees; latency native; every intermediate inspectable |
| Con | Qlib deducts its own flat fees (must be backed out); positions are integer share lots over a ¥1e8 account, so weights are derived not primitive; determinism hostage to Qlib internals | Must reimplement top-k dropout faithfully (~40 lines) |

**Choose B.** It is what "a deterministic evaluation engine — and only the engine — assigns
scores" means, and it is the only option that makes Properties 1–3 checkable by hand.

**But keep the Qlib path running unchanged as the comparability anchor.** The `.tex` says the
end-of-run evaluation "adopts [the baseline construction] unchanged for comparability", and the
scope paragraph says the protocol is instantiated with the published baseline's configuration
so its reported results stay comparable. So the final report carries three columns: Arm A under
`E_Θ`, Arm B under `E_Θ`, and the **Qlib standalone backtest** for both arms (the number
directly comparable to the paper's 0.0472 / 0.0459 / 4.68% / 0.6453 / 11.80%).

### Metric provenance split (a direct consequence of change #4)

| Class | Metrics | Depends on | Recomputed when zoo changes? |
| :--- | :--- | :--- | :--- |
| **Per-factor** | `ic`, `rank_ic`, `icir`, `rank_icir`, `ic_pos_frac`, `cx`, `is_oos_gap`, `decay_slope`, `ic_pers`, `turnover_solo` | `f` alone | No — cacheable by `md5(expr)` |
| **Repository** | `rho_max` | `f` and `zoo` signals | Yes |
| **Strategy-level** | `net_ir`, `net_arr`, `mdd`, `turnover_book`, `cost_bps` | `f`, `zoo`, via the combiner refit | Yes — one LightGBM fit per evaluation |

This split is also the performance story: only the third class costs a model fit, and that is
seconds on the train split.

---

## 3. THE PROTOCOL `Θ` — `quantaalpha/eval/protocol_csi300.yaml`

Every value below was read off the existing configs so the engine **nests** the baseline.

```yaml
protocol_id: csi300_v1
market: csi300
benchmark: SH000300          # conf_combined_factors.yaml:14, backtest.yaml:127
periods_per_year: 252

splits:                       # frozen; the engine never reads a window a caller hands it
  train:       ["2016-01-01", "2019-12-31"]   # conf_combined_factors.yaml:107
  valid:       ["2020-01-01", "2020-12-31"]   # :108
  inloop_test: ["2021-01-01", "2021-12-31"]   # :109  — NEVER widen
  final_test:  ["2022-01-01", "2025-12-26"]   # backtest.yaml:85
  search_is:  train           # IS during search
  search_oos: valid           # OOS proxy during search; final_test only in the end report

execution:                    # Eq. 4-5
  delta: 1
  fill_rule: open_next        # Φ = open(t+1); matches deal_price:open everywhere
  label_expr: "Ref($close, -2) / Ref($close, -1) - 1"   # backtest.yaml:55 — used for IC only

combiner:                     # procedure A — Eq. 2; FROZEN, seed included
  model: lightgbm
  seed: 42
  params:
    loss: mse
    learning_rate: 0.05       # ← unified; see Task 1
    max_depth: 8
    num_leaves: 210
    colsample_bytree: 0.8879
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    num_threads: 20
    early_stopping_round: 50
    num_boost_round: 500
  base_features:              # conf_combined_factors.yaml:28-29
    - "($close-$open)/$open"
    - "$volume/Mean($volume, 20)"
    - "($high-$low)/Ref($close, 1)"
    - "$close/Ref($close, 1)-1"
  fit_split: train

portfolio:                    # g — Eq. 6; the baseline's construction
  construction: topk_dropout
  topk: 50                    # conf_combined_factors.yaml:66, backtest.yaml:120
  n_drop: 5                   # :67, :121
  signed: false               # long-only + benchmark excess (baseline-comparable)
  gross_leverage: 1.0         # Σ|w| ≤ 1

costs:                        # Eq. 7
  kappa0: 0.0020              # = open_cost 0.0005 + close_cost 0.0015 → nests Qlib's flat fee
  kappa1: 0.10                # slippage ≈ 10% of daily vol per unit traded weight
  kappa2: 0.01                # calibrated: 1% of ADV ⇒ 10 bps of traded notional (derivation below)
  impact_exponent: 1.5        # φ ∝ |Δw|^{3/2}/√ADV
  beta_per_day: 0.0004        # ≈10%/yr borrow; inert while signed:false
  beta_offlist: .inf          # unshortable names
  vol_window: 20
  adv_window: 20

gates:                        # F_Θ — Eq. 9; absolute floors, train+valid only
  gamma_ic:  0.02
  gamma_ir:  0.20
  tau_max:   0.30
  rho_bar:   0.70
  gamma_cx:  250              # matches FACTOR_CoSTEER_SYMBOL_LENGTH_THRESHOLD in .env
  turnover_basis: book        # book | solo — see the note below

decay:                        # §3.4 item 7
  K: 4
  lambda_d: 0.5

overfit:
  delta_ref: 0.02             # reference IS→OOS RankIC gap for normalizing e_overfit

utility:
  mode: weighted_rank         # net_ir | net_arr | weighted_rank
  omega:                      # must sum to 1
    effectiveness: 0.30
    arr:           0.20
    stability:     0.15
    turnover:      0.10
    diversity:     0.15
    overfit:       0.10
    decay:         0.00       # optional diagnostic (Remark 1); off by default
```

**`κ₂` calibration (derivation — put this in a comment in `costs.py`).**
Let `adv_w = ADV_dollar / NAV` (ADV expressed as a portfolio weight) and participation
`p = |Δw| / adv_w`. Then `φ = |Δw|^{3/2} / √adv_w = |Δw|·√p`, so the impact charged **per unit
of traded notional** is `κ₂·φ/|Δw| = κ₂·√p`. Requiring 10 bps at `p = 1%` gives
`κ₂ · 0.1 = 0.0010` ⟹ **`κ₂ = 0.01`**. `NAV = 100_000_000` (`account:` in all three configs).

**Note on `turnover_basis`.** Under top-k dropout the book turnover is capped at
`n_drop/topk = 5/50 = 0.10` **by construction** — the `.tex` says as much — so `τ_max = 0.30`
never binds and the turnover *gate* is inert. That is a property of the chosen `g`, not a bug.
Capacity pressure comes from `κ₂` instead. Record **both** `turnover_book` (faithful to Eq. 6)
and `turnover_solo` (turnover of a book built from the candidate's own signal, which does
discriminate), let `Θ` select which one the gate reads, and default to `book`.

**Property 1 enforcement.** `Protocol` is a `@dataclass(frozen=True)`;
`Θ.hash = sha256(json.dumps(asdict(self), sort_keys=True, default=str))[:16]`, logged on every
evaluation. `LoopBase.run` pickles the whole loop after every step (`workflow.py:140`), so a
resumed session provably reuses the `Θ` it started with — and editing the YAML mid-run has no
effect. That is correct semantics, and the logged hash is what makes it visible rather than
silent.

---

## 4. RE-VERIFIED SEAM MAP (working tree, 2026-07-27)

> ⚠️ **Changed since plan v1** is marked. v1's numbers are stale for every file the uncommitted
> bring-up work touched.

### Plugin wiring / loop
- `pipeline/settings.py` — `AlphaAgentFactorBasePropSetting` **48**, `env_prefix="QLIB_FACTOR_"` at **50 / 63 / 76** (shared across all three settings classes — Hazard 1). Runner default `quantaalpha.factors.runner.QlibFactorRunner` at **56** *(v1 said 55)*; summarizer default `...feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback` at **57**. Singletons **116-120**.
- `pipeline/loop.py` — `AlphaAgentLoop.__init__` **54-131**; `quality_gate_config` param **66**, stored **82**; `consistency_enabled` read at **95** (default `False`), `complexity_enabled`/`redundancy_enabled` **96-97** (log-only); runner resolved **122**, summarizer **125**. Steps: `factor_propose` **143**, `factor_construct` **153**, `factor_calculate` **162**, `factor_backtest` **172** (calls `self.runner.develop(...)` at **176**, raises `FactorEmptyError` **179**, stores `self._last_experiment` **181**), `feedback` **186** (calls `self.summarizer.generate_feedback(...)` **187**). `_get_trajectory_data` **251**, returns `experiment` **260**. *(All step lines +2 vs v1.)*
- `utils/workflow.py` — `LoopMeta._get_steps` **26-42**, `__new__` **44-64** (non-underscore callables become steps — **any new method must be `_`-prefixed**), `LoopBase.run` **90-144**, `skip_loop_error` catch **116**, `CoderError` retry **121**, `dump` after every step **140**.
- `core/utils.py` — `import_class` **75**, `CacheSeedGen` **91-112**, `cache_with_pickle` **156**.
- `core/experiment.py` — `Experiment` **196**; `self.result: object = None` **212**, `self.sub_results: dict[str,float] = {}` **213**. No `backtest_results` attribute — the experiment object *is* the result carrier.
- `core/exception.py` — `CoderError` **1**, `CodeFormatError` **11**, `CustomRuntimeError` **17**, `NoOutputError` **23** (all retry-from-step-0); `CustomRunnerError` **29** (**not** a `CoderError` — would propagate); `FactorEmptyError` **35** (skip-loop).

### The runner (the main integration point) — **all line numbers changed**
- `factors/runner.py` — `QlibFactorRunner(CachedRunner[QlibFactorExperiment])` **37**.
  - `calculate_information_coefficient` **47**, `deduplicate_new_factors` **58** — per-datetime cross-sectional correlation machinery, still dead behind `if False:` at **151** *(v1 said 138)*. **This is the `ρ_max` seed.**
  - `@cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)` **75**, `develop(self, exp, use_local=True)` **76**.
  - `develop` recurses into `exp.based_experiments[-1]` **83-84**, builds `SOTA_factor = self.process_factor_data(exp.based_experiments)` **89** — **this is the repository signal panel, i.e. `zoo` in code** — and `new_factors = self.process_factor_data(exp)` **96**.
  - Writes `combined_factors_df.parquet` **172** *(v1: 159-161)*; config select `conf_baseline.yaml if len(exp.based_experiments)==0 else conf_combined_factors.yaml` **178** *(v1: 165)*; `exp.result = result` **200** *(v1: 187)*; `process_factor_data` **204** *(v1: 191)*.
- `components/runner/__init__.py` — `CachedRunner` **11**, `get_cache_key` **12-19** = `md5_hash("\n".join(task.get_task_information()))` over `based_experiments` + sub-tasks, **no objective or protocol hash** (Hazard 2); `assign_cached_result` **21**, copies `exp.result = cached_res.result` at **52**. `md5_hash` is `llm/client.py:29`.

### Metrics → trajectory → library
- `pipeline/evolution/controller.py` — `EvolutionConfig` **25**, `EvolutionController` **69**, `get_next_task` **134**, `get_all_tasks_for_current_phase` **167**, `advance_phase_after_parallel_completion` **293**, `report_task_complete` **681**, `create_trajectory_from_loop_result` **710** (takes `task/hypothesis/experiment/feedback` as separate args; sets `backtest_metrics = self._extract_metrics(backtest_result)` **766**), `_extract_metrics` **795** with `index_mapping` **813** and the two mapping loops **840** / **849**, `save_state` **877**.
  - ⚠️ `_extract_metrics` handles **only `pd.DataFrame` / `pd.Series`**. A plain `dict` result yields all-`None` metrics ⇒ `is_successful()` False. **The new runner must emit a `pd.Series`.**
- `pipeline/evolution/trajectory.py` — `RoundPhase` **21**, `StrategyTrajectory` **29**, `backtest_metrics` field **70**, `generate_id` **84**, `get_primary_metric` **90-92**, `is_successful` **94-97**, `to_dict` **131**, `from_dict` **140**; `TrajectoryPool` **148**, `select_parents_for_crossover` **251** with unseeded `random.shuffle` **330**/**338** — **dead code**; the controller uses `crossover_op.select_crossover_pairs`.
- `pipeline/evolution/crossover.py` — `generate_crossover_prompt_suffix` **242**, `select_crossover_pairs` **303**; **live** unseeded randomness at `random.shuffle` **370** (only when `prefer_diverse=False`), `random.sample` **412**/**432**, `random.choices` **502**.
- `pipeline/evolution/mutation.py` — `generate_mutation_prompt_suffix` **219**; unseeded `random.choice` **217** (fallback branch only).
- `factors/library.py` — `DEFAULT_FACTOR_CACHE_DIR` **19-21** (`FACTOR_CACHE_DIR`, default `data/results/factor_cache`); `FactorLibraryManager` **25**; `add_factors_from_experiment` **56**, calls `_extract_backtest_results` **74**, `factor_id = md5(f"{name}_{expr}")[:16]` **85**, writes key `"backtest_results"` **138**; `_sync_h5_to_md5_cache` **153** (`md5(expression)` full hex → `cache_dir/{md5}.pkl`, key computed **162**); `_extract_backtest_results` **295** *(v1: 294)* — **a `dict` passes through unchanged; a `Series`/`DataFrame` is flattened, floats rounded to 8 dp** ⇒ the whole vector stores with **no schema migration**.
- `factors/feedback.py` — `important_metrics` at **42-47**, duplicated at **88-93** (4 entries: three `1day.excess_return_without_cost.*` plus `"IC"`); `QlibFactorHypothesisExperiment2Feedback` **123**; `AlphaAgentQlibFactorHypothesisExperiment2Feedback` **215** (the wired summarizer), `generate_feedback` **216**, `json_mode=True` call **304**.

### Mining entry point — **all line numbers changed (file was reformatted)**
- `pipeline/factor_mining.py` — `force_timeout` **40**, `_run_branch` **94**, `_run_evolution_task` **119** (`quality_gate_cfg` param **127**, forwarded as `quality_gate_config` **186**), `_parallel_task_worker` **199**, `_serialize_task_for_parallel` **252**, `_run_tasks_parallel` **271**, `run_evolution_loop` **334** (`quality_gate_cfg` **340/346**, `parallel_enabled` **367**), parallel branch **452**, `create_trajectory_from_loop_result` calls **483** (parallel) / **523** (serial), sequential `quality_gate_cfg=` forward **521**, `main` **559**, config load **601**, `quality_gate_cfg` read **609**, `run_evolution_loop(...)` call **647**.
  - ⚠️ **Hazard 3 (live).** `_run_tasks_parallel` (**271-277**) neither accepts nor forwards `quality_gate_cfg`, and `configs/experiment.yaml` now sets `parallel_enabled: true`. Currently harmless *only* because the single threaded key `consistency_enabled` is `false` in YAML **and** `false` is the fallback default at `loop.py:95`. Flip it to `true` and the check silently will not run.

### LLM
- `llm/config.py` — `LLMSettings` **14** (no `env_prefix`; fields map from uppercased env names). `chat_model` **31**, `reasoning_model` **32**, `chat_max_tokens` **33**, `chat_temperature` **34**, `chat_stream` **35**, `chat_seed` **36**, `factor_mining_timeout` **41**, `chat_model_map` **65**, singleton **68**. **No `json_mode_supported` field** — a clean addition if ever needed.
- `llm/client.py` — `md5_hash` **29**, `robust_json_parse` **36**, `build_messages_and_create_chat_completion` **585**, `_create_chat_completion_inner_function` **748** with `json_mode: bool = False` **758**, `inspect.stack()[4]` **798/802**, `json_mode = None` when reasoning **806**, `if json_mode:` **853** → `kwargs["response_format"] = {"type":"json_object"}` **859**, JSON-repair block ~**897-925** (including the new trailing-comma fix at **921-922**).

### Gate / AST / config
- `factors/regulator/consistency_checker.py` — `FactorConsistencyChecker` **43**, `ComplexityChecker` **239** with hardcoded defaults `symbol_length_threshold=250` **245**, `base_features_threshold=6` **246**, `free_args_ratio_threshold=0.5` **247**; imports from `factor_ast` **262-264**.
- `factors/proposal.py` — `FactorRegulator` constructed **343**; `FactorQualityGate` lazy-loaded **358** with `complexity_enabled=True` **360** / `redundancy_enabled=True` **361**, both **hardcoded**.
- `factors/coder/factor_ast.py` — `parse_expression` **239**, `find_largest_common_subtree` **278**, `compare_expressions` **362**, `match_alphazoo` **370**, `count_free_args` **387** (**counts numeric constants / NumberNodes, not variable arguments**), `count_unique_vars` **426**, `count_all_nodes` **468**, `calculate_symbol_length` **482** (`len(expr.strip())`), `count_base_features` **496**.
- `factors/coder/config.py` — `FactorCoSTEERSettings` **7** (`env_prefix="FACTOR_CoSTEER_"`): `data_folder` **10**, `data_folder_debug` **13**, `python_bin` **25**, `factor_zoo_path` **28**, `duplication_threshold` **33**, `symbol_length_threshold` **37** (default 300), `base_features_threshold` **41**.
- `configs/experiment.yaml` — `parallel_execution: true` (execution block), `evolution.parallel_enabled: true`, `quality_gate` **102-116** (`consistency_enabled: false` **104**, `complexity_enabled` **107**, `redundancy_enabled` **110**, `max_correction_attempts` **116**), `factor.factors_per_hypothesis` **124**. **No `seed` field.** `configs/experiment_paper.yaml` = the same file with `num_directions: 10`, `max_rounds: 5`, `crossover_n: 10`, `factors_per_hypothesis: 3`.
- `configs/backtest.yaml` — `random_seed: 42` **11** (**dead** — not consumed anywhere under `quantaalpha/backtest/`), label **55**, splits **83-85**, `learning_rate: 0.1` **95**, `seed: 42` / `random_state: 42` **103-104**, `topk: 50` **120**, `n_drop: 5` **121**, `account` **126**, `benchmark` **127**, `deal_price: "open"` **130**, `open_cost` **131**, `close_cost` **132**, `min_cost` **133**.
- `factors/factor_template/conf_combined_factors.yaml` — benchmark **14**, base features **28-29**, label **30-31**, `TopkDropoutStrategy` **62** with `topk: 50` **66** / `n_drop: 5` **67**, `account` **71**, `deal_price: open` **76**, costs **77-79**, LGBM block **82-93** (`learning_rate: 0.05` **88**, **no seed**), splits **107-109**.
- `factors/factor_template/conf_baseline.yaml` — same shape: strategy **54-60**, costs **69-72**, `learning_rate: 0.05` **81**, **no seed**, splits **100-102**.

### Reusable seams
- `backtest/runner.py` — `BacktestRunner` **24**; `_create_dataset_with_computed_factors` **204**, per-datetime pct ranking `(x.rank(pct=True) - 0.5)` **341/346**, `PrecomputedDataHandler` **358-419**; `_compute_label` **429**; `_train_and_backtest` **479** → `LGBModel(**params)` **499**, `qlib_backtest(...)` **585**, IC/ICIR/RankIC/RankICIR **514-533**, ARR/IR/MDD/Calmar **684-698**.
- `backtest/custom_factor_calculator.py` — `DEFAULT_CACHE_DIR` **37**, `_get_cache_key = md5(expr)` **99-101**, `calculate_factor` **194**, `calculate_factors_batch` **307** (3-tier cache: `cache_location` H5 → MD5 pkl → recompute).
- `run.sh` — `FACTOR_CoSTEER_PYTHON_BIN` **51**, `MLFLOW_ALLOW_FILE_STORE` **55**, `EXPERIMENT_ID` default **66-69**, per-experiment `WORKSPACE_PATH` / `PICKLE_CACHE_FOLDER_PATH_STR` **73-78** (`EXPERIMENT_ID=shared` skips isolation), `FACTOR_LIBRARY_SUFFIX` **125**.
- `cli.py` — `app()` **29** (`fire.Fire` **30**) — the single chokepoint for both `run.sh` and `launcher.py`; the right place for the RNG seed.

---

## 5. NEW FILES

```
quantaalpha/eval/
├── __init__.py            # re-export EvaluationOperator, Protocol, load_protocol
├── protocol.py            # frozen Θ dataclass + content hash + load_protocol()
├── data.py                # Qlib panel loading: prices, ADV, benchmark, PIT universe  [NEW in v2]
├── execution.py           # Φ, δ, ỹ, drift, turnover                    (Eq. 4-6)
├── combiner.py            # procedure A: frozen LightGBM refit on zoo   (Eq. 2)   [NEW in v2]
├── portfolio.py           # g: top-k dropout (stateful) + signed variant (Eq. 6)  [NEW in v2]
├── costs.py               # κ₀ κ₁ κ₂ β, impact φ, trailing σ/ADV        (Eq. 7)
├── metrics.py             # the 7 quality dimensions                    (§3.4)
├── gates.py               # F_Θ feasibility                             (Eq. 9)
├── scoring.py             # R(f,m,zoo), e_j, U (3 modes)                (Eq. 10-12)
├── operator.py            # EvaluationOperator — the E_Θ entry point    (Eq. 13)
├── ledger.py              # append-only trial record incl. zoo_hash
└── protocol_csi300.yaml   # §3 above

quantaalpha/factors/net_cost_runner.py     # NetCostFactorRunner(QlibFactorRunner)
quantaalpha/factors/net_cost_feedback.py   # NetCostFactorFeedback(AlphaAgent...Feedback)
scripts/qa_eval_probe.py                   # manual validation / re-scoring driver (see §7)
scripts/qa_compare_arms.py                 # head-to-head table (Task 19)
```

**No `tests/` directory.** Validation is manual, specified per task in §6, with a whole-system
procedure in §7. (If these are ever wanted as automated tests, every snippet below drops into a
`pytest` function unchanged — but `pytest` is not currently installed and nothing here needs it.)

---

## 6. STEP-BY-STEP TASKS

Execute in order. Each task is atomic and independently checkable by hand. `$QA` below means the
repo root; run everything with the `quantaalpha` conda env active.

---

### Task 1 — Freeze the combiner procedure `A` (prerequisite)

`A` is now an element of `Θ`, and the `.tex` names the random seed explicitly. Today the two
paths disagree and the in-loop path is unseeded.

- **IMPLEMENT**
  1. `conf_combined_factors.yaml` LGBM kwargs (**82-93**): add `seed: 42` and `random_state: 42`.
  2. `conf_baseline.yaml` LGBM kwargs (**76-89**): the same two keys.
  3. Unify `learning_rate`: set `configs/backtest.yaml:95` to **0.05**, matching the two in-loop
     templates (the in-loop value is the paper's, and changing one file beats changing two).
  4. Record all of it verbatim in `protocol_csi300.yaml → combiner` (Task 2).
- **DO NOT** change `num_boost_round`, `early_stopping_round`, or the four base features — those
  are the baseline's, and their whole purpose is comparability.
- **MANUAL VALIDATION**
  ```bash
  cd $QA
  grep -n "learning_rate\|seed\|random_state" \
    quantaalpha/factors/factor_template/conf_combined_factors.yaml \
    quantaalpha/factors/factor_template/conf_baseline.yaml \
    configs/backtest.yaml
  ```
  Expect `learning_rate: 0.05` in all three, and `seed: 42` + `random_state: 42` in all three.
  Then prove the fit is now reproducible — run the standalone backtest twice and diff:
  ```bash
  for i in 1 2; do
    python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml \
      --factor-source custom --factor-json data/factorlib/all_factors_library.json
    cp data/results/backtest_v2_results/all_factors_library_backtest_metrics.json /tmp/qa_run$i.json
  done
  diff <(jq 'del(.elapsed_seconds)' /tmp/qa_run1.json) \
       <(jq 'del(.elapsed_seconds)' /tmp/qa_run2.json) && echo "DETERMINISTIC ✅"
  ```
  **If any metric differs, stop.** Something else is unseeded, and Property 2 cannot hold until
  it is found. (Prime suspect: LightGBM thread non-determinism — add
  `deterministic: true, force_row_wise: true` to the params.)

---

### Task 2 — `eval/protocol.py` + `eval/protocol_csi300.yaml`

- **IMPLEMENT** `@dataclass(frozen=True) Protocol` with nested frozen dataclasses mirroring §3
  (`Splits`, `Execution`, `Combiner`, `Portfolio`, `Costs`, `Gates`, `Decay`, `Overfit`,
  `Utility`). `hash` is a cached property:
  `sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:16]`.
  `load_protocol(path) -> Protocol` reads YAML and constructs; validate on load that
  `sum(omega.values()) == 1` (±1e-9) and that every split is a well-ordered date pair.
- **PATTERN** Plain dataclass, **not** `ExtendedBaseSettings` — `Θ` is loaded once from a file
  and must not be env-overridable, or Property 1 is unenforceable. (`QA_PROTOCOL` selects
  *which file*; it never overrides a field.)
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import dataclasses
  from quantaalpha.eval.protocol import load_protocol
  p1 = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  p2 = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  print('hash        :', p1.hash)
  print('stable      :', p1.hash == p2.hash)
  try:
      object.__setattr__(p1, '_x', 1) if False else setattr(p1, 'periods_per_year', 1)
      print('frozen      : NO ❌')
  except dataclasses.FrozenInstanceError:
      print('frozen      : yes ✅')
  print('omega sums  :', round(sum(p1.utility.omega.values()), 9))
  print('topk/n_drop :', p1.portfolio.topk, p1.portfolio.n_drop)
  print('kappas      :', p1.costs.kappa0, p1.costs.kappa1, p1.costs.kappa2)
  print('combiner sd :', p1.combiner.seed)
  PY
  ```
  Expect a 16-hex hash, `stable: True`, `frozen: yes ✅`, `omega 1.0`, `50 5`,
  `0.002 0.1 0.01`, `42`. Then prove sensitivity:
  ```bash
  sed 's/kappa2: 0.01/kappa2: 0.02/' quantaalpha/eval/protocol_csi300.yaml > /tmp/theta_mut.yaml
  python -c "
  from quantaalpha.eval.protocol import load_protocol
  a = load_protocol('quantaalpha/eval/protocol_csi300.yaml').hash
  b = load_protocol('/tmp/theta_mut.yaml').hash
  print(a, b, 'DIFFERENT ✅' if a != b else 'SAME ❌')"
  ```

---

### Task 3 — `eval/data.py` (Qlib panel loader)

The mining panel has no `$amount`, no `$vwap`, and no benchmark. The engine sources them from
Qlib, which also gives point-in-time CSI 300 membership.

- **IMPLEMENT**
  - `load_panel(theta, start, end) -> PanelBundle` returning aligned wide `(T × N)` frames:
    `open, close, high, low, volume, amount, vwap`, plus `universe` (bool mask of PIT CSI 300
    membership) — via `qlib.data.D.features(D.instruments("csi300"), [...])`. Qlib's
    `instruments("csi300")` is already point-in-time; **do not** substitute a static list.
  - `load_benchmark(theta, start, end) -> pd.Series` — daily return of `SH000300` from
    `D.features(["SH000300"], ["$close"])`, `pct_change()`.
  - `load_factor_signal(expr) -> pd.DataFrame` — read a mined factor's signal from
    `data/results/factor_cache/{md5(expr)}.pkl` (`library.py:153-176`), falling back to the
    workspace `result.h5`; return `(T × N)` aligned to the panel.
  - One `_align(df, panel)` helper that reindexes onto the panel's `(dates × instruments)` and
    masks non-members to `NaN`. **Everything downstream consumes aligned frames only.**
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_panel, load_benchmark
  th = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  p  = load_panel(th, '2019-01-01', '2019-12-31')
  b  = load_benchmark(th, '2019-01-01', '2019-12-31')
  print('dates        :', p.close.index.min().date(), '->', p.close.index.max().date())
  print('n names      :', p.close.shape[1], '| mean members/day:', round(p.universe.sum(1).mean(), 1))
  print('amount NaN%  :', round(p.amount.isna().mean().mean()*100, 2))
  print('vwap   NaN%  :', round(p.vwap.isna().mean().mean()*100, 2))
  print('bench len    :', len(b), '| ann. return:', round(((1+b).prod()**(252/len(b))-1)*100, 2), '%')
  print('PIT universe :', p.universe.sum(1).nunique() > 1)
  PY
  ```
  Expect ~244 trading days; mean members/day ≈ 300; low NaN rates for `amount`/`vwap`; a
  plausible 2019 CSI 300 return (**2019 was a strong year — roughly +25% to +40%**); and
  `PIT universe: True`, proving membership actually varies rather than being a static list.

---

### Task 4 — `eval/execution.py` (Eq. 4–6)

- **IMPLEMENT**
  - `fill_prices(panel, theta) -> pd.DataFrame` — for `fill_rule="open_next"`, `δ=1`:
    `P_fill[t] = open[t+1]`, i.e. `panel.open.shift(-1)`, adjusted by `$factor` for splits.
  - `realized_return(P_fill) -> ỹ` — `P_fill.shift(-1)/P_fill - 1` (Eq. 5).
    **This is the label/fill mismatch fix**: portfolio returns come from the fill series, not
    close-to-close. The close-to-close label stays in use for IC only (per §3.4 item 1).
  - `drift(w_prev, y_tilde) -> w_drift` — **compute from notionals, never from normalized
    weights**: `n = w_prev * (1 + ỹ)`, `w_drift = n / n.abs().sum()` for the long-only book. For
    the signed variant, `w_drift = n / NAV_post` with `NAV_post = n.sum()`; if
    `|NAV_post| < 1e-12`, carry `w_prev` forward and log a warning. (For a dollar-neutral book
    the normalizing denominator approaches zero and normalized-weight drift is numerically
    unstable — this is the trap.)
  - `turnover(w, w_drift) -> float` — `0.5 * (w - w_drift).abs().sum()` (Eq. 6).
- **PATTERN** Pure numpy/pandas. No QuantaAlpha imports except `protocol`.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import pandas as pd
  from quantaalpha.eval.execution import drift, turnover
  idx = list('ABCDE')
  w = pd.Series([0.2]*5, index=idx)
  y = pd.Series([0.10, -0.05, 0.00, 0.02, -0.02], index=idx)
  wd = drift(w, y)
  print('drift sums to 1      :', round(wd.sum(), 12))
  print('winner gained weight :', wd['A'] > w['A'], '| loser lost:', wd['B'] < w['B'])
  print('TO(w, w)   == 0      :', turnover(w, w) == 0.0)
  print('TO(w, wd) in [0,1]   :', 0 <= turnover(w, wd) <= 1, round(turnover(w, wd), 6))
  # full replacement of the book -> TO == 1
  old = pd.Series([0.2]*5 + [0.0]*5, index=list('ABCDEFGHIJ'))
  new = pd.Series([0.0]*5 + [0.2]*5, index=list('ABCDEFGHIJ'))
  print('full turnover == 1   :', round(turnover(new, old), 12))
  PY
  ```
  Expect `1.0`; `True | True`; `True`; an in-range turnover; and exactly `1.0` for a full
  replacement.

  Then the **no-look-ahead spot check** on the fill series:
  ```bash
  python - <<'PY'
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_panel
  from quantaalpha.eval.execution import fill_prices
  th    = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  full  = fill_prices(load_panel(th, '2019-01-01', '2019-12-31'), th)
  trunc = fill_prices(load_panel(th, '2019-01-01', '2019-06-28'), th)
  cut = trunc.index[-2]                      # last fully-formed row of the truncated panel
  a = full.loc[:cut]; b = trunc.loc[:cut]
  print('rows compared:', len(a), '| identical:', a.equals(b.reindex_like(a)))
  PY
  ```
  Expect `identical: True`. The final row of the truncated panel is excluded because
  `open_next` legitimately needs tomorrow — that is the *only* forward reference allowed, and it
  is the fill, not a feature.

---

### Task 5 — `eval/combiner.py` (Eq. 2, procedure `A`)

- **IMPLEMENT** `fit_predict(zoo_signals, candidate_signal, panel, theta) -> pd.DataFrame`:
  1. Build the design matrix: the four `base_features` from `Θ` evaluated on the panel,
     column-concatenated with every signal in `zoo_signals` **plus** `candidate_signal`.
  2. Preprocess exactly as the Qlib handler does: `Fillna`, `DropnaLabel`, `CSRankNorm` on both
     the feature and label groups (`conf_combined_factors.yaml:38-58`) — cross-sectional
     rank-normal per date.
  3. Label = `theta.execution.label_expr` evaluated on the panel.
  4. Fit `lightgbm` with `theta.combiner.params` + `seed`, on `theta.combiner.fit_split` **only**.
  5. Predict over the full requested window; return the `(T × N)` composite prediction `ŷ`.
  - Fit **once per evaluation**, on train only, never on valid/test. Cache the fitted model keyed
    by `(zoo_hash, candidate_md5, theta.hash)` so a repeated evaluation is free.
- **PATTERN** Mirrors `backtest/runner.py:479-503`, which already does `LGBModel(**params)` on a
  precomputed factor frame — read it before writing this.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_panel, load_factor_signal
  from quantaalpha.eval.combiner import fit_predict
  th  = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  pan = load_panel(th, '2016-01-01', '2020-12-31')
  f   = load_factor_signal("ZSCORE(TS_SUM(($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8), 10))")
  a = fit_predict({}, f, pan, th)
  b = fit_predict({}, f, pan, th)
  print('deterministic :', a.equals(b))
  print('shape         :', a.shape)
  print('predicts OOS  :', a.loc[th.splits.train[1]:].notna().any().any())
  PY
  ```
  Expect `deterministic: True` (the payoff from Task 1), a `(T × N)` frame, and non-null
  predictions past the train window. **If `deterministic` is False the seed did not reach
  LightGBM** — confirm both `seed` and `random_state` are passed, and add
  `deterministic: true, force_row_wise: true` if thread non-determinism is the cause.

---

### Task 6 — `eval/portfolio.py` (`g`, Eq. 6)

- **IMPLEMENT** `topk_dropout(pred, theta, prev_w=None) -> (w, w_drift)`, iterating dates:
  - Rank the cross-section of `ŷ_t` **within the PIT universe only**.
  - Hold the top `k`. On rebalance, drop at most `n_drop` of the currently-held names (the
    worst-ranked) and add the same number of the best-ranked non-held names. Equal-weight the
    resulting `k` names, `Σw = 1`.
  - Long-only when `signed: false`. When `signed: true`, mirror the construction on the bottom
    `k` with negative weights and scale so `Σ|w| = gross_leverage`.
  - Stateful: `w_t` depends on `w_drift_t` (Task 4), not on `ŷ_t` alone.
  - Assert `Σ|w_t| ≤ gross_leverage + 1e-9` on every date.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import numpy as np, pandas as pd
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.portfolio import topk_dropout
  th   = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  rng  = np.random.default_rng(0)
  dates = pd.date_range('2020-01-01', periods=60, freq='B')
  names = [f'S{i:03d}' for i in range(300)]
  pred  = pd.DataFrame(rng.normal(size=(60, 300)), index=dates, columns=names)
  w, wd = topk_dropout(pred, th)
  held  = (w != 0).sum(1)
  to    = 0.5 * (w - wd).abs().sum(1)
  print('always holds k=50 :', held.eq(th.portfolio.topk).all(), '| observed:', sorted(held.unique()))
  print('weights sum to 1  :', np.allclose(w.sum(1), 1.0))
  print('gross <= 1        :', bool((w.abs().sum(1) <= 1 + 1e-9).all()))
  print('max turnover      :', round(to.iloc[1:].max(), 4),
        '| structural cap n_drop/topk =', th.portfolio.n_drop / th.portfolio.topk)
  const = pd.DataFrame(np.tile(np.arange(300.), (60, 1)), index=dates, columns=names)
  wc, wdc = topk_dropout(const, th)
  print('constant pred TO  :', round((0.5*(wc-wdc).abs().sum(1)).iloc[2:].max(), 8), '(expect ~0)')
  PY
  ```
  Expect `always holds k=50: True`; weights summing to 1; gross ≤ 1; **max turnover ≤ 0.10** —
  the structural cap that makes `τ_max=0.30` inert (see §3) — and ~0 turnover under a constant
  prediction.

---

### Task 7 — `eval/costs.py` (Eq. 7)

- **IMPLEMENT**
  - `_trailing(df, window)` — the **single** shared helper, with an **explicit `.shift(1)`
    before rolling** so period `t` uses only data through `t-1`. Every trailing statistic goes
    through it. This is the one place a look-ahead bug can enter.
  - `trailing_vol(close, window)` → daily-return std; `trailing_adv(amount, window)` → mean
    dollar volume.
  - `impact(dw, adv_w, theta)` → `κ₂ · |Δw|^{1.5} / sqrt(adv_w)` with `adv_w = ADV / NAV`,
    guarded for `adv_w → 0` (clip at ~1e-8 and log the count of clipped cells).
  - `cost(w, w_drift, sigma, adv, theta) -> float` per date:
    `κ₀·TO + κ₁·Σσ|Δw| + κ₂·Σφ(Δw;ADV) + Σβ·max(0,−w)`.
  - `net_return(w, y_tilde, bench, c) -> pd.Series` — `(w·ỹ).sum(1) − bench − c` (Eq. 8).
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import pandas as pd
  from dataclasses import replace
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.costs import cost, impact
  th  = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  idx = list('ABCDE')
  w   = pd.Series([0.2]*5, index=idx)
  wd  = pd.Series([0.1, 0.3, 0.2, 0.2, 0.2], index=idx)
  sg  = pd.Series([0.02]*5, index=idx)
  adv = pd.Series([2e8]*5, index=idx)
  base = cost(w, wd, sg, adv, th)
  print('base cost (bps):', round(base*1e4, 3))
  free = replace(th, costs=replace(th.costs, kappa0=0, kappa1=0, kappa2=0, beta_per_day=0))
  print('zero-cost Θ    :', cost(w, wd, sg, adv, free), '(expect exactly 0.0)')
  for k in ('kappa0', 'kappa1', 'kappa2'):
      up = cost(w, wd, sg, adv, replace(th, costs=replace(th.costs, **{k: getattr(th.costs, k)*2})))
      print(f'  monotone in {k:7s}:', up > base)
  i1 = impact(pd.Series([0.01]*5, index=idx), pd.Series([2.0]*5, index=idx), th).sum()
  i2 = impact(pd.Series([0.02]*5, index=idx), pd.Series([2.0]*5, index=idx), th).sum()
  print('impact superlinear:', i2 > 2*i1, f'(ratio {i2/i1:.3f}, expect 2^1.5 = 2.828)')
  # κ₂ calibration: 1% participation should cost ~10 bps of traded notional
  dw, advw = pd.Series([0.001], index=['A']), pd.Series([0.1], index=['A'])   # p = 1%
  print('10bps @ 1% ADV :', round(float(impact(dw, advw, th).iloc[0] / dw.iloc[0]) * 1e4, 3), 'bps')
  PY
  ```
  Expect the zero-cost `Θ` returning exactly `0.0`; `monotone: True` for all three κ; an impact
  ratio ≈ **2.828**; and the calibration line printing ≈ **10.0 bps**. If the last one is off,
  `κ₂` is mis-scaled — re-derive from §3 before going further, because every downstream number
  depends on it.

---

### Task 8 — `eval/metrics.py` (§3.4, the 7 dimensions)

- **IMPLEMENT** two entry points, mirroring the provenance split in §2:
  - `per_factor_metrics(signal, expr, panel, theta) -> dict` — `ic`, `rank_ic` (per-date
    cross-sectional correlation of `z_t` against the **close-to-close label**, then time-mean),
    `icir`, `rank_icir`, `ic_pos_frac`, `cx`, `turnover_solo`,
    `is_oos_gap = rank_ic(IS) − rank_ic(OOS)`, and the decay block.
  - `strategy_metrics(w, w_drift, y_tilde, bench, costs, theta) -> dict` —
    `net_ir = mean(r_net)/std(r_net)·√252`, `net_arr = (Π(1+r_net))^(252/T) − 1`, `mdd`,
    `turnover_book`, `cost_bps`.
  - `rho_max(signal, zoo_signals) -> float` — **revive `runner.py:47-73`**: per-`datetime`
    cross-sectional correlation against each incumbent, time-averaged, then `max` of the absolute
    values. **Define `ρ_max = 0.0` for an empty zoo** (the first factor).
  - `cx(expr)` — from `factor_ast`: `calculate_symbol_length` (**482**), `count_base_features`
    (**496**), `count_free_args` (**387** — note it counts *numeric constants*, not variable
    arguments), `count_all_nodes` (**468**). Use `calculate_symbol_length` as the scalar `cx`
    compared against `γ_cx` (that is what the existing threshold means); carry the other three as
    diagnostics.
  - Decay: split the OOS window into `K` equal sub-periods; `IC^(k)` = mean IC in each; fit
    `IC^(k) = a + b·k` with `numpy.polyfit(deg=1)`. Report `decay_slope = b`,
    `persistence_ratio = IC^(K)/IC^(1)`, and `ic_pers = mean(IC)·(1 − λ_d·max(0, −b̂_norm))`
    where `b̂_norm = b·(K−1)/max(|mean(IC)|, 1e-8)` — the fitted fractional change in IC across
    the OOS window. Clip `b̂_norm` to `[-1, 1]` so `ic_pers` cannot go negative.
- **CRITICAL** The evaluation window is read from `theta.splits` — **never** accepted as a caller
  argument. Otherwise someone will eventually pass the test window.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import numpy as np, pandas as pd
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.metrics import rho_max, cx
  th = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  idx = pd.MultiIndex.from_product(
      [pd.date_range('2020-01-01', periods=50, freq='B'), [f'S{i}' for i in range(30)]],
      names=['datetime', 'instrument'])
  rng = np.random.default_rng(1)
  a = pd.Series(rng.normal(size=len(idx)), index=idx)
  b = pd.Series(rng.normal(size=len(idx)), index=idx)
  print('rho_max(f, {})    :', rho_max(a, {}),                  '(expect 0.0)')
  print('rho_max(f, {f})   :', round(rho_max(a, {'self': a}), 6), '(expect 1.0)')
  print('rho_max(f, {-f})  :', round(rho_max(a, {'neg': -a}), 6), '(expect 1.0 — absolute)')
  print('rho_max(f, {rnd}) :', round(rho_max(a, {'rnd': b}), 4),  '(expect near 0)')
  e = "ZSCORE(TS_SUM(($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8), 10))"
  print('cx                :', cx(e), '| gamma_cx =', th.gates.gamma_cx, '| passes:', cx(e) <= th.gates.gamma_cx)
  PY
  ```
  Expect `0.0`, `1.0`, `1.0`, ≈0, and a `cx` comfortably under 250.

  Then the **oracle check** — a factor equal to the forward label must top out on effectiveness:
  ```bash
  python - <<'PY'
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_panel
  from quantaalpha.eval.metrics import per_factor_metrics
  th  = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  pan = load_panel(th, '2016-01-01', '2019-12-31')
  oracle = pan.close.shift(-2) / pan.close.shift(-1) - 1     # == the label, by construction
  print('oracle rank_ic  :', round(per_factor_metrics(oracle, "ORACLE", pan, th)['rank_ic'], 4), '(expect ~1.0)')
  const = pan.close * 0 + 1
  print('constant rank_ic:', per_factor_metrics(const, "CONST", pan, th)['rank_ic'], '(expect ~0 or nan)')
  PY
  ```
  A rank IC well below 1.0 for the oracle means the label alignment or the shift is wrong —
  fix that before trusting any other number.

---

### Task 9 — `eval/gates.py` (`F_Θ`, Eq. 9)

- **IMPLEMENT** `feasible(m, theta) -> (bool, list[str])`, returning the verdict **and the list
  of failed gate names** (the summarizer needs the reason, not just the boolean):
  `rank_ic ≥ γ_ic`, `rank_icir ≥ γ_ir`, `turnover ≤ τ_max` (which turnover is selected by
  `theta.gates.turnover_basis`), `rho_max ≤ ρ̄`, `cx ≤ γ_cx`. Evaluated on **train/valid only**
  (Property 3) — the operator passes IS / OOS-proxy metrics, never test-window metrics.
  Keep the existing AST `RedundancyChecker` (`consistency_checker.py`) as a cheap pre-screen in
  the runner; `ρ_max` is authoritative.
- **PATTERN** Thresholds come from `Θ`, **not** from `configs/experiment.yaml` — that block is
  decorative (Hazard 4).
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  from dataclasses import replace
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.gates import feasible
  th   = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  good = dict(rank_ic=0.05, rank_icir=0.4, turnover_book=0.08, rho_max=0.3, cx=120)
  print('good         :', feasible(good, th))
  for k, v in [('rank_ic', 0.01), ('rank_icir', 0.1), ('rho_max', 0.9), ('cx', 400)]:
      print(f'fails {k:10s}:', feasible({**good, k: v}, th))
  tight = replace(th, gates=replace(th.gates, rho_bar=0.2))
  print('rho_bar 0.2  :', feasible(good, tight), '(was feasible; must now fail)')
  PY
  ```
  Expect `(True, [])` for `good`; each perturbation returning `(False, ['<that gate>'])`; and the
  tightened `Θ` flipping a previously-feasible factor to infeasible.

---

### Task 10 — `eval/scoring.py` (Eq. 10–12)

- **IMPLEMENT**
  - `rank(value, incumbent_values) -> float` = `(1/|zoo|)·Σ 1[value < v']` (Eq. 11); define
    `rank = 0.0` for an empty zoo so `e_j = 1.0` (the first factor is trivially best).
  - `dimension_scores(m, zoo_metrics, theta) -> dict[str, float]` — one `e_j` per dimension,
    using the **ranking scalar** below and sign-flipping lower-is-better dimensions **before**
    ranking:

    | Dimension | Ranking scalar | Direction |
    | :--- | :--- | :--- |
    | `effectiveness` | `net_ir` | higher better |
    | `arr` | `net_arr` | higher better |
    | `stability` | `rank_icir` | higher better |
    | `turnover` | `turnover_*` | **lower** better → negate |
    | `diversity` | `rho_max` | **lower** better → negate |
    | `overfit` | *not repository-ranked* — see below | — |
    | `decay` | `decay_slope` | higher better |

    > Effectiveness ranks on **net IR**, not RankIC. RankIC and IC are reported, and RankIC is
    > what the *gate* uses; the *score* is the realized economic edge. That is precisely what
    > makes the objective net-of-cost rather than a relabelled IC.

  - **Over-fitting risk is the documented exception** ("assessed from the factor's structure and
    refinement history rather than by repository rank"). Compute it directly in `[0,1]`:
    `e_overfit = 0.5·(1 − min(1, cx/γ_cx)) + 0.5·(1 − min(1, max(0, is_oos_gap)/δ_ref))`.
  - `utility(m, zoo_metrics, theta) -> float` — dispatch on `theta.utility.mode`:
    `net_ir` → `m['net_ir']`; `net_arr` → `m['net_arr']`; `weighted_rank` → `Σ ω_j e_j`.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.scoring import rank, dimension_scores, utility
  th = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  print('empty zoo     :', rank(0.5, []),      '(expect 0.0 -> e=1)')
  print('beats all     :', rank(9.0, [1,2,3]), '(expect 0.0 -> e=1)')
  print('beaten by all :', rank(0.0, [1,2,3]), '(expect 1.0 -> e=0)')
  print('median        :', round(rank(2.0, [1,2,3]), 4), '(expect 0.3333)')
  zoo = [dict(net_ir=0.5, net_arr=0.03, rank_icir=0.2, turnover_book=0.09,
              rho_max=0.5, cx=150, is_oos_gap=0.01, decay_slope=0.0) for _ in range(4)]
  best  = dict(net_ir=9.0,  net_arr=0.9,  rank_icir=9.0,  turnover_book=0.00,
               rho_max=0.0, cx=1,      is_oos_gap=-1.0, decay_slope=9.0)
  worst = dict(net_ir=-9.0, net_arr=-0.9, rank_icir=-9.0, turnover_book=0.99,
               rho_max=1.0, cx=10_000, is_oos_gap=9.0,  decay_slope=-9.0)
  eb, ew = dimension_scores(best, zoo, th), dimension_scores(worst, zoo, th)
  print('best  e_j     :', {k: round(v, 3) for k, v in eb.items()})
  print('worst e_j     :', {k: round(v, 3) for k, v in ew.items()})
  print('U(best)       :', round(utility(best,  zoo, th), 4), '(expect ~1.0)')
  print('U(worst)      :', round(utility(worst, zoo, th), 4), '(expect ~0.0)')
  print('all in [0,1]  :', all(0 <= v <= 1 for v in {**eb, **ew}.values()))
  PY
  ```
  Expect `e_j = 1.0` on every ranked dimension for `best`, `0.0` for `worst`, `U` bracketing
  `[0, 1]`, and `all in [0,1]: True`. **A `U(best)` below 0.95 means a sign flip is missing** —
  check `turnover` and `diversity` first.

---

### Task 11 — `eval/operator.py` + `eval/ledger.py`

- **IMPLEMENT**
  - `EvaluationOperator(theta: Protocol)` with
    `evaluate(candidate_signal, candidate_expr, zoo_signals, zoo_metrics) -> dict`:
    1. `zoo_hash = sha256("|".join(sorted(md5(expr) for expr in zoo_signals)))[:16]`
       (`"empty"` for the first factor).
    2. Per-factor metrics on IS (`theta.splits.search_is`) and the OOS proxy
       (`theta.splits.search_oos`).
    3. `rho_max` against `zoo_signals`.
    4. Combiner refit on `zoo ∪ {f}` → `ŷ` → `g` → `w, w_drift` → costs → `r_net` → strategy
       metrics.
    5. Gates → `feasible`, `failed_gates`.
    6. `e_j` and `U`.
    7. Return a flat dict: `{m_*, e_*, U, feasible, failed_gates, theta_hash, zoo_hash, zoo_size}`.
    8. `logger.info(f"E_theta eval: factor={expr[:40]} theta={theta.hash} zoo={zoo_hash} "
       f"U={U:.4f} feasible={feasible}")` — **on every call, unconditionally.** This log line is
       the primary whole-system signal (§7.1).
  - `Ledger(path)` — append-only JSONL, one row per evaluation:
    `{ts, factor_id, factor_expr, theta_hash, zoo_hash, zoo_size, metrics, e, U, feasible, failed_gates}`.
    Open in `"a"` mode and `flush()` per write. This is the reproducible trial record Properties
    1–2 make available, and the substrate for the (out-of-scope) deflated-Sharpe work.
- **PATTERN** Purity: identical `(f, zoo, Θ)` ⇒ identical output.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import json
  from dataclasses import replace
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_factor_signal
  from quantaalpha.eval.operator import EvaluationOperator
  th = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  f  = load_factor_signal("ZSCORE(TS_SUM(($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8), 10))")
  a = EvaluationOperator(th).evaluate(f, "EXPR_A", {}, [])
  b = EvaluationOperator(th).evaluate(f, "EXPR_A", {}, [])
  print('bit-identical :', json.dumps(a, sort_keys=True, default=str) ==
                            json.dumps(b, sort_keys=True, default=str))
  print('theta_hash    :', a['theta_hash'], '| zoo_hash:', a['zoo_hash'])
  print('metric keys   :', sorted(k for k in a if k.startswith(('m_', 'e_'))))
  print('U / feasible  :', round(a['U'], 4), a['feasible'], a['failed_gates'])
  c = EvaluationOperator(replace(th, costs=replace(th.costs, kappa2=0.05))).evaluate(f, "EXPR_A", {}, [])
  print('Θ changed hash:', a['theta_hash'] != c['theta_hash'])
  print('higher κ₂ ⇒ lower net_arr:', c['m_net_arr'] <= a['m_net_arr'])
  PY
  ```
  Expect `bit-identical: True`, both hashes present, and `higher κ₂ ⇒ lower net_arr: True`.

  Then the **nesting check** — the single most valuable validation in this whole plan, because it
  proves the new engine reduces to the old objective:
  ```bash
  python - <<'PY'
  from dataclasses import replace
  from quantaalpha.eval.protocol import load_protocol
  from quantaalpha.eval.data import load_factor_signal
  from quantaalpha.eval.operator import EvaluationOperator
  th   = load_protocol('quantaalpha/eval/protocol_csi300.yaml')
  free = replace(th, costs=replace(th.costs, kappa0=0, kappa1=0, kappa2=0, beta_per_day=0))
  f = load_factor_signal("ZSCORE(TS_SUM(($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8), 10))")
  a = EvaluationOperator(th).evaluate(f, "E", {}, [])
  b = EvaluationOperator(free).evaluate(f, "E", {}, [])
  print(f"net_arr   costed {a['m_net_arr']:+.6f}   frictionless {b['m_net_arr']:+.6f}")
  print(f"net_ir    costed {a['m_net_ir']:+.6f}   frictionless {b['m_net_ir']:+.6f}")
  print('cost_bps == 0 when all κ=0 :', abs(b['m_cost_bps']) < 1e-12)
  print('costs strictly reduce ARR  :', a['m_net_arr'] < b['m_net_arr'])
  print('IC unchanged (per-factor)  :', abs(a['m_rank_ic'] - b['m_rank_ic']) < 1e-12)
  PY
  ```
  All four must hold. The last is the provenance split working: per-factor metrics are invariant
  to the cost model; only strategy-level ones move.

---

### Task 12 — `factors/net_cost_runner.py` (the treatment-arm runner)

- **IMPLEMENT** `NetCostFactorRunner(QlibFactorRunner)`:
  - `__init__(self, scen)` — build
    `EvaluationOperator(load_protocol(os.environ.get("QA_PROTOCOL", "quantaalpha/eval/protocol_csi300.yaml")))`
    once; open the `Ledger` at `os.environ.get("QA_LEDGER", "data/results/eval_ledger.jsonl")`.
  - `get_cache_key(self, exp, **kwargs)` — `md5_hash(<base task-info string> + "|" + self.theta.hash)`.
    **This is the fix for Hazard 2**: the base key is task-info only, so without this the two arms
    silently share cached results.
  - `develop(self, exp, use_local=True)`, decorated
    `@cache_with_pickle(NetCostFactorRunner.get_cache_key, CachedRunner.assign_cached_result)`:
    1. `new_factors = self.process_factor_data(exp)` (**runner.py:204**) — raise
       `FactorEmptyError` if empty, preserving the skip-loop semantics at `workflow.py:116`.
    2. `zoo_signals = self.process_factor_data(exp.based_experiments)` when
       `exp.based_experiments` is non-empty (**runner.py:89** does exactly this); otherwise fall
       back to the `factor_cache/{md5}.pkl` store.
    3. `res = self.op.evaluate(new_factors, expr, zoo_signals, zoo_metrics)`; append to the ledger.
    4. Set `exp.result` to a **`pd.Series`** whose index carries **both** the seven canonical
       names — `IC`, `ICIR`, `RankIC`, `RankICIR`, `annualized_return`, `information_ratio`,
       `max_drawdown` — so `controller._extract_metrics` (**795**) keeps populating
       `backtest_metrics`, **and** the new keys `U`, `feasible`, `theta_hash`, `zoo_hash`,
       `rho_max`, `turnover_book`, `turnover_solo`, `cx`, `cost_bps`, `e_*`.
       Map `annualized_return ← net_arr` and `information_ratio ← net_ir`, so the reported
       headline numbers *are* the net-of-cost ones.
    5. `return exp` — `loop.py:176` and `assign_cached_result` (**:52**) then work unchanged.
  - **No non-underscore methods** on any `LoopMeta` class (`workflow.py:44-64`).
- **MANUAL VALIDATION**
  ```bash
  python -c "
  from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as N
  from quantaalpha.factors.runner import QlibFactorRunner as Q
  import inspect
  print('subclass      :', issubclass(N, Q))
  print('overrides key :', 'get_cache_key' in N.__dict__)
  print('overrides dev :', 'develop' in N.__dict__)
  print('theta in key  :', 'theta' in inspect.getsource(N.get_cache_key))"
  # the plugin actually resolves through the env var:
  QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner python -c "
  from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING as S
  from quantaalpha.core.utils import import_class
  print('resolved runner:', import_class(S.runner))"
  ```
  All four booleans `True`, and the last line must print `NetCostFactorRunner`. If it prints
  `QlibFactorRunner`, the env var is not reaching the settings singleton — check that it is
  exported (not just set) in the shell that launches `run.sh`.

---

### Task 13 — Extend `controller._extract_metrics` (additive, backward-compatible)

- **IMPLEMENT** In `pipeline/evolution/controller.py:795-859`, after the existing `index_mapping`
  loops (**840** / **849**), add a pass copying a whitelist of extra keys from the result's index
  into `metrics` when present: `U, feasible, theta_hash, zoo_hash, rho_max, turnover_book,
  turnover_solo, cx, cost_bps`, plus `e_effectiveness … e_decay`. Handle `pd.Series` and
  `pd.DataFrame` (first column). Coerce numerics with `float()`; keep `theta_hash`/`zoo_hash` as
  strings and `feasible` as `bool`. **Do not touch the canonical first-match-wins mapping** —
  Arm A's Qlib DataFrame has none of these keys and must behave exactly as before.
- **MANUAL VALIDATION**
  ```bash
  python - <<'PY'
  import pandas as pd
  from quantaalpha.pipeline.evolution.controller import EvolutionController, EvolutionConfig
  c = EvolutionController(EvolutionConfig(num_directions=1, max_rounds=1))
  new = pd.Series({'IC':0.02,'ICIR':0.1,'RankIC':0.03,'RankICIR':0.2,
                   'annualized_return':0.05,'information_ratio':0.4,'max_drawdown':-0.1,
                   'U':0.71,'feasible':True,'theta_hash':'abc123','zoo_hash':'def456',
                   'rho_max':0.31,'turnover_book':0.09,'cx':120})
  m = c._extract_metrics(new)
  print('canonical kept :', m['RankIC'] == 0.03, m['information_ratio'] == 0.4)
  print('extras present :', m.get('U'), m.get('feasible'), m.get('theta_hash'), m.get('rho_max'))
  old = pd.DataFrame({'v': {'IC':0.02, 'Rank IC':0.03, 'annualized_return':0.05}})
  print('control arm    :', c._extract_metrics(old))
  print('no leakage     :', 'U' not in c._extract_metrics(old))
  PY
  ```
  Expect the canonical keys unchanged, the extras surfaced for the Series, and
  `no leakage: True` for the DataFrame.

---

### Task 14 — Arm-selectable primary metric (`trajectory.py`)

- **IMPLEMENT** In `pipeline/evolution/trajectory.py`, add module-level:
  ```python
  _PRIMARY_METRIC   = os.environ.get("QA_PRIMARY_METRIC", "RankIC")
  _REQUIRE_FEASIBLE = os.environ.get("QA_REQUIRE_FEASIBLE", "false").lower() in ("1", "true", "yes")
  ```
  `get_primary_metric()` (**90-92**) → `self.backtest_metrics.get(_PRIMARY_METRIC)`.
  `is_successful()` (**94-97**) → primary is not None **and** primary > 0 **and**
  (`not _REQUIRE_FEASIBLE or bool(self.backtest_metrics.get("feasible", True))`).
  Arm A sets neither (defaults reproduce today's behaviour exactly); Arm B sets
  `QA_PRIMARY_METRIC=U`, `QA_REQUIRE_FEASIBLE=true`. **Keep `RankIC` in `backtest_metrics` for
  both arms** so they stay mutually reportable.
  Also fix the **three** log strings that hardcode `RankIC=` — `factor_mining.py:492` (parallel
  branch), **:531** (serial branch), **:549** (the "Top N trajectories" summary) — to print
  `{_PRIMARY_METRIC}=`. Otherwise Arm B's logs claim to show RankIC while showing `U`, which will
  waste someone's afternoon.
- **MANUAL VALIDATION**
  ```bash
  # Arm B semantics
  QA_PRIMARY_METRIC=U QA_REQUIRE_FEASIBLE=true python - <<'PY'
  from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory as T, RoundPhase
  t = T(trajectory_id='x', direction_id=0, round_idx=0, phase=RoundPhase.ORIGINAL)
  t.backtest_metrics = {'RankIC': 0.10, 'U': 0.62, 'feasible': True}
  print('primary           :', t.get_primary_metric(), '(expect 0.62)')
  print('success           :', t.is_successful(),      '(expect True)')
  t.backtest_metrics['feasible'] = False
  print('infeasible blocked:', not t.is_successful(),  '(expect True)')
  PY
  # Arm A semantics (defaults) — must be byte-for-byte today's behaviour
  python - <<'PY'
  from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory as T, RoundPhase
  t = T(trajectory_id='x', direction_id=0, round_idx=0, phase=RoundPhase.ORIGINAL)
  t.backtest_metrics = {'RankIC': 0.10, 'U': 0.62, 'feasible': False}
  print('primary :', t.get_primary_metric(), '(expect 0.10)')
  print('success :', t.is_successful(),      '(expect True — feasibility ignored)')
  PY
  ```

---

### Task 15 — `factors/net_cost_feedback.py` + de-duplicate the metric list

**This is the highest-leverage change in the plan.** Today the generator is literally shown
flat-fee performance, and the feedback that shapes the next hypothesis never mentions cost.
`woolly-orbiting-pie.md` calls this "a large part of the contribution's actual mechanism".

- **IMPLEMENT**
  1. In `factors/feedback.py`, hoist the 4-element list duplicated at **42-47** and **88-93**
     into one module constant `FRICTIONLESS_METRICS`, referenced from both branches.
  2. `NetCostFactorFeedback(AlphaAgentQlibFactorHypothesisExperiment2Feedback)` overriding the
     `combined_result` payload passed to `factor_feedback_generation`, so the LLM sees:
     - `U` and each `e_j` **with its dimension name**, ordered worst-first (the model should see
       what to improve, not a scalar);
     - the raw `m(f)` for `net_ir`, `net_arr`, `mdd`, `turnover_book`, `cost_bps`, `rho_max`, `cx`;
     - `feasible`, and when false **`failed_gates` with the threshold that was missed** —
       "RankICIR 0.11 < γ_ir 0.20". A rejection reason is far more actionable than a score;
     - `theta_hash`, so the transcript records which protocol produced the feedback.
  3. Keep `json_mode=True` on the call (**feedback.py:304**) — it works with the current minimax
     setup given the trailing-comma repair at `client.py:921`.
- **MANUAL VALIDATION**
  ```bash
  grep -c "excess_return_without_cost" quantaalpha/factors/feedback.py    # expect 3 (one constant)
  python -c "
  from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback as N
  from quantaalpha.factors.feedback import AlphaAgentQlibFactorHypothesisExperiment2Feedback as A
  print('subclass:', issubclass(N, A))"
  ```
  Then **read the actual prompt text** — the point of the task is what the model sees:
  ```bash
  python - <<'PY'
  from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback
  fb = NetCostFactorFeedback.__new__(NetCostFactorFeedback)
  demo = {'U': 0.42, 'feasible': False, 'failed_gates': ['rank_icir'], 'theta_hash': 'abc123',
          'e_effectiveness': 0.8, 'e_turnover': 0.1, 'e_diversity': 0.2, 'e_arr': 0.5,
          'e_stability': 0.15, 'e_overfit': 0.6,
          'm_net_ir': 0.31, 'm_net_arr': 0.021, 'm_mdd': -0.14, 'm_turnover_book': 0.098,
          'm_cost_bps': 18.4, 'm_rho_max': 0.66, 'm_cx': 180, 'm_rank_icir': 0.11}
  print(fb._format_metric_block(demo))     # name the helper with a leading underscore
  PY
  ```
  Read the output. It must name the weakest dimensions, state the failed gate **with its
  threshold**, and quote cost in basis points. **If a human could not tell from it what to change
  about the factor, rewrite it** — that is the entire mechanism of the contribution.

---

### Task 16 — One RNG seeding entry point + close the parallel gate gap

- **IMPLEMENT**
  1. `cli.py:app()` (**29**): before `fire.Fire(...)`, read `seed` from `configs/experiment.yaml`
     (via `load_run_config`, as `factor_mining.py:601` does) or `QA_SEED`, defaulting to 42; then
     `random.seed(s)` and `numpy.random.seed(s)`. Log the seed.
  2. Add `seed: 42` to `configs/experiment.yaml` and `configs/experiment_paper.yaml`.
  3. `run.sh`: `export PYTHONHASHSEED=${QA_SEED:-42}` **before** the `quantaalpha` invocation —
     it must be set pre-interpreter to have any effect.
  4. Add a comment noting that `TrajectoryPool.select_parents_for_crossover` (`trajectory.py:251`,
     shuffles at **330**/**338**) is dead; the live unseeded paths are `crossover.py:370/412/432/502`
     and `mutation.py:217`.
  5. **Close Hazard 3**: give `_run_tasks_parallel` (`factor_mining.py:271`) a `quality_gate_cfg`
     parameter, thread it through `_parallel_task_worker` (**199**) into `_run_evolution_task`
     (**119**), and pass it at the call site (~**468**). `parallel_enabled` is now `true`, so
     leaving this is a live footgun the moment `consistency_enabled` is switched on.
- **MANUAL VALIDATION**
  ```bash
  grep -n "PYTHONHASHSEED" run.sh
  grep -n "seed:" configs/experiment.yaml configs/experiment_paper.yaml
  for i in 1 2; do QA_SEED=7 python -c "
  import quantaalpha.cli, random, numpy
  print(random.random(), numpy.random.random())"; done      # both lines must match
  python -c "
  import inspect, quantaalpha.pipeline.factor_mining as fm
  print('parallel forwards quality_gate_cfg:',
        'quality_gate_cfg' in inspect.getsource(fm._run_tasks_parallel))"
  ```

---

### Task 17 — Arm A (control): decide the scale, then run

**Decision required before running.** A completed 12-factor smoke run already exists
(`data/factorlib/all_factors_library.json` — IC 0.04943 / RankIC 0.04790 / ARR 3.79% /
IR 0.4705 / MDD −13.54%).

| | Reuse the existing smoke run | Re-run at paper scale (`experiment_paper.yaml`) |
| :--- | :--- | :--- |
| Cost | free | 10 directions × 5 rounds × 3 factors — hours of local inference |
| Claim strength | weak — 12 factors is a thin repository, and both `ρ_max` and relative rank are repository-size-sensitive | strong — ~150 factors, matches the paper's protocol |

**Recommendation: re-run at paper scale.** `U` is defined by *rank against the repository*
(Eq. 11), so with `|zoo| = 12` the scoring resolution is 1/12 and the diversity gate has almost
nothing to bite on. Whichever is chosen, **both arms must use the same scale**.

- **IMPLEMENT**
  ```bash
  # .env first: CHAT_TEMPERATURE=0.0, CHAT_SEED=42
  EXPERIMENT_ID=armA_control ./run.sh "<the paper's initial direction>" armA
  python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml \
    --factor-source combined --factor-json all_factors_library_armA.json
  ```
  `run.sh:73-78` already isolates `WORKSPACE_PATH` and `PICKLE_CACHE_FOLDER_PATH_STR` per
  `EXPERIMENT_ID` — that is what keeps the arms' pickle caches apart today (Hazard 2 is the
  belt-and-braces version). **Preserve Arm A's factor library JSON**; Task 19 re-scores it.
- **MANUAL VALIDATION** The run completes; the library JSON is non-empty and roughly the expected
  size; the standalone backtest prints IC / RankIC / ARR / IR / MDD. **No assertion on absolute
  values** — the model is minimax-m3, not the paper's GPT-5.2, so divergence from
  0.0472 / 0.0459 / 4.68% / 0.6453 / 11.80% is expected. Only the A-vs-B delta is claimed.

---

### Task 18 — Arm B (treatment): run under the new objective

- **IMPLEMENT**
  ```bash
  EXPERIMENT_ID=armB_treatment \
  QA_PRIMARY_METRIC=U QA_REQUIRE_FEASIBLE=true \
  QA_PROTOCOL=quantaalpha/eval/protocol_csi300.yaml \
  QA_LEDGER=data/results/ledger_armB.jsonl \
  QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner \
  QLIB_FACTOR_SUMMARIZER=quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback \
  ./run.sh "<the paper's initial direction>" armB
  ```
  Same config, same scale, same seed, same model as Arm A. **Only the objective differs.**
- **MANUAL VALIDATION** — §7, which is written for exactly this run. Do §7.1 and §7.5 **while the
  run is still going**; if either fails, kill it rather than burn hours on an inert change.

---

### Task 19 — Head-to-head under one `E_Θ`

- **IMPLEMENT** `scripts/qa_compare_arms.py`, producing a markdown table. Score **both** factor
  sets with the **same** `EvaluationOperator` on `final_test` (2022-01-01 → 2025-12-26) —
  otherwise it compares engines, not objectives — and add the Qlib standalone backtest as the
  comparability anchor.

  | Metric | Arm A (RankIC) | Arm B (`U`) | Paper (GPT-5.2) |
  | :--- | :--- | :--- | :--- |
  | IC / RankIC | | | 0.0472 / 0.0459 |
  | ICIR / RankICIR | | | — |
  | **Net IR** (E_Θ) | | | — |
  | **Net ARR** (E_Θ) | | | — |
  | MDD | | | 11.80% |
  | Mean turnover (book / solo) | | | — |
  | Mean `ρ_max` | | | — |
  | Mean `cx` | | | — |
  | IS→OOS gap | | | — |
  | Qlib ARR / IR (anchor) | | | 4.68% / 0.6453 |

- **INTERPRETATION — read before drawing conclusions.**
  - **Expected:** Arm A wins on raw IC/RankIC; Arm B wins on net IR / net ARR and shows lower mean
    turnover and lower `ρ_max`. That gap *is* the §3.2 claim.
  - **If Arm B also wins on raw RankIC, be suspicious, not pleased.** Likeliest causes, in order:
    the feasibility gate is not filtering (check `failed_gates` frequency in the ledger);
    `ρ_max` was computed against an empty zoo throughout (check `zoo_size` — it must grow); or
    Arm B read Arm A's pickle cache (check the two `theta_hash` and
    `PICKLE_CACHE_FOLDER_PATH_STR` values differ).
  - **If the arms produce near-identical factors**, the objective is not reaching the generator —
    go straight to §7.5.
- **MANUAL VALIDATION**
  ```bash
  wc -l data/results/ledger_armB.jsonl         # one row per evaluation
  python - <<'PY'
  import json, collections
  rows = [json.loads(l) for l in open('data/results/ledger_armB.jsonl')]
  print('rows          :', len(rows))
  print('theta hashes  :', set(r['theta_hash'] for r in rows), '(expect exactly one)')
  print('zoo grows     :', [r['zoo_size'] for r in rows][:12], '...')
  print('feasible rate :', round(sum(r['feasible'] for r in rows)/len(rows), 3))
  print('gate failures :', collections.Counter(g for r in rows for g in r['failed_gates']))
  us = [r['U'] for r in rows if r['feasible']]
  print('U range       :', round(min(us), 4), '->', round(max(us), 4))
  PY
  ```
  Exactly one `theta_hash` across the run (Property 1 held); `zoo_size` strictly growing; and a
  feasible rate **strictly between 0 and 1** — 0 means the gates are too tight, 1 means they are
  inert; either way the gate is not doing work.

---

## 7. WHOLE-SYSTEM VALIDATION — "is the change live, and is it working?"

Run these against the **Arm B** run. They are ordered so the first failure tells you where the
wiring broke. Set `RUN=log/<the Arm B log dir>`.

### 7.1 Is the new engine being called at all?
```bash
grep -rc "E_theta eval:" $RUN | grep -v ':0$'
grep -rm3 "E_theta eval:" $RUN
```
**Expect** one line per factor evaluated, each carrying `theta=<hash>` and `zoo=<hash>`.
**Zero lines ⇒ the plugin swap did not take.** Confirm with:
```bash
python -c "from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING as S; print(S.runner)"
```
(run it with the same env exported). This is the single most common failure and it is silent —
the loop keeps working, just with the old objective.

### 7.2 Is the protocol frozen for the whole run?
```bash
grep -rho "theta=[0-9a-f]*" $RUN | sort -u
```
**Expect exactly one hash.** More than one means the YAML was edited mid-run, or two protocols
are in play — Property 1 violated, and the run is not reportable.

### 7.3 Did the metric vector reach the trajectory pool?
```bash
python - <<'PY'
import json, glob
p = sorted(glob.glob('log/*/trajectory_pool.json'))[-1]
d = json.load(open(p)); tr = d['trajectories']
tr = list(tr.values()) if isinstance(tr, dict) else tr
print('pool:', p, '| trajectories:', len(tr))
for t in tr:
    m = t.get('backtest_metrics') or {}
    print(f"  {t['trajectory_id']:14s} phase={str(t.get('phase')):10s} "
          f"RankIC={m.get('RankIC')} U={m.get('U')} feasible={m.get('feasible')} "
          f"rho={m.get('rho_max')} TO={m.get('turnover_book')}")
PY
```
**Expect** every scored trajectory to carry **both** `RankIC` and `U`, plus `feasible`, `rho_max`,
`turnover_book`.
- `U` absent/`None` ⇒ **Task 13 did not land** — `_extract_metrics` is dropping the extras.
- `backtest_metrics` empty entirely ⇒ the runner returned a `dict`, not a `pd.Series` (the
  documented failure mode — `controller.py:795` is pandas-only).

### 7.4 Did selection actually use `U`?
```bash
grep -rE "Task done|Trajectory done" $RUN | head
```
**Expect** the printed metric label to be `U=` (Task 14 fixed it) and its value to match the pool.
Then confirm ordering: the trajectory with the highest `U` — not the highest `RankIC` — should be
selected as a mutation parent in the next round. Cross-check `$RUN/evolution_state.json` against
the pool.

### 7.5 Did the generator actually see cost?
This is the mechanism check. The others prove plumbing; this one proves *contribution*.
```bash
grep -rioE "net_ir|net-of-cost|turnover|cost_bps|rho_max|failed gate|basis point" $RUN \
  | sed 's/.*://' | sort | uniq -c | sort -rn
```
**Expect** cost vocabulary present in the logged feedback. **Zero hits ⇒ the summarizer swap did
not take**, and the whole objective change is inert from the generator's point of view — Arm B
will be Arm A with extra logging. Then read one prompt end to end and ask whether *you* could act
on it.

### 7.6 Did the library store the vector?
```bash
python - <<'PY'
import json
d = json.load(open('data/factorlib/all_factors_library_armB.json'))
fs = d['factors']; fs = list(fs.values()) if isinstance(fs, dict) else fs
print('factors:', len(fs))
br = fs[0].get('backtest_results', {})
print('keys   :', sorted(br))
for k in ('U','feasible','theta_hash','zoo_hash','rho_max','turnover_book','cx'):
    print(f'  {k:14s}: {br.get(k)}')
PY
```
**Expect** the full vector, stored without any schema migration (`library.py:295` passes dicts
through and flattens Series).

### 7.7 Three adversarial probes (cheap, and they catch real bugs)

The strongest "is it working" evidence available without a test suite. Each is a single
re-scoring pass over Arm B's existing library — **no mining, no LLM spend.**

```bash
# (a) NEGATIVE CONTROL — zero the costs; net metrics must collapse onto gross
sed -e 's/kappa0: 0.0020/kappa0: 0.0/' -e 's/kappa1: 0.10/kappa1: 0.0/' \
    -e 's/kappa2: 0.01/kappa2: 0.0/'   -e 's/beta_per_day: 0.0004/beta_per_day: 0.0/' \
    quantaalpha/eval/protocol_csi300.yaml > /tmp/theta_free.yaml
LIB=data/factorlib/all_factors_library_armB.json
python scripts/qa_eval_probe.py --library $LIB --protocol /tmp/theta_free.yaml            --out /tmp/scores_free.json
python scripts/qa_eval_probe.py --library $LIB --protocol quantaalpha/eval/protocol_csi300.yaml --out /tmp/scores_costed.json
python -c "
import json
a=json.load(open('/tmp/scores_costed.json')); b=json.load(open('/tmp/scores_free.json'))
worse=sum(a[k]['m_net_arr'] < b[k]['m_net_arr'] for k in a)
print(f'costed ARR < frictionless ARR for {worse}/{len(a)} factors  (expect ALL)')
print('mean cost drag (bps/yr):', round(sum(b[k]['m_net_arr']-a[k]['m_net_arr'] for k in a)/len(a)*1e4, 1))"

# (b) SENSITIVITY — 10x the capacity knob; the U ranking must reshuffle
sed 's/kappa2: 0.01/kappa2: 0.10/' quantaalpha/eval/protocol_csi300.yaml > /tmp/theta_k10.yaml
python scripts/qa_eval_probe.py --library $LIB --protocol /tmp/theta_k10.yaml --out /tmp/scores_k10.json
python -c "
import json
from scipy.stats import spearmanr
a=json.load(open('/tmp/scores_costed.json')); c=json.load(open('/tmp/scores_k10.json'))
ks=sorted(a)
print('U rank corr (k2 x10):', round(spearmanr([a[k]['U'] for k in ks],[c[k]['U'] for k in ks]).statistic, 4))
print('  -> expect < 1.0; exactly 1.0 means kappa2 is not reaching U')"

# (c) GATE KILL SWITCH — an impossible redundancy ceiling must reject everything
sed 's/rho_bar:   0.70/rho_bar:   -1.0/' quantaalpha/eval/protocol_csi300.yaml > /tmp/theta_kill.yaml
python scripts/qa_eval_probe.py --library $LIB --protocol /tmp/theta_kill.yaml --out /tmp/scores_kill.json
python -c "
import json; s=json.load(open('/tmp/scores_kill.json'))
print('feasible count:', sum(v['feasible'] for v in s.values()), '/', len(s), '(expect 0)')"
```

`scripts/qa_eval_probe.py` is a thin loop: load the library JSON, pull each factor's signal via
`eval.data.load_factor_signal`, evaluate against the growing zoo under the given protocol, dump
`{factor_id: metrics}` to JSON. ~40 lines. It is a **manual diagnostic driver**, not a test
suite, and it doubles as the scoring engine for Task 19.

### 7.8 The one-line summary

After Arm B, all five of these should hold:

> The ledger has one row per evaluated factor, all carrying the **same** `theta_hash` and a
> **growing** `zoo_size`; the trajectory pool carries `U` **and** `RankIC` for every scored
> trajectory; the feedback text mentions cost; the feasible rate is strictly between 0 and 1; and
> zeroing the cost coefficients strictly raises `net_arr` for **every** factor.

If all five hold, the objective change is **live**. If Arm B's factors then show lower turnover
and lower `ρ_max` than Arm A's at comparable RankIC, the change is also **working**.

---

## 8. HAZARDS (each verified in the tree)

1. **`QLIB_FACTOR_` is a shared env prefix.** `AlphaAgentFactorBasePropSetting`,
   `FactorBasePropSetting`, and `FactorBackTestBasePropSetting` all declare it
   (`settings.py:50/63/76`), so `QLIB_FACTOR_RUNNER` swaps the runner in **all three** at once.
   Convenient here, but the env var cannot give the mining loop and the backtest loop different
   runners. If the arms ever need to diverge, edit `settings.py` or add a distinct prefix.
2. **Pickle-cache collision.** `CachedRunner.get_cache_key` (`components/runner/__init__.py:12-19`)
   hashes task-info strings only — no objective, no protocol. Without the Task 12 override the
   treatment arm silently serves control-arm results. Belt: `Θ.hash` in the key. Braces: per-arm
   `PICKLE_CACHE_FOLDER_PATH_STR`, which `run.sh:73-78` already does per `EXPERIMENT_ID`.
3. **Parallel mode drops `quality_gate_cfg`** (`factor_mining.py:271`), and `parallel_enabled` is
   now `true`. Harmless today only because `consistency_enabled: false` coincides with the
   fallback default at `loop.py:95`. Fixed in Task 16.
4. **`configs/experiment.yaml`'s `quality_gate` and `factor.complexity` blocks are decorative.**
   Only `consistency_enabled` is threaded; `complexity_enabled`/`redundancy_enabled` are hardcoded
   `True` at `proposal.py:360-361`; `consistency_strict_mode`, `max_correction_attempts`, and
   every complexity threshold are never read. **The new gates must read `Θ`, not this file.**
5. **`configs/backtest.yaml:11 random_seed: 42` is dead** — not consumed anywhere under
   `quantaalpha/backtest/`. Task 16 adds the real seeding entry point at `cli.py:29`.
6. **`_extract_metrics` is pandas-only** (`controller.py:795`). A `dict` result ⇒ all-`None`
   metrics ⇒ `is_successful()` False ⇒ silently zero successful trajectories. Emit a `pd.Series`.
7. **`daily_pv.h5` has `$return`, not `$vwap`.** The paper's six base features are
   O/H/L/C/V/**VWAP**; the mining prompts and panel use `$return` as the sixth. The evaluation
   engine sidesteps this by reading prices from Qlib (which has both), but the *mined factor pool*
   is still a `$return` pool, not the paper's `$vwap` pool. State this in the write-up; do not
   silently claim parity.
8. **Loop helpers must be `_`-prefixed** (`workflow.py:44-64`), or they silently become pipeline
   steps.

---

## 9. ACCEPTANCE CRITERIA

- [ ] `E_Θ` is a pure function of `(f, zoo, Θ)` — two identical calls return byte-identical output.
- [ ] `Θ` is frozen (`FrozenInstanceError` on mutation), hashed, and logged at **every**
      evaluation; a mutated YAML yields a different hash.
- [ ] `zoo_hash` is recorded on every ledger row, and `zoo_size` grows across the run.
- [ ] The combiner refit is deterministic — the same standalone backtest run twice gives identical
      metrics (Task 1).
- [ ] Zero-cost / zero-latency `Θ` **nests** the frictionless objective: `cost_bps == 0`,
      per-factor metrics unchanged, strategy metrics strictly better.
- [ ] No look-ahead: truncating the panel leaves earlier fill prices and metrics unchanged.
- [ ] The portfolio map is the baseline's top-50 / drop-5 construction; `Σ|w| ≤ 1`; book turnover
      ≤ `n_drop/topk`.
- [ ] Arm B optimizes `U` (`QA_PRIMARY_METRIC=U`); Arm A optimizes `RankIC` (defaults) — one
      `StrategyTrajectory` class serves both.
- [ ] The library JSON stores the full metric vector with no schema migration.
- [ ] The pickle cache key includes `Θ.hash`.
- [ ] The feedback the LLM sees is net-of-cost and dimension-targeted, with failed gates and their
      thresholds — verified by reading an actual logged prompt (§7.5).
- [ ] No edits to the loop, the DSL, CoSTEER, or the mutation/crossover prompt bodies — only
      `QLIB_FACTOR_*` env swaps plus the bounded edits in Tasks 1, 13, 14, 15, 16.
- [ ] The in-loop test window (2021) is not widened; `final_test` is touched only in Task 19.
- [ ] Both arms complete at the same scale; the comparison table and both ledgers exist.
- [ ] All three §7.7 adversarial probes behave as specified.

## 10. OPEN DECISIONS (need a human call)

1. **Arm scale** — reuse the 12-factor smoke library, or re-run at paper scale?
   *Recommendation: paper scale*, because `U` is repository-relative and 12 incumbents give a
   scoring resolution of 1/12. Blocks Tasks 17–19 only; everything before them is unaffected.
2. **`signed: false` (long-only, baseline-comparable) vs `signed: true` (dollar-neutral
   long–short).** The `.tex` admits both; long-only is what keeps the numbers comparable to the
   paper, and A-share shorting is economically strained anyway. *Recommendation: long-only as
   primary, signed as a single sensitivity pass* — and note that `β` is inert under long-only, so
   the borrow term is untested until that pass runs.
3. **`turnover_basis: book` vs `solo`.** `book` is faithful to Eq. 6 but structurally capped at
   0.10, so the gate never binds. *Recommendation: keep `book`, record `solo`, and report that the
   capacity pressure in this instantiation comes from `κ₂`, not from the turnover gate.*
4. **`κ₂` sensitivity sweep.** It is the capacity knob and the most sensitive parameter.
   *Recommendation: report `κ₂ ∈ {0.005, 0.01, 0.02, 0.05}` rather than a single point estimate.*

## 11. OUT OF SCOPE

Per the paper's own scoping ("Scope and roadmap"): the falsification critic (deflated Sharpe,
multiple-testing correction) and the decay-monitored deployment library. The ledger built in
Task 11 — carrying `theta_hash`, `zoo_hash`, and the full metric vector per trial — is the
substrate both will need, and is built now so that work is not blocked later.
