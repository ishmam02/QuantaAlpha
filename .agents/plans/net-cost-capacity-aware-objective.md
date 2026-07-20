# Feature: Net-of-Cost, Capacity-Aware Objective for QuantaAlpha

Read these files before implementing. Pay attention to naming of existing utils, types, and models.

This plan elaborates `woolly-orbiting-pie.md` (strategy) against the formal objective in
`problem_formulation.tex`, using `Get_Started.md` as the codebase map. Every `file:line`
reference below was verified against the working tree on 2026-07-20.

## Feature Description

QuantaAlpha currently optimizes a **frictionless** objective: `f* = argmax L(f(X), y) − λR(f)`
where `L` is RankIC. `StrategyTrajectory.get_primary_metric()` returns `backtest_metrics.get("RankIC")`
(`trajectory.py:90-92`) and `is_successful()` is `RankIC > 0` (`trajectory.py:94-97`). The feedback
the LLM sees is four `excess_return_without_cost.*` string literals (`feedback.py:42-47` and `88-93`).

We replace this with the paper's objective `f* = argmax_{f ∈ F_Θ} U(m(f))`, `m(f) = E_Θ(f)`, evaluated
by a **frozen, deterministic engine** `E_Θ` that prices transaction cost **and** execution latency
inside the objective, scores factors on a 7-dimensional quality vector, ranks them relative to the
evolving repository, and enforces hard feasibility gates. The **factor-generation process is held
fixed** — we change only *what is optimized* and *how it is measured*. No edits to the loop, the DSL,
CoSTEER, or the mutation/crossover operators' prompts.

Three deliverables, run as a controlled A/B:
1. Point the LLM at **Ollama Cloud / `kimi-k2.5:cloud`**.
2. Re-run the paper's CSI 300 experiment as a **control arm** (same model, same protocol, old objective).
3. Implement the new objective and run the **treatment arm** for a like-for-like comparison.

## User Story

As a quant researcher, I want the mining loop to optimize a net-of-cost, capacity-aware utility
computed by a frozen deterministic engine, so that the factors it discovers are tradeable after
costs and latency rather than illiquid, high-turnover correlations that vanish once trading is priced in.

## Problem Statement

Maximizing RankIC systematically rewards signals concentrated in small/illiquid names with high
turnover — exactly the signals whose apparent edge is consumed by spread, slippage, impact, and
latency. The current pipeline (a) shows the generator cost-free performance, (b) has no slippage /
impact / borrow / shorting, (c) trains on close-to-close labels but executes at the open (label/fill
mismatch), and (d) uses AST-subtree redundancy where the formulation requires cross-sectional
correlation `ρ_max`.

## Solution Statement

Add a self-contained `quantaalpha/eval/` package implementing `E_Θ` as a pure function of `(f, Θ)`,
wire it into the loop via the existing plugin mechanism (`QLIB_FACTOR_RUNNER` / `QLIB_FACTOR_SUMMARIZER`
env vars — `settings.py:50,63,76`), switch the trajectory's primary metric to `U` (arm-selectable),
store the metric vector in the factor library with no schema migration, add a single RNG-seeding
entry point for a clean A/B, and report both arms side-by-side on the 2022–2025 test window.

## Feature Metadata
- **Type**: Enhancement (new objective + evaluation engine; generation process held fixed)
- **Complexity**: High
- **Systems Affected**: `quantaalpha/eval/` (new), `quantaalpha/factors/` (runner, feedback, library), `quantaalpha/pipeline/evolution/trajectory.py`, `quantaalpha/llm/{config,client}.py`, `configs/`, `pyproject.toml`, `tests/eval/` (new), `run.sh`, `launcher.py`/`cli.py`
- **Dependencies**: `numpy>=1.24,<2.0`, `pandas>=1.5,<3.0` (both pinned in `requirements.txt:22-23`), `scipy` (**absent — add**), `pyqlib` (unpinned, `requirements.txt:59`), `rdagent==0.8.0` (`requirements.txt:57`), `pytest`+`coverage` (present in `requirements/test.txt:1-3` but orphaned — wire it)
- **Paper**: arXiv:2602.07085, §3 (`problem_formulation.tex`)

---

## CONTEXT REFERENCES

### Files to Read Before Implementing
- `woolly-orbiting-pie.md` — the strategy this plan elaborates (Phases 0–5, decisions, risks).
- `problem_formulation.tex` — the formal objective: Eq. 4–5 (execution/fill), Eq. 6 (turnover), Eq. 7 (cost), Eq. 8 (net return), §3.4 (7 quality dims), Eq. 9 (feasible set `F_Θ`), Eq. 10–12 (rank/score/utility), Eq. 13 (protocol `Θ`), Properties 1–3 (Immutability / Determinism / Point-in-time).
- `Get_Started.md` — codebase map (§5 mining loop, §6 plugin system, §7 LoopMeta, §8 evolution, §9 DSL, §10 quality gate, §14 standalone backtest).
- `quantaalpha/pipeline/settings.py:48-84` — `AlphaAgentFactorBasePropSetting` / `FactorBasePropSetting` / `FactorBackTestBasePropSetting`, all `env_prefix="QLIB_FACTOR_"` (lines 50, 63, 76); the runner default is `quantaalpha.factors.runner.QlibFactorRunner` (line 55), summarizer default `quantaalpha.factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback` (line 57). Singletons at 116-120.
- `quantaalpha/pipeline/loop.py:53-131` — `AlphaAgentLoop.__init__` resolves dotted paths via `import_class`; `self.runner = import_class(PROP_SETTING.runner)(scen)` (122), `self.summarizer = import_class(PROP_SETTING.summarizer)(scen)` (125). The 5 steps: `factor_propose` (141-149), `factor_construct` (151-158), `factor_calculate` (160-167), `factor_backtest` (170-182, calls `self.runner.develop(prev_out["factor_calculate"], use_local=...)` at 176, stores `self._last_experiment = exp` at 181, raises `FactorEmptyError` if None at 179), `feedback` (184-249, calls `self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)` at 187). `_get_trajectory_data` (251-268) returns `hypothesis/experiment/feedback/...`.
- `quantaalpha/utils/workflow.py:44-65` (`LoopMeta` step discovery — non-underscore callables only), `:90-144` (`LoopBase.run` — skip_loop_error at 116-120 advances loop_idx, CoderError at 121-124 retries same loop from step 0, `dump` after every step at 140).
- `quantaalpha/core/utils.py:75-88` (`import_class`), `:156-201` (`cache_with_pickle` decorator factory — keys on `hash_func(*args,**kwargs)`, short-circuits on None or `cache_with_pickle=False`, folder `pickle_cache_folder_path_str/<mod>.<name>/`).
- `quantaalpha/core/experiment.py:196-214` — `Experiment` has `result: object = None` (212) and `sub_results: dict[str,float]` (213). **No `backtest_results` attribute** — the experiment object *is* the backtest result.
- `quantaalpha/core/exception.py:1-44` — `CoderError` (1-8) + `CodeFormatError`/`CustomRuntimeError`/`NoOutputError` (11-26) trigger retry-from-step-0; `FactorEmptyError` (35-38) triggers skip-loop; `CustomRunnerError` (29-32) is NOT a `CoderError` and would propagate.
- `quantaalpha/factors/runner.py:37-189` — `QlibFactorRunner(CachedRunner[QlibFactorExperiment])`; `develop(self, exp, use_local=True)` (75-189) decorated `@cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)` (75); sets `exp.result = result` (187) from `exp.experiment_workspace.execute(...)`; writes `combined_factors_df.parquet` (159-161); config-select `conf_baseline.yaml if len(based_experiments)==0 else conf_combined_factors.yaml` (165). **Dead correlation machinery**: `calculate_information_coefficient` (47-56, per-column `.corr()`) and `deduplicate_new_factors` (58-73, `groupby("datetime").parallel_apply(...).mean()` → `.unstack().max(axis=0)`) behind `if False:` at **line 138** — this is the ρ_max reuse seed (per-datetime cross-sectional corr → time-mean → max). `process_factor_data` (191-234).
- `quantaalpha/components/runner/__init__.py:11-54` — `CachedRunner.get_cache_key` (12-19) = `md5_hash("\n".join(task.get_task_information()))` over based_experiments + exp sub_tasks; **NO objective/protocol hash** (the collision hazard). `assign_cached_result` (21-53) copies `exp.result = cached_res.result` (52). `md5_hash` at `llm/client.py:29-33`.
- `quantaalpha/factors/regulator/consistency_checker.py` — `ComplexityChecker` (239-305) imports `calculate_symbol_length, count_base_features, count_free_args, count_all_nodes` from `factor_ast` (260-266) with hardcoded thresholds 250/6/0.5. `RedundancyChecker` (308-361) is AST-subtree-based via `match_alphazoo`. `FactorQualityGate` (364-454) takes only `*_enabled` flags. **Wiring gap**: only `consistency_enabled` is read from config (`loop.py:95`, forwarded at `loop.py:114-116`); `complexity_enabled`/`redundancy_enabled` are hardcoded `True` at `proposal.py:358-362`; `consistency_strict_mode`, `max_correction_attempts`, and ALL `factor.complexity`/`factor.duplication` thresholds in `configs/experiment.yaml:116-140` are defined but **never read**. ⇒ The new `F_Θ` gates must read their own `protocol_csi300.yaml`, not the unwired YAML.
- `quantaalpha/factors/coder/factor_ast.py` — reuse exports: `calculate_symbol_length(expr)->int` (482-493, `len(expr.strip())`), `count_base_features(expr)->int` (496-510, counts `$`-VarNodes), `count_free_args(expr)->int` (387-398, **counts numeric constants / NumberNodes, not variable args**), `count_all_nodes(expr)->int` (468-479), `count_unique_vars(expr)->int` (426-439), `find_largest_common_subtree(root1,root2)->Optional[SubtreeMatch]` (278-360), `match_alphazoo(prop_expr, factor_df)->(int,subtree,alpha)` (370-383), `compare_expressions(expr1,expr2)->Optional[SubtreeMatch]` (362-366), `parse_expression(text)->Node` (239).
- `quantaalpha/factors/coder/config.py:7-47` — `FactorCoSTEERSettings` (`env_prefix="FACTOR_CoSTEER_"`, line 8): `data_folder` (10-11, default `git_ignore_folder/factor_implementation_source_data`, env `FACTOR_CoSTEER_DATA_FOLDER`), `data_folder_debug` (13-14, default `..._debug`, env `FACTOR_CoSTEER_DATA_FOLDER_DEBUG`), `factor_zoo_path` (28-31), `duplication_threshold` (33-35), `symbol_length_threshold` (37-39), `base_features_threshold` (41-44). Singleton `FACTOR_COSTEER_SETTINGS` at 47.
- `quantaalpha/factors/feedback.py` — `QlibFactorHypothesisExperiment2Feedback` (123), `AlphaAgentQlibFactorHypothesisExperiment2Feedback` (215, the one wired in `settings.py:57`), `QlibModelHypothesisExperiment2Feedback` (342). `process_results` (26-56, no-SOTA branch) hardcodes the metric list at **42-47**; the SOTA branch duplicates it at **88-93** — identical 4-element list (`1day.excess_return_without_cost.max_drawdown` / `.information_ratio` / `.annualized_return` / `IC`). `generate_feedback` renders `factor_feedback_generation` prompts (146-164 / 275-293); the AlphaAgent variant injects `complexity_feedback` (239-272). No `_extract_backtest_results` here (that's in `library.py:294-325`).
- `quantaalpha/factors/library.py:25-343` — `FactorLibraryManager.add_factors_from_experiment` (56-150) builds `factor_entry` (119-140) with key `backtest_results`. `_extract_backtest_results` (294-325): for a `dict` result → **passes through unchanged** (322-323); for `pd.Series`/`DataFrame` → flat dict, floats rounded to 8 dp, non-floats (str/bool) kept as-is. ⇒ A Series result stores the full metric vector in the library JSON with **no schema migration**. `_sync_h5_to_md5_cache` (152-176): `md5(expression)` (full hex) → `cache_dir/{md5}.pkl`. Default cache dir `os.environ.get("FACTOR_CACHE_DIR","data/results/factor_cache")` (19-22). `factor_id = md5(f"{factor_name}_{factor_expr}")[:16]` (85-87).
- `quantaalpha/pipeline/evolution/trajectory.py` — `RoundPhase` (21-25); `StrategyTrajectory` (28-81, `backtest_metrics` at 70); `get_primary_metric` (90-92) → `backtest_metrics.get("RankIC")`; `is_successful` (94-97) → `rank_ic is not None and rank_ic > 0`; `generate_id` (83-88); `TrajectoryPool.select_parents_for_crossover` (251-339) has unseeded `random.shuffle` at **330** and **338** — but this method is **dead** (the controller uses `crossover_op.select_crossover_pairs`); `_save`/`_load` (341-382); `get_statistics` (384-392); `cleanup_file` (401-408).
- `quantaalpha/pipeline/evolution/controller.py` — `EvolutionConfig` (24-66); `create_trajectory_from_loop_result` (710-793) takes `task/hypothesis/experiment/feedback` as **separate args** (the caller unpacks `traj_data` at `factor_mining.py:448-453`/`484-489`) and sets `backtest_metrics = self._extract_metrics(experiment.result)`; `_extract_metrics` (795-859) returns a fixed 7-key dict `{IC,ICIR,RankIC,RankICIR,annualized_return,information_ratio,max_drawdown}` init None, maps via `index_mapping` (first match wins), handles `pd.DataFrame`/`pd.Series` — **a plain `dict` result yields all-None metrics** (the integration hazard); `report_task_complete` (681-708); `get_next_task` (134-165); `advance_phase_after_parallel_completion` (293-356); `save_state` (877-903).
- `quantaalpha/pipeline/evolution/mutation.py` — `MutationOperator.generate_mutation_prompt_suffix(parent)` (219-267) → `## Mutation Round Guidance` block; `_generate_fallback_hypothesis` (192-217) uses unseeded `random.choice` at **line 217** (only in the `else` fallback branch).
- `quantaalpha/pipeline/evolution/crossover.py` — `CrossoverOperator.generate_crossover_prompt_suffix(parents)` (242-301) → `## Crossover Round Guidance` block; `select_crossover_pairs` (303-373): `prefer_diverse=True` (default) → deterministic; `prefer_diverse=False` → unseeded `random.shuffle(all_combos)` at **370**; `_select_candidates_by_strategy` uses `random.sample` at 412/432; `_weighted_sample` uses `random.choices` at 502. (The live unseeded paths are here, not in the dead `TrajectoryPool` method.)
- `quantaalpha/pipeline/factor_mining.py:514-657` — `main`; `quality_gate_cfg` read at 553, passed to `run_evolution_loop` (587) → `_run_evolution_task` (482) → `AlphaAgentLoop(quality_gate_config=...)` (178). **Parallel-path gap**: `_run_tasks_parallel` (257-315) at 434-441 does NOT receive/forward `quality_gate_cfg` — silently dropped in parallel mode.
- `quantaalpha/llm/config.py:14-68` — `LLMSettings` has **no `env_prefix`** (fields map from uppercased env vars). **No `json_mode_supported` field** (clean addition). Singleton `LLM_SETTINGS` at 68. Key fields: `chat_model` (31, default `gpt-4-turbo`), `reasoning_model` (32), `chat_max_tokens` (33, 3000), `chat_temperature` (34, 0.5), `chat_stream` (35, True), `chat_seed` (36, None), `openai_api_key` (28), `openai_base_url` (29), `embedding_model` (45, ""), `embedding_base_url` (49), `factor_mining_timeout` (41, 999999), `chat_model_map` (65, "{}").
- `quantaalpha/llm/client.py` — `build_messages_and_create_chat_completion` (585-608); `response_format={"type":"json_object"}` at **line 859**, gated by the per-call `json_mode: bool = False` param on `_create_chat_completion_inner_function` (758) — **not unconditional**, but callers pass `json_mode=True` (proposal.py:274/307/442, feedback.py:175/304/382, regulator/consistency_checker.py:104, evolution/crossover.py:173, evolution/mutation.py:140); when `reasoning_flag` is True, `json_mode=None` (806). `robust_json_parse` (36-121): direct → json fence → balanced braces → LaTeX-escape repair → loose regex → raise. `chat_model_map` uses `inspect.stack()[4]` (798, 802 — fragile). `md5_hash` (29-33).
- `quantaalpha/app/utils/health_check.py:43-49` — `health_check` checks **docker + ports only** (default port 19899); does NOT check Qlib data or env. ⇒ The real Phase 0 data-path gate is a `--dry-run` standalone backtest.
- `quantaalpha/backtest/runner.py` — `BacktestRunner` (24); reuse seam `PrecomputedDataHandler` inner class (358-417) feeds precomputed factor DataFrames to Qlib; `_create_dataset_with_computed_factors` (204-427) does per-datetime pct ranking `(x.rank(pct=True)-0.5)` (339-347); `_train_and_backtest` (479-674) produces `IC/ICIR/Rank IC/Rank ICIR` (514-537) and `annualized_return/information_ratio/max_drawdown/calmar_ratio` (653-667) via `LGBModel` (498-503) + `qlib.backtest.backtest` (585-613).
- `quantaalpha/backtest/custom_factor_calculator.py` — 3-tier cache in `calculate_factors_batch` (307-470): H5 `cache_location` (342-351) → MD5 `.pkl` (353-360, key `md5(expr)` at 99-101, dir `FACTOR_CACHE_DIR` at 37) → recompute with `signal.alarm(120)` (383-425). `calculate_factor` (194-261) reuses `expr_parser` + `function_lib` via `eval`.
- `configs/experiment.yaml` — `evolution.enabled:true`, `max_rounds:3`, `num_directions:2`, `quality_gate` (107-119, largely decorative — see wiring gap above), `factor.complexity` (131-134, decorative), `backtest.qlib.config_name: conf_baseline.yaml` (151). **No `seed` field.**
- `configs/backtest.yaml` — `random_seed:42` (11, **dead** — not consumed by `quantaalpha/backtest/`); `learning_rate:0.1` (95); `label` (55, with spaces around `/`); `deal_price:"open"` (130); segments train 2016-2020 / valid 2021 / test 2022-2025 (83-85).
- `quantaalpha/factors/factor_template/conf_baseline.yaml` — `learning_rate:0.05` (81); `label` (27); `deal_price:open` (69). `conf_combined_factors.yaml` — `learning_rate:0.05` (88); `label` (31); `deal_price:open` (76); segments train 2016-2019 / valid 2020 / **test 2021 only** (107-109) — the in-loop test window that must NOT be widened.
- `pyproject.toml` — only `docs` optional-dependency wired (no `test`, no `[tool.pytest.ini_options]`).
- `requirements.txt:22-23,57,59` — `numpy>=1.24,<2.0`, `pandas>=1.5,<3.0`, `rdagent==0.8.0`, `pyqlib` (unpinned). **`scipy` absent.** `requirements/test.txt:1-3` — `coverage`, `pytest` (orphaned).

### New Files to Create
- `quantaalpha/eval/__init__.py` — package init, re-export `EvaluationOperator`, `Protocol`, `load_protocol`.
- `quantaalpha/eval/protocol.py` — `@dataclass(frozen=True) Protocol` + `Θ.hash` (SHA-256 of canonical JSON) + `load_protocol(yaml_path)`.
- `quantaalpha/eval/execution.py` — fill rule `Φ`, latency `δ`, realized return `ỹ`, portfolio map `g`, drift, turnover `TO_t`.
- `quantaalpha/eval/costs.py` — `κ₀/κ₁/κ₂/β`, impact `φ ∝ |Δw|^{3/2}/√ADV`, shared `_trailing()` helper, `c_t`.
- `quantaalpha/eval/metrics.py` — the 7 quality dimensions; reuses `factor_ast` for `cx(f)`; revives `runner.py:47-73` correlation machinery for `ρ_max`; loads incumbents from `factor_cache/{md5}.pkl`.
- `quantaalpha/eval/gates.py` — `F_Θ` feasibility (Eq. 9), evaluated on train/valid only.
- `quantaalpha/eval/scoring.py` — `R(f,m,zoo)` (Eq. 11), `e_j = 1-R` (Eq. 10), sign-flip lower-is-better dims, `U = Σ ω_j e_j` (Eq. 12).
- `quantaalpha/eval/operator.py` — `EvaluationOperator` (the `E_Θ` entry point; pure function of `(f, Θ)`; logs `Θ.hash` every call).
- `quantaalpha/eval/ledger.py` — append-only trial record `(factor_id, Θ.hash, m(f), timestamp)`.
- `quantaalpha/eval/protocol_csi300.yaml` — the frozen CSI 300 protocol (values in Task 6).
- `quantaalpha/factors/net_cost_runner.py` — `NetCostFactorRunner(QlibFactorRunner)`; `NetCostFactorRunner.get_cache_key` appends `Θ.hash`.
- `quantaalpha/factors/net_cost_feedback.py` — `NetCostFactorFeedback(AlphaAgentQlibFactorHypothesisExperiment2Feedback)` surfacing per-dimension scores.
- `scripts/smoke_llm.py` — one plain + one JSON-mode completion against the configured provider.
- `requirements/dev.txt` — `pytest`, `coverage`, `scipy` (and pin `pyqlib` defensively).
- `tests/eval/__init__.py`, `tests/eval/conftest.py`, `tests/eval/test_*.py` — the unit invariants + e2e.
- `tests/eval/fixtures/` — tiny synthetic price panel for deterministic unit tests.

### Documentation to Read
- `problem_formulation.tex` §3.3 (Eq. 4–8), §3.4 (7 dims, Eq. 10–12), §3.5 (Eq. 9, Eq. 13, Properties 1–3) — the contract for `E_Θ`.
- `Get_Started.md` §6 (plugin system), §7 (LoopMeta — helpers must be `_`-prefixed), §8 (evolution — mutation/crossover emit *prompt suffixes*, not factor edits), §10 (quality gate threshold drift), §14 (two backtest paths) — the integration hazards.
- Ollama Cloud OpenAI-compatibility docs (the base URL `https://ollama.com/v1` and `response_format` support are **unverified** — confirm in Task 4 before any LLM spend).

### Patterns to Follow
- **Plugin swap via env var** (existing pattern): override a stage by setting `QLIB_FACTOR_RUNNER` / `QLIB_FACTOR_SUMMARIZER` (`settings.py:50`, `Get_Started.md` §6). No hardcoded imports in `loop.py`.
- **`import_class`** (`core/utils.py:75-88`): any replacement class must be importable as `module.path.ClassName` and accept the same constructor kwargs (`scen` for runner/summarizer — `loop.py:122,125`).
- **`@cache_with_pickle(hash_func, post_process_func)`** (`core/utils.py:156-201`, applied at `runner.py:75`): the subclass's overridden `develop` must re-apply the decorator with its own `get_cache_key`.
- **Loop helpers are `_`-prefixed** (`workflow.py:59`, `loop.py:253-254`): any new method on a `LoopMeta` class must start with `_` or it silently becomes a pipeline step.
- **Trajectory metric flow**: `exp.result` → `controller._extract_metrics` (`controller.py:795-859`) → `backtest_metrics` → `get_primary_metric`/`is_successful` (`trajectory.py:90-97`). The new runner emits a `pd.Series` whose index carries both the 7 canonical names (so `_extract_metrics` keeps working) **and** `U/feasible/theta_hash/rho_max/turnover_mean/cx/e_1..e_7`; `_extract_metrics` is extended (additively, backward-compatible) to surface the extra keys.
- **Library passthrough** (`library.py:322-323`): a Series `exp.result` stores the full vector in the library JSON with no migration.
- **Mutation/crossover = prompt suffixes** (`mutation.py:219-267`, `crossover.py:242-301`): do not manipulate factor expressions in the evolution operators.
- **Frozen Θ** (Property 1): `@dataclass(frozen=True)`; `Θ.hash` logged at every evaluation; loaded once from YAML at startup; editing the YAML mid-run has no effect on a resumed session (the pickled loop captures Θ — `workflow.py:140`).

---

## STEP-BY-STEP TASKS

Execute every task in order. Each task is atomic and independently testable.

### Task 1 — Wire dev/test tooling into the package
- **IMPLEMENT**: Create `requirements/dev.txt` with `pytest`, `coverage`, `scipy` (pin `scipy>=1.10,<1.13` for `numpy<2.0`), and `pyqlib==0.9.6` (defensive pin against the unpinned `requirements.txt:59`). In `pyproject.toml`, add `[tool.setuptools.dynamic.optional-dependencies].test = {file = ["requirements/test.txt"]}` and a `[tool.pytest.ini_options]` section with `testpaths=["tests"]`, `addopts="-q"`, `filterwarnings=["ignore::DeprecationWarning"]`.
- **PATTERN**: `pyproject.toml` already wires `docs = {file=["requirements/docs.txt"]}` (line ~35); mirror that for `test`.
- **VALIDATE**: `pip install -e .[test] && python -c "import scipy, pytest; print(scipy.__version__, pytest.__version__)"`.

### Task 2 — Bring up the environment + validate the data path
- **IMPLEMENT**: `conda create -n quantaalpha python=3.10 -y && conda activate quantaalpha` (3.12 will not work — `numpy<2.0`/`pyqlib`). `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . && pip install -r requirements.txt && pip install -e .[test]`. Download `cn_data.zip`, `daily_pv.h5`, `daily_pv_debug.h5` from HuggingFace `QuantaAlpha/qlib_csi300`; place per README §3 (rename `daily_pv_debug.h5` → `daily_pv.h5` inside the debug folder). `cp configs/.env.example .env`; fill `QLIB_DATA_DIR`, `DATA_RESULTS_DIR`, paths. Add `FACTOR_CoSTEER_DATA_FOLDER`/`FACTOR_CoSTEER_DATA_FOLDER_DEBUG` if non-default.
- **PATTERN**: `Get_Started.md` §2; data folder resolution at `factors/coder/config.py:10-14`.
- **VALIDATE**: `quantaalpha health_check` (docker+ports only — `health_check.py:43-49`); then the **real** data-path gate: `python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source alpha158_20 --dry-run -v` (must report factors loaded + Qlib data reachable, no LLM spend).

### Task 3 — Add a global `json_mode_supported` flag to the LLM client
- **IMPLEMENT**: Add `json_mode_supported: bool = True` to `LLMSettings` (`llm/config.py`, near line 36). In `llm/client.py:853-859`, change the guard to `if json_mode and LLM_SETTINGS.json_mode_supported:` before setting `kwargs["response_format"] = {"type":"json_object"}`. When false, rely on `robust_json_parse` (36-121) which already handles fences / balanced braces / LaTeX-escape. Leave the per-call `json_mode` param (758) untouched.
- **PATTERN**: `LLMSettings` has no `env_prefix` (`config.py:14-66`) so the new field maps directly to `JSON_MODE_SUPPORTED`. This is the exact mitigation the woolly plan specifies for Ollama Cloud rejecting `response_format`.
- **VALIDATE**: `JSON_MODE_SUPPORTED=true python -c "from quantaalpha.llm.config import LLM_SETTINGS; assert LLM_SETTINGS.json_mode_supported"`; `python -c "import ast; ast.parse(open('quantaalpha/llm/client.py').read())"`.

### Task 4 — Point the LLM at Ollama Cloud / kimi-k2.5 and smoke-test
- **IMPLEMENT**: Add to `.env`: `OPENAI_API_KEY=<ollama-cloud-key>`, `OPENAI_BASE_URL=https://ollama.com/v1`, `CHAT_MODEL=kimi-k2.5:cloud`, `REASONING_MODEL=kimi-k2.5:cloud`, `CHAT_SEED=42`, `CHAT_TEMPERATURE=0.0`, `CHAT_STREAM=False`, `CHAT_MAX_TOKENS=8000`. Create `scripts/smoke_llm.py` that issues (a) one plain completion and (b) one JSON-mode completion (`json_mode=True`) and prints which succeeded. **Verify the base URL with a raw `curl` first** (unverified during planning); if `response_format` is rejected, set `JSON_MODE_SUPPORTED=false` and confirm `robust_json_parse` still recovers. Determine whether the default coder path (`QlibFactorParser`, template-first) ever calls embeddings, or only CoSTEER's RAG; if embeddings are needed and Ollama Cloud doesn't serve them for this model, set `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` to a separate provider.
- **PATTERN**: `build_messages_and_create_chat_completion` (`client.py:585-608`); `chat_model_map` selects by caller class via `inspect.stack()[4]` (`client.py:798`) — keep `CHAT_MODEL` as the fallback. Context-window risk: `prompts.yaml` is 34 KB; `proposal.py` already has `is_input_length_error` handling.
- **VALIDATE**: `python scripts/smoke_llm.py` prints success for at least one of the two modes; `curl -sS https://ollama.com/v1/models` (or the documented endpoint) returns a model list including `kimi-k2.5:cloud`.

### Task 5 — Freeze the shared Θ baseline (config unify + control-arm run)
- **IMPLEMENT**: Unify `learning_rate` to a single frozen value (set both `conf_baseline.yaml:81` and `conf_combined_factors.yaml:88` to match `configs/backtest.yaml:95` = `0.1`, **or** set `configs/backtest.yaml:95` to `0.05` — pick one and record it as a `protocol_csi300.yaml` field in Task 6; the paper's baseline is the reference). Do **not** widen the in-loop test window (`conf_combined_factors.yaml:107-109`, test 2021 only — `Get_Started.md` §5, the point-in-time invariant). Run the **unmodified** control arm: `EXPERIMENT_ID=baseline_kimi ./run.sh "<paper's initial direction>"` (per-experiment workspace/cache isolation is already in `run.sh:54-69`). Then the standalone backtest on 2022–2025: `python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source combined --factor-json all_factors_library.json`. Record IC / RankIC / ARR / IR / MDD; compare to paper 0.0472 / 0.0459 / 4.68% / 0.6453 / 11.80% as a sanity reference only. **Preserve the baseline's factor library JSON** — the treatment arm is scored against the same repository semantics, and baseline factors are re-scored under `E_Θ` in Task 20.
- **PATTERN**: `run.sh` exports `WORKSPACE_PATH`/`PICKLE_CACHE_FOLDER_PATH_STR` per `EXPERIMENT_ID` (54-69); `EXPERIMENT_ID=shared` skips isolation. Cache-key hazard: `CachedRunner.get_cache_key` (`components/runner/__init__.py:12-19`) is task-info-only, so per-arm `PICKLE_CACHE_FOLDER_PATH_STR` is what keeps the arms apart today.
- **VALIDATE**: control-arm run completes; `all_factors_library.json` (or the suffixed name) is non-empty; standalone backtest prints IC/RankIC/ARR/IR/MDD. No assertion on absolute values (model differs from paper).

### Task 6 — `quantaalpha/eval/protocol.py` + `protocol_csi300.yaml`
- **IMPLEMENT**: `@dataclass(frozen=True) Protocol` collecting splits (`train/valid/test` date tuples), gates (`gamma_ic, gamma_ir, tau_max, rho_bar, gamma_cx`), cost coefficients (`kappa0, kappa1, kappa2, beta_per_day, beta_offlist`), `delta` + `fill_rule` (`"open_next"`), portfolio-map config (`long_short=True, gross_leverage=1.0`), `weights omega` (7-dim), `is_window`/`oos_window` selectors, and `periods_per_year=252`. `Θ.hash` = `hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:16]`. `load_protocol(path) -> Protocol` reads YAML and constructs. Write `protocol_csi300.yaml` with the woolly plan's defaults: δ=1, Φ=open(t+1); κ₀=0.0020; κ₁=0.10; κ₂ calibrated so 1% ADV ≈ 10bps (start `0.1`); β≈0.0004/day, ∞ off-list; γ_ic=0.02, γ_ir=0.20; τ_max=0.30; ρ̄=0.70; γ_cx=200; ω effectiveness-weighted; splits train 2016-2019 / valid 2020 / in-loop-test 2021 / final-test 2022-2025; learning_rate frozen per Task 5.
- **PATTERN**: frozen dataclass + canonical-JSON hash = mechanical enforcement of Property 1 (Immutability). `ExtendedBaseSettings` is *not* used here — `Protocol` is plain, loaded once, not env-driven.
- **VALIDATE**: `python -c "from quantaalpha.eval.protocol import load_protocol; p=load_protocol('quantaalpha/eval/protocol_csi300.yaml'); print(p.hash); p2=load_protocol('quantaalpha/eval/protocol_csi300.yaml'); assert p.hash==p2.hash"`; and a mutated YAML produces a different hash.

### Task 7 — `quantaalpha/eval/execution.py`
- **IMPLEMENT**: `fill_prices(P, t, delta, rule)` (Eq. 4; default δ=1, Φ=open(t+1) — matches existing `deal_price:open`, `backtest.yaml:130`). `realized_return(P_fill)` → `ỹ_{i,t} = P_fill_{i,t+1}/P_fill_{i,t} - 1` (Eq. 5) — **this fixes the label/fill mismatch**: returns come from the fill series, not close-to-close. `portfolio_map(pred) -> w` (Eq. 6): rank cross-sectionally, centre, normalize `Σ|w|=1`, long top / short bottom, dollar-neutral. `drift(w, y_tilde)` → `w_drift` computed from drifted **notionals** over post-drift NAV (the normalizing denominator `Σw(1+ỹ)≈0` makes normalized-weight drift unstable). `turnover(w, w_drift) -> TO_t` (Eq. 6) = `0.5·Σ|w - w_drift|`.
- **PATTERN**: pure-numpy, no QuantaAlpha imports; takes aligned `(N×T)` price/signal panels.
- **VALIDATE**: `pytest tests/eval/test_execution.py` — dollar-neutrality `Σw_t≈0` and `Σ|w_t|≤1` per date; `TO∈[0,1]`; held-constant portfolio → `TO=0`.

### Task 8 — `quantaalpha/eval/costs.py`
- **IMPLEMENT**: `cost(w, w_drift, sigma, adv, theta) -> c_t` (Eq. 7) = `kappa0*TO + kappa1*Σ sigma_{i,t}|Δw| + kappa2*Σ phi(Δw;ADV) + Σ beta_{i,t} max(0,-w)`, with `phi(Δw;ADV) ∝ |Δw|^{3/2}/sqrt(ADV)` (square-root impact). `trailing_vol(P, window)` and `trailing_adv(volume, window)` via a shared `_trailing(series, window)` helper with an **explicit shift** (`series.shift(window)`) so no look-ahead enters (Property 3). `borrow_rate(beta_per_day, offlist_mask)`.
- **PATTERN**: `_trailing` with explicit shift is the look-ahead guard — the strongest source of silent PIT bugs.
- **VALIDATE**: `pytest tests/eval/test_costs.py` — `c_t` non-decreasing in each κ; doubling `Δw` more than doubles the impact term (super-linearity); `c_t=0` when all κ=0 and β=0.

### Task 9 — `quantaalpha/eval/metrics.py`
- **IMPLEMENT**: The 7 dimensions on `m(f)` over the protocol's IS window:
  1. **Effectiveness** — `IC_t=corr(z_t,y_{t+1})`, `RankIC_t`, net IR `= mean(r_net)/std(r_net)*sqrt(P)` from `execution`/`costs`.
  2. **Annualized return** — `ARR=(Π(1+r_net))^(P/T_eval)-1`, plus MDD.
  3. **Stability** — `ICIR=mean(IC)/std(IC)`, `RankICIR`, `frac(IC>0)`.
  4. **Turnover** — `mean(TO_t)`.
  5. **Diversity** — `ρ_max = max_{f'∈zoo} |corr(z^f, z^{f'})|` per-date cross-sectional, time-averaged, max over incumbents. Revive the dead machinery from `runner.py:47-73` (`groupby("datetime").apply(corr).mean()` → max), loading incumbent signals from `factor_cache/{md5}.pkl` (`library.py:152-176`, dir `FACTOR_CACHE_DIR` default `data/results/factor_cache`).
  6. **Over-fitting risk** — `cx(f)` from `factor_ast`: `calculate_symbol_length`, `count_base_features`, `count_free_args` (note: counts numeric constants), `count_all_nodes`; plus `Δ_{IS→OOS} = m_IS - m_OOS`.
  7. **Decay resistance (optional)** — partition OOS into K sub-periods, fit `IC^(k)=a+b·k`, report slope `b` and `IC^pers`.
  - **IS→OOS resolution**: during search, IS=train, OOS proxy=valid/in-loop-test; only the final report uses 2022–2025. The window is a **protocol field the operator reads** (`protocol.py`), never a caller argument.
- **PATTERN**: import `calculate_symbol_length, count_base_features, count_free_args, count_all_nodes` from `quantaalpha.factors.coder.factor_ast` (the only QuantaAlpha import in `eval/`). Reuse `pd.DataFrame.groupby("datetime").corr` for `ρ_max`.
- **VALIDATE**: `pytest tests/eval/test_metrics.py` — a factor equal to forward return scores max effectiveness; a constant factor scores `ρ_max` against itself = 1; `cx` agrees with `factor_ast` on a fixed expression.

### Task 10 — `quantaalpha/eval/gates.py`
- **IMPLEMENT**: `feasible(m, theta) -> bool` implementing `F_Θ` (Eq. 9): `RankIC≥gamma_ic AND RankICIR≥gamma_ir AND mean(TO)≤tau_max AND rho_max≤rho_bar AND cx≤gamma_cx`, all thresholds read from `Θ`. Gates evaluated on **train/valid only** (Property 3). Keep the AST redundancy check (`RedundancyChecker`, `consistency_checker.py:308-361`) as a **cheap pre-screen** in the runner; `ρ_max` is authoritative.
- **PATTERN**: thresholds are `Θ` fields (not the unwired `experiment.yaml:116-140` block — that is decorative, per the wiring gap).
- **VALIDATE**: `pytest tests/eval/test_gates.py` — a factor above all thresholds is feasible; relaxing any threshold in a mutated `Θ` admits a previously-rejected factor.

### Task 11 — `quantaalpha/eval/scoring.py`
- **IMPLEMENT**: `rank(f, m_j, zoo) = (1/|zoo|)·Σ 1[m(f) < m(f')]` (Eq. 11), `e_j = 1 - rank` ∈ [0,1] (Eq. 10). Sign-flip dimensions where lower is better (turnover, ρ_max, cx, IS→OOS degradation) **before** ranking. `utility(m, theta) = Σ ω_j e_j` (Eq. 12). Over-fitting risk (dim 6) is scored from structure/history, **not** repository rank (§3.4).
- **PATTERN**: `zoo` is the moving reference set; the same accumulated library underlies both combination (frozen, Eq. 2) and scoring (moving, Eq. 11).
- **VALIDATE**: `pytest tests/eval/test_scoring.py` — a factor beating every incumbent on a dimension scores `e_j=1`; the worst scores `0`; `U` monotonic in each `e_j`.

### Task 12 — `quantaalpha/eval/operator.py` + `ledger.py`
- **IMPLEMENT**: `EvaluationOperator(theta: Protocol)` with `evaluate(factor_signal, factor_expr, zoo_signals) -> dict` returning `{m_1..m_7, e_1..e_7, U, feasible, theta_hash, rho_max, turnover_mean, cx}` — a pure function of `(f, Θ)` (Property 2). Logs `Θ.hash` on every call. `Ledger(path)` append-only writes `(factor_id, theta_hash, m(f), timestamp)` as JSONL; the substrate for the future deflated-Sharpe / multiple-testing work (out of scope per the paper's roadmap).
- **PATTERN**: purity = identical inputs ⇒ identical outputs; `Θ.hash` logged every call so a stale-protocol resume is visible (`workflow.py:140` pickles the whole loop including Θ).
- **VALIDATE**: `pytest tests/eval/test_operator.py` — two calls with identical input return bit-identical output; a mutated `Θ` changes `theta_hash` and the output.

### Task 13 — `tests/eval/` unit invariants (greenfield)
- **IMPLEMENT**: `tests/eval/conftest.py` builds a tiny synthetic `(N×T)` price/volume panel (deterministic). Tests:
  1. **Nesting** (most valuable): with `κ₀=κ₁=κ₂=β=0`, `δ=0`, `r_net` == gross frictionless return to fp tolerance.
  2. **Dollar neutrality** after drift.
  3. **Turnover bounds** `TO∈[0,1]`; constant portfolio → `TO=0`.
  4. **Cost monotonicity** + impact super-linearity.
  5. **Determinism**: `E_Θ(f)` twice → bit-identical; mutated `Θ` → different hash.
  6. **No look-ahead**: truncate price panel after `T` ⇒ no `m(f)` on `≤T` changes.
  7. **Scoring**: best-on-dimension → `e_j=1`; worst → `0`.
- **PATTERN**: `pytest` is now wired (Task 1); no existing tests to follow — greenfield.
- **VALIDATE**: `pytest tests/eval/ -q` — all 7 invariants pass.

### Task 14 — `quantaalpha/factors/net_cost_runner.py` (the treatment-arm runner)
- **IMPLEMENT**: `NetCostFactorRunner(QlibFactorRunner)`. Reuse `process_factor_data` (`runner.py:191-234`) to get the factor signal panel. Construct `EvaluationOperator(theta=load_protocol(...))` once in `__init__` (θ loaded from a path via env `QA_PROTOCOL` or default `quantaalpha/eval/protocol_csi300.yaml`). Override `develop(self, exp, use_local=True)`:
  - compute the signal panel,
  - run `E_Θ.evaluate(signal, factor_expr, zoo_signals)` where `zoo_signals` are loaded from `factor_cache/{md5}.pkl` for incumbent factors (reuse `library.py` paths),
  - set `exp.result` to a `pd.Series` whose index carries the 7 canonical names (`IC/ICIR/RankIC/RankICIR/annualized_return/information_ratio/max_drawdown`, so `controller._extract_metrics` at `795-859` still populates `backtest_metrics` for reporting) **plus** `U/feasible/theta_hash/rho_max/turnover_mean/cx/e_1..e_7`,
  - append `(factor_id, theta_hash, m(f))` to the ledger,
  - return `exp` (so `factor_backtest` at `loop.py:176` and `assign_cached_result` at `components/runner/__init__.py:52` both work unchanged).
  - Re-apply `@cache_with_pickle(NetCostFactorRunner.get_cache_key, CachedRunner.assign_cached_result)` and override `get_cache_key` to append `Θ.hash` to the task-info string (mitigates the collision hazard at `components/runner/__init__.py:12-19`). Raise `FactorEmptyError` (`exception.py:35-38`) if the signal panel is empty (keeps the skip-loop semantics at `workflow.py:116-120`).
- **PATTERN**: subclass + plugin swap; `exp.result` as a Series keeps `_extract_metrics` and `_extract_backtest_results` (`library.py:294-325`) both working. The `Θ.hash` in the cache key is the objective-aware cache isolation.
- **VALIDATE**: `python -c "from quantaalpha.factors.net_cost_runner import NetCostFactorRunner; from quantaalpha.factors.runner import QlibFactorRunner; assert issubclass(NetCostFactorRunner, QlibFactorRunner)"`; `QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner python -c "import os; from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING; import quantaalpha.core.utils as u; print(u.import_class(ALPHA_AGENT_FACTOR_PROP_SETTING.runner))"`.

### Task 15 — Extend `controller._extract_metrics` to surface the new vector
- **IMPLEMENT**: In `quantaalpha/pipeline/evolution/controller.py:795-859`, after the canonical `index_mapping` loop, add an additive pass that copies a whitelist of extra keys (`U`, `feasible`, `theta_hash`, `rho_max`, `turnover_mean`, `cx`, `e_1..e_7`) from `result.index` into `metrics` if present (for `pd.Series`/`pd.DataFrame`). Do **not** change behaviour for the control arm (its Qlib DataFrame result has none of these keys). Keep the 7 canonical keys first-match-wins as-is.
- **PATTERN**: backward-compatible additive edit; the control arm's path is unchanged.
- **VALIDATE**: `pytest tests/eval/test_extract_metrics.py` — a Series with `RankIC` + `U` populates both; a DataFrame without extras behaves exactly as before.

### Task 16 — Arm-selectable primary metric in `trajectory.py`
- **IMPLEMENT**: In `quantaalpha/pipeline/evolution/trajectory.py:90-97`, read a module-level primary-metric key and a require-feasible flag from env: `_PRIMARY_METRIC = os.environ.get("QA_PRIMARY_METRIC", "RankIC")`, `_REQUIRE_FEASIBLE = os.environ.get("QA_REQUIRE_FEASIBLE", "false").lower() in ("1","true","yes")`. `get_primary_metric` returns `self.backtest_metrics.get(_PRIMARY_METRIC)`. `is_successful` = `primary is not None and primary > 0` AND (`not _REQUIRE_FEASIBLE or self.backtest_metrics.get("feasible", True)`). Control arm sets neither (defaults to RankIC, feasibility off); treatment arm sets `QA_PRIMARY_METRIC=U` and `QA_REQUIRE_FEASIBLE=true`. Keep `RankIC` in `backtest_metrics` so both arms stay reportable.
- **PATTERN**: small direct edit per the woolly plan; env selection matches the existing env-prefix pattern and keeps one `StrategyTrajectory` class serving both arms.
- **VALIDATE**: `QA_PRIMARY_METRIC=U python -c "from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory as T; t=T(trajectory_id='x',direction_id=0,round_idx=0,phase=__import__('quantaalpha.pipeline.evolution.trajectory',fromlist=['RoundPhase']).RoundPhase.ORIGINAL); t.backtest_metrics={'RankIC':0.1,'U':-0.2,'feasible':True}; assert t.get_primary_metric()==-0.2 and not t.is_successful()"`; same `t` with `QA_PRIMARY_METRIC=RankIC` (unset) → `get_primary_metric()==0.1` and successful.

### Task 17 — `quantaalpha/factors/net_cost_feedback.py` + de-duplicate the metric list
- **IMPLEMENT**: In `quantaalpha/factors/feedback.py`, hoist the duplicated 4-element list (`42-47` and `88-93`) into one module-level constant `FRICTIONLESS_METRICS` and reference it in both branches. Create `NetCostFactorFeedback(AlphaAgentQlibFactorHypothesisExperiment2Feedback)` overriding `generate_feedback` (or `process_results`) to surface **dimension-targeted** feedback — `m(f)`, `e_j`, `U`, `Θ.hash`, and which gate failed (if any) — instead of the 4 frictionless metrics. This is the mechanism the woolly plan flags as "a large part of the contribution's actual mechanism": the feedback that shapes the next hypothesis now mentions cost.
- **PATTERN**: `AlphaAgentQlibFactorHypothesisExperiment2Feedback` (`feedback.py:215`) is the wired summarizer (`settings.py:57`); its `generate_feedback` (275-293) renders `factor_feedback_generation` prompts — override the `combined_result` payload.
- **VALIDATE**: `python -c "from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback; from quantaalpha.factors.feedback import AlphaAgentQlibFactorHypothesisExperiment2Feedback; assert issubclass(NetCostFactorFeedback, AlphaAgentQlibFactorHypothesisExperiment2Feedback)"`; `grep -n "excess_return_without_cost" quantaalpha/factors/feedback.py` shows the literal in exactly one constant.

### Task 18 — Single RNG-seeding entry point + flag the parallel-path gap
- **IMPLEMENT**: In `quantaalpha/cli.py:app()` (29-37, the chokepoint for both `run.sh` and `launcher.py`), seed `random`/`numpy.random` from a new `experiment.yaml` `seed: 42` field (read via `load_run_config` like the other sections in `factor_mining.py:549-553`) or env `QA_SEED`. In `run.sh`, `export PYTHONHASHSEED=${QA_SEED:-42}` **before** `python` starts (must be set pre-interpreter). This covers the live unseeded paths: `crossover.py:370` (only `prefer_diverse=False`), `mutation.py:217` (only fallback). Note in a code comment that `TrajectoryPool.select_parents_for_crossover` (`trajectory.py:330,338`) is dead code (controller uses `crossover_op.select_crossover_pairs`). Add a `# TODO(parallel):` note at `factor_mining.py:434-441` that `_run_tasks_parallel` drops `quality_gate_cfg` — do not fix here unless the treatment arm uses parallel mode (default `parallel_enabled:false`, `experiment.yaml:75`).
- **PATTERN**: `core/utils.py:102-105` already reseeds global `random` at import via `CacheSeedGen` (for cache, not reproducibility) — this is a separate, reproducibility-focused seed at the entry point.
- **VALIDATE**: `QA_SEED=42 python -c "import random; from quantaalpha.cli import app" ` (import smoke); `grep -n PYTHONHASHSEED run.sh` shows the export; two consecutive `EXPERIMENT_ID=seedtest ./run.sh "<dir>"` runs (stub LLM) produce identical trajectory IDs where the path is deterministic.

### Task 19 — End-to-end wiring test (1 loop, 1 factor, stub LLM, no spend)
- **IMPLEMENT**: `tests/eval/test_e2e_wiring.py` runs a 1-loop / 1-factor mining loop on `daily_pv_debug.h5` with a **stub LLM** (monkeypatch `APIBackend.build_messages_and_create_chat_completion` to return canned hypothesis/expression JSON) and `QA_PRIMARY_METRIC=U`, `QA_REQUIRE_FEASIBLE=true`, `QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner`, `QLIB_FACTOR_SUMMARIZER=quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback`. Assert: the trial ledger gains exactly one row carrying the expected `Θ.hash`; the trajectory's `backtest_metrics` contains both `RankIC` and `U`; the factor library JSON entry's `backtest_results` carries the full vector.
- **PATTERN**: this proves the plugin wiring end-to-end without LLM spend — the woolly plan's stated e2e.
- **VALIDATE**: `pytest tests/eval/test_e2e_wiring.py -q` passes.

### Task 20 — Head-to-head comparison under the same `E_Θ`
- **IMPLEMENT**: Score **both** factor sets under the same `E_Θ` (else it compares engines, not objectives): Arm A (baseline, RankIC objective) and Arm B (treatment, `U` objective) both final-scored by `EvaluationOperator`. Report side-by-side on 2022–2025: IC, RankIC, ICIR, **net IR**, **net ARR**, MDD, mean turnover, `ρ_max`, complexity, IS→OOS degradation — plus the paper's published numbers as a third reference column. Expected/interesting result: Arm A wins on raw IC/RankIC, Arm B wins on net IR/ARR after costs (the gap §3.2 argues exists). **If Arm B also wins on raw RankIC, suspect a bug** (e.g., `ρ_max` not enforced, or `feasible` not gating). Every evaluation appends to the ledger (the reproducible trial record).
- **PATTERN**: the standalone backtest (`configs/backtest.yaml`, `backtest/runner.py`) remains Qlib-based for the *reported* numbers; `E_Θ` is the *comparable* scorer across arms. The ledger is the substrate for the out-of-scope falsification critic.
- **VALIDATE**: a comparison table (markdown or CSV) is produced; ledger JSONL has one row per evaluated factor with a `Θ.hash`; sanity check that Arm B's net IR/ARR ordering is consistent with its `U` ordering.

---

## TESTING STRATEGY

### Unit Tests (`tests/eval/`, greenfield — Task 13)
- Nesting (reduces to frictionless at zero cost/latency), dollar neutrality, turnover bounds, cost monotonicity + impact super-linearity, determinism + hash sensitivity, no-look-ahead, scoring bounds. These are the invariants that actually catch bugs.

### Integration Tests
- `test_extract_metrics.py` (Task 15): Series with extras vs DataFrame without.
- `test_e2e_wiring.py` (Task 19): plugin swap + ledger + library vector, stub LLM, no spend.

### Edge Cases
- Empty signal panel → `FactorEmptyError` (skip-loop, `workflow.py:116-120`).
- `ρ_max` against an empty zoo (first factor) → define `ρ_max=0` (no incumbents).
- Dollar-neutral denominator `Σw(1+ỹ)≈0` → drift from notionals, not normalized weights (Task 7).
- Resumed session with an edited `protocol_csi300.yaml` → `Θ.hash` logged at every evaluation surfaces the stale protocol (Property 1, `workflow.py:140`).
- Ollama Cloud rejects `response_format` → `JSON_MODE_SUPPORTED=false` + `robust_json_parse` (Task 4).
- `QA_PRIMARY_METRIC` unset for control arm → RankIC path unchanged.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```bash
ruff check quantaalpha/eval quantaalpha/factors/net_cost_runner.py quantaalpha/factors/net_cost_feedback.py tests/eval
ruff format --check quantaalpha/eval quantaalpha/factors/net_cost_runner.py quantaalpha/factors/net_cost_feedback.py tests/eval
python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('quantaalpha/eval').rglob('*.py')]"
```

### Level 2: Unit Tests
```bash
pytest tests/eval/ -q
```

### Level 3: Integration / Wiring
```bash
# LLM smoke (Task 4)
python scripts/smoke_llm.py
# E2E wiring, stub LLM, no spend (Task 19)
QA_PRIMARY_METRIC=U QA_REQUIRE_FEASIBLE=true \
QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner \
QLIB_FACTOR_SUMMARIZER=quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback \
pytest tests/eval/test_e2e_wiring.py -q
```

### Level 4: Manual Validation (full arms — LLM spend)
```bash
# Control arm (Task 5)
EXPERIMENT_ID=baseline_kimi ./run.sh "<paper's initial direction>"
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source combined --factor-json all_factors_library.json
# Treatment arm (Task 14-17 wired via env)
EXPERIMENT_ID=treatment_kimi QA_PRIMARY_METRIC=U QA_REQUIRE_FEASIBLE=true \
QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner \
QLIB_FACTOR_SUMMARIZER=quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback \
./run.sh "<paper's initial direction>"
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source combined --factor-json all_factors_library_treatment.json
# Head-to-head (Task 20): score both under E_Θ, produce the comparison table.
```

---

## ACCEPTANCE CRITERIA
- [ ] `E_Θ` is a pure function of `(f, Θ)` (Property 2) — proven by the determinism unit test.
- [ ] `Θ` is frozen, hashed, and logged at every evaluation (Property 1); a mutated YAML changes the hash.
- [ ] No look-ahead: truncating the panel after `T` leaves `m(f)` on `≤T` unchanged (Property 3).
- [ ] Zero-cost/zero-latency `E_Θ` nests the frictionless objective (nesting test passes).
- [ ] Treatment arm optimizes `U` (env `QA_PRIMARY_METRIC=U`); control arm optimizes RankIC (default) — same `StrategyTrajectory` class.
- [ ] The factor library JSON stores the full metric vector with no schema migration.
- [ ] The pickle cache key includes `Θ.hash` (no arm collision).
- [ ] The feedback the LLM sees is net-of-cost / dimension-targeted, not the 4 frictionless literals.
- [ ] No edits to the loop, the DSL, CoSTEER, or the mutation/crossover prompt bodies — only `QLIB_FACTOR_*` env swaps + the small `trajectory.py`/`controller.py`/`feedback.py` edits.
- [ ] The in-loop test window (2021) is not widened.
- [ ] Both arms run to completion on CSI 300; the comparison table and the ledger are produced.
- [ ] All Level 1–3 validation commands pass.

## COMPLETION CHECKLIST
- [ ] Tasks 1–2: env + data path verified (`--dry-run` backtest on `alpha158_20`).
- [ ] Task 3–4: `json_mode_supported` flag + Ollama Cloud smoke test passes.
- [ ] Task 5: control arm run + standalone backtest; baseline factor library preserved.
- [ ] Tasks 6–12: `quantaalpha/eval/` package complete; `Θ.hash` stable & sensitive.
- [ ] Task 13: 7 unit invariants pass.
- [ ] Tasks 14–17: runner + summarizer wired; `_extract_metrics` extended; `get_primary_metric`/`is_successful` arm-selectable; metric list de-duplicated.
- [ ] Task 18: single seeding entry point; `PYTHONHASHSEED` in `run.sh`.
- [ ] Task 19: e2e wiring test passes with stub LLM.
- [ ] Task 20: head-to-head table + ledger produced.
- [ ] All validation commands (L1–L3) pass; L4 manual runs complete.
- [ ] Ready for `/commit`.