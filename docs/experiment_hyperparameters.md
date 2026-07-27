# AlphaAgent Experiment Hyperparameter Configuration Documentation

> This document details all hyperparameter settings in `run_experiment.sh` and its related configuration files.

---

## Table of Contents

1. [Model Configuration](#1-model-configuration)
2. [Planning Configuration](#2-planning-configuration-planning)
3. [Execution Configuration](#3-execution-configuration-execution)
4. [Evolution Configuration](#4-evolution-configuration-evolution)
5. [Factor Generation Configuration](#5-factor-generation-configuration-factor)
6. [Backtest Configuration](#6-backtest-configuration-backtest)
7. [Data Configuration](#7-data-configuration-data)
8. [Model Training Configuration](#8-model-training-configuration-lgbmodel)
9. [Trading Strategy Configuration](#9-trading-strategy-configuration-strategy)
10. [LLM Configuration](#10-llm-configuration)
11. [Logging Configuration](#11-logging-configuration-logging)
12. [Path Configuration](#12-path-configuration)

---

## 1. Model Configuration

### 1.1 LLM Model Presets

Quickly switch models via the `MODEL_PRESET` environment variable:

| Preset name | Reasoning model | Chat model | API endpoint |
|---------|---------|---------|----------|
| `gemini` | `google/gemini-3-pro-preview` | `google/gemini-3-pro-preview` | OpenRouter |
| `deepseek` | `deepseek/deepseek-v3.2` | `deepseek/deepseek-v3.2` | OpenRouter |
| `deepseek_aliyun` | `deepseek-v3.2` | `deepseek-v3.2` | Alibaba Cloud DashScope |
| `claude` | `anthropic/claude-sonnet-4.5` | `anthropic/claude-sonnet-4.5` | OpenRouter |
| `gpt` | `openai/gpt-5.2` | `openai/gpt-5.2` | OpenRouter |
| `qwen` | `qwen3-235b-a22b-instruct-2507` | `qwen3-235b-a22b-instruct-2507` | Alibaba Cloud DashScope |

### 1.2 Environment Variable Overrides

```bash
REASONING_MODEL=<model_name>   # Reasoning model
CHAT_MODEL=<model_name>        # Chat model
OPENAI_API_KEY=<api_key>       # API key
OPENAI_BASE_URL=<base_url>     # API endpoint
```

---

## 2. Planning Configuration (Planning)

> Config file: `alphaagent/app/qlib_rd_loop/run_config.yaml`

| Parameter | Default | Type | Description |
|--------|--------|------|------|
| `enabled` | `true` | bool | Enable parallel planning |
| `num_directions` | **10** | int | 🔥**Hyperparam** Number of parallel exploration directions |
| `max_attempts` | `5` | int | Maximum planning retries |
| `use_llm` | `true` | bool | Whether to use the LLM to generate directions |
| `allow_fallback` | `true` | bool | Whether to fall back to built-in templates if the LLM fails |
| `prompt_file` | `planning_prompts.yaml` | str | Planning prompt file path |

---

## 3. Execution Configuration (Execution)

| Parameter | Default | Type | Description |
|--------|--------|------|------|
| `max_loops` | `11` | int | Maximum number of loop iterations |
| `steps_per_loop` | `5` | int | Steps per loop (fixed: propose hypothesis / build factor / calculate / backtest / feedback) |
| `step_n` | `null` | int | Total number of steps (highest priority; overrides max_loops × steps_per_loop) |
| `use_local` | `true` | bool | Use the local environment for backtesting (vs Docker) |
| `parallel_execution` | `false` | bool | Whether to run branches in parallel using multiprocessing |
| `branch_log_root` | `/mnt/DATA/quantagent/AlphaAgent/log` | str | Branch log root directory |
| `branch_log_prefix` | `branch` | str | Branch log prefix |

---

## 4. Evolution Configuration (Evolution)

> 🔥 Core hyperparameter section

| Parameter | Default | Type | Description |
|--------|--------|------|------|
| `enabled` | `true` | bool | Enable evolution mode |
| `mutation_enabled` | `true` | bool | Enable the mutation phase |
| `crossover_enabled` | `true` | bool | Enable the crossover phase |
| `max_rounds` | **11** | int | 🔥**Hyperparam** Maximum number of evolution rounds (including the original round) |
| `crossover_size` | **2** | int | 🔥**Hyperparam** Number of parents selected for each crossover (2 or 3) |
| `crossover_n` | **10** | int | 🔥**Hyperparam** Number of combinations generated per crossover round |
| `parallel_enabled` | `false` | bool | Parallel execution within the evolution phase |
| `prefer_diverse_crossover` | `true` | bool | Prefer diverse crossover combinations |
| `parent_selection_strategy` | `best` | str | Parent selection strategy |
| `top_percent_threshold` | `0.3` | float | Top-percent threshold (used with the `top_percent_plus_random` strategy) |
| `fresh_start` | `true` | bool | Whether to start from an empty trajectory pool |
| `cleanup_on_finish` | `false` | bool | Whether to clean up the trajectory pool after the experiment |
| `prompt_file` | `evolution_prompts.yaml` | str | Evolution prompt file path |

### 4.1 Parent Selection Strategies

| Strategy | Description |
|--------|------|
| `best` | Prefer the best-performing trajectories |
| `random` | Random selection |
| `weighted` | Performance-weighted sampling (higher performance → higher weight) |
| `weighted_inverse` | Inverse-performance-weighted sampling (encourages exploring poor trajectories) |
| `top_percent_plus_random` | Top 30% guaranteed + remaining filled randomly |

### 4.2 Evolution Workflow

```
Round 0: Original round  → generate initial factors
Round 1: Mutation round   → mutate existing factors
Round 2: Crossover round  → combine different factors
Round 3: Mutation round
Round 4: Crossover round
...
```

---

## 5. Factor Generation Configuration (Factor)

| Parameter | Default | Type | Description |
|--------|--------|------|------|
| `factors_per_hypothesis` | **3** | int | 🔥 Number of factors generated per hypothesis |

### 5.1 Complexity Constraints

| Parameter | Default | Description |
|--------|--------|------|
| `symbol_length_threshold` | **250** | 🔥 Maximum character length of a factor expression (key parameter to prevent overfitting) |
| `base_features_threshold` | `6` | Maximum number of distinct base features ($close, $open, etc.) |
| `free_args_ratio_threshold` | `0.5` | Maximum ratio of free parameters (numeric constants / total nodes) |

### 5.2 Duplication Check

| Parameter | Default | Description |
|--------|--------|------|
| `duplication.enabled` | `true` | Enable the duplication check |
| `duplication.threshold` | `5` | Duplicate subtree size threshold |
| `duplication.factor_zoo_path` | `null` | Factor library file path |

---

## 6. Backtest Configuration (Backtest)

| Parameter | Default | Type | Description |
|--------|--------|------|------|
| `use_docker` | `false` | bool | Use a Docker environment for backtesting |
| `timeout` | **800** | int | 🔥 Timeout per backtest (seconds) |
| `qlib.config_name` | `conf.yaml` | str | Qlib configuration file name |

---

## 7. Data Configuration (Data)

> Config file: `alphaagent/scenarios/qlib/experiment/factor_template/conf.yaml`

### 7.1 Qlib Initialization

| Parameter | Value | Description |
|--------|-----|------|
| `provider_uri` | `~/.qlib/qlib_data/cn_data` | Qlib data path |
| `region` | `cn` | Market region (cn/us) |

### 7.2 Market Configuration

| Parameter | Value | Description |
|--------|-----|------|
| `market` | **csi300** | 🔥 Stock pool (CSI 300) |
| `benchmark` | **SH000300** | 🔥 Benchmark index (CSI 300 index) |

### 7.3 Time Range

| Dataset | Time range | Description |
|--------|----------|------|
| **Overall data** | 2016-01-01 ~ 2025-12-26 | Full data range |
| **Training set** | 2016-01-01 ~ 2020-12-31 | Model training (5 years) |
| **Validation set** | 2021-01-01 ~ 2021-12-31 | Model validation (1 year) |
| **Test set** | 2022-01-01 ~ 2025-12-26 | Backtest evaluation (~4 years) |

### 7.4 Data Processors

```yaml
learn_processors:
  - Fillna (feature)      # Fill missing values
  - ProcessInf            # Handle infinite values
  - DropnaLabel           # Drop empty labels
  - CSRankNorm (feature)  # Cross-sectional rank normalization (features)
  - CSRankNorm (label)    # Cross-sectional rank normalization (label)

infer_processors:
  - Fillna (feature)
  - ProcessInf
  - CSRankNorm (feature)
  - CSRankNorm (label)
```

### 7.5 Label Definition

```python
label = "Ref($close, -2) / Ref($close, -1) - 1"  # T+2 daily return
```

---

## 8. Model Training Configuration (LGBModel)

> LightGBM model hyperparameters

| Parameter | Default | Description |
|--------|--------|------|
| `loss` | `mse` | Loss function |
| `learning_rate` | **0.05** | 🔥 Learning rate |
| `max_depth` | **8** | 🔥 Maximum tree depth |
| `num_leaves` | **210** | 🔥 Number of leaves |
| `colsample_bytree` | `0.8879` | Column sampling ratio |
| `subsample` | `0.8789` | Row sampling ratio |
| `lambda_l1` | `205.6999` | L1 regularization |
| `lambda_l2` | `580.9768` | L2 regularization |
| `num_threads` | `20` | Number of parallel threads |
| `early_stopping_round` | **50** | 🔥 Early-stopping rounds |
| `num_boost_round` | **500** | 🔥 Maximum number of boosting iterations |

---

## 9. Trading Strategy Configuration (Strategy)

### 9.1 TopkDropout Strategy

| Parameter | Default | Description |
|--------|--------|------|
| `topk` | **50** | 🔥 Number of held stocks |
| `n_drop` | **5** | 🔥 Number of stocks dropped per rebalance |

### 9.2 Transaction Costs

| Parameter | Default | Description |
|--------|--------|------|
| `account` | `100000000` | Initial capital (100 million) |
| `limit_threshold` | `0.095` | Price-limit threshold (9.5%) |
| `deal_price` | `open` | Execution price (open price) |
| `open_cost` | **0.0005** | 🔥 Buy cost (0.05%) |
| `close_cost` | **0.0015** | 🔥 Sell cost (0.15%) |
| `min_cost` | `5` | Minimum transaction cost (yuan) |

---

## 10. LLM Configuration

| Parameter | Default | Description |
|--------|--------|------|
| `factor_mining_timeout` | `999999` | Total factor-mining timeout (seconds) |
| `max_retries` | `3` | Maximum API call retries |
| `retry_delay` | `1.0` | Retry interval (seconds) |
| `json_mode_strict` | `true` | JSON-mode strictness |

---

## 11. Logging Configuration (Logging)

| Parameter | Default | Description |
|--------|--------|------|
| `level` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `save_snapshots` | `true` | Save intermediate session snapshots |
| `save_trajectory_pool` | `true` | Save the trajectory pool to JSON |

---

## 12. Path Configuration

### 12.1 Workspace Paths

```bash
# Auto-generated (default)
WORKSPACE_PATH=/mnt/DATA/quantagent/QuantaAlpha/QuantaAlpha_workspace_exp_YYYYMMDD_HHMMSS
PICKLE_CACHE_FOLDER_PATH=/mnt/DATA/quantagent/AlphaAgent/pickle_cache_exp_YYYYMMDD_HHMMSS

# Manual specification
EXPERIMENT_ID=my_exp bash run_experiment.sh "direction"

# Shared-directory mode
EXPERIMENT_ID=shared bash run_experiment.sh "direction"
```

### 12.2 Output Files

| File | Path | Description |
|------|------|------|
| Factor library | `all_factors_library.json` | Default output |
| Factor library (with suffix) | `all_factors_library_{suffix}.json` | Suffix-specified output |
| Config file | `alphaagent/app/qlib_rd_loop/run_config.yaml` | Main config file |
| Backtest config | `alphaagent/scenarios/qlib/experiment/factor_template/conf.yaml` | Qlib config |

---

## Appendix: Key Hyperparameter Summary

| Category | Parameter | Default | Impact |
|------|------|--------|------|
| **Planning** | `num_directions` | 10 | Number of initial exploration directions |
| **Evolution** | `max_rounds` | 11 | Total evolution rounds |
| **Evolution** | `crossover_size` | 2 | Number of crossover parents |
| **Evolution** | `crossover_n` | 10 | Number of crossover combinations per round |
| **Factor** | `factors_per_hypothesis` | 3 | Number of factors per hypothesis |
| **Factor** | `symbol_length_threshold` | 250 | Maximum expression length |
| **Model** | `learning_rate` | 0.05 | LGB learning rate |
| **Model** | `num_boost_round` | 500 | LGB iterations |
| **Strategy** | `topk` | 50 | Number of held stocks |
| **Strategy** | `n_drop` | 5 | Number of rebalanced stocks per day |
| **Data** | `market` | csi300 | Stock pool |
| **Data** | `train` | 2016-2020 | Training set range |
| **Data** | `test` | 2022-2025 | Test set range |

---

*Document generated: 2026-01-24*