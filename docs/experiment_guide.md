# Quanta Alpha Quantitative Factor Mining Experiment Guide

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Experimental Approach and Methodology](#2-experimental-approach-and-methodology)
- [3. Main Experiment Workflow](#3-main-experiment-workflow)
- [4. Independent Backtest Framework](#4-independent-backtest-framework)
- [5. Evaluation Metrics](#5-evaluation-metrics)
- [6. Experiment Configuration and Hyperparameters](#6-experiment-configuration-and-hyperparameters)
- [7. Data and Backtest Settings](#7-data-and-backtest-settings)
- [8. Experimental Conclusions and Analysis](#8-experimental-conclusions-and-analysis)

---

## 1. Project Overview

### 1.1 Project Goals

**Quanta Alpha** is an LLM-driven quantitative factor auto-mining system. Its core goals are:

- **Automated factor discovery**: Use an LLM to generate market hypotheses and convert them into computable factor expressions
- **Evolutionary optimization**: Iteratively improve factor quality through mutation and crossover operations
- **End-to-end backtest validation**: Backtest factors based on the Qlib framework to evaluate their predictive power and investment value

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Quanta Alpha System Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User input (exploration direction)                            │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────┐                                               │
│   │   Planning   │ ──→ Generate N parallel exploration directions│
│   └──────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              Evolution Controller                         │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │
│   │  │ Original │→│ Mutation │→│ Crossover│→ loop...        │  │
│   │  └──────────┘  └──────────┘  └──────────┘               │  │
│   └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │           QuantAgentLoop (5-step loop)                    │  │
│   │  1. factor_propose    → LLM generates market hypothesis   │  │
│   │  2. factor_construct  → LLM generates factor expressions │  │
│   │  3. factor_calculate  → Parse and compute factor values   │  │
│   │  4. factor_backtest   → Qlib backtest                     │  │
│   │  5. feedback          → LLM analyzes feedback + adds factor│  │
│   └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────┐                                               │
│   │ Factor lib   │ ──→ Archive of all valid factors             │
│   │  (JSON)      │                                               │
│   └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Core File Structure

| File / Script | Function |
|-----------|----------|
| `run_experiment.sh` | Main experiment entry script; activates the environment and calls `alphaagent mine` |
| `backtest_v2/run_backtest.py` | Independent backtest tool; batch-backtests the factor library |
| `run_config.yaml` | Main experiment config file (planning, evolution, backtest parameters) |
| `backtest_v2/config.yaml` | Independent backtest config file |
| `all_factors_library_*.json` | Factor library output files |

---

## 2. Experimental Approach and Methodology

### 2.1 Core Method: LLM-Driven Evolutionary Factor Mining

Traditional factor mining relies on manual experience and domain knowledge, which is inefficient and hard to scale. Quanta Alpha adopts a novel approach:

#### 2.1.1 Hypothesis-Driven

```
User input (e.g. "momentum strategy")
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  LLM generates a market hypothesis       │
   │  e.g.:                                   │
   │  "The momentum effect of past N-day      │
   │   returns is significant in the A-share  │
   │   market; short-term momentum may        │
   │   reverse while medium-term momentum     │
   │   remains effective."                    │
   └─────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  LLM converts the hypothesis into        │
   │  factor expressions, e.g.:               │
   │  ($close - Ref($close, 5)) / Ref($close, 5)  │
   │  Rank(Mean($close/Ref($close,1)-1, 20))      │
   └─────────────────────────────────────────┘
```

#### 2.1.2 Evolutionary Optimization

Borrowing from genetic algorithms, factors are optimized via **mutation** and **crossover**:

```
             ┌─────────────────────────────────────────────┐
             │            Evolution Workflow                │
             ├─────────────────────────────────────────────┤
             │                                             │
  Round 0    │   Original round: N directions explored in    │
  (Original) │   parallel → generate N initial trajectories│
             │                                             │
             │              │                              │
             │              ▼                              │
             │                                             │
  Round 1    │   Mutation round: "mutate" the original     │
  (Mutation) │   trajectories                              │
             │   → generate orthogonal strategies based on │
             │     original trajectories                   │
             │   → avoid re-exploring the same paths       │
             │                                             │
             │              │                              │
             │              ▼                              │
             │                                             │
  Round 2    │   Crossover round: select K parents to       │
  (Crossover)│   combine                                   │
             │   → fuse the strengths of different          │
             │     trajectories                            │
             │   → generate new hybrid strategies           │
             │                                             │
             │              │                              │
             │              ▼                              │
             │                                             │
  Round 3+   │   continue mutation → crossover → mutation  │
             │   → ... until max rounds reached             │
             │                                             │
             └─────────────────────────────────────────────┘
```

### 2.2 Methodological Advantages

| Advantage | Description |
|------|------|
| **Interpretability** | LLM-generated factors come with hypothesis explanations, making the factor logic easy to understand |
| **Diversity** | The evolution mechanism ensures factor diversity and avoids local optima |
| **Automation** | The full pipeline is automated, reducing manual intervention |
| **Iterative optimization** | The feedback mechanism lets the system learn from failures and keep improving |

---

## 3. Main Experiment Workflow

### 3.1 Launch Command

```bash
# Basic usage
./run_experiment.sh "your exploration direction description"

# Examples
./run_experiment.sh "short-term reversal factor based on volume-price relationship"
./run_experiment.sh "market sentiment factor using volatility and volume"
```

### 3.2 The Five-Step Loop

Each exploration round contains 5 core steps:

#### Step 1: factor_propose (hypothesis generation)
- **Input**: exploration direction + historical trajectory (trace)
- **Process**: call the LLM to generate a market hypothesis
- **Output**: structured hypothesis description (hypothesis)
- **Core module**: `QuantAgentHypothesisGen`

```python
# Pseudocode example
hypothesis = llm.generate(
    prompt="""
    Based on the following exploration direction, generate a verifiable market hypothesis:
    Direction: {direction}
    Historical experience: {trace.history}

    Please describe:
    1. The core logic of the hypothesis
    2. The expected market phenomenon
    3. Possible verification methods
    """
)
```

#### Step 2: factor_construct (factor construction)
- **Input**: market hypothesis
- **Process**: the LLM converts the hypothesis into 2-3 factor expressions
- **Output**: list of factor expressions
- **Core module**: `QuantAgentHypothesis2FactorExpression`

```
# Factor expression syntax examples
$close                              # closing price
Ref($close, 5)                      # closing price 5 days ago
Mean($volume, 20)                   # 20-day average volume
Rank($close / Ref($close, 1) - 1)  # daily return rank
Std($close / Ref($close, 1) - 1, 20)  # 20-day return standard deviation
```

#### Step 3: factor_calculate (factor calculation)
- **Input**: factor expression
- **Process**: parse the expression and compute factor values
- **Output**: factor value matrix (time × stock)
- **Core module**: `QlibFactorParser`

##### 3.1 Calculation Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Factor Calculation Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Factor expression input                                                │
│   e.g.: "RANK(DELTA($close, 5) / TS_STD($close, 20))"                    │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  1. Expression preprocessing (expr_parser.py)                     │  │
│   │     - Parenthesis balance check                                    │  │
│   │     - Invalid operator check                                       │  │
│   │     - Unary-minus preprocessing (e.g. "* -$close" → "* (-1 * $close)")│
│   └──────────────────────────────────────────────────────────────────┘  │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  2. Expression parsing (pyparsing library)                        │  │
│   │     - Build the AST (abstract syntax tree)                         │  │
│   │     - Operator precedence: * / → + - → > < >= <= == != → && → ||  │  │
│   │     - Convert infix expressions to function-call form              │  │
│   │       e.g.: "$close + $open" → "ADD($close, $open)"                │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  3. Factor validation (factor_regulator.py)                      │  │
│   │     - Parseability check                                           │  │
│   │     - Duplicate subtree detection (vs existing factor library)   │  │
│   │     - Complexity constraints:                                      │  │
│   │       · Symbol length (SL) ≤ 300                                   │  │
│   │       · Base feature count (ER) ≤ 6                                │  │
│   │       · Free-parameter ratio < 50%                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  4. Load market data (Qlib)                                       │  │
│   │     - Data source: daily_pv.h5 (daily data)                       │  │
│   │     - Contains: $open, $high, $low, $close, $volume                │  │
│   │     - Index: MultiIndex (instrument, datetime)                     │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  5. Recursively compute factor values (function_lib.py)           │  │
│   │     - Variable replacement: "$close" → "df['$close']"              │  │
│   │     - Execute the parsed expression with eval()                    │  │
│   │     - All functions automatically handle groupby('instrument')     │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  6. Result output and caching                                      │  │
│   │     - Output: result.h5 (HDF5 format)                             │  │
│   │     - Index: MultiIndex (instrument, datetime)                     │  │
│   │     - Data type: float64                                           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

##### 3.2 Expression Parser Implementation

The expression parser is built on the **pyparsing** library and supports full arithmetic, comparison, logical, and conditional expressions:

```python
# Core parsing logic (expr_parser.py)

# 1. Define basic elements
var = Combine(Optional("$") + Word(alphas, alphanums + "_"))  # variable: $close, volume
number = Regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")   # number: 1.5, -3, 1e-8

# 2. Define operator precedence (low to high)
expr = infixNotation(operand, [
    (mul_div,      2, LEFT,  parse_arith_op),       # * /
    (add_minus,    2, LEFT,  parse_arith_op),       # + -
    (comparison,   2, LEFT,  parse_comparison_op),  # > < >= <= == !=
    (logical_and,  2, LEFT,  parse_logical),         # && &
    (logical_or,   2, LEFT,  parse_logical),         # || |
    (conditional,  3, RIGHT, parse_conditional)      # ? :
])

# 3. Convert operators to function calls
# e.g.: "$close + $open"  →  "ADD($close, $open)"
#       "$close > $open"  →  "GT($close, $open)"
#       "A ? B : C"       →  "WHERE(A, B, C)"
```

##### 3.3 Supported Factor Function Library

The system has a rich built-in factor computation library (`function_lib.py`), divided into the following categories:

**Time-series functions (TS_*)** — grouped by stock, computed along the time axis

| Function | Description | Example |
|------|------|------|
| `DELTA(df, p)` | p-period difference | `DELTA($close, 5)` 5-day close change |
| `DELAY(df, p)` | delay by p periods | `DELAY($close, 1)` previous day's close |
| `TS_MEAN(df, p)` | p-period rolling mean | `TS_MEAN($volume, 20)` 20-day avg volume |
| `TS_STD(df, p)` | p-period rolling std | `TS_STD($close, 20)` 20-day volatility |
| `TS_MAX(df, p)` | p-period rolling max | `TS_MAX($high, 10)` 10-day high |
| `TS_MIN(df, p)` | p-period rolling min | `TS_MIN($low, 10)` 10-day low |
| `TS_RANK(df, p)` | p-period rolling rank | `TS_RANK($close, 20)` 20-day rank |
| `TS_CORR(df1, df2, p)` | p-period rolling correlation | `TS_CORR($close, $volume, 20)` |
| `TS_SUM(df, p)` | p-period rolling sum | `TS_SUM($volume, 5)` 5-day volume |
| `TS_ARGMAX(df, p)` | argmax position | `TS_ARGMAX($close, 20)` days since the highest point |
| `TS_ARGMIN(df, p)` | argmin position | `TS_ARGMIN($close, 20)` days since the lowest point |
| `TS_PCTCHANGE(df, p)` | p-period return | `TS_PCTCHANGE($close, 5)` 5-day return |

**Cross-sectional functions (CS_*)** — grouped by date, computed across stocks

| Function | Description | Example |
|------|------|------|
| `RANK(df)` | Cross-sectional rank (percentage) | `RANK($close)` close-price rank |
| `ZSCORE(df)` | Cross-sectional standardization | `ZSCORE($volume)` volume Z-score |
| `MEAN(df)` | Cross-sectional mean | `MEAN($close)` market-wide average price |
| `STD(df)` | Cross-sectional std | `STD($close)` market-wide price dispersion |
| `SCALE(df)` | Cross-sectional scaling | `SCALE($close)` normalization |

**Math functions**

| Function | Description | Example |
|------|------|------|
| `ABS(df)` | Absolute value | `ABS(DELTA($close, 1))` |
| `LOG(df)` | Natural log | `LOG($volume)` |
| `SQRT(df)` | Square root | `SQRT($volume)` |
| `SIGN(df)` | Sign function | `SIGN(DELTA($close, 1))` |
| `POW(df, n)` | Power | `POW($close, 2)` |
| `EXP(df)` | Exponential | `EXP($close / 100)` |

**Technical indicator functions**

| Function | Description | Example |
|------|------|------|
| `SMA(df, m)` | Simple moving average | `SMA($close, 20)` |
| `EMA(df, p)` | Exponential moving average | `EMA($close, 12)` |
| `WMA(df, p)` | Weighted moving average | `WMA($close, 10)` |
| `MACD(price, s, l)` | MACD indicator | `MACD($close, 12, 26)` |
| `RSI(price, p)` | Relative strength index | `RSI($close, 14)` |
| `BB_UPPER/MIDDLE/LOWER` | Bollinger bands | `BB_UPPER($close, 20)` |
| `DECAYLINEAR(df, p)` | Linear decay weighting | `DECAYLINEAR($close, 10)` |

**Regression functions**

| Function | Description | Example |
|------|------|------|
| `REGBETA(y, x, p)` | Rolling regression coefficient | `REGBETA($close, $volume, 20)` |
| `REGRESI(y, x, p)` | Rolling regression residual | `REGRESI($close, MEAN($close), 20)` |

**Logical and conditional functions**

| Function | Description | Example |
|------|------|------|
| `GT(a, b)` | Greater than | `GT($close, $open)` |
| `LT(a, b)` | Less than | `LT($close, DELAY($close, 1))` |
| `AND(a, b)` | Logical AND | `AND(GT($close, $open), GT($volume, 1e8))` |
| `OR(a, b)` | Logical OR | `OR(GT($close, $open), LT($low, $open))` |
| `WHERE(cond, t, f)` | Conditional selection | `WHERE(GT($close, $open), $high, $low)` |

##### 3.4 Factor Complexity Regularization

To prevent factors from being overly complex or duplicating existing ones, the system implements complexity regularization (`FactorRegulator`):

```
Complexity penalty: R_g(f, h) = α₁·SL(f) + α₂·PC(f) + α₃·ER(f, h)

where:
- SL(f): Symbol Length — number of characters in the expression
- PC(f): Parameter Complexity — ratio of free parameters
- ER(f, h): Expression Redundancy — number of base features
```

**Validation rules**:

| Metric | Threshold | Description |
|------|------|------|
| Symbol length (SL) | ≤ 300 | Expression must not be too long |
| Base feature count (ER) | ≤ 6 | At most 6 raw features |
| Free-parameter ratio | < 50% | Numeric constants must not be too prevalent |
| Duplicate subtree size | ≤ 8 | Overlap with existing factors must not be too large |

##### 3.5 Computation Execution and Caching

The final execution of factor computation uses Python's `eval()`:

```python
# Computation template (template.jinja2)
def calculate_factor(expr: str, name: str):
    # 1. Load data
    df = pd.read_hdf('./daily_pv.h5', key='data')

    # 2. Symbol replacement
    expr = parse_symbol(expr, df.columns)   # TRUE → True, $close → close
    expr = parse_expression(expr)            # parse into function-call form

    # 3. Variable replacement
    for col in df.columns:
        expr = expr.replace(col[1:], f"df['{col}']")  # close → df['$close']

    # 4. Execute computation
    df[name] = eval(expr)  # execute the parsed expression
    result = df[name].astype(np.float64)

    # 5. Save result
    result.to_hdf('result.h5', key='data')
```

**Caching mechanism**:
- Results are saved in HDF5 format (`result.h5`)
- Workspace path: `/mnt/DATA/quantagent/QuantaAlpha/QuantaAlpha_workspace/{UUID}/`
- The independent backtest framework can reuse these results via the cache-extraction tool

#### Step 4: factor_backtest (factor backtest)
- **Input**: computed factor values
- **Process**: ML-based backtest via Qlib
- **Output**: backtest metrics (IC, ICIR, returns, etc.)
- **Core module**: `QlibFactorRunner`

```yaml
# Main program backtest config (conf.yaml)
Training set: 2016-01-01 ~ 2020-12-31  # 5 years
Validation set: 2021-01-01 ~ 2021-12-31  # 1 year
Test set: 2022-01-01 ~ 2025-12-26  # ~4 years

Model: LightGBM
Strategy: TopkDropoutStrategy (Top50, Drop5)
```

#### Step 5: feedback (feedback and library insertion)
- **Input**: backtest results + hypothesis
- **Process**: the LLM analyzes the results and generates feedback
- **Output**: feedback report + factor written to the library
- **Core module**: `QuantAgentQlibFactorHypothesisExperiment2Feedback`

```python
# Factor library entry structure
factor_entry = {
    "factor_name": "momentum_5d",
    "factor_expression": "($close - Ref($close, 5)) / Ref($close, 5)",
    "hypothesis": "short-term momentum effect...",
    "direction": "momentum strategy",
    "evolution_phase": "original" | "mutation" | "crossover",
    "metrics": {
        "RankIC": 0.05,
        "RankICIR": 0.8,
        "annualized_return": 0.15,
        ...
    }
}
```

### 3.3 Evolution Controller

The evolution controller (`EvolutionController`) manages the entire evolution process:

```
┌─────────────────────────────────────────────────────────────┐
│                  Evolution Controller                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TrajectoryPool (trajectory pool)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Trajectory 1: direction=0, phase=original, ic=0.03  │   │
│  │ Trajectory 2: direction=1, phase=original, ic=0.05  │   │
│  │ Trajectory 3: direction=0, phase=mutation, ic=0.04  │   │
│  │ Trajectory 4: parents=[1,2], phase=crossover, ic=0.06│  │
│  │ ...                                                  │  │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Task scheduling logic                                       │
│  ├─ Original: create an initial task for each planning dir  │
│  ├─ Mutation: generate a mutation task for each trajectory│
│  └─ Crossover: select K parent combinations, generate tasks│
│                                                              │
│  Parent selection strategies                                 │
│  ├─ best: prefer the best-performing trajectories            │
│  ├─ weighted: performance-weighted sampling (poor ones      │
│  │   more likely to be selected, encouraging exploration)   │
│  └─ random: random selection                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Crossover-round evaluator — evaluation mechanism

**1. Visible evaluation metrics**

The crossover-round evaluator can see 7 metrics for each trajectory:

| Metric | Description | Use |
|--------|-------------|-----|
| IC | Information coefficient | Correlation between factor and return |
| ICIR | IC information ratio | Stability of IC |
| RankIC | Rank IC | A more robust factor-correlation metric |
| RankICIR | Rank IC information ratio | Stability of RankIC |
| annualized_return | Annualized excess return | Strategy profitability |
| information_ratio | Information ratio | Risk-adjusted return |
| max_drawdown | Maximum drawdown | Strategy risk |

**2. Primary evaluation metric**

Currently RankIC is used as the primary evaluation metric (the `get_primary_metric()` method):

`trajectory.py` lines 90-92:
```python
def get_primary_metric(self) -> Optional[float]:
    """Get the primary metric (RankIC) for comparison."""
    return self.backtest_metrics.get("RankIC")
```

**3. Which decisions the evaluation is used for**

| Decision point | Usage |
|----------------|-------|
| Parent selection | Use `get_primary_metric()` (RankIC) to sort or weight according to `parent_selection_strategy` |
| Combination scoring | Evaluate combination quality by average RankIC in `select_crossover_pairs` |
| Diversity preference | Combine `direction_id` diversity + phase diversity + average performance into a composite score |

**4. Trajectory information visible to the LLM**

When generating crossover prompts, the LLM can see:

```
### Parent 1: Original Round
**Direction ID**: 0
**Hypothesis**: price-volume factor mining...
**Factors**:
  - ROC60_Factor: RANK(TS_PCTCHANGE($close, 60))...
**Metrics**:
  - IC: 0.0053
  - ICIR: 0.0418
  - RankIC: 0.0220
  - RankICIR: 0.1789
  - annualized_return: 0.068
  - information_ratio: 1.12
  - max_drawdown: -0.05
**Feedback**: The results show...
```

**5. Evaluation flowchart**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Crossover-round evaluation flow                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Candidate trajectory pool (from previous rounds)            │
│     ├── Trajectory A: RankIC=0.22, direction=0, phase=original │
│     ├── Trajectory B: RankIC=0.18, direction=1, phase=original │
│     ├── Trajectory C: RankIC=0.25, direction=0, phase=mutation│
│     └── Trajectory D: RankIC=0.15, direction=1, phase=mutation│
│                                                                  │
│  2. Filter by parent_selection_strategy                          │
│     ├── best: take Top-N by RankIC descending                   │
│     ├── weighted: higher RankIC → higher weight                 │
│     ├── weighted_inverse: lower RankIC → higher weight (explore)│
│     └── top_percent_plus_random: top 30% guaranteed + random   │
│                                                                  │
│  3. Generate combinations and score                              │
│     ├── Combination 1: [A, C] → score = diversity + avg_metric  │
│     ├── Combination 2: [A, D] → score = ...                    │
│     └── ...                                                      │
│                                                                  │
│  4. Select the Top-N combinations as crossover parents           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Independent Backtest Framework

### 4.1 Design Purpose

The in-loop backtest (`factor_backtest`) during the main experiment is used to quickly evaluate a single factor. The independent backtest framework (`backtest_v2`) is used to:

1. **Batch evaluation**: uniformly backtest the entire factor library
2. **Long-period validation**: validate factor stability over a longer test period (2022-2025)
3. **Combination effect**: evaluate the overall performance of multi-factor combinations
4. **Baseline comparison**: compare with the official Qlib factor library (Alpha158)

### 4.2 Backtest Configuration

```yaml
# backtest_v2/config.yaml key configuration

Data configuration:
  Data source: ~/.qlib/qlib_data/cn_data
  Market: csi300 (CSI 300 constituents)
  Data range: 2016-01-01 ~ 2025-12-26

Dataset split:
  Training set: 2016-01-01 ~ 2020-12-31  # model learns historical patterns
  Validation set: 2021-01-01 ~ 2021-12-31  # model tuning
  Test set: 2022-01-01 ~ 2025-12-26  # out-of-sample validation (final evaluation)

Model configuration:
  Model type: LightGBM
  Learning rate: 0.1
  Max depth: 8
  Number of leaves: 210
  Early-stopping rounds: 50
  Max iterations: 500

Backtest strategy:
  Strategy: TopkDropoutStrategy
  - topk: 50      # hold the 50 highest-rated stocks
  - n_drop: 5     # drop the 5 lowest-rated stocks at each rebalance

Transaction costs:
  Buy: 0.05%
  Sell: 0.15%
  Minimum: 5 yuan
```

### 4.3 Usage

```bash
# Backtest a single factor library
python backtest_v2/run_backtest.py \
    -c backtest_v2/config.yaml \
    --factor-source custom \
    --factor-json /path/to/factors.json

# Compare with Alpha158
python backtest_v2/run_backtest.py \
    -c backtest_v2/config.yaml \
    --factor-source alpha158_20

# Batch backtest
./batch_backtest.sh
```

### 4.4 Differences Between the Two Backtests

| Feature | In-loop backtest | Independent backtest framework |
|------|---------------|--------------|
| Purpose | Quickly evaluate a single factor | Batch-evaluate a factor library |
| Backtest period | 2021 (validation set) | 2022-2025 (test set) |
| Number of factors | 2-3 per round | The whole library |
| Cache | Auto-cached to workspace | Independent cache directory |
| Output | Written to factor library JSON | Independent metrics JSON |

---

## 5. Evaluation Metrics

### 5.1 Important note: return metrics are all excess returns

⚠️ **Note**: The return-type metrics output by the backtest framework (`annualized_return`, `max_drawdown`, etc.) **are all excess returns** — i.e. performance relative to the benchmark (CSI 300 index) net of cost, not absolute returns.

**Formula**:
```python
excess return = portfolio return - benchmark return - transaction cost
excess_return = portfolio_return - bench_return - cost
```

All risk-analysis metrics (annualized return, max drawdown, information ratio, etc.) are computed from this excess-return series.

### 5.2 Predictive-Power Metrics

#### IC (Information Coefficient)
```
Definition: Pearson correlation between factor values and future returns
Range: [-1, 1]
Interpretation:
  IC > 0.03: the factor has some predictive power
  IC > 0.05: the factor has fairly strong predictive power
  IC > 0.10: the factor has very strong predictive power (rare)

Formula:
  IC_t = Corr(Factor_t, Return_{t+1})
  IC = Mean(IC_t)  # average over all times
```

#### ICIR (IC Information Ratio)
```
Definition: ratio of the mean of IC to its standard deviation
Formula: ICIR = Mean(IC) / Std(IC)
Interpretation:
  ICIR > 0.5: the factor's predictive power is stable
  ICIR > 1.0: the factor's predictive power is very stable

Significance: a factor with a high but unstable IC may be worse than one with a moderate but stable IC
```

#### Rank IC
```
Definition: Spearman correlation between factor rank and return rank
Advantage: more robust to outliers, better suited to real investment scenarios
```

#### Rank ICIR
```
Definition: ratio of the mean of Rank IC to its standard deviation
Formula: RankICIR = Mean(RankIC) / Std(RankIC)
```

### 5.3 Return Metrics (all excess returns)

#### Excess annualized return (Annualized Return)
```
Definition: the strategy's annualized excess return relative to the benchmark
Formula: Ann_Excess_Return = (1 + Total_Excess_Return)^(252/Trading_Days) - 1
Example:
  0.18 (18%) means the strategy beats the benchmark by 18% per year

Note: transaction costs are already deducted from this metric
```

#### Information Ratio
```
Definition: ratio of excess return to tracking error
Formula: IR = Mean(Excess_Return) / Std(Excess_Return) × √252
Interpretation:
  IR > 1.0: the strategy significantly beats the benchmark
  IR > 2.0: the strategy performs excellently

Significance: how much excess return you get per unit of tracking-error risk
```

#### Excess max drawdown (Max Drawdown)
```
Definition: the largest peak-to-trough decline of the excess-return curve
Formula: MDD = min((Excess_Cumulative_t - Excess_Peak) / Excess_Peak)
Example:
  -0.09 (-9%) means the excess return fell 9% from its peak

Note: this is a drawdown relative to the benchmark, not the portfolio's absolute drawdown
```

#### Calmar Ratio
```
Definition: ratio of excess annualized return to excess max drawdown
Formula: Calmar = Ann_Excess_Return / |MDD|
Interpretation:
  Calmar > 1.0: good risk-adjusted return
  Calmar > 2.0: excellent risk-adjusted return
```

### 5.4 Metric Interpretation Example

Using actual backtest results (the `QA_phase_mutation` factor library):

```json
{
  "IC": 0.1277,                    // very strong predictive power
  "ICIR": 0.8738,                 // stable predictive power
  "Rank IC": 0.1242,              // strong rank predictive power
  "Rank ICIR": 0.8566,            // stable rank prediction
  "annualized_return": 0.1827,    // excess annualized return 18.27%
  "information_ratio": 2.2588,    // information ratio 2.26
  "max_drawdown": -0.0894,        // excess max drawdown 8.94%
  "calmar_ratio": 2.0447          // risk-adjusted return 2.04
}
```

**Overall assessment**:
- This factor library performs excellently over the 2022-2025 test period
- IC > 0.12 indicates the factor combination has very strong return-prediction power
- **Excess annualized return 18.27%**: the strategy beats the CSI 300 index by 18.27% per year on average (net of transaction costs)
- **Information ratio 2.26**: per 1% of tracking error, the strategy earns 2.26% of excess return (IR > 2 is usually considered excellent)
- **Excess max drawdown 8.94%**: the maximum decline relative to the benchmark is kept within 9%
- **Calmar ratio 2.04**: the excess return is twice the max drawdown, a good risk-return profile

### 5.5 Benchmark-Comparison Summary

| Metric type | Specific metric | Benchmark comparison method |
|----------|----------|--------------|
| Predictive power | IC, ICIR, Rank IC, Rank ICIR | Directly compute the factor-return correlation |
| Return | annualized_return | **Excess** annualized = portfolio annualized - benchmark annualized - cost |
| Risk | max_drawdown | Max drawdown of the **excess**-return curve |
| Composite | information_ratio, calmar_ratio | Computed from excess returns |

**Benchmark settings**:
```yaml
benchmark: SH000300  # CSI 300 index
market: csi300       # CSI 300 constituents
```

---

## 6. Experiment Configuration and Hyperparameters

### 6.1 Core Hyperparameters

#### Planning stage (Planning)
```yaml
planning:
  num_directions: 10    # number of parallel exploration directions
                        # larger → broader initial exploration, but more resource use
                        # recommended range: 5-15
```

#### Evolution stage (Evolution)
```yaml
evolution:
  max_rounds: 5         # max number of evolution rounds
                        # larger → deeper exploration, but longer runtime
                        # recommended range: 3-7

  crossover_size: 2     # number of parents per crossover
                        # 2: pairwise crossover (most common)
                        # 3: three-way crossover (more diverse)

  crossover_n: 10       # number of combinations generated per crossover round
                        # larger → broader crossover exploration
                        # recommended range: 5-15

  parent_selection_strategy: best  # parent selection strategy
                        # best: prefer the best trajectories
                        # weighted: weighted sampling (encourages exploration)
                        # random: random selection
```

#### Execution stage (Execution)
```yaml
execution:
  max_loops: 7          # max number of loops per trajectory
                        # total steps = max_loops × 5

  steps_per_loop: 5     # fixed at 5 (5-step loop)
```

### 6.2 Configuration Examples

**Exploration mode** (breadth-first):
```yaml
planning:
  num_directions: 15
evolution:
  max_rounds: 3
  crossover_n: 15
  parent_selection_strategy: weighted
```

**Depth mode** (depth-first):
```yaml
planning:
  num_directions: 5
evolution:
  max_rounds: 7
  crossover_n: 5
  parent_selection_strategy: best
```

**Balanced mode** (recommended):
```yaml
planning:
  num_directions: 10
evolution:
  max_rounds: 5
  crossover_n: 10
  parent_selection_strategy: best
```

---

## 7. Data and Backtest Settings

### 7.1 Data Source

```
Data source: Qlib China A-share data
Path: ~/.qlib/qlib_data/cn_data

Contains:
- Daily quotes: open, high, low, close, volume
- Fundamentals such as market cap and valuation
- CSI 300 constituent list
```

### 7.2 Market Settings

```yaml
Market: csi300 (CSI 300 constituents)
Benchmark: SH000300 (CSI 300 index)

Reasons for this choice:
- Good liquidity, low transaction cost
- Highly representative, covers core blue chips
- High data quality, few outliers
```

### 7.3 Time Split

```
┌───────────────────────────────────────────────────────────────┐
│                        Timeline                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  2016        2020        2021        2022        2025         │
│    │──────────┼───────────┼───────────┼───────────┤          │
│    │ Training │ Validation│         Test set       │          │
│    │  5 years │  1 year   │         4 years        │          │
│    │          │           │                       │          │
│    │ Model    │ Tuning    │   Out-of-sample eval   │          │
│    │ learning │           │   (final)              │          │
│    │ history  │           │                       │          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Design considerations**:
- **Training set (2016-2020)**: 5 years of data is enough for the model to learn market patterns
- **Validation set (2021)**: prevents overfitting, used for early stopping and hyperparameter tuning
- **Test set (2022-2025)**: fully out-of-sample, evaluates true generalization ability

### 7.4 Trading Strategy

```yaml
Strategy: TopkDropoutStrategy
Parameters:
  topk: 50      # hold the 50 highest-rated stocks
  n_drop: 5     # drop the 5 lowest-rated stocks at each rebalance

Logic:
1. Each trading day, compute a prediction score for all stocks
2. Select the top 50 by score to build the portfolio
3. At each rebalance:
   - keep stocks in the old portfolio that are still in the Top 50
   - sell the 5 lowest-rated stocks
   - buy the stocks newly entering the Top 50
4. Equal-weight holdings

Advantages:
- Avoids overtrading (only drops the worst)
- Keeps the portfolio stable
- Reduces transaction costs
```

---

## 8. Experimental Conclusions and Analysis

### 8.1 Experiment Design Summary

1. **Input**: the user provides an exploration direction (e.g. "momentum strategy", "value factor")
2. **Process**:
   - Planning generates 10 parallel directions
   - 5 evolution rounds (original → mutation → crossover → mutation → crossover)
   - 7 loops per round, each generating 2-3 factors
3. **Output**: a factor library JSON (containing hundreds of validated factors)

### 8.2 Key Findings

#### Impact of the Evolution Phase on Factor Quality

| Evolution phase | Factor characteristics | Typical performance |
|----------|----------|----------|
| Original | Basic exploration | IC distribution is wide, contains many noisy factors |
| Mutation | Targeted improvement | Targeted refinement on the original basis, IC improves slightly |
| Crossover | Combinatorial innovation | Fuses the strengths of multiple directions, may produce breakthrough factors |

#### Factor Selection Strategy
- **RankIC ranking**: select the factors with the strongest predictive power
- **Phase filtering**: factors from different phases may have different characteristics
- **Random sampling**: validate the overall quality of the factor library

### 8.3 Practical Recommendations

1. **First run**: use the balanced-mode configuration and run the full pipeline
2. **Baseline comparison**: always compare with Alpha158
3. **Multiple validations**: verify across different time windows and markets
4. **Factor selection**: focus on factors with RankIC > 0.03 and ICIR > 0.5

### 8.4 Limitations and Future Improvements

| Limitation | Possible improvement |
|--------|-----------|
| LLM-generated factors may be biased | Add stronger factor-regularization constraints |
| Backtests may overfit | Use more out-of-sample validation |
| High compute-resource consumption | Optimize the caching mechanism, incremental computation |
| Simplified transaction costs | Consider a more realistic slippage model |

---

## Appendix

### A. Common Commands Cheat Sheet

```bash
# Run the main experiment
./run_experiment.sh "your exploration direction"

# Independent backtest
python backtest_v2/run_backtest.py -c backtest_v2/config.yaml \
    --factor-source custom \
    --factor-json /path/to/factors.json

# Batch backtest
./batch_backtest.sh

# View the factor library
python show_all_factors.py

# Clear the cache
./clear_cache.sh
```

### B. Directory Structure

```
AlphaAgent/                      # Quanta Alpha main directory
├── run_experiment.sh            # main experiment entry
├── batch_backtest.sh            # batch backtest script
├── clear_cache.sh               # cache-clearing script
├── all_factors_library_*.json  # factor library output
├── factor_library/              # factor library samples
├── backtest_v2/                 # independent backtest framework
│   ├── run_backtest.py          # backtest entry
│   ├── config.yaml              # backtest config
│   └── ...
├── alphaagent/                  # core code
│   ├── app/                     # app entry
│   │   └── qlib_rd_loop/        # main loop
│   │       ├── run_config.yaml
│   │       ├── factor_mining.py
│   │       └── ...
│   ├── components/              # components
│   │   ├── workflow/            # workflow
│   │   ├── coder/               # factor coding
│   │   └── proposal/            # proposal generation
│   └── scenarios/               # scenario config
│       └── qlib/                # Qlib scenario
└── /mnt/DATA/quantagent/        # data storage
    └── AlphaAgent/
        ├── factor_cache/        # factor cache
        ├── backtest_v2_results/ # backtest results
        └── QuantaAlpha_workspace/  # workspace
```

### C. Metric Cheat Sheet

| Metric | Name | Type | Good threshold | Description |
|------|--------|------|----------|------|
| IC | Information coefficient | Predictive power | > 0.05 | Factor-return correlation |
| ICIR | IC information ratio | Prediction stability | > 0.5 | Stability of IC |
| Rank IC | Rank IC | Predictive power | > 0.05 | Rank correlation, more robust |
| Rank ICIR | Rank IC information ratio | Prediction stability | > 0.5 | Stability of Rank IC |
| annualized_return | **Excess** annualized return | Return | > 10% | Annualized excess vs benchmark |
| information_ratio | Information ratio | Risk-adjusted return | > 1.0 | Excess return / tracking error |
| max_drawdown | **Excess** max drawdown | Risk | > -15% | Max drop of the excess-return curve |
| calmar_ratio | Calmar ratio | Risk-adjusted return | > 1.0 | Excess annualized / |max drawdown| |

### D. References

1. Qlib: An AI-oriented Quantitative Investment Platform (Microsoft Research)
2. LightGBM: A Highly Efficient Gradient Boosting Decision Tree
3. Factor Investing: From Traditional to Alternative Risk Premia

---

*Document version: 1.1*
*Project name: Quanta Alpha*
*Updated: 2026-01-17*