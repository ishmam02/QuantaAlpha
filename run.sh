#!/bin/bash
# QuantaAlpha main experiment runner
#
# Usage:
#   ./run.sh "initial direction"                    # default experiment
#   ./run.sh "initial direction" "suffix"           # with factor library suffix
#   CONFIG=configs/experiment.yaml ./run.sh "direction"
#
# Examples:
#   ./run.sh "price-volume factor mining"
#   ./run.sh "momentum reversal factors" "exp_momentum"
#
# This runs the single objective: the net-of-cost, capacity-aware mean-variance
# construction with the ICIR linear combiner (E_Theta, problem_formulation.tex
# section 3). It swaps the runner and summarizer through the existing plugin env
# vars; the generation process (loop, DSL, CoSTEER, mutation and crossover) sits
# behind it.

# =============================================================================
# Locate project root
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =============================================================================
# Load .env configuration
# =============================================================================
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Error: .env file not found"
    echo "Please run: cp configs/.env.example .env"
    exit 1
fi

# QA_CHAT_SEED overrides CHAT_SEED *after* .env is sourced. Sourcing .env
# assigns CHAT_SEED unconditionally, so an exported value is silently clobbered
# -- which would have left every replication of a paper run on an identical LLM
# seed, varying only the evolution operators' RNG. The 4.05pp of run-to-run
# variance measured here is generation noise, and sampling it means resampling
# the generator, not just the parent-selection shuffle.
if [ -n "${QA_CHAT_SEED:-}" ]; then
    export CHAT_SEED="${QA_CHAT_SEED}"
fi

# QA_CHAT_MODEL / QA_REASONING_MODEL override CHAT_MODEL / REASONING_MODEL
# *after* .env is sourced. Same clobber problem as QA_CHAT_SEED: sourcing .env
# assigns CHAT_MODEL/REASONING_MODEL unconditionally, so a pre-exported value is
# silently overwritten. Used for per-run model swaps (e.g. a glm-vs-kimi smoke
# comparison) without editing .env.
if [ -n "${QA_CHAT_MODEL:-}" ]; then
    export CHAT_MODEL="${QA_CHAT_MODEL}"
fi
if [ -n "${QA_REASONING_MODEL:-}" ]; then
    export REASONING_MODEL="${QA_REASONING_MODEL}"
fi

# Evolve sequentially, not in parallel within a round. Two reasons, and the
# second is what crashed a run.
#
# 1. Feedback can only teach a batch generated after it, so batches produced
#    concurrently inside a round never see each other's verdicts.
# 2. With num_directions=10 and parallel_enabled, one run forks ~16 Python
#    processes, each loading pandas, qlib and panel data. That many
#    interpreters on an 8-core/16 GB box exhausted memory ~35 minutes in.
export QA_SEQUENTIAL_EVOLUTION="${QA_SEQUENTIAL_EVOLUTION:-true}"

# =============================================================================
# Activate conda environment
# =============================================================================
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null

if [ $? -ne 0 ]; then
    source activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null
fi

if ! command -v quantaalpha &> /dev/null; then
    echo "Error: quantaalpha command not found. Please install: pip install -e ."
    exit 1
fi

# Pin the factor-execution interpreter to the activated conda python.
# FACTOR_COSTEER_SETTINGS.python_bin defaults to bare "python"; factor.py:execute runs it
# via /bin/sh (shell=True), where "python" resolves to a non-conda interpreter missing
# quantaalpha's deps → ImportError → "No valid factor data found to merge". Pin the absolute path.
export FACTOR_CoSTEER_PYTHON_BIN="$(command -v python)"

# Qlib's qrun uses mlflow's filesystem tracking backend (./mlruns); newer mlflow disables it
# (maintenance mode). Allow it so the in-loop backtest (qrun) can create experiments/recorders.
export MLFLOW_ALLOW_FILE_STORE=true

# Thread budget. The tracked configs ask for 20 LightGBM threads, which
# oversubscribes an 8-core box 2.5x on its own and thrashes badly once several
# instances share it. QA_THREADS caps every numeric backend to one instance's
# fair share; the drivers set it from core count / parallelism.
if [ -n "${QA_THREADS:-}" ]; then
    export OMP_NUM_THREADS="${QA_THREADS}"
    export MKL_NUM_THREADS="${QA_THREADS}"
    export OPENBLAS_NUM_THREADS="${QA_THREADS}"
    export NUMEXPR_NUM_THREADS="${QA_THREADS}"
    export LGBM_NUM_THREADS="${QA_THREADS}"
fi

# Determinism. PYTHONHASHSEED must be set BEFORE the interpreter starts to have
# any effect -- setting it inside Python is too late. cli.py:app() seeds the
# Python and NumPy RNGs from `seed:` in the run config (or QA_SEED).
export PYTHONHASHSEED="${QA_SEED:-42}"

echo "Python: $(python --version)"
echo "QuantaAlpha: $(which quantaalpha)"
echo ""

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"configs/experiment.yaml"}

# -----------------------------------------------------------------------------
# Experiment identity
# -----------------------------------------------------------------------------
if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="meanvar_$(date +%Y%m%d_%H%M%S)"
fi
export EXPERIMENT_ID

RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    export WORKSPACE_PATH="${RESULTS_BASE}/workspace_${EXPERIMENT_ID}"
    export PICKLE_CACHE_FOLDER_PATH_STR="${RESULTS_BASE}/pickle_cache_${EXPERIMENT_ID}"
    mkdir -p "${WORKSPACE_PATH}" "${PICKLE_CACHE_FOLDER_PATH_STR}"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Workspace: ${WORKSPACE_PATH}"
fi

# =============================================================================
# Validate Qlib data
# =============================================================================
QLIB_DATA="${QLIB_DATA_DIR:-}"
if [ -z "${QLIB_DATA}" ]; then
    echo "Error: QLIB_DATA_DIR not set. Please set Qlib data path in .env"
    echo "Example: QLIB_DATA_DIR=/path/to/qlib/cn_data"
    exit 1
fi
if [ ! -d "${QLIB_DATA}" ]; then
    echo "Error: Qlib data directory does not exist: ${QLIB_DATA}"
    echo "Please check QLIB_DATA_DIR path in .env"
    exit 1
fi
# Validate required subdirectories
for subdir in calendars features instruments; do
    if [ ! -d "${QLIB_DATA}/${subdir}" ]; then
        echo "Error: Qlib data directory missing ${subdir}/: ${QLIB_DATA}"
        echo "Valid Qlib data dir must contain calendars/, features/, instruments/"
        exit 1
    fi
done
echo "Qlib data validated: ${QLIB_DATA}"

# Ensure Qlib data symlink
if [ -n "${QLIB_DATA}" ]; then
    QLIB_SYMLINK_DIR="$HOME/.qlib/qlib_data"
    # Resolve to an ABSOLUTE path. A relative symlink target resolves relative to
    # ~/.qlib/qlib_data/ (doubled/broken), which makes the in-loop qrun backtest load
    # a non-existent data dir (data_path=~/.qlib/qlib_data/data/qlib/cn_data).
    QLIB_DATA_ABS="$(cd "${QLIB_DATA}" && pwd)"
    mkdir -p "${QLIB_SYMLINK_DIR}"
    if [ ! -L "${QLIB_SYMLINK_DIR}/cn_data" ] || [ "$(readlink "${QLIB_SYMLINK_DIR}/cn_data")" != "${QLIB_DATA_ABS}" ]; then
        ln -sfn "${QLIB_DATA_ABS}" "${QLIB_SYMLINK_DIR}/cn_data"
    fi
fi

# =============================================================================
# Parse arguments and run
# =============================================================================
# Deliberately generic. It names the DATA and the framing, and asserts nothing
# about which effects exist on this market -- no horizon, no reversal-versus-
# continuation prior, no index name. Every admitted factor so far picks h=1 with
# a negative t (short-horizon reversal), and putting that in the direction would
# convert a finding into a tautology: a search told to look for reversal that
# finds reversal has demonstrated nothing about learning.
#
# It also matters less than it used to. With max_rounds 15 and scheduled reseed
# every 4 rounds, most directions in a run are generated by the system from
# measured outcomes rather than supplied here.
DIRECTION="${1:-cross-sectional equity factors from daily price and volume}"
LIBRARY_SUFFIX="$2"

# Factor library filename: data/factorlib/all_factors_library[_<suffix>].json
# (loop.py:224-232). The library is upserted, never truncated -- library.py
# loads the existing file, mutates the dict, and rewrites it -- so a shared
# filename ACCUMULATES factors across runs. Default to a per-run suffix so a
# run leaves any existing all_factors_library.json untouched.
#
# Precedence: positional $2 > a pre-set FACTOR_LIBRARY_SUFFIX > EXPERIMENT_ID.
# Exporting FACTOR_LIBRARY_SUFFIX="" selects the bare all_factors_library.json
# and accumulates into it.
if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
elif [ -z "${FACTOR_LIBRARY_SUFFIX+x}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${EXPERIMENT_ID}"
fi

# =============================================================================
# Wire the objective
# =============================================================================
# NOTE: QLIB_FACTOR_ is a shared env prefix across all three settings classes
# (settings.py:50/63/76), so this swaps the runner everywhere at once. That
# is what we want here.
export QLIB_FACTOR_RUNNER="quantaalpha.factors.net_cost_runner.NetCostFactorRunner"
export QLIB_FACTOR_SUMMARIZER="quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback"
# Fitness for parent selection. U was the obvious choice and the wrong one:
# measured on the mean_variance run it correlates +0.41 with repository SIZE
# and only +0.50 with the marginal contribution it is meant to stand in for,
# because a candidate ranks above more incumbents as the zoo fills with
# mediocre ones. Replaying that run, switching to delta_net_ir changes the
# bred parent in 7 of 7 groups and lifts the mean contribution of what gets
# bred from +0.00384 to +0.06507.
export QA_PRIMARY_METRIC="${QA_PRIMARY_METRIC:-delta_net_ir}"
# Rejection is FEEDBACK, not exclusion. Filtering rejected factors out of
# the parent pool would keep the generator from ever seeing why they were
# rejected, so it could not learn to clear the bar -- the gate would only
# shrink the repository instead of raising the quality of what enters it.
# U already ranks parents, so a rejected factor is deprioritised without
# being silenced. Set to true to restore hard exclusion.
export QA_REQUIRE_FEASIBLE="${QA_REQUIRE_FEASIBLE:-false}"
# Absolute paths: the loop runs factor code inside per-factor workspaces, and
# a relative protocol/ledger path would resolve against whatever the cwd
# happens to be at that moment. The default protocol is the ICIR linear
# combiner + soft-penalty mean-variance construction (the cap-sweep-validated
# config); override with QA_PROTOCOL to use another variant.
export QA_PROTOCOL="${QA_PROTOCOL:-${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml}"
export QA_LEDGER="${QA_LEDGER:-${SCRIPT_DIR}/${RESULTS_BASE#./}/ledger_${EXPERIMENT_ID}.jsonl}"
mkdir -p "$(dirname "${QA_LEDGER}")"
# Budget follows the outcome. QA_TARGET_MINED counts the LIBRARY (everything
# the search produced); QA_TARGET_ZOO counts ADMITTED factors. Use the first.
#
# 150 was always a MINING number -- it is expected_factor_count(10,10,5,3), and
# that function's docstring says "how many factors a run of this shape is
# expected to MINE". Setting it as QA_TARGET_ZOO asked for 150 admitted out of
# 150 generated: a 100% admission rate, i.e. a gate that never rejects. (With
# crossover_n=5 the shape generates 105, so it was asking for 143%.) The run of
# 2026-08-15 stopped at 9 admitted / 108 mined and could not have done better:
# admission is MARGINAL contribution to the existing book, so each admission
# raises the bar for the next and the rate falls toward zero by construction.
#
# So: target the library, and let the zoo settle wherever the objective puts it
# (~6-15). Select the deliverable afterwards on standalone significance --
# scripts/qa_screen_library.py -- which is how a desk builds a factor library.
# Measured on the 2026-08-15 run: 11 of 101 mined factors clear Harvey-Liu-Zhu
# t>3.0 on their own merit, a ~10% hit rate, so ~1500 mined yields ~150 usable.
export QA_TARGET_MINED="${QA_TARGET_MINED:-150}"
# Left unset by default: with QA_TARGET_MINED driving the budget, an admission
# target can only stop the run early on a number it cannot reach.
export QA_TARGET_ZOO="${QA_TARGET_ZOO:-}"
export QA_MAX_ROUNDS_CAP="${QA_MAX_ROUNDS_CAP:-60}"

# Factor execution + the coder's evolving loop run over this many processes
# (RD_AGENT_SETTINGS.multi_proc_n). 2 on a 16 GB box: measured 8.1 GB peak at 2
# vs 9.2 GB at 3, and 3 swaps. Independent of QA_SEED_WORKERS, which sizes the
# eval fork pool.
export MULTI_PROC_N="${MULTI_PROC_N:-2}"

# -----------------------------------------------------------------------------
# Preflight: fail before any LLM spend, not five minutes into the run
# -----------------------------------------------------------------------------
THETA_HASH=$(python - <<'PY' 2>/dev/null
import os, sys
try:
    from quantaalpha.core.utils import import_class
    from quantaalpha.eval.protocol import load_protocol
    theta = load_protocol(os.environ["QA_PROTOCOL"])
    import_class(os.environ["QLIB_FACTOR_RUNNER"])
    import_class(os.environ["QLIB_FACTOR_SUMMARIZER"])
    print(theta.hash)
except Exception as exc:
    print(f"PREFLIGHT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
)
if [ -z "${THETA_HASH}" ]; then
    echo "Error: preflight failed."
    echo "  Re-run this to see the error:"
    echo "    python -c \"from quantaalpha.eval.protocol import load_protocol; load_protocol('${QA_PROTOCOL}')\""
    exit 1
fi

echo ""
echo "========================================================================"
echo "  OBJECTIVE: net-of-cost, capacity-aware mean-variance (E_Theta, ICIR)"
echo "------------------------------------------------------------------------"
echo "  objective    : ${QA_PRIMARY_METRIC}   (feasibility enforced: ${QA_REQUIRE_FEASIBLE})"
echo "  protocol     : ${QA_PROTOCOL}"
echo "  theta hash   : ${THETA_HASH}"
echo "  runner       : ${QLIB_FACTOR_RUNNER##*.}"
echo "  summarizer   : ${QLIB_FACTOR_SUMMARIZER##*.}"
echo "  ledger       : ${QA_LEDGER}"
echo "========================================================================"
echo "Starting experiment..."
echo "Config: ${CONFIG_PATH}"
echo "Data: ${QLIB_DATA}"
echo "Results: ${RESULTS_BASE}"
echo "Factor library suffix: ${FACTOR_LIBRARY_SUFFIX}"
echo "----------------------------------------"

if [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    quantaalpha mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi
