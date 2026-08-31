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

# =============================================================================
# Parity with main -- the comparison arm must differ in the ALGORITHM, not the
# environment. Every override below exists because sourcing .env assigns the
# variable unconditionally, so a pre-exported value is silently clobbered.
# =============================================================================
# Per-run LLM swaps without editing .env (a model change is a confound, so both
# arms must be pointable at one model from outside).
if [ -n "${QA_CHAT_SEED:-}" ]; then
    export CHAT_SEED="${QA_CHAT_SEED}"
fi
if [ -n "${QA_CHAT_MODEL:-}" ]; then
    export CHAT_MODEL="${QA_CHAT_MODEL}"
fi
if [ -n "${QA_REASONING_MODEL:-}" ]; then
    export REASONING_MODEL="${QA_REASONING_MODEL}"
fi

# Thread budget. The tracked configs ask for 20 LightGBM threads, which
# oversubscribes an 8-core box 2.5x on its own and thrashes once several
# instances share it.
if [ -n "${QA_THREADS:-}" ]; then
    export OMP_NUM_THREADS="${QA_THREADS}"
    export MKL_NUM_THREADS="${QA_THREADS}"
    export OPENBLAS_NUM_THREADS="${QA_THREADS}"
    export NUMEXPR_NUM_THREADS="${QA_THREADS}"
    export LGBM_NUM_THREADS="${QA_THREADS}"
fi

# Determinism. PYTHONHASHSEED must be set BEFORE the interpreter starts; setting
# it inside Python is too late.
export PYTHONHASHSEED="${QA_SEED:-42}"

# Refuse the treatment arm's wiring. This branch is the BASELINE: it has no
# quantaalpha/eval protocol and no net-cost runner, so if a shell that last ran
# main still exports these, the baseline would either crash on an import that
# does not exist here or, worse, half-adopt the treatment's evaluation and stop
# being a baseline. Unset rather than trust the caller's shell to be clean.
unset QLIB_FACTOR_RUNNER QLIB_FACTOR_SUMMARIZER QA_PROTOCOL QA_LEDGER \
      QA_PRIMARY_METRIC QA_REQUIRE_FEASIBLE QA_TARGET_ZOO

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

echo "Python: $(python --version)"
echo "QuantaAlpha: $(which quantaalpha)"
echo ""

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"configs/experiment.yaml"}

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
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
DIRECTION="$1"
LIBRARY_SUFFIX="$2"

# Per-run library. Without a suffix the library is the SHARED
# data/factorlib/all_factors_library.json, and FactorLibraryManager._load()
# reads it before appending -- so a second baseline run starts holding the
# first run's factors, reaches the target without mining them, and reports a
# 150-factor result it did not produce. Defaulting the suffix to EXPERIMENT_ID
# makes every run write its own file and start genuinely empty.
#
# Precedence: positional $2 > a pre-set FACTOR_LIBRARY_SUFFIX > EXPERIMENT_ID.
# Exporting FACTOR_LIBRARY_SUFFIX="" deliberately selects the shared file.
if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
elif [ -z "${FACTOR_LIBRARY_SUFFIX+x}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${EXPERIMENT_ID}"
fi

# Factor-count target. max_rounds alone gives an UPPER estimate
# (D + D + C*(R-2) batches x factors_per_hypothesis); every factor whose
# implementation fails to yield a usable signal is dropped with nothing to
# replace it, so a run configured for exactly 150 finishes short. The
# controller extends past max_rounds until the library actually holds this
# many, bounded by QA_MAX_ROUNDS_CAP.
export QA_TARGET_MINED="${QA_TARGET_MINED:-150}"
export QA_MAX_ROUNDS_CAP="${QA_MAX_ROUNDS_CAP:-60}"

echo ""
echo "Starting experiment..."
echo "Config: ${CONFIG_PATH}"
echo "Data: ${QLIB_DATA}"
echo "Results: ${RESULTS_BASE}"
echo "----------------------------------------"

if [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    quantaalpha mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi
