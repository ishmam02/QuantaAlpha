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
# ARM SELECTION (QA_ARM)
#   The default is the TREATMENT arm: the net-of-cost, capacity-aware objective
#   of problem_formulation.tex section 3, evaluated by E_Theta.
#
#     ./run.sh "price-volume factor mining"                  # treatment (default)
#     QA_ARM=control ./run.sh "price-volume factor mining"   # published baseline
#
#   Treatment swaps only the runner and the summarizer, through the existing
#   plugin env vars; the generation process (loop, DSL, CoSTEER, mutation and
#   crossover prompt bodies) is identical in both arms. For a valid A/B the two
#   arms must share the model, the seed, the config and the scale -- only the
#   objective may differ.

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
# Arm selection
# -----------------------------------------------------------------------------
# Treatment is the default: this is the net-of-cost objective branch, and the
# baseline is reachable with QA_ARM=control. The banner below prints the active
# arm unconditionally so a run can never be misattributed after the fact.
case "$(echo "${QA_ARM:-treatment}" | tr '[:upper:]' '[:lower:]')" in
    treatment|b|new|netcost|net_cost) QA_ARM_RESOLVED="treatment" ;;
    control|a|baseline|rankic)        QA_ARM_RESOLVED="control" ;;
    *)
        echo "Error: unknown QA_ARM='${QA_ARM}' (expected 'treatment' or 'control')"
        exit 1
        ;;
esac

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="${QA_ARM_RESOLVED}_$(date +%Y%m%d_%H%M%S)"
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

if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
else
    # Keep the two arms' factor libraries apart by default; the head-to-head in
    # scripts/qa_compare_arms.py needs both, and a shared filename loses one.
    export FACTOR_LIBRARY_SUFFIX="${EXPERIMENT_ID}"
fi

# =============================================================================
# Wire the objective
# =============================================================================
if [ "${QA_ARM_RESOLVED}" = "treatment" ]; then
    # NOTE: QLIB_FACTOR_ is a shared env prefix across all three settings classes
    # (settings.py:50/63/76), so this swaps the runner everywhere at once. That
    # is what we want here.
    export QLIB_FACTOR_RUNNER="quantaalpha.factors.net_cost_runner.NetCostFactorRunner"
    export QLIB_FACTOR_SUMMARIZER="quantaalpha.factors.net_cost_feedback.NetCostFactorFeedback"
    export QA_PRIMARY_METRIC="${QA_PRIMARY_METRIC:-U}"
    export QA_REQUIRE_FEASIBLE="${QA_REQUIRE_FEASIBLE:-true}"
    # Absolute paths: the loop runs factor code inside per-factor workspaces, and
    # a relative protocol/ledger path would resolve against whatever the cwd
    # happens to be at that moment.
    export QA_PROTOCOL="${QA_PROTOCOL:-${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300.yaml}"
    export QA_LEDGER="${QA_LEDGER:-${SCRIPT_DIR}/${RESULTS_BASE#./}/ledger_${EXPERIMENT_ID}.jsonl}"
    mkdir -p "$(dirname "${QA_LEDGER}")"
fi

# -----------------------------------------------------------------------------
# Preflight: fail before any LLM spend, not five minutes into the run
# -----------------------------------------------------------------------------
if [ "${QA_ARM_RESOLVED}" = "treatment" ]; then
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
        echo "Error: treatment-arm preflight failed."
        echo "  Re-run this to see the error:"
        echo "    python -c \"from quantaalpha.eval.protocol import load_protocol; load_protocol('${QA_PROTOCOL}')\""
        exit 1
    fi
fi

echo ""
echo "========================================================================"
if [ "${QA_ARM_RESOLVED}" = "treatment" ]; then
    echo "  ARM: TREATMENT  --  net-of-cost, capacity-aware objective (E_Theta)"
    echo "------------------------------------------------------------------------"
    echo "  objective    : ${QA_PRIMARY_METRIC}   (feasibility enforced: ${QA_REQUIRE_FEASIBLE})"
    echo "  protocol     : ${QA_PROTOCOL}"
    echo "  theta hash   : ${THETA_HASH}"
    echo "  runner       : ${QLIB_FACTOR_RUNNER##*.}"
    echo "  summarizer   : ${QLIB_FACTOR_SUMMARIZER##*.}"
    echo "  ledger       : ${QA_LEDGER}"
else
    echo "  ARM: CONTROL  --  published RankIC objective (unmodified pipeline)"
    echo "------------------------------------------------------------------------"
    echo "  objective    : RankIC   (stock runner/summarizer, no feasibility gate)"
fi
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
