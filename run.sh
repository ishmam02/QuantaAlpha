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

# QA_CHAT_SEED overrides CHAT_SEED *after* .env is sourced. Sourcing .env
# assigns CHAT_SEED unconditionally, so an exported value is silently clobbered
# -- which would have left every replication of a paper run on an identical LLM
# seed, varying only the evolution operators' RNG. The 4.05pp of run-to-run
# variance measured here is generation noise, and sampling it means resampling
# the generator, not just the parent-selection shuffle.
if [ -n "${QA_CHAT_SEED:-}" ]; then
    export CHAT_SEED="${QA_CHAT_SEED}"
fi

# Applies to BOTH arms. Two reasons, and the second is what crashed a run.
#
# 1. Feedback can only teach a batch generated after it, so batches produced
#    concurrently inside a round never see each other's verdicts.
# 2. With num_directions=10 and parallel_enabled, ONE arm forks ~16 Python
#    processes, each loading pandas, qlib and panel data. Two instances put 32
#    interpreters on an 8-core/16 GB box and exhausted memory ~35 minutes in.
#    Setting this only for the treatment arm, as it was, left the control arm
#    running fully parallel -- and both instances were mining the control arm
#    when the machine died.
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

# Factor library filename: data/factorlib/all_factors_library[_<suffix>].json
# (loop.py:224-232). The library is upserted, never truncated -- library.py
# loads the existing file, mutates the dict, and rewrites it -- so a shared
# filename ACCUMULATES factors across runs.
#
# Default to a per-run suffix: the two arms must not pool their factors, the
# Task 19 head-to-head needs both libraries side by side, and it leaves an
# existing all_factors_library.json untouched.
#
# Precedence: positional $2 > a pre-set FACTOR_LIBRARY_SUFFIX > EXPERIMENT_ID.
# Exporting FACTOR_LIBRARY_SUFFIX="" selects the bare all_factors_library.json
# and accumulates into it, which is the pre-arm behaviour.
if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
elif [ -z "${FACTOR_LIBRARY_SUFFIX+x}" ]; then
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
