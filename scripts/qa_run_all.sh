#!/bin/bash
# Run the whole paper-scale experiment on one machine, as fast as it can take it.
#
#   ./scripts/qa_run_all.sh "price-volume factor mining"
#
# Runs seeds CONCURRENTLY with full isolation, which is the only lever that
# shortens wall-clock here: the mining loop spends most of its time waiting on
# LLM API calls, so a second instance costs far less than a second machine and
# overlaps almost perfectly with the first's idle time.
#
# Sizing is measured against the host, not assumed. On this box (8 cores,
# 16 GB) two instances is the honest ceiling:
#
#   * CPU -- Theta asks LightGBM for 20 threads, already 2.5x oversubscribed on
#     8 cores. QA_THREADS divides the real core count between instances.
#   * RAM -- one instance peaks around 5-6 GB (aligned repository ~2 GB at 150
#     factors, plus the combiner's training matrix and panel data). Three would
#     sit near the 16 GB limit with nothing spare, and an OOM on day three
#     costs more than the day it would have saved.
#
# Override with QA_PARALLEL=N if you know better than the default.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

# Re-exec under caffeinate so a multi-day run is not ended by the machine
# sleeping. nohup only survives the terminal closing; it does nothing about
# system sleep, and the default macOS behaviour on battery is to sleep.
#   -i  no idle sleep      -m  no disk sleep      -s  no system sleep (on AC)
# This does NOT defeat closing the lid: that sleeps the machine regardless
# unless an external display is attached (clamshell) or pmset disablesleep is
# set. The pre-flight below says so explicitly rather than letting you find out
# on day three.
if [ -z "${QA_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export QA_CAFFEINATED=1
    exec caffeinate -ims "$0" "$@"
fi

DIRECTION="${1:?usage: $0 \"<research direction>\"}"
SEEDS="${QA_SEEDS:-42 7 13}"
N_SEEDS=$(echo ${SEEDS} | wc -w | tr -d ' ')

CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)"
RAM_GB="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 17179869184) / 1073741824 ))"
# ~6 GB per instance, and never more instances than seeds or cores/2.
BY_RAM=$(( RAM_GB / 6 )); [ "${BY_RAM}" -lt 1 ] && BY_RAM=1
BY_CPU=$(( CORES / 4 ));  [ "${BY_CPU}" -lt 1 ] && BY_CPU=1
PARALLEL="${QA_PARALLEL:-$(( BY_RAM < BY_CPU ? BY_RAM : BY_CPU ))}"
[ "${PARALLEL}" -gt "${N_SEEDS}" ] && PARALLEL="${N_SEEDS}"
THREADS=$(( CORES / PARALLEL )); [ "${THREADS}" -lt 1 ] && THREADS=1

echo "========================================================================"
echo "  FULL EXPERIMENT -- ${N_SEEDS} seed(s), ${PARALLEL} at a time"
echo "------------------------------------------------------------------------"
echo "  host      : ${CORES} cores, ${RAM_GB} GB RAM"
echo "  seeds     : ${SEEDS}"
echo "  parallel  : ${PARALLEL}  (${THREADS} threads each)"
echo "  arm B g   : ${QA_ARM_B_CONSTRUCTIONS:-topk_dropout mean_variance}"
echo "========================================================================"

# Fail in seconds, once, before any instance starts.
[ -f "${SCRIPT_DIR}/.env" ] && { set -a; . "${SCRIPT_DIR}/.env"; set +a; }
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || true
if ! PYTHONPATH="${SCRIPT_DIR}" python scripts/qa_preflight.py \
        --config "${CONFIG_PATH:-configs/experiment_paper.yaml}"; then
    echo ""
    echo "Pre-flight FAILED -- nothing started."
    exit 1
fi

# --- power: an 8-day run must survive the machine being left alone ---
if command -v pmset >/dev/null 2>&1; then
    ON_AC=$(pmset -g batt 2>/dev/null | grep -c "AC Power")
    LID_SLEEPS=$(pmset -g custom 2>/dev/null | awk '/AC Power/,0' | awk '/ disablesleep/{print $2}')
    echo ""
    echo "POWER"
    if [ "${ON_AC}" -eq 0 ]; then
        echo "  !! ON BATTERY -- macOS sleeps on battery by default and the run will"
        echo "     stop. Plug in before leaving this."
    else
        echo "  on AC power, running under caffeinate (no idle/disk/system sleep)"
    fi
    if [ "${LID_SLEEPS:-0}" != "1" ]; then
        echo "  !! CLOSING THE LID WILL STILL SLEEP THE MACHINE and pause the run."
        echo "     Leave it open, or attach an external display, or run:"
        echo "         sudo pmset -a disablesleep 1     # undo: sudo pmset -a disablesleep 0"
        echo "     The run resumes on wake, but hours of wall-clock are lost."
    fi
fi

mkdir -p data/results
PIDS=()
LAUNCHED=0
for SEED in ${SEEDS}; do
    # Throttle to PARALLEL concurrent instances.
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${PARALLEL}" ]; do sleep 30; done

    LOG="/tmp/qa_full_seed_${SEED}.log"
    echo "launching seed ${SEED} -> ${LOG}"
    # Every shared mutable path is made per-instance. The factor-signal cache is
    # keyed by expression MD5 and the flat-fee cache is a read-modify-write of
    # one file: two instances sharing either will corrupt or clobber it.
    QA_SEEDS="${SEED}" \
    QA_INSTANCE="s${SEED}" \
    FACTOR_CACHE_DIR="${SCRIPT_DIR}/data/results/factor_cache_s${SEED}" \
    QA_THREADS="${THREADS}" \
    nohup ./scripts/qa_paper_experiment.sh "${DIRECTION}" > "${LOG}" 2>&1 &
    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
    sleep 5     # distinct STAMP: it has one-second resolution
done

echo ""
echo "${LAUNCHED} instance(s) launched. Waiting for all to finish..."
for pid in "${PIDS[@]}"; do wait "${pid}"; done

echo ""
echo "========================================================================"
echo "  ALL SEEDS COMPLETE"
echo "========================================================================"
ls -1d data/results/paper_* 2>/dev/null | sed 's/^/  /'
echo ""
echo "Per-seed logs: /tmp/qa_full_seed_*.log"
echo "Read each run's paired_summary.md; the effect is the PAIRED difference,"
echo "not either arm's own number."
