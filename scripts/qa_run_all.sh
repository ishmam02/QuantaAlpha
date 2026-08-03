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

# Size by PROCESSES, not by instances. An earlier version budgeted ~6 GB per
# instance and picked two on a 16 GB box -- but an instance is not one process.
# With num_directions=10 and parallel evolution each instance forked ~16 Python
# interpreters, every one of them loading pandas, qlib and panel data. Two
# instances meant 32 interpreters, memory was exhausted about 35 minutes in and
# the machine went down with nothing to show for it.
#
# QA_SEQUENTIAL_EVOLUTION (set in run.sh, both arms) now holds an instance to
# roughly PROC_PER_INSTANCE interpreters. Budget ~1.2 GB each, keep 4 GB for the
# OS, and never exceed one instance per 2 cores.
PROC_PER_INSTANCE="${QA_PROC_PER_INSTANCE:-3}"
GB_PER_PROC=2
USABLE_GB=$(( RAM_GB - 4 )); [ "${USABLE_GB}" -lt 2 ] && USABLE_GB=2
BY_RAM=$(( USABLE_GB / (PROC_PER_INSTANCE * GB_PER_PROC) )); [ "${BY_RAM}" -lt 1 ] && BY_RAM=1
BY_CPU=$(( CORES / (PROC_PER_INSTANCE * 2) ));               [ "${BY_CPU}" -lt 1 ] && BY_CPU=1
PARALLEL="${QA_PARALLEL:-$(( BY_RAM < BY_CPU ? BY_RAM : BY_CPU ))}"
[ "${PARALLEL}" -gt "${N_SEEDS}" ] && PARALLEL="${N_SEEDS}"
THREADS=$(( CORES / (PARALLEL * PROC_PER_INSTANCE) )); [ "${THREADS}" -lt 1 ] && THREADS=1

echo "========================================================================"
echo "  FULL EXPERIMENT -- ${N_SEEDS} seed(s), ${PARALLEL} at a time"
echo "------------------------------------------------------------------------"
echo "  host      : ${CORES} cores, ${RAM_GB} GB RAM"
echo "  seeds     : ${SEEDS}"
echo "  parallel  : ${PARALLEL} instance(s) x ~${PROC_PER_INSTANCE} proc = "\
     "$(( PARALLEL * PROC_PER_INSTANCE )) interpreters, ${THREADS} thread(s) each"
echo "  arm B g   : ${QA_ARM_B_CONSTRUCTIONS:-topk_dropout mean_variance}"
echo "========================================================================"

# Hoisted here rather than left to run.sh: the process fan-out check runs in
# THIS shell and must see the same value the mining runs will use. Without it
# pre-flight measured the parallel configuration, failed, and refused to start
# a run that would in fact have been sequential and safe.
export QA_SEQUENTIAL_EVOLUTION="${QA_SEQUENTIAL_EVOLUTION:-true}"

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

    # NOT /tmp: macOS clears it on reboot, so the crash that ends a run also
    # destroys the only record of why. The per-seed logs written under
    # data/results survived; these did not.
    mkdir -p data/results/logs
    LOG="data/results/logs/qa_full_seed_${SEED}.log"
    echo "launching seed ${SEED} -> ${LOG}"
    # Every shared mutable path is made per-instance. The factor-signal cache is
    # keyed by expression MD5 and the flat-fee cache is a read-modify-write of
    # one file: two instances sharing either will corrupt or clobber it.
    QA_SEEDS="${SEED}" \
    QA_INSTANCE="s${SEED}" \
    FACTOR_CACHE_DIR="${SCRIPT_DIR}/data/results/factor_cache_s${SEED}" \
    QA_THREADS="${THREADS}" \
    QA_REUSE_A="${QA_REUSE_A:-}" \
    nohup ./scripts/qa_paper_experiment.sh "${DIRECTION}" > "${LOG}" 2>&1 &
    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
    sleep 5     # distinct STAMP: it has one-second resolution
done

# --- background compactor -------------------------------------------------
# library.py caches each signal exactly as computed: the full 2008-2026 x 5982
# grid, ~228 MB, where every consumer reads the 2427 x 622 evaluation panel --
# 12 MB. Left alone a seed accumulates ~13 GB/hour and fills the disk mid-run,
# which has already happened twice.
#
# Compacting at the source would mean a qlib init and a panel load inside every
# factor write, in BOTH arms -- the control arm otherwise never touches E_Theta.
# A periodic sweep is the cheaper trade: it skips files younger than 60s and
# replaces atomically, so a miner reading a signal sees the old file or the new
# one, never a partial.
(
    sleep "${QA_COMPACT_INTERVAL:-1800}"
    while pgrep -f "qa_paper_experiment.sh" >/dev/null 2>&1; do
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_compact_cache.py --yes \
            >> data/results/logs/compact_auto.log 2>&1
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_prune_cache.py --orphans --yes \
            >> data/results/logs/compact_auto.log 2>&1
        # Workspaces and pickle caches dwarf the signal cache -- a finished arm
        # leaves ~87 GB of them against ~2 GB of compacted signals -- and the
        # compactor never touched them. Reaped only for stamps whose process has
        # exited AND whose every factor resolves in factor_cache.
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_reap_scratch.py --yes \
            >> data/results/logs/compact_auto.log 2>&1
        echo "[$(date '+%F %H:%M')] free: $(df -h . | awk 'NR==2{print $4}')" \
            >> data/results/logs/compact_auto.log
        sleep "${QA_COMPACT_INTERVAL:-1800}"
    done
) &
COMPACTOR_PID=$!
echo "  background compactor: pid ${COMPACTOR_PID}, every ${QA_COMPACT_INTERVAL:-1800}s"

echo ""
echo "${LAUNCHED} instance(s) launched. Waiting for all to finish..."
for pid in "${PIDS[@]}"; do wait "${pid}"; done
kill "${COMPACTOR_PID}" 2>/dev/null || true

echo ""
echo "========================================================================"
echo "  ALL SEEDS COMPLETE"
echo "========================================================================"
ls -1d data/results/paper_* 2>/dev/null | sed 's/^/  /'
echo ""
echo "Per-seed logs: data/results/logs/qa_full_seed_*.log"
echo "Read each run's paired_summary.md; the effect is the PAIRED difference,"
echo "not either arm's own number."
