#!/bin/bash
# Append a snapshot of the treatment arm's admission trajectory to a log.
# Written for cron, which is a hostile environment: no conda, no PYTHONPATH,
# a near-empty PATH, and a shell that will hand the script a literal
# "ledger_treatment_*.jsonl" when nothing matches.
#
#   ./scripts/qa_watch_ledger.sh          # one snapshot, to stdout + log
#   crontab -e  ->  */30 * * * * /full/path/scripts/qa_watch_ledger.sh
#
# Picks the NEWEST ledger rather than globbing: several accumulate across seeds
# and constructions, and qa_analyze_ledger takes exactly one.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

OUT="${ROOT}/data/results/logs/admission_watch.log"
mkdir -p "$(dirname "${OUT}")"

# Absolute interpreter: cron's PATH will not find the conda env.
PY="${QA_PYTHON:-/opt/anaconda3/envs/quantaalpha/bin/python}"
[ -x "${PY}" ] || PY="$(command -v python3 || command -v python)"

LEDGER="$(ls -t "${ROOT}"/data/results/ledger_treatment_*.jsonl 2>/dev/null | head -1)"
{
    echo "===================================================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    if [ -z "${LEDGER}" ]; then
        echo "no treatment ledger yet (Arm B has not recorded an evaluation)"
    else
        echo "ledger: $(basename "${LEDGER}")"
        PYTHONPATH="${ROOT}" "${PY}" "${ROOT}/scripts/qa_analyze_ledger.py" "${LEDGER}" 2>&1 \
            | grep -vE "^\s*$" 
    fi
    # One line on whether the run is even alive -- an unchanging admission
    # table means something very different if the process died an hour ago.
    if pgrep -f "qa_paper_experiment|quantaalpha mine" >/dev/null 2>&1; then
        echo "[run is alive]"
    else
        echo "[!! no mining process found -- the run has stopped]"
    fi
} | tee -a "${OUT}"
