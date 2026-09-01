#!/bin/bash
# Resume the stopped production mine EXACTLY where it left off.
#
# Resumption is by EXPERIMENT_ID, not by a flag: run.sh derives the ledger,
# library and trajectory-pool paths from it, and `evolution.fresh_start: false`
# in configs/experiment_paper.yaml makes the controller LOAD that pool instead
# of starting empty. Reusing the id is therefore the whole mechanism -- and
# using a NEW id would silently start from round 0 with an empty pool while
# leaving the old run's files untouched and unread.
#
# Nothing is deleted on start: the library is upserted, never truncated.
#
# Usage:  scripts/qa_resume.sh [EXPERIMENT_ID]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXP="${1:-meanvar_20260828_194432}"
POOL="data/results/trajectory_pool_${EXP}.json"
LIB="data/factorlib/all_factors_library_${EXP}.json"

[ -f "$POOL" ] || { echo "no pool at $POOL -- wrong EXPERIMENT_ID?"; exit 1; }
echo "resuming ${EXP}"
echo "  pool    $(python3 -c "import json;print(len(json.load(open('$POOL'))['trajectories']))" 2>/dev/null) trajectories"
echo "  library $(python3 -c "import json;d=json.load(open('$LIB'));f=d.get('factors',d);print(len(f))" 2>/dev/null) factors"

screen -dmS qa_mine bash -lc \
  "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate quantaalpha && \
   EXPERIMENT_ID=${EXP} $(pwd)/scripts/qa_mine.sh"
sleep 2
screen -ls | grep qa_mine && echo "  -> attach with: screen -r qa_mine"
