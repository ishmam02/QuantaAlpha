#!/usr/bin/env bash
# Poll the LLM endpoint until the session quota resets, then launch the full
# mine exactly once. Emits one line per poll so a Monitor can report progress.
#
# Why poll instead of sleeping a fixed 30 minutes: the reset is approximate, and
# launching into a still-throttled endpoint is what produced the 2026-08-23 run
# where the reseed failed all 3 attempts on 429s and the mine stalled.
set -uo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a
PY=/opt/anaconda3/envs/quantaalpha/bin/python3.10

probe() {
  PYTHONPATH=. timeout 150 "$PY" - <<'PYEOF' 2>/dev/null
from quantaalpha.llm.client import APIBackend
try:
    APIBackend().build_messages_and_create_chat_completion(
        user_prompt="Reply with exactly: OK", system_prompt="You reply tersely.")
    print("READY")
except Exception:
    print("BLOCKED")
PYEOF
}

MAX_WAIT_S=$((150 * 60))   # give up after 2.5h rather than poll forever
START=$(date +%s)
n=0
while true; do
  n=$((n + 1))
  out="$(probe | tail -1)"
  now=$(date +%s); elapsed=$(( (now - START) / 60 ))
  if [ "$out" = "READY" ]; then
    echo "QUOTA READY after ${elapsed}m (${n} probes) -- launching full mine"
    break
  fi
  if [ $((now - START)) -ge $MAX_WAIT_S ]; then
    echo "GAVE UP: quota still blocked after ${elapsed}m (${n} probes); NOT launching"
    exit 1
  fi
  echo "poll ${n}: still rate-limited (${elapsed}m elapsed)"
  sleep 180
done

export EXPERIMENT_ID="full_$(date +%Y%m%d_%H%M%S)"
echo "EXPERIMENT_ID=${EXPERIMENT_ID}"
nohup ./scripts/qa_mine_full.sh > "data/results/${EXPERIMENT_ID}.launch.log" 2>&1 &
echo "LAUNCHED pid=$! log=data/results/${EXPERIMENT_ID}.log"
