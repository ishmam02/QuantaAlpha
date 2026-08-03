#!/bin/bash
# Wait for the running experiment to finish, then re-mine Arm A and Arm B
# (topk_dropout) at the current CHAT_MAX_TOKENS.
#
# Why: seed 42's three arms were mined under two different token budgets --
# mean_variance at 65536, the other two at 16000. That is not a nuisance, it is
# a confound: the SAME mean_variance construction moved 1.9pp of net ARR when
# only the token budget changed (-4.64% -> -2.72% at 66 batches). Any pairing
# across those arms measures the budget as much as the objective, so seed 42
# cannot support a comparison until all three share one setting.
#
#   ./scripts/qa_queue_remine.sh <pid-to-wait-for>

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"
WAIT_PID="${1:?usage: $0 <pid>}"
LOG=data/results/logs/remine.log
mkdir -p data/results/logs

{
  echo "=== queued $(date '+%F %H:%M') -- waiting on pid ${WAIT_PID} ==="
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
  echo "=== predecessor finished $(date '+%F %H:%M'); starting re-mine ==="

  # Reclaim before starting: the mean_variance arm leaves ~50 GB of raw
  # signals, and the re-mine needs room for two more arms.
  PYTHONPATH="${SCRIPT_DIR}" python scripts/qa_compact_cache.py --yes 2>&1 | tail -3
  PYTHONPATH="${SCRIPT_DIR}" python scripts/qa_prune_cache.py --orphans --yes 2>&1 | tail -2
  echo "free: $(df -h . | awk 'NR==2{print $4}')"

  # Retire the 16000-token arms BEFORE the re-mine. Two reasons:
  #
  #  1. qa_make_report groups rows by `__label__`, which names the
  #     CONFIGURATION, not the run. Old and new "Arm A (mined)" rows would be
  #     averaged into one column -- silently mixing two token budgets, which is
  #     the exact confound this re-mine exists to remove.
  #  2. The flat-fee cache is a single file per instance, so rows written by
  #     the confounded comparison would persist into the clean one.
  #
  # Archived rather than deleted: topk@16000 against topk@65536 is a second,
  # independent measurement of the token-budget effect on a different
  # construction, which corroborates the 1.9pp seen on mean_variance.
  ARCHIVE="data/results/archive_16k_tokens"
  mkdir -p "${ARCHIVE}"
  for f in data/factorlib/*_20260731_191752*.json; do
      [ -e "$f" ] && mv "$f" "${ARCHIVE}/" && echo "archived $(basename "$f")"
  done
  mv data/results/ledger_treatment_20260731_191752.jsonl "${ARCHIVE}/" 2>/dev/null
  rm -f data/results/backtest_v2_raw_s42.json data/results/backtest_v2_raw.json
  echo "purged the flat-fee cache so the clean comparison recomputes from scratch"

  # No QA_REUSE_A: Arm A is re-mined too, so all three arms share one budget.
  # QA_ARM_B_CONSTRUCTIONS=topk_dropout mines Arm B under the published
  # construction; mean_variance is already done at this budget.
  QA_SEEDS="42" \
  QA_ARM_B_CONSTRUCTIONS="topk_dropout" \
  ./scripts/qa_run_all.sh "price-volume factor mining"
  echo "=== re-mine finished $(date '+%F %H:%M') ==="
} >> "${LOG}" 2>&1
