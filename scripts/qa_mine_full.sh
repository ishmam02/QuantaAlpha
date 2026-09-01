#!/usr/bin/env bash
# Full production mine -- the verification run for the 2026-08-23 fix set.
#
# What this run is testing (all four land together, so the run has to separate
# their effects -- see the post-run analysis):
#   * mutation redesign  : refine / orthogonal / admitted-push
#   * direction reseed   : learning-aware, informed directions
#   * operator diversity : de-primed seeds + the ADMITTED-PUSH prescription fix
#   * crossover Eq.7     : two-best parents + strength diagnosis, no splice
#
# ONE mine at a time. Three concurrent mines exhausted the Ollama session quota
# on 2026-08-23 and starved the reseed of every one of its 3 attempts, which is
# why that run produced no reseed directions at all.
set -euo pipefail
cd "$(dirname "$0")/.."

export EXPERIMENT_ID="${EXPERIMENT_ID:-full_$(date +%Y%m%d_%H%M%S)}"
export CONFIG_PATH="${CONFIG_PATH:-configs/experiment_paper.yaml}"

# Full prompt/response capture -- Phase 3 renders the live prompt HTML from this.
export QA_FULL_LLM_LOG=1
# Per-segment ablation so the refine diagnosis routes on the broken part.
export QA_ABLATION_DIAGNOSIS=1
# Keep the LLM diagnosis + strength diagnosis on (both default on; explicit here
# so the run is self-documenting).
export QA_LLM_DIAGNOSIS=1
export QA_LLM_STRENGTH_DIAGNOSIS=1
# Thread budget. 5 seed workers x QA_THREADS each is the real peak, and there
# are exactly 5 admission test_seeds so the worker count cannot go higher --
# extra workers would idle (net_cost_runner._seed_workers). At QA_THREADS=4 the
# peak was 5x4 = 20 threads on 8 cores: 2.5x oversubscribed during seed bursts
# (the exact thrash run.sh's own comment warns about) and idle between them.
# 5 x 2 = 10 is a 1.25x peak, which keeps the cores busy without thrashing.
export QA_THREADS="${QA_THREADS:-2}"
export QA_SEED_WORKERS="${QA_SEED_WORKERS:-5}"   # = len(test_seeds); more would idle

LOG="data/results/${EXPERIMENT_ID}.log"
mkdir -p data/results

echo "=============================================="
echo " experiment : ${EXPERIMENT_ID}"
echo " config     : ${CONFIG_PATH}"
echo " log        : ${LOG}"
echo " started    : $(date -u +%FT%TZ)"
echo "=============================================="

# APPEND, never truncate. `>` cost the first 8 hours of this run's log on a
# resume: the ledger and pool survived (those are the data), but every rendered
# prompt, every per-stage timing and every gate reason before the restart was
# destroyed -- and those are exactly what the post-run analysis reads.
exec ./run.sh "cross-sectional equity factors from daily price and volume" \
  >> "${LOG}" 2>&1
