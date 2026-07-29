#!/bin/bash
# Full paper-scale experiment: replicated A/B, comparisons, capacity, report.
#
#   ./scripts/qa_paper_experiment.sh "price-volume factor mining"
#   QA_SEEDS="42 7 13" ./scripts/qa_paper_experiment.sh "..."
#
# To survive a closed terminal (this runs for days):
#   nohup ./scripts/qa_paper_experiment.sh "..." > /tmp/qa_paper.log 2>&1 &
#
# This runs for DAYS. Everything it depends on is checked first, because every
# check in qa_preflight.py corresponds to something that has already cost a
# full run in this project.
#
# What it produces, per generation seed:
#   * both arms mined back-to-back under one config, one Theta, one seed
#   * the E_Theta comparison (full cost model) and the flat-fee comparison
#   * the admission trajectory (does U's feedback teach?)
# and once at the end:
#   * the construction sweep (topk_dropout vs mean_variance)
#   * the capacity curve (net IR vs NAV) for every library
#   * a summary of the PAIRED difference across seeds
#
# Why replication is not optional here: an unchanged control arm moved 4.05pp
# of annualised return between two runs, which is larger than any arm
# difference measured. A single pair cannot separate the objective from that
# noise. The paired difference DOES cancel it -- it reproduced to 0.39pp across
# two runs -- but only for arms mined in the SAME run, which is why each seed
# runs both arms together.

set -uo pipefail   # NOT -e: a failed seed must not end a days-long run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

DIRECTION="${1:?usage: $0 \"<research direction>\"}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_paper.yaml}"
QA_SEEDS="${QA_SEEDS:-42 7 13}"
RUN_ID="paper_$(date +%Y%m%d_%H%M%S)"
OUT="data/results/${RUN_ID}"
mkdir -p "${OUT}"

[ -f "${SCRIPT_DIR}/.env" ] && { set -a; . "${SCRIPT_DIR}/.env"; set +a; }

# QA_CHAT_SEED overrides CHAT_SEED *after* .env is sourced. Sourcing .env
# unconditionally assigns CHAT_SEED, so an exported value is silently clobbered
# -- which would have left every replication of a paper run using an identical
# LLM seed, varying only the evolution operators' RNG. The measured 4.05pp of
# run-to-run variance is generation noise, and sampling it properly means
# resampling the generator, not just the parent-selection shuffle.
if [ -n "${QA_CHAT_SEED:-}" ]; then
    export CHAT_SEED="${QA_CHAT_SEED}"
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || \
    source activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || true
PY="$(command -v python)"
export MLFLOW_ALLOW_FILE_STORE=true

echo "========================================================================"
echo "  PAPER-SCALE EXPERIMENT  ${RUN_ID}"
echo "------------------------------------------------------------------------"
echo "  direction : ${DIRECTION}"
echo "  config    : ${CONFIG_PATH}"
echo "  seeds     : ${QA_SEEDS}"
echo "  output    : ${OUT}"
echo "========================================================================"

# ---- pre-flight: fail in seconds rather than after days ----
if ! PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_preflight.py --config "${CONFIG_PATH}" \
        | tee "${OUT}/preflight.txt"; then
    echo ""
    echo "Pre-flight FAILED. Nothing has been run. Fix the checks above first."
    exit 1
fi

LIBS=()
for SEED in ${QA_SEEDS}; do
    echo ""
    echo "########################################################################"
    echo "#  GENERATION SEED ${SEED}"
    echo "########################################################################"
    # Both arms in ONE invocation: run-to-run variance is common-mode and only
    # cancels in the paired difference when the pair is mined together.
    # One seed failing must not end a multi-day run: record it and carry on,
    # because the remaining seeds are still worth having and the paired summary
    # reports on however many completed.
    # Vary BOTH seeds: QA_SEED reseeds the evolution operators, QA_CHAT_SEED
    # the generator itself. Holding the latter fixed would sample a much
    # narrower slice of the variation that actually dominates this study.
    if QA_SEED="${SEED}" QA_CHAT_SEED="${SEED}" CONFIG_PATH="${CONFIG_PATH}" \
       ./scripts/qa_run_arms.sh "${DIRECTION}" 2>&1 | tee "${OUT}/seed_${SEED}.log"; then
        echo "seed ${SEED}: completed" >> "${OUT}/status.txt"
    else
        echo "seed ${SEED}: FAILED (rc=$?) -- continuing with the remaining seeds" \
            | tee -a "${OUT}/status.txt"
    fi

    STAMP="$(grep -oE 'stamp     : [0-9_]+' "${OUT}/seed_${SEED}.log" | tail -1 | awk '{print $3}')"
    if [ -z "${STAMP}" ]; then
        echo "seed ${SEED}: no stamp found, skipping its analysis" >> "${OUT}/status.txt"
        continue
    fi
    echo "${SEED} ${STAMP}" >> "${OUT}/stamps.txt"

    A="data/factorlib/all_factors_library_control_${STAMP}.json"
    B="data/factorlib/all_factors_library_treatment_${STAMP}_zoo.json"
    [ -f "${B}" ] || B="data/factorlib/all_factors_library_treatment_${STAMP}.json"
    LIBS+=(--library "${A}" --label "ArmA-s${SEED}" --library "${B}" --label "ArmB-s${SEED}")

    # Does U's feedback actually teach? Only answerable per run.
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_analyze_ledger.py \
        "data/results/ledger_treatment_${STAMP}.jsonl" \
        --out "${OUT}/admission_seed_${SEED}.md" || true

    for f in arm_comparison backtest_v2_comparison; do
        [ -f "data/results/${f}_${STAMP}.md" ] && cp "data/results/${f}_${STAMP}.md" "${OUT}/" || true
    done
done

# ---- construction sweep: does g change the conclusion? ----
# Runs before capacity because it answers a prior question: whether the cost
# model can act on the book at all. Under top-k dropout turnover is pinned, so
# a capacity number measured there says more about n_drop than about capacity.
if [ ${#LIBS[@]} -gt 0 ]; then
    echo ""
    echo "========================================================================"
    echo "  CONSTRUCTION SWEEP -- topk_dropout vs mean_variance"
    echo "========================================================================"
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_construction_sweep.py \
        "${LIBS[@]}" --seeds "${QA_SEEDS_COMBINER:-42}" \
        --out "${OUT}/construction.md" || true
fi

# ---- capacity: one curve per library, no new mining ----
if [ ${#LIBS[@]} -gt 0 ]; then
    echo ""
    echo "========================================================================"
    echo "  CAPACITY SWEEP -- net IR versus NAV"
    echo "========================================================================"
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_capacity_sweep.py \
        "${LIBS[@]}" --navs "${QA_NAVS:-1e7,1e8,1e9,1e10}" \
        --out "${OUT}/capacity.md" || true
fi

# ---- the paired difference across seeds ----
echo ""
echo "========================================================================"
echo "  PAIRED SUMMARY ACROSS SEEDS"
echo "========================================================================"
PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_paired_summary.py \
    --stamps "${OUT}/stamps.txt" --out "${OUT}/paired_summary.md" || true

echo ""
echo "Run status:"
[ -f "${OUT}/status.txt" ] && sed 's/^/  /' "${OUT}/status.txt"
echo ""
echo "Artefacts in ${OUT}:"
ls -1 "${OUT}" | sed 's/^/  /'
echo ""
echo "Read paired_summary.md first: a single seed cannot separate the objective"
echo "from generation noise, and the paired difference is what does."
