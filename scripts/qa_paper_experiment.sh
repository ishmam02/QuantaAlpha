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
# Arm B is mined once per construction. g is not a reporting choice for the
# treatment arm: its in-loop evaluation builds the book through g, so under
# topk_dropout -- where turnover is pinned at n_drop/topk -- the capacity-aware
# objective is optimising against a book it cannot change the trading of. That
# variant isolates the OBJECTIVE (only the objective differs from Arm A);
# mean_variance is the treatment the formulation actually proposes. Both pair
# against the SAME Arm A from the same seed, so within-run pairing holds.
QA_ARM_B_CONSTRUCTIONS="${QA_ARM_B_CONSTRUCTIONS:-topk_dropout mean_variance}"
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
echo "  Arm B g   : ${QA_ARM_B_CONSTRUCTIONS}"
N_SEEDS=$(echo ${QA_SEEDS} | wc -w | tr -d ' ')
N_CONS=$(echo ${QA_ARM_B_CONSTRUCTIONS} | wc -w | tr -d ' ')
echo "  mines     : $((N_SEEDS)) Arm A + $((N_SEEDS * N_CONS)) Arm B = $((N_SEEDS + N_SEEDS * N_CONS)) total"
echo "  output    : ${OUT}"
echo "========================================================================"

# Hoisted here, not left to run.sh, for two reasons: the pre-flight check for
# process fan-out runs in THIS shell and must see the same value the mining
# runs will use, and every child inherits it so both arms are covered.
export QA_SEQUENTIAL_EVOLUTION="${QA_SEQUENTIAL_EVOLUTION:-true}"

# ---- pre-flight: fail in seconds rather than after days ----
if ! PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_preflight.py --config "${CONFIG_PATH}" \
        | tee "${OUT}/preflight.txt"; then
    echo ""
    echo "Pre-flight FAILED. Nothing has been run. Fix the checks above first."
    exit 1
fi

# Emit a protocol with `construction` overridden. Theta remains frozen and
# hashed for each run -- the variants are different protocols, not a mutated
# one, so a comparison that accidentally mixed them is detectable by hash.
protocol_for () {
    local construction="$1"
    local base="${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300.yaml"
    [ "${construction}" = "topk_dropout" ] && { echo "${base}"; return; }
    local out="${SCRIPT_DIR}/${OUT}/protocol_${construction}.yaml"
    "${PY}" - "${base}" "${out}" "${construction}" <<'PYPROTO'
import sys, yaml
src, dst, construction = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src))
cfg.setdefault("portfolio", {})["construction"] = construction
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PYPROTO
    echo "${out}"
}

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
    # QA_REUSE_A lets an already-mined control library stand in, so a change
    # that only affects the treatment arm does not pay to re-mine a control it
    # cannot have altered. Arm A's pipeline never reads Theta's scoring, so
    # reusing it is exact, not an approximation -- but the pairing is then
    # across runs for that seed, and run-to-run generation variance only
    # cancels for arms mined together. Say so in the write-up.
    ARM_A_LIB="${QA_REUSE_A:-}"
    if [ -n "${ARM_A_LIB}" ]; then
        if [ -f "${ARM_A_LIB}" ]; then
            echo "  reusing Arm A: ${ARM_A_LIB} ($(count_factors "${ARM_A_LIB}") factors)"
        else
            echo "  !! QA_REUSE_A=${ARM_A_LIB} not found; will mine Arm A instead"
            ARM_A_LIB=""
        fi
    fi
    for CONSTRUCTION in ${QA_ARM_B_CONSTRUCTIONS}; do
        PROTO="$(protocol_for "${CONSTRUCTION}")"
        TAG="seed_${SEED}_${CONSTRUCTION}"
        echo ""
        echo "--- Arm B construction: ${CONSTRUCTION} ---"

        # First construction mines Arm A too; later ones reuse it, so every
        # Arm B variant is paired against the SAME control from this seed.
        if [ -z "${ARM_A_LIB}" ]; then
            ARMS="control treatment"; REUSE=""
        else
            ARMS="treatment"; REUSE="${ARM_A_LIB}"
        fi

        if QA_SEED="${SEED}" QA_CHAT_SEED="${SEED}" CONFIG_PATH="${CONFIG_PATH}" \
           QA_PROTOCOL="${PROTO}" QA_ARMS="${ARMS}" QA_REUSE_A="${REUSE}" \
           ./scripts/qa_run_arms.sh "${DIRECTION}" 2>&1 | tee "${OUT}/${TAG}.log"; then
            echo "${TAG}: completed" >> "${OUT}/status.txt"
        else
            echo "${TAG}: FAILED -- continuing" | tee -a "${OUT}/status.txt"
        fi

        STAMP="$(grep -oE 'stamp     : [0-9_]+' "${OUT}/${TAG}.log" | tail -1 | awk '{print $3}')"
        if [ -z "${STAMP}" ]; then
            echo "${TAG}: no stamp, skipping analysis" >> "${OUT}/status.txt"
            continue
        fi

        A="${REUSE:-data/factorlib/all_factors_library_control_${STAMP}.json}"
        [ -z "${ARM_A_LIB}" ] && [ -f "${A}" ] && ARM_A_LIB="${A}"
        B="data/factorlib/all_factors_library_treatment_${STAMP}_zoo.json"
        [ -f "${B}" ] || B="data/factorlib/all_factors_library_treatment_${STAMP}.json"

        # Only the construction-matched variant is a clean objective-only A/B,
        # so that is the one the paired summary reads.
        if [ "${CONSTRUCTION}" = "topk_dropout" ]; then
            echo "${SEED} ${STAMP}" >> "${OUT}/stamps.txt"
            LIBS+=(--library "${A}" --label "ArmA-s${SEED}")
        fi
        LIBS+=(--library "${B}" --label "ArmB-${CONSTRUCTION}-s${SEED}")

        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_analyze_ledger.py \
            "data/results/ledger_treatment_${STAMP}.jsonl" \
            --out "${OUT}/admission_${TAG}.md" || true

        # Reclaim orphaned signals as we go. Each mined factor caches ~228 MB
        # at full 2008-2026 x 5982-instrument resolution, so a seed costs ~100 GB
        # and three exhaust a 1 TB disk mid-run. Orphans -- pickles no library
        # references -- are always safe; a seed's whole cache is only dropped
        # after its comparisons have written their results.
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_prune_cache.py --orphans --yes || true

        for f in arm_comparison backtest_v2_comparison; do
            [ -f "data/results/${f}_${STAMP}.md" ] && \
                cp "data/results/${f}_${STAMP}.md" "${OUT}/${f}_${TAG}.md" || true
        done
    done
done

# ---- reclaim the finished seeds' signal caches ----
# Safe only here: the comparisons above have written their markdown and JSON,
# so the signals are needed solely to RE-score, and every factor can be
# recomputed from the expression its library stores. Set QA_KEEP_CACHE=1 to
# retain them for re-scoring at a different Theta.
if [ -z "${QA_KEEP_CACHE:-}" ]; then
    for SEED in ${SEEDS}; do
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_prune_cache.py \
            --seed "${SEED}" --yes || true
    done
fi

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
