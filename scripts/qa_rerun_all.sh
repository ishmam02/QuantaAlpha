#!/bin/bash
# Re-run the experiment after the evaluator and selection fixes.
#
#   ./scripts/qa_rerun_all.sh "price-volume factor mining"
#
# Shape:
#   seed 42          Arm B only, both constructions (Arm A reused -- SEE BELOW)
#   seeds 7, 13      full: Arm A + Arm B topk_dropout + Arm B mean_variance
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE REUSING ARM A ON SEED 42
#
# Arm A's generation is NOT unchanged. Four things reach the control arm, not
# the one:
#
#   1. factor_ast.py unary-minus fix. Expressions whose root is a negation
#      counted zero nodes and were rejected; they now count. ACCEPTS MORE.
#   2. factor_zoo_path is set. It was null, which made FactorRegulator.alphazoo
#      an empty frame and the duplication gate inert. It now fires against
#      Alpha158(20). REJECTS MORE.
#   3. duplication.threshold 5 -> 8, loosening that newly-live gate.
#   4. evolution.mutation_top_fraction 0.5. Mutation used to breed every
#      trajectory; it now breeds the top half by fitness -- RankIC for the
#      control arm, delta_net_ir for the treatment arm.
#
# All four are generation-side and therefore SHOULD apply to both arms: the A/B
# holds generation fixed and varies only the objective, so a change that
# reached one arm alone would confound it. But it does mean the existing
# 151-factor Arm A was mined under a different generation process, and pairing
# it with a freshly-mined Arm B is a cross-generation comparison, not a clean
# A/B.
#
# QA_REUSE_A is therefore OFF by default and seed 42 re-mines Arm A too. Set
# QA_REUSE_A=<path> only if you want the older library and accept the caveat.
# ---------------------------------------------------------------------------

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${QA_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export QA_CAFFEINATED=1
    exec caffeinate -ims "$0" "$@"
fi

DIRECTION="${1:?usage: $0 \"<research direction>\"}"
PRIMARY_SEED="${QA_PRIMARY_SEED:-42}"
# One seed by default. Generation noise is roughly 9x combiner-seed noise, so
# extra seeds are how a difference becomes resolvable -- but they triple a
# multi-day run, and there is no point paying for resolution before knowing
# whether the effect is there at all. Add them once seed 42 says something:
#   QA_OTHER_SEEDS="7 13" ./scripts/qa_rerun_all.sh "..."
OTHER_SEEDS="${QA_OTHER_SEEDS:-}"
CONSTRUCTIONS="${QA_ARM_B_CONSTRUCTIONS:-topk_dropout mean_variance}"

[ -f "${SCRIPT_DIR}/.env" ] && { set -a; . "${SCRIPT_DIR}/.env"; set +a; }
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || true
PY="$(command -v python)"
[ -z "${PY}" ] && { echo "no python after activating ${CONDA_ENV_NAME:-quantaalpha}"; exit 1; }

# Sequential evolution, and the process fan-out check below must see it.
export QA_SEQUENTIAL_EVOLUTION="${QA_SEQUENTIAL_EVOLUTION:-true}"
CONFIG="${CONFIG_PATH:-configs/experiment_paper.yaml}"

echo "========================================================================"
echo "  RE-RUN after the evaluator + selection fixes"
echo "------------------------------------------------------------------------"
echo "  seed ${PRIMARY_SEED}    : $( [ -n "${QA_REUSE_A:-}" ] && echo "Arm B only (reusing ${QA_REUSE_A})" || echo "Arm A + Arm B" )"
echo "  seeds ${OTHER_SEEDS:-(none -- set QA_OTHER_SEEDS to add)}"
echo "  Arm B g   : ${CONSTRUCTIONS}"
echo "  config    : ${CONFIG}"
echo ""
echo "  What changed since the last run:"
echo "    evaluator  benchmark aligned to the fill rule (net_arr/net_ir were"
echo "               understated in EVERY previously reported figure)"
echo "    selection  mutation parents ranked on fitness (were unranked)"
echo "    fitness    delta_net_ir for Arm B (was U, which drifts with |zoo|)"
echo "    novelty    alphazoo seeded from Alpha158(20) (the gate was inert)"
echo "    parser     unary minus no longer silently rejected"
echo "========================================================================"

if ! PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_preflight.py --config "${CONFIG}"; then
    echo ""
    echo "Pre-flight FAILED -- nothing started."
    exit 1
fi

# Seed the library and the alphazoo before mining, so round 0 starts from the
# Alpha158(20) pool the paper describes rather than from nothing.
echo ""
echo "Seeding the Alpha158(20) pool..."
PYTHONPATH="${SCRIPT_DIR}" "${PY}" - <<'PYSEED'
from quantaalpha.factors.seed_pool import write_alphazoo_csv
print("  alphazoo ->", write_alphazoo_csv("data/factorlib/alpha158_20_seed_pool.csv"))
PYSEED

# The alphazoo above only feeds the novelty gate. Seeding the LIBRARY is the
# other half of "the initial seed factor pool is derived from Alpha158(20)":
# without it the search starts from an empty repository and the seed pool is a
# thing candidates are compared against rather than a thing they build on.
# Applied per seed and to BOTH arms, since it is generation-side.
if [ "${QA_SEED_LIBRARY:-true}" = "true" ]; then
    for seed in ${PRIMARY_SEED} ${OTHER_SEEDS}; do
        PYTHONPATH="${SCRIPT_DIR}" \
        FACTOR_CACHE_DIR="${SCRIPT_DIR}/data/results/factor_cache_s${seed}" \
        "${PY}" scripts/qa_seed_library.py \
            --library "data/factorlib/seed_pool_s${seed}.json" \
            >> data/results/logs/seed_library.log 2>&1 \
            && echo "  seed ${seed} library seeded" \
            || echo "  seed ${seed} library seeding FAILED (see logs/seed_library.log)"
    done
fi

mkdir -p data/results/logs

# --- disk maintenance, for the whole run --------------------------------
# NOT optional. A live arm writes ~14 GB/hour -- workspace result.h5 plus the
# coder.factor.execute memo -- and this disk has filled mid-run twice. The
# sweep compacts cached signals, drops orphans, prunes the running arm's memo
# and reaps finished stamps; each refuses anything still in use.
#
# Gated on THIS script still being alive, not on an arm being mid-run: the
# sweeper starts before the first arm does, so a `pgrep qa_run_arms.sh`
# condition would find nothing and exit on its first evaluation.
QA_PARENT_PID=$$
(
    while kill -0 "${QA_PARENT_PID}" 2>/dev/null; do
        sleep "${QA_COMPACT_INTERVAL:-1800}"
        for tool in qa_compact_cache.py "qa_prune_cache.py --orphans" \
                    qa_reap_scratch.py qa_prune_pickle_cache.py; do
            # shellcheck disable=SC2086
            PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/${tool} --yes \
                >> data/results/logs/compact_auto.log 2>&1
        done
        echo "[$(date '+%F %H:%M')] free: $(df -h . | awk 'NR==2{print $4}')" \
            >> data/results/logs/compact_auto.log
    done
) &
SWEEPER=$!
trap 'kill "${SWEEPER}" 2>/dev/null || true' EXIT
echo "  disk sweeper: pid ${SWEEPER}, every ${QA_COMPACT_INTERVAL:-1800}s"

run_seed () {
    local seed="$1" arms="$2" label="$3"
    # Arm A is mined ONCE per seed and reused for the remaining constructions.
    # The control arm runs the stock Qlib runner and never touches Theta, so its
    # library does not depend on Arm B's portfolio map -- mining it per
    # construction produced two identical libraries at roughly five hours each.
    local reuse_a="${QA_REUSE_A:-}"
    for construction in ${CONSTRUCTIONS}; do
        local this_arms="${arms}"
        if [ -n "${reuse_a}" ]; then
            this_arms="treatment"
        fi
        local tag="rerun_seed_${seed}_${construction}"
        local log="data/results/logs/${tag}.log"
        echo ""
        echo "--- seed ${seed} | ${construction} | arms='${this_arms}' | ${label} -> ${log}"
        local proto="${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300.yaml"
        if [ "${construction}" != "topk_dropout" ]; then
            proto="${SCRIPT_DIR}/data/results/protocol_${construction}_rerun.yaml"
            PYTHONPATH="${SCRIPT_DIR}" "${PY}" - "${construction}" "${proto}" <<'PYPROTO'
import sys, yaml
construction, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("quantaalpha/eval/protocol_csi300.yaml"))
cfg.setdefault("portfolio", {})["construction"] = construction
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"  wrote {dst}")
PYPROTO
        fi
        QA_SEED="${seed}" QA_CHAT_SEED="${seed}" \
        CONFIG_PATH="${CONFIG}" QA_PROTOCOL="${proto}" \
        QA_ARMS="${this_arms}" QA_REUSE_A="${reuse_a}" \
        FACTOR_CACHE_DIR="${SCRIPT_DIR}/data/results/factor_cache_s${seed}" \
        QA_INSTANCE="s${seed}" \
        ./scripts/qa_run_arms.sh "${DIRECTION}" > "${log}" 2>&1 \
            && echo "    completed" || echo "    FAILED -- continuing"

        # Capture Arm A once so the next construction reuses it rather than
        # spending another five hours reproducing the same library.
        if [ -z "${reuse_a}" ]; then
            local stamp
            stamp="$(grep -oE 'stamp     : [0-9_]+' "${log}" | tail -1 | awk '{print $3}')"
            if [ -n "${stamp}" ] && [ -f "data/factorlib/all_factors_library_control_${stamp}.json" ]; then
                reuse_a="data/factorlib/all_factors_library_control_${stamp}.json"
                echo "    Arm A mined -> reusing ${reuse_a} for the remaining constructions"
            fi
        fi
        # The paired summary reads stamps, not libraries: it is the only figure
        # that resolves across GENERATION seeds rather than combiner seeds, and
        # generation noise is ~9x the combiner kind.
        grep -oE 'stamp     : [0-9_]+' "${log}" | tail -1 | awk '{print $3}' \
            >> data/results/stamps.txt 2>/dev/null || true
    done
}

if [ -n "${QA_REUSE_A:-}" ]; then
    run_seed "${PRIMARY_SEED}" "treatment" "Arm B only, Arm A reused"
else
    run_seed "${PRIMARY_SEED}" "control treatment" "full"
fi
for seed in ${OTHER_SEEDS}; do
    run_seed "${seed}" "control treatment" "full"
done

# --- the headline: full cost model, BOTH constructions, EVERY arm -------
# qa_run_arms.sh compares the arms once per construction, so each run produces
# one full-cost table for the construction it was given. Neither run produces
# the grid, and the grid is the thing worth reading: it holds cost model and
# construction as the only moving parts, from one set of fits, so top-k against
# mean-variance is a like-for-like comparison rather than two tables that also
# differ in when they were produced.
echo ""
echo "========================================================================"
echo "  HEADLINE: full cost model x construction, all arms"
echo "========================================================================"
LIB_A="$(ls -t data/factorlib/all_factors_library_control_*.json 2>/dev/null | head -1)"
LIB_B="$(ls -t data/factorlib/all_factors_library_treatment_*_zoo.json 2>/dev/null | head -1)"
if [ -n "${LIB_A}" ] && [ -n "${LIB_B}" ]; then
    for cfgk in baseline seed arm-a arm-b; do
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_cost_construction_matrix.py \
            --arm-a "${LIB_A}" --arm-b "${LIB_B}" \
            --protocol "${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300.yaml" \
            --seeds 42,1,7 --only "${cfgk}" \
            --out data/results/cost_construction_matrix.md \
            >> data/results/logs/cost_matrix.log 2>&1 \
            && echo "  ${cfgk} priced" || echo "  ${cfgk} FAILED"
    done
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_cost_construction_matrix.py \
        --protocol "${SCRIPT_DIR}/quantaalpha/eval/protocol_csi300.yaml" \
        --seeds 42,1,7 --render --out data/results/cost_construction_matrix.md \
        && echo "  -> data/results/cost_construction_matrix.md"
else
    echo "  no finished libraries yet; skipping"
fi

# --- capacity, and the paired difference across generation seeds --------
# Both were only ever wired into the old driver stack. Neither is redundant:
# capacity_sweep answers "at what NAV does this die", which nothing else asks,
# and paired_summary is the only figure that resolves across GENERATION seeds.
# It needs more than one seed to say anything -- with a single seed it reports
# that and stops, which is the honest output rather than a missing file.
echo ""
echo "========================================================================"
echo "  CAPACITY and PAIRED DIFFERENCE"
echo "========================================================================"
LIBS_FOR_CAP="$(ls -t data/factorlib/all_factors_library_treatment_*_zoo.json 2>/dev/null | head -2)"
if [ -n "${LIBS_FOR_CAP}" ]; then
    # shellcheck disable=SC2086
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_capacity_sweep.py \
        ${LIBS_FOR_CAP} --navs "${QA_NAVS:-1e7,1e8,1e9,1e10}" \
        --out data/results/capacity.md \
        && echo "  -> data/results/capacity.md" || echo "  capacity sweep failed"
fi
if [ -s data/results/stamps.txt ]; then
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_paired_summary.py \
        --stamps data/results/stamps.txt --out data/results/paired_summary.md \
        && echo "  -> data/results/paired_summary.md" || echo "  paired summary failed"
fi

# --- reference: the same engine, priced under a flat fee ----------------
# Not a result, a reference point. The in-loop objective and the headline both
# charge the full cost model; this reprices the identical books with kappa1,
# kappa2 and borrow zeroed so the distance between the two is legible as one
# number rather than inferred across tables. It uses OUR engine throughout --
# Qlib's own backtest is a separate question and is deliberately not mixed in
# here, because it cannot charge slippage or impact at all.
echo ""
echo "========================================================================"
echo "  REFERENCE: flat fee, same engine, same books"
echo "========================================================================"
for seed in ${PRIMARY_SEED} ${OTHER_SEEDS}; do
    lib_a="data/factorlib/all_factors_library_control_$(ls -t data/factorlib 2>/dev/null | grep -oE "control_[0-9_]+\.json" | head -1 | sed 's/control_//;s/\.json//')"
    for construction in ${CONSTRUCTIONS}; do
        out="data/results/flatfee_reference_seed_${seed}_${construction}.json"
        PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_flatfee_reference.py \
            --seed "${seed}" --construction "${construction}" --out "${out}" \
            >> data/results/logs/flatfee_reference.log 2>&1 \
            && echo "  seed ${seed} ${construction} -> ${out}" \
            || echo "  seed ${seed} ${construction}: skipped (no library yet)"
    done
done

# --- reference: Qlib's own backtest, the path the paper's numbers come from ---
# A DIFFERENT ENGINE, not this experiment priced differently. It charges a
# commission and price limits and models no slippage, impact or borrow, so it
# cannot score the objective these arms were mined under. It is here because the
# published figures come from it and a comparable number is worth having.
#
# Read it with the caveat this repository measured: on the Arm B library it
# reports RankIC 0.1179 while the library's best individual factor manages
# 0.0505 and none exceeds 0.10, so its headline is not attainable from these
# inputs. Treat it as "what the published path says", not as ground truth.
echo ""
echo "========================================================================"
echo "  REFERENCE: Qlib's own backtest (the published path)"
echo "========================================================================"
for lib in data/factorlib/all_factors_library_*_zoo.json \
           data/factorlib/all_factors_library_control_*.json; do
    [ -f "${lib}" ] || continue
    case "${lib}" in *_zoo.json|*control_*) ;; *) continue ;; esac
    echo "  ${lib}"
    PYTHONPATH="${SCRIPT_DIR}" "${PY}" -m quantaalpha.backtest.run_backtest \
        -c configs/backtest.yaml \
        --factor-source custom \
        --factor-json "${lib}" \
        >> data/results/logs/qlib_reference.log 2>&1 \
        && echo "    done" || echo "    FAILED (see data/results/logs/qlib_reference.log)"
done
echo "  metrics: data/results/backtest_v2_results/*_backtest_metrics.json"

echo ""
echo "========================================================================"
echo "  DONE. Logs: data/results/logs/rerun_seed_*.log"
echo ""
echo "  The measurement that decides whether the search now LEARNS is not the"
echo "  admission rate -- that rose 54%->78% on the last run purely because the"
echo "  U bar softened as the zoo filled. It is whether delta_net_ir acquires a"
echo "  positive slope across batches, currently t=+0.42:"
echo "    python scripts/qa_analyze_ledger.py data/results/ledger_treatment_<stamp>.jsonl"
echo "========================================================================"
