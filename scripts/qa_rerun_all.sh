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
OTHER_SEEDS="${QA_OTHER_SEEDS:-7 13}"
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
echo "  seeds ${OTHER_SEEDS} : Arm A + Arm B"
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

mkdir -p data/results/logs
run_seed () {
    local seed="$1" arms="$2" label="$3"
    for construction in ${CONSTRUCTIONS}; do
        local tag="rerun_seed_${seed}_${construction}"
        local log="data/results/logs/${tag}.log"
        echo ""
        echo "--- seed ${seed} | ${construction} | ${label} -> ${log}"
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
        QA_ARMS="${arms}" QA_REUSE_A="${QA_REUSE_A:-}" \
        FACTOR_CACHE_DIR="${SCRIPT_DIR}/data/results/factor_cache_s${seed}" \
        QA_INSTANCE="s${seed}" \
        ./scripts/qa_run_arms.sh "${DIRECTION}" > "${log}" 2>&1 \
            && echo "    completed" || echo "    FAILED -- continuing"
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
