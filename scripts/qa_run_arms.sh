#!/bin/bash
# Run BOTH arms back-to-back under identical conditions, then compare.
#
#   ./scripts/qa_run_arms.sh "price-volume factor mining"
#   CONFIG_PATH=configs/experiment_paper.yaml ./scripts/qa_run_arms.sh "..."
#
# The two arms must differ ONLY in the objective. This script enforces that by
# driving both from one invocation with one config, one seed and one Theta --
# rather than trusting two hand-typed command lines to match.
#
# Arm A (control)   : RankIC objective, stock runner/summarizer, flat-fee Qlib
# Arm B (treatment) : U objective, E_Theta with the full cost model, net-of-cost
#                     feedback to the generator
#
# Everything else -- model, seed, temperature, directions, rounds, factors per
# hypothesis, evaluation windows -- is shared.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

DIRECTION="${1:?usage: $0 \"<research direction>\"}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# run.sh activates conda inside its own subshell, which does nothing for the
# comparison steps we run here. Without this, bare `python` resolves to the base
# interpreter and the post-run comparison dies on `No module named 'qlib'` --
# after both arms have already spent their LLM budget.
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
if ! "${PY}" -c "import qlib" 2>/dev/null; then
    echo "Error: ${PY} cannot import qlib. Activate the quantaalpha environment"
    echo "       (or set CONDA_ENV_NAME) before running -- the comparison step"
    echo "       would otherwise fail only after both arms have finished."
    exit 1
fi
export MLFLOW_ALLOW_FILE_STORE=true

# One protocol hash for both arms. Read it once and fail early if Theta is
# broken, rather than discovering it half way through the first run.
THETA=$(PYTHONPATH="${SCRIPT_DIR}" "${PY}" -c "
from quantaalpha.eval.protocol import load_protocol, default_protocol_path
print(load_protocol(default_protocol_path()).hash)" 2>/dev/null | tail -1)
if [ -z "${THETA}" ]; then
    echo "Error: could not load the evaluation protocol. Aborting before either arm runs."
    exit 1
fi

echo "========================================================================"
echo "  A/B RUN  --  both arms, one config, one seed, one Theta"
echo "------------------------------------------------------------------------"
echo "  direction : ${DIRECTION}"
echo "  config    : ${CONFIG_PATH}"
echo "  theta     : ${THETA}"
echo "  stamp     : ${STAMP}"
grep -E "^CHAT_TEMPERATURE|^CHAT_SEED|^CHAT_MODEL" .env 2>/dev/null | sed 's/^/  /'
echo "========================================================================"

# Refuse to run a comparison that cannot be valid.
TEMP=$(grep -E "^CHAT_TEMPERATURE" .env | cut -d= -f2 | tr -d ' ')
if [ "${TEMP:-1}" != "0.0" ] && [ "${TEMP:-1}" != "0" ]; then
    echo "Error: CHAT_TEMPERATURE=${TEMP}. The arms would diverge through sampling"
    echo "       noise rather than the objective. Set CHAT_TEMPERATURE=0.0 in .env."
    exit 1
fi
if ! grep -qE "^CHAT_SEED" .env 2>/dev/null; then
    echo "Error: CHAT_SEED is not set in .env; the arms would not be reproducible."
    exit 1
fi

# Arm B's admission gate rejects a fraction of what it mines, so at equal
# generation budget it finishes with fewer factors than Arm A -- and factor
# count moves the combiner independently of factor quality. Measured: at 18 vs
# 11 the arms' net-ARR gap read as 4.9x the seed noise and "resolvable"; at
# 18 vs 17 the same libraries gave 1.9x and "not resolvable". Most of the
# apparent deficit was the six missing factors, not their quality.
#
# The budget therefore follows the OUTCOME, not an estimate: Arm A runs first,
# we count what it produced, and Arm B keeps mining until its repository holds
# that many ADMITTED factors (QA_MAX_ROUNDS_CAP bounds the cost). An assumed
# multiplier cannot do this -- the admission rate is unknown before the run and
# drifts as the repository improves, which is the point of an adaptive bar.
#
# This deliberately breaks budget parity: Arm B searches more than Arm A.
# Equalising output rather than input is defensible -- the question is "N
# factors chosen by U vs N chosen by RankIC" -- but any Arm B advantage is
# then partly bought with extra search, and the report must state both budgets.

count_factors () {
    "${PY}" - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(len(d.get("factors", d) or {}))
except Exception:
    print(0)
PY
}
run_arm () {
    local arm="$1" label="$2"
    local id="${arm}_${STAMP}"
    echo ""
    echo "########################################################################"
    echo "#  ${label}   (EXPERIMENT_ID=${id})"
    echo "########################################################################"
    # QA_TARGET_ZOO is passed per arm, never inherited. `export` in
    # setup_treatment persists for the rest of the shell, so once the treatment
    # arm ran first the control arm silently picked up its target -- and since
    # the control arm writes no ledger, replay_repository reported 0 admitted
    # and it extended to the round cap. Observed: Arm A ran 7 rounds against a
    # configured 5 and reached 194 factors on its way to ~450.
    QA_ARM="${arm}" \
    QA_TARGET_ZOO="${_ARM_TARGET:-}" \
    EXPERIMENT_ID="${id}" \
    FACTOR_LIBRARY_SUFFIX="${id}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    ./run.sh "${DIRECTION}" 2>&1 | tee "/tmp/qa_${id}.log"
}

# QA_ARMS selects which arms to mine; QA_REUSE_A/B supply an already-mined
# library for an arm being skipped. Re-running one arm after a code change that
# cannot affect the other is far cheaper than re-mining both -- but note the
# pairing is then across runs, and run-to-run generation variance is large (it
# only cancels for arms mined in the SAME run).
QA_ARMS="${QA_ARMS:-control treatment}"
# QA_ARM_ORDER decides which arm mines first. Treatment leads by default: it is
# the arm carrying whatever change is under test, so a mistake in it surfaces in
# the first hour rather than after the control has spent three. The arms are
# independent -- Arm B's factor target comes from the config, not from Arm A's
# output -- so the order affects only when you learn something, never what.
QA_ARM_ORDER="${QA_ARM_ORDER:-treatment control}"

setup_treatment () {
    # Target comes from the CONFIG, not from Arm A's realised output.
    # Deriving it from Arm A would couple Arm B's budget to Arm A's own
    # run-to-run noise (measured at 4pp on return, and one factor short of
    # the nominal count on two of three runs) and would stop Arm B from
    # running on its own. The config determines the shape of the search, so
    # it determines the count both arms are aiming at.
    if [ -z "${QA_TARGET_ZOO:-}" ]; then
        N_EXP="$(PYTHONPATH="${SCRIPT_DIR}" "${PY}" - "${CONFIG_PATH}" <<'PYCFG' 2>/dev/null
import sys, yaml
from quantaalpha.pipeline.evolution.controller import expected_factor_count
c = yaml.safe_load(open(sys.argv[1])); ev = c.get("evolution", {})
print(expected_factor_count(
c.get("planning", {}).get("num_directions", 2),
ev.get("crossover_n", 2),
ev.get("max_rounds", 3),
c.get("factor", {}).get("factors_per_hypothesis", 1)))
PYCFG
)"
        case "${N_EXP}" in ''|*[!0-9]*) N_EXP=0 ;; esac
        [ "${N_EXP}" -gt 0 ] && _ARM_TARGET="${N_EXP}"
    else
        _ARM_TARGET="${QA_TARGET_ZOO}"
    fi
    if [ -n "${_ARM_TARGET:-}" ]; then
        # Deliberately NOT defaulted here: the controller falls back to
        # max_rounds * 3, which scales with the configured search, whereas a
        # fixed number here would be too tight at paper scale and wastefully
        # loose at test scale. Export only if the caller pinned one.
        [ -n "${QA_MAX_ROUNDS_CAP:-}" ] && export QA_MAX_ROUNDS_CAP
        echo ""
        echo "  !! BUDGET PARITY BROKEN BY DESIGN"
        echo "     Arm B mines until |zoo| >= ${_ARM_TARGET} admitted factors"
        echo "     (the count this config is expected to produce), capped at"
        echo "     ${QA_MAX_ROUNDS_CAP:-max_rounds x3} rounds."
        echo "     Arm B searches more than Arm A -- report both budgets."
    fi
}

for _arm in ${QA_ARM_ORDER}; do
    case " ${QA_ARMS} " in
        *" ${_arm} "*) ;;
        *) continue ;;
    esac
    _ARM_TARGET=""          # cleared per arm: only the treatment arm has a target
    if [ "${_arm}" = "treatment" ]; then
        setup_treatment
        run_arm treatment "ARM B -- treatment (net-of-cost U objective)"
    else
        run_arm control "ARM A -- control (RankIC objective)"
    fi
done

LIB_A="${QA_REUSE_A:-data/factorlib/all_factors_library_control_${STAMP}.json}"
LIB_B="${QA_REUSE_B:-data/factorlib/all_factors_library_treatment_${STAMP}.json}"

# Prefer the zoo subset (the effective-alpha repository) only when it actually
# holds factors. The control arm runs the stock runner, which never sets
# `in_zoo`/`feasible`, so its _zoo.json is empty BY CONSTRUCTION -- not because
# its factors were rejected. Preferring it unconditionally compared zero Arm A
# factors against Arm B's full repository and handed Arm B a meaningless win.
pick_library () {
    local full="$1" zoo="${1%.json}_zoo.json"
    if [ -f "${zoo}" ] && [ "$(count_factors "${zoo}")" -gt 0 ]; then
        echo "${zoo}"
    else
        echo "${full}"
    fi
}
LIB_A="$(pick_library "${LIB_A}")"
LIB_B="$(pick_library "${LIB_B}")"
echo "  Arm A library: ${LIB_A}  ($(count_factors "${LIB_A}") factors)"
echo "  Arm B library: ${LIB_B}  ($(count_factors "${LIB_B}") factors)"

echo ""
echo "========================================================================"
echo "  BOTH ARMS COMPLETE -- comparing under E_Theta (full cost model)"
echo "========================================================================"
PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_compare_arms.py \
    --arm-a "${LIB_A}" \
    --arm-b "${LIB_B}" \
    --out "data/results/arm_comparison_${STAMP}.md"

# The second table. E_Theta charges slippage and impact; the published numbers
# do not, so an arm can only be set against the paper through the flat-fee path.
# Both references (alpha158_20, base features) are scored here too -- an arm that
# does not clear four hand-written expressions has not earned its LLM spend.
echo ""
echo "========================================================================"
echo "  BACKTEST V2 -- flat-fee cost model (paper-comparable)"
echo "========================================================================"
PYTHONPATH="${SCRIPT_DIR}" "${PY}" scripts/qa_backtest_all.py \
    --arm-a "${LIB_A}" \
    --arm-b "${LIB_B}" \
    --seeds "${BT_SEEDS:-42,1,7}" \
    --out "data/results/backtest_v2_comparison_${STAMP}.md"

echo ""
echo "Artefacts:"
echo "  logs      : /tmp/qa_control_${STAMP}.log , /tmp/qa_treatment_${STAMP}.log"
echo "  libraries : ${LIB_A}"
echo "              ${LIB_B}"
echo "  ledger    : data/results/ledger_treatment_${STAMP}.jsonl  (Arm B only)"
echo "  E_Theta   : data/results/arm_comparison_${STAMP}.md"
echo "  flat-fee  : data/results/backtest_v2_comparison_${STAMP}.md"
