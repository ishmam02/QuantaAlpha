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
# generation budget it ends with fewer factors than Arm A -- and factor count
# moves the combiner independently of factor quality, which confounds the
# comparison. QA_TREATMENT_ROUND_MULTIPLIER scales the treatment arm's evolution
# rounds so both arms finish with a similar ADMITTED count.
#
# This deliberately breaks budget parity: Arm B then searches more than Arm A.
# Equalising output rather than input is a defensible choice -- the question is
# "18 factors chosen by U vs 18 chosen by RankIC" -- but any Arm B advantage is
# then partly bought with extra search, and the report must say so. Setting it
# to 1.0 restores strict budget parity at the cost of unequal counts.
treatment_config () {
    local mult="${QA_TREATMENT_ROUND_MULTIPLIER:-1.0}"
    if [ "${mult}" = "1.0" ] || [ "${mult}" = "1" ]; then
        echo "${CONFIG_PATH}"; return
    fi
    local out="${SCRIPT_DIR}/data/results/experiment_treatment_${STAMP}.yaml"
    mkdir -p "$(dirname "${out}")"
    "${PY}" - "${CONFIG_PATH}" "${out}" "${mult}" <<'PYEOF'
import math, sys, yaml
src, dst, mult = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = yaml.safe_load(open(src))
ev = cfg.setdefault("evolution", {})
base = int(ev.get("max_rounds", 3))
ev["max_rounds"] = max(base, math.ceil(base * mult))
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"treatment rounds: {base} -> {ev['max_rounds']} (x{mult})", file=sys.stderr)
PYEOF
    echo "${out}"
}

run_arm () {
    local arm="$1" label="$2"
    local id="${arm}_${STAMP}"
    local cfg="${CONFIG_PATH}"
    if [ "${arm}" = "treatment" ]; then
        cfg="$(treatment_config)"
        if [ "${cfg}" != "${CONFIG_PATH}" ]; then
            echo ""
            echo "  !! BUDGET PARITY BROKEN: treatment arm uses ${cfg}"
            echo "     (QA_TREATMENT_ROUND_MULTIPLIER=${QA_TREATMENT_ROUND_MULTIPLIER})"
            echo "     Arm B searches more than Arm A; report both budgets."
        fi
    fi
    echo ""
    echo "########################################################################"
    echo "#  ${label}   (EXPERIMENT_ID=${id})"
    echo "########################################################################"
    QA_ARM="${arm}" \
    EXPERIMENT_ID="${id}" \
    FACTOR_LIBRARY_SUFFIX="${id}" \
    CONFIG_PATH="${cfg}" \
    ./run.sh "${DIRECTION}" 2>&1 | tee "/tmp/qa_${id}.log"
}

# QA_ARMS selects which arms to mine; QA_REUSE_A/B supply an already-mined
# library for an arm being skipped. Re-running one arm after a code change that
# cannot affect the other is far cheaper than re-mining both -- but note the
# pairing is then across runs, and run-to-run generation variance is large (it
# only cancels for arms mined in the SAME run).
QA_ARMS="${QA_ARMS:-control treatment}"
case " ${QA_ARMS} " in *" control "*) run_arm control "ARM A -- control (RankIC objective)";; esac
case " ${QA_ARMS} " in *" treatment "*) run_arm treatment "ARM B -- treatment (net-of-cost U objective)";; esac

LIB_A="${QA_REUSE_A:-data/factorlib/all_factors_library_control_${STAMP}.json}"
LIB_B="${QA_REUSE_B:-data/factorlib/all_factors_library_treatment_${STAMP}.json}"

# Prefer the zoo subset (the effective-alpha repository) only when it actually
# holds factors. The control arm runs the stock runner, which never sets
# `in_zoo`/`feasible`, so its _zoo.json is empty BY CONSTRUCTION -- not because
# its factors were rejected. Preferring it unconditionally compared zero Arm A
# factors against Arm B's full repository and handed Arm B a meaningless win.
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
