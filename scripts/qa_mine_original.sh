#!/bin/bash
# BASELINE arm launcher -- the `original` branch (paper reproduction).
#
# Exists because a bare ./run.sh on this branch defaults CONFIG_PATH to
# configs/experiment.yaml, whose shape is num_directions 2 / max_rounds 3 /
# crossover_n 2 / factors_per_hypothesis 1 -- six factors, not 150. The paper
# shape lives in configs/experiment_paper.yaml and has to be selected
# explicitly, exactly as scripts/qa_mine.sh does for the treatment arm.
#
# A separate file from qa_mine.sh on purpose. qa_mine.sh is untracked, so it
# sits in the working tree whichever branch is checked out, and it exports
# treatment-only knobs (QA_MIN_MARGINAL_ER, QA_MAX_LIBRARY,
# QA_RESTORE_CAP_EVICTED). Nothing reads those here, but "which launcher did
# that run use" should not be a question anyone has to reconstruct later.
#
# Held IDENTICAL to the treatment arm (these are confounds, not treatments):
#   model, chat seed, numeric seed, factor-count target, direction string.
# Differing BY DESIGN: the evolution algorithm, the runner, the admission gate.
#
# Usage:
#   scripts/qa_mine_original.sh
#   screen -dmS qa_orig bash -lc 'scripts/qa_mine_original.sh'
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Refuse to run from the wrong branch: this launcher pins the baseline's config
# shape, and silently applying it to the treatment branch would produce a run
# that is neither arm.
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "${BRANCH}" != "original" ] && [ -z "${QA_ALLOW_ANY_BRANCH:-}" ]; then
    echo "Error: on branch '${BRANCH}', expected 'original'."
    echo "  git checkout original      # or set QA_ALLOW_ANY_BRANCH=1 to override"
    exit 1
fi

export CONFIG_PATH="configs/experiment_paper.yaml"

# --- held identical to scripts/qa_mine.sh (the treatment arm) --------------
export QA_SEED="${QA_SEED:-42}"
export QA_CHAT_SEED="${QA_CHAT_SEED:-42}"
export QA_CHAT_MODEL="${QA_CHAT_MODEL:-glm-5.2:cloud}"
export QA_REASONING_MODEL="${QA_REASONING_MODEL:-glm-5.2:cloud}"
export QA_TARGET_MINED="${QA_TARGET_MINED:-150}"
export QA_MAX_ROUNDS_CAP="${QA_MAX_ROUNDS_CAP:-60}"

echo "arm       : BASELINE (original)"
echo "config    : ${CONFIG_PATH}"
echo "model     : ${QA_CHAT_MODEL}"
echo "target    : ${QA_TARGET_MINED} mined factors (cap ${QA_MAX_ROUNDS_CAP} rounds)"
echo "library   : data/factorlib/all_factors_library_${EXPERIMENT_ID:-<auto>}.json"
echo ""

exec caffeinate -i ./run.sh "cross-sectional equity factors from daily price and volume"
