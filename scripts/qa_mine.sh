#!/bin/bash
# Full production mine launcher.
#
# Wraps run.sh fixing the PRODUCTION config + seed, so a bare run.sh (which
# defaults CONFIG_PATH to the small configs/experiment.yaml: num_directions 2,
# max_rounds 3) is never launched by accident when a full 150-factor mine is
# intended.
#
# Survives a laptop lid close and the Claude session exiting when launched
# detached:
#     screen -dmS qa_mine bash -lc "$(pwd)/scripts/qa_mine.sh"   # run from the repo root
#   - screen detaches the mine from the terminal; the session reparents to
#     launchd, so it keeps running even if this Claude session / terminal dies.
#   - caffeinate -i holds an idle-sleep assertion for the mine's lifetime, so
#     the box stays awake while the lid is OPEN and the user is away (a ~10-30h
#     mine would otherwise stall on the ~10min idle-sleep timeout). It does NOT
#     block lid-CLOSE sleep: closing the lid still sleeps the box, the mine
#     pauses, and it resumes cleanly on wake -- it does not die. (To mine
#     overnight with the lid physically closed, use caffeinate -s on AC power
#     or an external-display clamshell; that is not the default because a
#     closed-awake laptop can overheat.)
#
# Reattach / watch:
#     screen -r qa_mine        # attach to the live mine
#     tail -f data/results/ledger_meanvar_*.jsonl
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Production config: num_directions 10, max_rounds 15, crossover_n 10.
export CONFIG_PATH="configs/experiment_paper.yaml"

# Seed. QA_CHAT_SEED overrides CHAT_SEED *after* run.sh sources .env (which
# assigns it unconditionally); QA_SEED drives the Python/NumPy RNG + PYTHONHASHSEED.
export QA_SEED="${QA_SEED:-42}"
export QA_CHAT_SEED="${QA_CHAT_SEED:-42}"

# Target the LIBRARY (everything mined), not the zoo. ~150 mined yields a
# usable standalone-significance-screened library; the zoo settles wherever
# the objective puts it. run.sh defaults this to 150 already -- set it
# explicitly so the intent is unambiguous.
export QA_TARGET_MINED="${QA_TARGET_MINED:-150}"

# Generation + reasoning model: glm-5.2:cloud. Measured head-to-head on the
# same prompt, glm-5.3:cloud is ~5.8x slower per call (28.6s vs 4.9s mean) and
# ~2.6x more tokens, and its in-loop admits were NOT measurably more
# crash-robust than 5.2's (10/11 of 5.3's admitted factors are crash-fragile,
# ic_crash<0). On an LLM-dominant pipeline (~75-85% of per-factor wall is LLM)
# that 5.8x turned a 12-13h mine into a multi-day one, so we switched back to
# 5.2 to resume at production cadence. CHAT_MAX_TOKENS stays at 131072 in .env
# (harmless headroom for 5.2). .env also pins this, but set the override vars
# too so an .env edit cannot silently swap the model mid-mine. To re-try 5.3
# later, launch inline with `QA_CHAT_MODEL=glm-5.3:cloud QA_REASONING_MODEL=glm-5.3:cloud`.
export QA_CHAT_MODEL="${QA_CHAT_MODEL:-glm-5.2:cloud}"
export QA_REASONING_MODEL="${QA_REASONING_MODEL:-glm-5.2:cloud}"

# Marginal effective-rank admission gate. A factor can pass the pairwise
# rho_max gate (no SINGLE held factor duplicates it) yet be correlated with the
# COLLECTIVE book and add few independent directions. This rejects such a
# factor at the margin (the redundancy the pairwise gate misses). The rejection
# is fed back to the generator per-factor -- the marginal_er number + a
# glossary lesson ("redundant at the margin ... collective ... yours to
# determine") -- so the model learns the bar, not just hits it. 0.1 is the
# conservative start (reject factors adding < 0.1 independent directions); set
# QA_MIN_MARGINAL_ER=0 to disable. The threshold is read from the ENV at
# process start, so a running mine cannot pick up a change -- relaunch to
# change it. Default-off via env keeps the frozen protocol hash unchanged.
export QA_MIN_MARGINAL_ER="${QA_MIN_MARGINAL_ER:-0.1}"

# Library cap (zoo size). The admission gates (|delta_t|, monotonicity,
# mechanism+sign, FDR, rho_max, rho_within, marginal_er) are the quality
# filter; this cap only bounds the count and evicts by |t_nw| (a different
# criterion than admission). QA_MAX_LIBRARY overrides admission.max_library
# (default 40); 0 = uncapped (no count eviction). Uncapped is safe: the gates
# self-limit -- rho_max and marginal_er tighten as the zoo fills the direction
# space, so the zoo plateaus where non-redundant strong factors run out, not at
# infinity. Under shrinkage=0.5 net_ir grows with zoo size with diminishing
# returns, so uncapping captures that (small) gain instead of truncating it.
# Read at process start -- relaunch to change. Default 0 (uncapped) via env
# keeps the frozen protocol hash unchanged.
export QA_MAX_LIBRARY="${QA_MAX_LIBRARY:-0}"

# Restore factors the count cap evicted. library_cap evictions removed factors
# that PASSED the gates -- displaced by the count limit, not by any quality
# gate -- so with the cap uncapped they belong back in the zoo.
# replay_repository skips library_cap evictions when this is set, so the runner
# rehydrate + evolution controller + _zoo.json writer all re-include them
# consistently. 1 = restore, unset = byte-identical old behavior.
export QA_RESTORE_CAP_EVICTED="${QA_RESTORE_CAP_EVICTED:-1}"

# Hold an idle-sleep assertion for the mine's lifetime (see header). exec so
# caffeinate replaces this shell and the assertion releases exactly when the
# mine exits.
exec caffeinate -i ./run.sh "cross-sectional equity factors from daily price and volume"