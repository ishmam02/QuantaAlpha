"""The baseline arm must be a BASELINE: unseeded, environment-matched to the
treatment, and actually 150 factors.

Three properties, each of which was violable on this branch:

O1-O3  NO SEED CONTEXT. The LLM must never be handed a pre-existing factor.
       There is no seed pool on this branch, but seeding can arrive three other
       ways: the RAG channel, the factor library being fed back into a prompt,
       or -- the live defect -- a SHARED library file that a second run loads
       and appends to, so it starts holding the previous run's factors.

O4-O8  ALWAYS 150. `max_rounds` sizes an UPPER estimate
       (D + D + C*(R-2) batches x factors_per_hypothesis = exactly 150 for the
       paper shape); every factor whose implementation fails to produce a
       usable signal is dropped and nothing replaces it, so the run reliably
       finishes short. The budget has to follow the outcome, and it has to stay
       bounded -- the phase getters re-enter themselves on every transition and
       terminate ONLY on this rule.

O9-O11 ENVIRONMENT PARITY. A model, seed or evaluation difference between the
       arms is a confound, not a treatment. The treatment's wiring in
       particular must be refused rather than trusted absent: this branch has
       no quantaalpha/eval protocol and no net-cost runner.

Run:  python tests/evolution/test_original_arm_parity.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from quantaalpha.pipeline.evolution.controller import EvolutionController  # noqa: E402

RUN_SH = (ROOT / "run.sh").read_text()

# ---------------------------------------------------------------------------
# O1 -- no seed-pool machinery, and no published-factor set, anywhere.
# ---------------------------------------------------------------------------
assert not (ROOT / "quantaalpha" / "factors" / "seed_pool.py").exists(), \
    "O1: seed_pool.py exists on the baseline branch"
# Alpha158/360 DO live in quantaalpha/backtest/ -- that is the standalone
# `quantaalpha backtest --factor-source alpha158_20` CLI, a published baseline
# to score a factor set AGAINST. That is legitimate and stays. What must hold
# is ISOLATION: the mining loop must never reach it, or the generator would be
# primed with 20 public factors it did not mine.
MINING = ("pipeline", "factors", "components", "core", "llm", "coder")
hits = []
for sub in MINING:
    for f in (ROOT / "quantaalpha" / sub).rglob("*.py"):
        t = f.read_text(errors="ignore")
        if re.search(r"[Aa]lpha1?58|[Aa]lpha360|seed_pool|QA_SEED_POOL", t):
            hits.append(f.relative_to(ROOT))
assert not hits, f"O1: published-factor references on the MINING path: {hits}"

importers = []
for f in (ROOT / "quantaalpha").rglob("*.py"):
    if f.relative_to(ROOT).parts[1] == "backtest":
        continue
    if re.search(r"factor_loader|FactorLoader", f.read_text(errors="ignore")):
        importers.append(f.relative_to(ROOT))
assert not importers, \
    f"O1: the published-factor loader is reachable outside backtest/: {importers}"

paper = (ROOT / "configs" / "experiment_paper.yaml").read_text()
assert "factor_source" not in paper, \
    "O1: the mining config declares a factor_source -- the run would start seeded"
print("O1 PASS  no seed pool; Alpha158 confined to the offline backtest CLI, "
      "unreachable from the mining path and absent from the mining config")

# ---------------------------------------------------------------------------
# O2 -- the RAG channel is the documented way to inject prior knowledge into
# the hypothesis prompt. On the factor path it must be None at every site.
# ---------------------------------------------------------------------------
prop = (ROOT / "quantaalpha" / "factors" / "proposal.py").read_text()
rag_assigns = re.findall(r'"RAG"\s*:\s*([^,\n]+)', prop)
assert rag_assigns, "O2: no RAG assignment found -- prompt shape changed, re-check"
bad = [v.strip() for v in rag_assigns if v.strip() != "None"]
assert not bad, f"O2: factor proposal injects RAG content: {bad}"
print(f"O2 PASS  RAG is None at all {len(rag_assigns)} factor-proposal sites")

# ---------------------------------------------------------------------------
# O3 -- the library is written, never read back into a prompt.
# ---------------------------------------------------------------------------
assert "FactorLibraryManager" not in prop, \
    "O3: proposal.py imports the factor library -- prior factors can reach the prompt"
print("O3 PASS  the factor library never reaches the hypothesis prompt")

# ---------------------------------------------------------------------------
# O4-O8 -- the round budget.
# A stub carries only what the rule reads, so the test needs no LLM, no pool
# and no operators, and cannot pass because some unrelated default changed.
# ---------------------------------------------------------------------------
SUFFIX = "_pytest_parity_probe"
LIB = ROOT / "data" / "factorlib" / f"all_factors_library_{SUFFIX}.json"


def stub(round_idx: int, max_rounds: int = 5):
    s = SimpleNamespace(_current_round=round_idx,
                        config=SimpleNamespace(max_rounds=max_rounds))
    s._library_size = lambda: EvolutionController._library_size(s)
    return s


def exhausted(s) -> bool:
    return EvolutionController._rounds_exhausted(s)


def write_lib(n: int):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    LIB.write_text(json.dumps({"metadata": {}, "factors": {str(i): {} for i in range(n)}}))


try:
    os.environ["FACTOR_LIBRARY_SUFFIX"] = SUFFIX
    os.environ.pop("QA_MAX_ROUNDS_CAP", None)

    # O4 -- below max_rounds it never stops, target or no target.
    os.environ["QA_TARGET_MINED"] = "150"
    write_lib(0)
    assert exhausted(stub(4)) is False, "O4: stopped before max_rounds"
    print("O4 PASS  below max_rounds the run always continues")

    # O5 -- at max_rounds but SHORT of the target, it extends. This is the
    # property max_rounds alone cannot express and the whole point of the fix.
    write_lib(143)
    assert exhausted(stub(5)) is False, \
        "O5: stopped at max_rounds holding 143/150 -- the run finishes short"
    print("O5 PASS  143/150 at max_rounds -> extends instead of finishing short")

    # O6 -- target reached, it stops (and does not overrun forever).
    write_lib(150)
    assert exhausted(stub(5)) is True, "O6: did not stop after reaching the target"
    write_lib(151)
    assert exhausted(stub(5)) is True, "O6: did not stop past the target"
    print("O6 PASS  150/150 -> stops")

    # O7 -- THE RECURSION BOUND. The phase getters re-enter themselves on every
    # transition; if this ever returns False at the cap the run recurses until
    # RecursionError. A previous branch reached round 1116 this way.
    write_lib(10)
    os.environ["QA_MAX_ROUNDS_CAP"] = "12"
    assert exhausted(stub(12)) is True, "O7: ran past the cap -- recursion is unbounded"
    assert exhausted(stub(99)) is True, "O7: ran past the cap"
    assert exhausted(stub(11)) is False, "O7: stopped one round early"
    print("O7 PASS  the cap bounds the extension (and the getters' recursion)")

    # O8 -- with no target the rule is the ORIGINAL one, exactly. A stop rule
    # that quietly changed unconfigured behaviour would silently alter every
    # existing run of this branch.
    os.environ.pop("QA_TARGET_MINED", None)
    os.environ.pop("QA_MAX_ROUNDS_CAP", None)
    write_lib(0)
    for r, mx in ((0, 5), (4, 5), (5, 5), (9, 5), (2, 3), (3, 3)):
        s = stub(r, mx)
        assert exhausted(s) is (r >= mx), \
            f"O8: unconfigured behaviour changed at round {r}/{mx}"
    # a malformed target must not extend the run either
    os.environ["QA_TARGET_MINED"] = "not-an-int"
    assert exhausted(stub(5)) is True, "O8: a malformed target extended the run"
    os.environ["QA_TARGET_MINED"] = "0"
    assert exhausted(stub(5)) is True, "O8: a zero target extended the run"
    print("O8 PASS  no target -> byte-identical to the old max_rounds rule")

    # O8b -- an unreadable library must stop, never extend blindly.
    os.environ["QA_TARGET_MINED"] = "150"
    LIB.write_text("{ this is not json")
    assert exhausted(stub(5)) is True, "O8b: extended on an unreadable library"
    print("O8b PASS  an unreadable library stops rather than extending blindly")
finally:
    os.environ.pop("FACTOR_LIBRARY_SUFFIX", None)
    os.environ.pop("QA_TARGET_MINED", None)
    os.environ.pop("QA_MAX_ROUNDS_CAP", None)
    LIB.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# O9 -- every run gets its OWN library. This is the seeding defect: without a
# suffix the library is the shared all_factors_library.json, which
# FactorLibraryManager._load() reads before appending.
# ---------------------------------------------------------------------------
assert 'FACTOR_LIBRARY_SUFFIX="${EXPERIMENT_ID}"' in RUN_SH, \
    "O9: run.sh does not default the library suffix -- runs share a library file"
print("O9 PASS  run.sh gives each run its own library (no cross-run inheritance)")

# ---------------------------------------------------------------------------
# O10 -- the treatment arm's wiring is refused, not merely absent. This branch
# has no quantaalpha/eval protocol and no net-cost runner.
# ---------------------------------------------------------------------------
assert not (ROOT / "quantaalpha" / "eval" / "protocol.py").exists(), \
    "O10: eval/protocol.py exists -- this is no longer the baseline branch"
unset_block = re.search(r"unset ((?:[A-Z_]+[ \\\n]*)+)", RUN_SH)
assert unset_block, "O10: no unset block in run.sh"
cleared = set(unset_block.group(1).split())
for var in ("QLIB_FACTOR_RUNNER", "QLIB_FACTOR_SUMMARIZER", "QA_PROTOCOL"):
    assert var in cleared, f"O10: run.sh does not unset {var} (a dirty shell contaminates the baseline)"
print("O10 PASS  run.sh refuses the treatment arm's runner/summarizer/protocol")

# ---------------------------------------------------------------------------
# O11 -- determinism + LLM overrides exist, so both arms can be pinned to one
# model and one seed from outside .env.
# ---------------------------------------------------------------------------
for needle, why in (
    ('export PYTHONHASHSEED=', "PYTHONHASHSEED (must be set before interpreter start)"),
    ('export CHAT_MODEL="${QA_CHAT_MODEL}"', "QA_CHAT_MODEL override"),
    ('export CHAT_SEED="${QA_CHAT_SEED}"', "QA_CHAT_SEED override"),
    ('OMP_NUM_THREADS', "thread pinning"),
):
    assert needle in RUN_SH, f"O11: run.sh is missing {why}"
print("O11 PASS  determinism + model/seed overrides match the treatment arm")

print("\nALL PASS")
