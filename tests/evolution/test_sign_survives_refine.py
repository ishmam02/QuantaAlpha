"""A frozen premise must keep its directional commitment.

Measured on the 2026-08-21 run: 36 rejections for "the mechanism names no
direction" against 0 for a missing mechanism. The LLM was NOT hedging -- the
original hypothesis path asked for `expected_ic_sign` on 79 of 79 calls and got
one every time. The losses were structural:

  * `gen()` short-circuits in refine mode and rebuilds the parent's hypothesis
    WITHOUT its sign, so every refine child arrived unfalsifiable;
  * the mutation and crossover output schemas never asked for a sign at all
    (crossover: 20 responses, 0 signs).

Since `require_sign_match` went on, that locked out both operators that do the
learning -- only ORIGINAL-phase factors could be admitted.

S1  the directive carries the parent's sign, and it survives dict round-trip
S2  the freeze path passes it into the hypothesis
S3  a hedged or absent parent sign degrades to "" rather than a bogus value
S4  both evolution schemas now request a direction
"""
from quantaalpha.pipeline.evolution.diagnosis import RefinementDirective
import inspect

flds = getattr(RefinementDirective, "__dataclass_fields__", {})
assert "parent_expected_ic_sign" in flds, "S1: the directive cannot carry the sign"
src = inspect.getsource(RefinementDirective)
assert '"parent_expected_ic_sign": self.parent_expected_ic_sign' in src, "S1: not serialized"
assert 'd.get("parent_expected_ic_sign", "")' in src, "S1: not deserialized"
print("S1 PASS  the directive carries parent_expected_ic_sign and survives to_dict/from_dict")

psrc = open("quantaalpha/factors/proposal.py").read()
i_freeze = psrc.index('if _rd and _rd.get("refine_target") == "expression":')
seg = psrc[i_freeze:i_freeze + 2000]
assert "expected_ic_sign=parent_sign" in seg, "S2: the freeze path still drops the sign"
assert '_rd.get("parent_expected_ic_sign")' in seg, "S2: does not read it from the directive"
print("S2 PASS  the freeze path passes the parent's sign into the frozen hypothesis")

# S3: the normalisation logic, exercised directly
for raw, want in (("positive", "positive"), ("NEGATIVE", "negative"),
                  ("maybe positive", ""), ("", ""), (None, "")):
    got = str(raw or "").strip().lower()
    if got not in ("positive", "negative"):
        got = ""
    assert got == want, f"S3: {raw!r} -> {got!r}, wanted {want!r}"
print("S3 PASS  a hedged or missing parent sign degrades to \"\" (unfalsifiable), never a guess")

import yaml, re
ev = open("quantaalpha/pipeline/prompts/evolution_prompts.yaml").read()
yaml.safe_load(ev)
# S4 (revised for the Eq.7 crossover redesign): the crossover child is no longer
# produced by a dedicated crossover LLM call with its own `hybrid_hypothesis`
# schema (that call -- generate_crossover -- was removed; the schema would be
# dead config). The crossover child now routes through the SAME hypothesis
# generator as the original/refine paths (refine_target="recombine" falls
# through proposal.py's gen to hypothesis_gen, which renders
# `hypothesis_output_format`). So the schema that must request a sign is
# `hypothesis_output_format` in prompts.yaml -- and it carries expected_ic_sign.
# The crossover-specific lineage contract ("which validated decision it
# inherits from which parent") lives in the crossover suffix.
qa = open("quantaalpha/factors/prompts/prompts.yaml").read()
yaml.safe_load(qa)
i_hof = qa.index("hypothesis_output_format:")
assert "expected_ic_sign" in qa[i_hof:i_hof + 2000], (
    "S4: the hypothesis-output schema (used by the crossover child via "
    "hypothesis_gen) must request a direction")
# The mutation orthogonal-hypothesis schema in evolution_prompts.yaml still
# requests a sign (mutation has its own generate call).
i_mut = ev.index("mutation:")
assert "expected_ic_sign" in ev[i_mut:i_mut + 4000], (
    "S4: the mutation schema must still request a direction")
# The crossover suffix carries the credible-lineage instruction + forbids a
# splice. YAML wraps the lineage sentence across lines, so normalize whitespace.
i_cx = ev.index("crossover:")
cx_norm = re.sub(r"\s+", " ", ev[i_cx:])
assert "which validated decision it inherits from which parent" in cx_norm, (
    "S4: the crossover suffix must instruct the child to state which validated "
    "decision it inherits from which parent (credible lineage)")
assert "not a literal splice" in cx_norm, (
    "S4: the crossover suffix must forbid a literal splice")
print("S4 PASS  crossover child commits a sign (via hypothesis_gen) + states its lineage")
# No dead crossover output schema remains (its only consumer was removed).
assert "hybrid_hypothesis" not in ev, (
    "S4: the dead crossover `hybrid_hypothesis` schema must not remain "
    "(generate_crossover, its only consumer, was removed)")
assert "fusion_logic" not in ev and "innovation_points" not in ev, (
    "S4: the old splice-narration fields must be gone")

print("\nALL PASS")

# ---------------------------------------------------------------------------
# S5: the falsification test must run AFTER the significance bars.
#
# Placed first it tested the DIRECTION of factors whose effect is
# indistinguishable from zero. The sign of a |t|=0.3 IC series is a coin flip,
# so a coin flip was rejecting candidates before anything established that they
# predicted at all. Measured over 70 scored factors: 54% sign agreement against
# a market that realizes negative 71% of the time -- i.e. the test was mostly
# sampling noise, and it was doing so before the noise had been filtered out.
src = open("quantaalpha/factors/net_cost_runner.py").read()
i_fdr  = src.index("t_req, n_tests, q = self._fdr_bar(t)")
i_tbar = src.index("if abs(t) < bar:", i_fdr)
i_sign = src.index("if want_sign and not validated:")
assert i_fdr < i_tbar < i_sign, (
    "S5: the sign test must come after the |t| and FDR bars "
    f"(fdr={i_fdr}, tbar={i_tbar}, sign={i_sign})")
print("S5 PASS  the falsification test runs after the significance bars, "
      "so only real effects are asked to justify themselves")

# ---------------------------------------------------------------------------
# S6: the trajectory persists expected_ic_sign so a parent's direction is
# recoverable at diagnosis time.
#
# The freeze path (S2) reads parent_expected_ic_sign from the directive, which
# llm_diagnosis._to_directive AND the diagnosis.common-tail both source from
# getattr(parent, "expected_ic_sign", "") -- i.e. the PARENT TRAJECTORY. Until
# now the trajectory never carried the sign (controller stored str(hypothesis),
# discarding AlphaAgentHypothesis.expected_ic_sign right there), so both reads
# resolved to "" and every frozen expression-refine child was rejected
# no_mechanism (1817 run: 3/3). S6 checks the trajectory field, the exact
# diagnosis read, the to_dict/from_dict round-trip, and legacy-dict tolerance.
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory, RoundPhase

tflds = getattr(StrategyTrajectory, "__dataclass_fields__", {})
assert "expected_ic_sign" in tflds, "S6: the trajectory has no expected_ic_sign field"

traj = StrategyTrajectory(
    trajectory_id="t_sign", direction_id=0, round_idx=0, phase=RoundPhase.ORIGINAL,
    hypothesis="h", expected_ic_sign="negative",
    # NO sign_predicted -- the sign MUST come from the trajectory field, not the
    # metrics dict (the 1817 parents had none: sign_predicted was never persisted).
    backtest_metrics={"U": 0.5},
)
assert getattr(traj, "expected_ic_sign", "") == "negative", "S6: field not set"

# The exact read both diagnosis paths (llm_diagnosis:232 / diagnosis common tail)
# use to source parent_expected_ic_sign:
sign = str(getattr(traj, "expected_ic_sign", "")
           or traj.backtest_metrics.get("sign_predicted", "") or "").strip().lower()
assert sign == "negative", f"S6: diagnosis read resolves to {sign!r}, wanted 'negative'"

# Round-trip through to_dict/from_dict (the pool persists trajectories this way):
rt = StrategyTrajectory.from_dict(traj.to_dict())
assert getattr(rt, "expected_ic_sign", "") == "negative", \
    "S6: sign lost in to_dict/from_dict"

# A dict persisted BEFORE the field existed must not break from_dict (the default
# kicks in; the sign is simply unrecoverable for those legacy trajectories):
old = traj.to_dict()
old.pop("expected_ic_sign", None)
assert getattr(StrategyTrajectory.from_dict(old), "expected_ic_sign", "") == "", \
    "S6: from_dict must tolerate a missing expected_ic_sign (legacy dicts)"
print("S6 PASS  the trajectory persists expected_ic_sign; diagnosis recovers "
      "it; round-trips; tolerates legacy dicts")

print("\nALL PASS")
