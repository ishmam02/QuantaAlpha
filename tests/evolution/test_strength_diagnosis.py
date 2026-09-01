"""Strength diagnosis (mutation's diagnosis, inverted) for the Eq. 7 crossover.

``diagnose_strength`` locates what each of the two best parents' measurements
VALIDATED -- the construction sub-pattern that drove the book, the strongest
scored dimension, the strategic repair action validated on the lineage -- so the
crossover LLM can author a NEW child from complementary validated ideas. It is
the inverse of ``diagnose_parent``: it locates strengths, not shortfalls, and
under the same discipline: DIAGNOSE, never PRESCRIBE (state the measured
strength and where; never a remedy; close "how to combine that is yours to
determine") and NO hardcoded market priors.
"""
import json
from dataclasses import dataclass, field

import quantaalpha.pipeline.evolution.strength_diagnosis as sd
from quantaalpha.pipeline.evolution.strength_diagnosis import (
    StrengthDirective, diagnose_strength, strongest_dimensions,
    _build_strength_target, repair_actions_summary, lineage_validated_segments,
    build_segments,
)


@dataclass
class P:
    trajectory_id: str = "A"
    hypothesis: str = "overnight gap reverses next day"
    factors: list = field(default_factory=lambda: [
        {"name": "Overnight_5D",
         "expression": "RANK(TS_SUM(($open - DELAY($close,1)) / DELAY($close,1), 5))"}])
    backtest_metrics: dict = field(default_factory=lambda: {
        "U": 1.2, "rank_ic": -0.045, "turnover_book": 0.03,
        # Real e_* schema (diagnosis._DIMENSION_CATEGORY keys): effectiveness/arr/
        # stability -> signal, turnover -> cost, overfit -> overfit, diversity ->
        # redundancy, decay -> decay.
        "e_effectiveness": 0.90, "e_turnover": 0.30, "e_overfit": 0.80,
        "e_diversity": 0.50, "e_decay": 0.40,
        "rho_max": 0.20, "admitted": True, "verdict": "admitted",
    })
    expected_ic_sign: str = "negative"
    hypothesis_details: dict = field(default_factory=lambda: {
        "reason": "overnight gap reverts intraday",
        "concise_observation": "gap vs prior close",
        "concise_justification": "liquidity provision at the open",
        "concise_knowledge": "amihud illiquidity",
    })
    refine_actions: list = field(default_factory=list)


def _seg(parent, **overrides):
    s = build_segments(parent)[0]
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# --------------------------------------------------------------------------
# D1: strongest_dimensions -- high e_* kept, strongest first; always >=1
# --------------------------------------------------------------------------
m = {"e_effectiveness": 0.90, "e_turnover": 0.30, "e_overfit": 0.80, "U": 1.0}
dims = strongest_dimensions(m)
assert dims[0] == ("effectiveness", 0.90), f"D1: strongest first; got {dims[0]}"
assert ("overfit", 0.80) in dims, "D1: second-strong kept"
assert ("turnover", 0.30) not in [d for d, _ in dims], (
    "D1: a weak dimension below the threshold is dropped when stronger ones exist")
# No e_* at all -> empty
assert strongest_dimensions({"U": 1.0}) == [], "D1: no e_* -> empty"
# All below threshold -> the single strongest still returned (always >=1)
solo = strongest_dimensions({"e_effectiveness": 0.10, "U": 1.0})
assert solo == [("effectiveness", 0.10)], f"D1: always at least the strongest; got {solo}"
print("D1 PASS  strongest_dimensions: high e_* kept, strongest first, always >=1")

# --------------------------------------------------------------------------
# D2: _build_strength_target -- signal names the construction that drove the book
# --------------------------------------------------------------------------
p = P()
seg = _seg(p, combiner_weight=0.62, weight_stability=0.8, rank_ic=-0.045,
           turnover_share=0.03)
tgt = _build_strength_target("effectiveness", 0.90, [seg], p.backtest_metrics)
assert tgt["factor"] == "Overnight_5D", f"D2: signal target names the factor; got {tgt['factor']}"
assert tgt["op"] == "TS_SUM", f"D2: signal target names the temporal op; got {tgt['op']}"
assert "drove the book" in tgt["mechanism_hint"], "D2: signal hint locates the construction"
assert "combiner weight" in tgt["mechanism_hint"], "D2: signal hint carries the combiner credit"
# measurement-only: no remedy token
for bad in [" use ", " try ", "should", "lengthen", "simplify", "blend"]:
    assert bad not in tgt["mechanism_hint"].lower(), f"D2: prescription leaked: {bad!r}"
print("D2 PASS  _build_strength_target(signal): names the driving construction, no remedy")

# --------------------------------------------------------------------------
# D3: _build_strength_target -- cost names the cheapest + most stable component
# --------------------------------------------------------------------------
seg_c = _seg(p, turnover_share=0.012)
tgt_c = _build_strength_target("turnover", 0.30, [seg_c], p.backtest_metrics)
assert tgt_c["factor"] == "Overnight_5D", "D3: cost target names the cheapest factor"
assert "cheapest component" in tgt_c["mechanism_hint"], "D3: cost hint locates the cheapest"
assert "most stable" in tgt_c["mechanism_hint"], "D3: cost hint locates the most stable"
print("D3 PASS  _build_strength_target(cost): names cheapest + most stable")

# --------------------------------------------------------------------------
# D4: _build_strength_target -- overfit names the most parsimonious component
# --------------------------------------------------------------------------
seg_o = build_segments(p)[0]  # depth/n_free_params from the real AST
tgt_o = _build_strength_target("overfit", 0.80, [seg_o], p.backtest_metrics)
assert "parsimonious" in tgt_o["mechanism_hint"], "D4: overfit hint names parsimony"
assert "free parameters" in tgt_o["mechanism_hint"], "D4: overfit hint names the parameter count"
print("D4 PASS  _build_strength_target(overfit): names the parsimonious component")

# --------------------------------------------------------------------------
# D5: repair_actions_summary -- admitted repair surfaced; rejected repair NOT
# --------------------------------------------------------------------------
admitted_anc = P(trajectory_id="AD1")
admitted_anc.backtest_metrics = dict(admitted_anc.backtest_metrics, admitted=True)
admitted_anc.refine_actions = [{
    "verdict": "marginal", "weakness_dimension": "cost",
    "refine_target": "expression", "mechanism_hint": "widened the window",
    "target_subtree_signatures": ["TS_SUM(...,5)"],
}]
rejected_anc = P(trajectory_id="RE1")
rejected_anc.backtest_metrics = {"U": 0.1, "admitted": False, "verdict": "marginal"}
rejected_anc.refine_actions = [{
    "verdict": "marginal", "weakness_dimension": "signal",
    "refine_target": "expression", "mechanism_hint": "flipped the sign",
    "target_subtree_signatures": ["RANK(...)"],
}]
# The parent itself is admitted but has no refine_actions; only the admitted
# ancestor's repair is validated.
repairs = repair_actions_summary(P(trajectory_id="A"), [admitted_anc, rejected_anc])
assert len(repairs) == 1, f"D5: only the admitted repair surfaced; got {len(repairs)}"
assert repairs[0]["weakness_dimension"] == "cost", "D5: the cost repair is the validated one"
assert repairs[0]["admitted_on"] == "AD1", "D5: carries the ancestor it was admitted on"
# A parent with no admitted refine ancestor -> empty
assert repair_actions_summary(P(), [rejected_anc]) == [], (
    "D5: a rejected ancestor's repair is NOT surfaced")
print("D5 PASS  repair_actions_summary: admitted repair surfaced, rejected not")

# --------------------------------------------------------------------------
# D6: lineage_validated_segments -- construction patterns aggregate by recurrence
# --------------------------------------------------------------------------
lineage = lineage_validated_segments(P(trajectory_id="A"), [admitted_anc])
assert lineage["construction_patterns"], "D6: construction patterns aggregated"
# Both admitted (parent + ancestor) share the TS_SUM signature -> recurrence >=2
sigs = {r["signature"]: r for r in lineage["construction_patterns"]}
assert any("TS_SUM" in s for s in sigs), "D6: the TS_SUM construction pattern aggregated"
ts_sum_rec = [r for r in lineage["construction_patterns"] if "TS_SUM" in r["signature"]][0]
assert ts_sum_rec["recurrence"] >= 2, f"D6: recurrence counted across admitted; got {ts_sum_rec['recurrence']}"
assert lineage["repair_actions"], "D6: validated repairs threaded"
print("D6 PASS  lineage_validated_segments: construction + repair + hypothesis aggregated")

# --------------------------------------------------------------------------
# D7: diagnose_strength table fallback -- structure deterministic, prose
# diagnose-never-prescribe, no market prior
# --------------------------------------------------------------------------
import os
os.environ["QA_LLM_STRENGTH_DIAGNOSIS"] = "0"
try:
    d = diagnose_strength(P(), [admitted_anc])
    assert d is not None, "D7: a parent with U must return a directive"
    assert d.refine_target.value == "recombine", "D7: strength routes as recombine"
    assert d.strongest_dimension == "effectiveness", (
    f"D7: strongest is the top e_* (effectiveness); got {d.strongest_dimension}")
    assert d.targets, "D7: located targets present"
    assert d.targets[0]["factor"] == "Overnight_5D", "D7: first target names the factor"
    txt = d.directive_text
    # diagnose-never-prescribe: closes "yours to determine", no remedy token
    assert "yours to determine" in txt.lower(), "D7: must close 'yours to determine'"
    for bad in [" use ", " try ", "should", "lengthen", "simplify", "blend linearly",
                "csi300", "china", "shanghai", "mean-revert"]:
        assert bad not in txt.lower(), f"D7: prescription/market-prior leaked: {bad!r}"
    # no market prior in the system prompt either
    for bad in ["csi300", "china", "shanghai"]:
        assert bad not in sd._SYSTEM_STRENGTH.lower(), f"D7: market prior in system: {bad!r}"
    print("D7 PASS  diagnose_strength table fallback: deterministic, no prescription, no market prior")
finally:
    os.environ["QA_LLM_STRENGTH_DIAGNOSIS"] = "1"

# --------------------------------------------------------------------------
# D8: diagnose_strength LLM path -- prose authored, structure kept; a fabricated
# strongest dimension is NOT accepted (never invent a dimension the metrics lack)
# --------------------------------------------------------------------------
captured = {}


class _FakeBackend:
    def build_messages_and_create_chat_completion(self, *, user_prompt, system_prompt,
                                                  json_mode=False, **kw):
        captured["user"] = user_prompt
        captured["system"] = system_prompt
        return json.dumps({
            "strength": "The premise produced a measurable edge at the open.",
            "strongest": "overfit",  # a real scored dimension -> ACCEPTED
            "directive": ("The measurements locate the strength in the overnight-gap "
                          "construction. rank_ic -0.045 across seeds. How to combine "
                          "that is yours to determine."),
        })


# Patch the import site (diagnose_strength imports APIBackend lazily).
import quantaalpha.llm.client as llm_client
llm_client.APIBackend = _FakeBackend
try:
    d = diagnose_strength(P(), [admitted_anc])
    assert "yours to determine" in d.directive_text, "D8: LLM directive applied"
    assert d.mechanism_hint == "The premise produced a measurable edge at the open.", (
        "D8: LLM strength applied to mechanism_hint")
    assert d.strongest_dimension == "overfit", (
        f"D8: a real scored dimension is accepted; got {d.strongest_dimension}")
    # The structured targets are STILL the deterministic table (hybrid: prose only)
    assert d.targets[0]["factor"] == "Overnight_5D", (
        "D8: deterministic structure preserved under the LLM prose")
    print("D8 PASS  diagnose_strength LLM path: prose authored, structure kept")
finally:
    from quantaalpha.llm.client import APIBackend as _Real
    llm_client.APIBackend = _Real

# --------------------------------------------------------------------------
# D9: a fabricated strongest dimension is rejected (never invent a dimension)
# --------------------------------------------------------------------------
class _FakeBackend2:
    def build_messages_and_create_chat_completion(self, *, user_prompt, system_prompt,
                                                  json_mode=False, **kw):
        return json.dumps({
            "strength": "edge", "strongest": "momentum",  # not a scored e_* dim
            "directive": "How to combine that is yours to determine.",
        })


llm_client.APIBackend = _FakeBackend2
try:
    d = diagnose_strength(P(), [admitted_anc])
    assert d.strongest_dimension == "effectiveness", (
        f"D9: a fabricated 'momentum' must not override the real strongest; "
        f"got {d.strongest_dimension}")
    print("D9 PASS  fabricated strongest dimension rejected")
finally:
    from quantaalpha.llm.client import APIBackend as _Real
    llm_client.APIBackend = _Real

# --------------------------------------------------------------------------
# D10: no objective vector -> None (nothing to diagnose)
# --------------------------------------------------------------------------
no_u = P()
no_u.backtest_metrics = {"rank_ic": 0.01}  # no U
assert diagnose_strength(no_u) is None, "D10: no U -> None"
print("D10 PASS  no objective vector -> None")

# --------------------------------------------------------------------------
# D11: the LLM prompt carries the measurement + lineage, no market prior
# --------------------------------------------------------------------------
assert "U" in captured["user"] or "objective" in captured["user"].lower(), (
    "D11: the prompt carries the measurement")
assert "admitted" in captured["user"].lower() or "lineage" in captured["user"].lower(), (
    "D11: the prompt carries the lineage")
for bad in ["csi300", "china", "shanghai", "mean-revert", "reversal prior"]:
    assert bad not in captured["user"].lower(), f"D11: market prior in prompt: {bad!r}"
    assert bad not in captured["system"].lower(), f"D11: market prior in system: {bad!r}"
print("D11 PASS  prompt carries measurement + lineage, no market prior")

# --------------------------------------------------------------------------
# D12: recurrence counts DISTINCT ADMITTED TRAJECTORIES, not occurrences, and a
# solo parent never fabricates a lineage claim.
#
# The parent's expression contains DELAY($close,1) TWICE. Counting occurrences
# made a parent with NO ancestors report "recurred in 2 admitted ancestors" --
# a validation the measurement does not have. A lineage claim needs the pattern
# in >=2 admitted trajectories.
# --------------------------------------------------------------------------
os.environ["QA_LLM_STRENGTH_DIAGNOSIS"] = "0"
try:
    solo = P(trajectory_id="SOLO")
    lin_solo = lineage_validated_segments(solo, [])
    for rec in lin_solo["construction_patterns"]:
        assert rec["recurrence"] == 1, (
            f"D12: a solo parent must count 1 trajectory, not occurrences; "
            f"{rec['signature'][:40]} -> {rec['recurrence']}")
    d_solo = diagnose_strength(solo, [])
    assert "consistently contributed" not in d_solo.directive_text, (
        "D12: a solo parent must NOT claim a construction consistently contributed")

    # Two admitted trajectories sharing the pattern -> a real lineage claim.
    twin = P(trajectory_id="TWIN")
    lin_two = lineage_validated_segments(solo, [twin])
    shared = [r for r in lin_two["construction_patterns"] if "TS_SUM" in r["signature"]][0]
    assert shared["recurrence"] == 2, (
        f"D12: two admitted trajectories -> recurrence 2; got {shared['recurrence']}")
    d_two = diagnose_strength(solo, [twin])
    assert "recurs in 2 admitted trajectories" in d_two.directive_text, (
        "D12: a genuine cross-trajectory recurrence must be reported")
    print("D12 PASS  recurrence counts trajectories; solo parent claims no lineage")
finally:
    os.environ["QA_LLM_STRENGTH_DIAGNOSIS"] = "1"

# --------------------------------------------------------------------------
# D13: a CROSSOVER child is not a "validated repair action".
#
# A crossover task carries a refine_directive (refine_target="recombine",
# verdict="crossover"), so _create_trajectory writes a refine_actions entry for
# it exactly as it does for a refine child. But a recombination repaired no
# diagnosed weakness (weakness_dimension is None) -- surfacing it as "a
# strategic repair action validated in a previously successful trajectory"
# would credit the lineage with a fix that never happened.
# --------------------------------------------------------------------------
cx_child = P(trajectory_id="CX1")
cx_child.refine_actions = [{
    "round_idx": 1, "verdict": "crossover", "weakness_dimension": None,
    "refine_target": "recombine", "mechanism_hint": "",
    "target_subtree_signatures": [],
}]
reps = repair_actions_summary(P(trajectory_id="K"), [cx_child, admitted_anc])
assert all(r["refine_target"] != "recombine" for r in reps), (
    "D13: a crossover child must NOT be reported as a validated repair")
assert len(reps) == 1 and reps[0]["weakness_dimension"] == "cost", (
    f"D13: the genuine refine repair must still surface; got {reps}")
print("D13 PASS  a crossover child is not counted as a validated repair action")

print("\nALL PASS")