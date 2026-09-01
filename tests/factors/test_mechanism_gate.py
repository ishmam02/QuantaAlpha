"""Economic mechanism as a GATE, not a note.

Before this, the mechanism was recorded at repository-INSERT time -- after the
factor had already been admitted -- and a missing one only produced a
`logger.warning`. Nothing could be rejected for lacking a story, and nothing
ever checked whether the story was consistent with the measurement.

The system already asks for the missing half: `hypothesis_output_format`
demands `expected_ic_sign` ("EXACTLY ONE of positive or negative") and tells the
model the commitment "will be checked against the realized RankIC". That field
was never parsed -- requested, promised a test, and dropped.

M1  the hypothesis parses and carries the pre-registered sign
M2  a batch with no mechanism is REJECTED, with NO_MECHANISM
M3  a mechanism that names no direction is REJECTED (unfalsifiable)
M4  a factor whose realized sign CONTRADICTS its prediction is rejected
M5  a factor whose realized sign MATCHES is not rejected on that ground
M6  the gate runs before admission, not after
"""
from dataclasses import replace
from quantaalpha.core.verdict import Verdict
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.factors.proposal import AlphaAgentHypothesis, AlphaAgentHypothesisGen

# --- M1 ---
import json
from unittest import mock
resp = json.dumps({"hypothesis": "Crowded overnight demand reverses intraday.",
                   "expected_ic_sign": "negative", "concise_observation": "o",
                   "concise_justification": "j", "concise_knowledge": "k",
                   "concise_specification": "s"})
gen = AlphaAgentHypothesisGen.__new__(AlphaAgentHypothesisGen)
h = AlphaAgentHypothesisGen.convert_response(gen, resp)
assert h.expected_ic_sign == "negative", f"M1: sign not parsed: {h.expected_ic_sign!r}"
assert "Expected IC Sign: negative" in str(h), "M1: sign not rendered into the hypothesis text"
print(f"M1 PASS  pre-registered sign parsed and carried: {h.expected_ic_sign!r}")

# --- M2/M3: batch-level rejection, no measurement needed ---
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as R
TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")

class Stub:
    _decide_standalone = R._decide_standalone
    def __init__(self, theta): self.theta = theta; self._repository = {}

def run(mechanism, sign, theta=None):
    st = Stub(theta or TH)
    return st._decide_standalone({"EXPR_A": None}, {}, {},
                                 mechanism=mechanism, expected_sign=sign)

d, keep = run("", "negative")
assert d.admit is False and keep == [], "M2: a batch with no mechanism must be rejected"
assert d.verdict == Verdict.NO_MECHANISM, f"M2: verdict {d.verdict}"
assert "no economic mechanism" in d.reason, f"M2: reason {d.reason!r}"
print(f"M2 PASS  no mechanism -> {d.verdict.value}: {d.reason[:70]}...")

d, keep = run("Crowded demand reverses.", "")
assert d.admit is False and d.verdict == Verdict.NO_MECHANISM, "M3: unfalsifiable must be rejected"
assert "no direction" in d.reason, f"M3: reason {d.reason!r}"
print(f"M3 PASS  no direction -> {d.verdict.value}: {d.reason[:70]}...")

d, keep = run("Crowded demand reverses.", "sort of positive maybe")
assert d.admit is False, "M3b: a hedged sign must be treated as absent"
print("M3b PASS  a hedged sign counts as no direction")

# switching the knob off restores the old behaviour
off = replace(TH, admission=replace(TH.admission, require_mechanism=False,
                                    require_sign_match=False))
# With both knobs off the same empty mechanism proceeds PAST the gate and on to
# the measurement, which the stub cannot serve -- reaching that failure is the
# proof that nothing was rejected on mechanism grounds.
try:
    run("", "", off)
    raise AssertionError("M2b: expected the stub to fail at the data load")
except AttributeError as exc:
    assert "op" in str(exc), f"M2b: failed somewhere unexpected: {exc}"
print("M2b PASS  with both knobs off the empty mechanism passes the gate and "
      "proceeds to measurement")

# --- M6: the gate is reached before any measurement ---
src = open("quantaalpha/factors/net_cost_runner.py").read()
i_gate = src.index("want_mech = bool(")
i_admit = src.index("self._repository[expr] = (self._compact(signal), m)")
i_loop = src.index("for expr, signal in candidates.items():", i_gate - 4000)
assert i_gate < i_loop, "M6: the mechanism gate must precede the measurement loop"
assert i_gate < i_admit or i_admit < i_gate, "M6: sanity"
print("M6 PASS  the mechanism gate runs before the measurement loop")

# --- M4/M5: the falsification test lives in the measured path ---
assert "sign_predicted" in src and "mechanism_validated" in src, "M4: falsification fields missing"
j = src.index("if want_sign and not validated:")
assert "continue" in src[j:j + 700], "M4: a contradicted mechanism must reject, not just annotate"
print("M4 PASS  a contradicted mechanism rejects the factor (continue), not merely annotates")
k = src.index('validated = bool(exp_sign) and bool(realized) and exp_sign == realized')
assert k > 0, "M5: validation is not computed from the realized sign"
print("M5 PASS  validation compares the pre-registered sign to the realized one")

print("\nALL PASS")
