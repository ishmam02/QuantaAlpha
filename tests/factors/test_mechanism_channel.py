"""The mechanism must actually ARRIVE, not merely be gated on.

Measured 2026-08-21 across every ledger and factor library on disk: the
`mechanism` field is non-empty ZERO times. `convert_response` builds the
experiment from the response text alone and never sees the hypothesis, so
`exp.hypothesis` was unset for every factor this system has ever mined.

While a missing mechanism only logged a warning that was invisible. The moment
it became an admission gate it rejected 100% of candidates -- which is how a
gate can be simultaneously correct and catastrophic: it was measuring a broken
channel, not a quality signal.

C1  the constructor attaches the hypothesis to the experiment it returns
C2  the runner's extraction recovers the mechanism text from that experiment
C3  it recovers the pre-registered sign too
C4  a gate reading a properly-wired experiment does NOT reject for "no mechanism"
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

src = (ROOT / "quantaalpha/factors/proposal.py").read_text()
assert "exp.hypothesis = hypothesis" in src, \
    "C1: the constructor does not attach the hypothesis -- mechanism will always be empty"
i_attach = src.index("exp.hypothesis = hypothesis")
i_return = src.index("return exp", i_attach)
assert i_attach < i_return, "C1: attached after return"
print("C1 PASS  the constructor attaches the hypothesis before returning the experiment")

from quantaalpha.factors.proposal import AlphaAgentHypothesis
class FakeExp:
    pass
exp = FakeExp()
# The mechanism must STATE its direction in words. Since the 2026-08-24 sign-flip
# hardening the gate derives the sign from this text rather than trusting the
# expected_ic_sign field, so a mechanism with no readable direction is rejected -- which
# is what a bare "Crowded overnight demand reverses intraday." used to hit here.
MECH = ("Crowded overnight demand reverses intraday, so a high value predicts lower "
        "forward returns.")
exp.hypothesis = AlphaAgentHypothesis(
    MECH, "o", "j", "k", "s", expected_ic_sign="negative")

# the exact extraction the runner performs
hyp_obj = getattr(exp, "hypothesis", None)
mech = getattr(hyp_obj, "hypothesis", None) or (hyp_obj if isinstance(hyp_obj, str) else None)
sign = str(getattr(hyp_obj, "expected_ic_sign", "") or "").lower()
assert mech and mech.strip(), f"C2: extraction returned {mech!r}"
print(f"C2 PASS  runner extraction recovers: {mech[:44]!r}")
assert sign == "negative", f"C3: sign {sign!r}"
print(f"C3 PASS  runner extraction recovers the pre-registered sign: {sign!r}")

# C4: that mechanism must clear the presence gate
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.core.verdict import Verdict
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as R
TH = load_protocol(str(ROOT / "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml"))
class Stub:
    _decide_standalone = R._decide_standalone
    def __init__(self): self.theta = TH; self._repository = {}
# The sentinel below must sit OUTSIDE the try. Raising an AssertionError inside it is
# caught by its own `except Exception` and re-raised as "rejected a properly-wired
# mechanism", which reports a clean pass-through as a rejection and hides the real state.
_outcome = _err = None
try:
    _outcome = Stub()._decide_standalone(
        {"EXPR": None}, {}, {}, mechanism=mech, expected_sign=sign)
except AttributeError as e:
    _err = e
except Exception as e:                                    # noqa: BLE001
    raise AssertionError(f"C4: rejected a properly-wired mechanism: {e!r}")

if _err is not None:
    # Reached the data load and tripped on the stub's missing collaborator: the gate let
    # the mechanism through, which is the property under test.
    assert "op" in str(_err), f"C4: failed before the data load: {_err}"
else:
    # Returned a verdict instead. That is a pass only if the gate did not reject it for a
    # missing or directionless mechanism -- the exact failure this file exists to catch.
    decision = _outcome[0] if isinstance(_outcome, tuple) else _outcome
    verdict = str(getattr(decision, "verdict", "") or "")
    assert "mechanism" not in verdict.lower(), (
        f"C4: a properly-wired mechanism was rejected as {verdict!r}: "
        f"{getattr(decision, 'reason', '')}")
print("C4 PASS  a properly-wired mechanism clears the presence gate and reaches measurement")

print("\nALL PASS")
