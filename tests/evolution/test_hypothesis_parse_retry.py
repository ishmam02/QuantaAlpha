"""The hypothesis-revise path must not die on a malformed LLM response.

On the 2026-08-23 push smoke, one REFINE hypothesis-revise call (target=
hypothesis) returned a full Python function instead of a JSON object.
``robust_json_parse`` repaired nothing (no JSON object present), raised
``JSONDecodeError``, and ``gen()`` only retried on input-length errors -- so
the parse error propagated, the task died, and the round lost its child. The
non-blocking run continued, but the child was gone.

The fix has two pieces, each guarded here:

  R1  ``_gen_with_parse_retry`` re-prompts a bounded number of times on
      unparseable output; a later attempt that returns valid JSON succeeds.
  R2  after ``MAX_PARSE_RETRIES`` it re-raises the last parse error (so the
      caller can degrade safely rather than loop forever).
  R3  ``gen()`` for the hypothesis-revise path degrades to the FROZEN parent
      premise (expression-refine semantics) when the LLM is persistently
      unparseable -- the child is scored instead of lost.
  R4  ``gen()`` for a FRESH generation (no parent to fall back to) re-raises;
      a fresh-gen failure is a real error, not something to paper over.
  R5  ``_refine_hypothesis_fallback`` returns None for every non-revise path
      (expression-refine, sign-flip, fresh gen) so the caller re-raises.
"""
import json
from types import SimpleNamespace

from quantaalpha.core.proposal import Trace
import quantaalpha.factors.proposal as _proposal
from quantaalpha.factors.proposal import (
    AlphaAgentHypothesisGen, AlphaAgentHypothesis, MAX_PARSE_RETRIES,
)

_real = _proposal.APIBackend

# A response with NO JSON object -- the 2026-08-23 failure mode (a Python
# function, no braces for robust_json_parse's last-resort regex to find).
MALFORMED = "def calculate_vwap_dev_depth5():\n    depth = 5\n    return 0"
VALID = '{"hypothesis": "revamped premise", "concise_observation": "", '\
        '"concise_justification": "", "concise_knowledge": "", '\
        '"concise_specification": "", "expected_ic_sign": "negative"}'


class _Queue:
    """Fake APIBackend: ``APIBackend()`` returns this instance; each call pops
    the next queued response. Raises if the queue empties (the caller retried
    more than expected)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def build_messages_and_create_chat_completion(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def _patch(queue):
    _proposal.APIBackend = lambda: queue


def _restore():
    _proposal.APIBackend = _real


def _new_gen(**attrs):
    g = AlphaAgentHypothesisGen.__new__(AlphaAgentHypothesisGen)
    g.scen = SimpleNamespace(get_scenario_all_desc=lambda **k: "")
    g.targets = []
    g.potential_direction = None
    for k, v in attrs.items():
        setattr(g, k, v)
    return g


# ---------------------------------------------------------------------------
# R1 -- a transient malformed response is recovered by re-prompting.
# ---------------------------------------------------------------------------
q = _Queue([MALFORMED, MALFORMED, VALID])
_patch(q)
try:
    g = _new_gen()
    h = g._gen_with_parse_retry("sys", "usr", json_flag=True)
finally:
    _restore()
assert q.calls == 3, f"R1: expected 3 LLM calls (2 malformed + 1 valid), got {q.calls}"
assert h.hypothesis == "revamped premise", "R1: the valid response must be parsed"
assert h.expected_ic_sign == "negative", "R1: the sign must survive the parse"
print(f"R1 PASS  re-prompt recovers a transient malformed response ({q.calls} calls)")

# ---------------------------------------------------------------------------
# R2 -- persistent malformation re-raises after MAX_PARSE_RETRIES.
# ---------------------------------------------------------------------------
q = _Queue([MALFORMED] * MAX_PARSE_RETRIES)
_patch(q)
try:
    g = _new_gen()
    try:
        g._gen_with_parse_retry("sys", "usr", json_flag=True)
        raise AssertionError("R2: persistent malformation must re-raise")
    except (json.JSONDecodeError, ValueError):
        pass
finally:
    _restore()
assert q.calls == MAX_PARSE_RETRIES, \
    f"R2: expected exactly {MAX_PARSE_RETRIES} calls, got {q.calls}"
print(f"R2 PASS  re-raises after {MAX_PARSE_RETRIES} retries (no infinite loop)")

# ---------------------------------------------------------------------------
# R5 -- the fallback returns None for every non-revise path.
# ---------------------------------------------------------------------------
assert _new_gen()._refine_hypothesis_fallback() is None, \
    "R5: a fresh gen (no refine_directive) has no parent to fall back to"
g_expr = _new_gen(refine_directive={"refine_target": "expression"})
assert g_expr._refine_hypothesis_fallback() is None, \
    "R5: an expression-refine (frozen premise) does not use the revise fallback"
g_sign = _new_gen(refine_directive={"refine_target": "sign"})
assert g_sign._refine_hypothesis_fallback() is None, \
    "R5: a sign-flip refine does not use the revise fallback"
print("R5 PASS  fallback is None for expression / sign / fresh paths")

# ---------------------------------------------------------------------------
# R3 -- gen() degrades to the frozen parent premise on the revise path.
# ---------------------------------------------------------------------------
q = _Queue([MALFORMED] * MAX_PARSE_RETRIES)
_patch(q)
try:
    g = _new_gen(
        refine_directive={
            "refine_target": "hypothesis",
            "parent_hypothesis": "Amihud illiquidity predicts continuation.",
            "parent_expected_ic_sign": "positive",
            "directive_text": "The continuation premise is not supported.",
        },
        parent_prefix={
            "hypothesis": "Amihud illiquidity predicts continuation.",
            "expected_ic_sign": "positive",
        },
    )
    h = g.gen(Trace(scen=None))
finally:
    _restore()
assert isinstance(h, AlphaAgentHypothesis), \
    "R3: gen() must return a hypothesis, not raise, on the revise path"
assert h.hypothesis == "Amihud illiquidity predicts continuation.", \
    "R3: the frozen PARENT premise is used (expression-refine semantics)"
assert h.expected_ic_sign == "positive", \
    "R3: the parent's pre-registered direction travels with the frozen premise"
assert q.calls == MAX_PARSE_RETRIES, \
    f"R3: the re-prompt budget is exhausted before degrading ({q.calls} calls)"
print("R3 PASS  hypothesis-revise degrades to frozen parent premise (child is scored, not lost)")

# ---------------------------------------------------------------------------
# R4 -- gen() for a FRESH generation re-raises (a real error, not papered over).
# ---------------------------------------------------------------------------
q = _Queue([MALFORMED] * MAX_PARSE_RETRIES)
_patch(q)
try:
    g = _new_gen()  # no refine_directive -> fresh generation
    try:
        g.gen(Trace(scen=None))
        raise AssertionError("R4: a fresh-gen parse failure must re-raise")
    except (json.JSONDecodeError, ValueError):
        pass
finally:
    _restore()
assert q.calls == MAX_PARSE_RETRIES, \
    f"R4: the re-prompt budget is exhausted before re-raising ({q.calls} calls)"
print("R4 PASS  fresh-gen parse failure re-raises (no silent fallback to a non-existent parent)")

print("\nALL PASS")