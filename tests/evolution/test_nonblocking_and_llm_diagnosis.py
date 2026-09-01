"""The search must accumulate what it measures, and diagnose it with the LLM.

Every assertion here fails against the previous behaviour. That is the point:
the two changes were claimed before without a test that could tell.

  N1-N3  admission observes instead of blocking, WITHOUT losing the verdict
  N4     pathologies still block (validity, not quality)
  L1-L4  the LLM authors the diagnosis, and degrades safely when it cannot
  P1     the diagnosis is shown the population, which it never was
"""
import sys
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from quantaalpha.core.verdict import Verdict
from quantaalpha.eval.admission import decide
from quantaalpha.eval.protocol import load_protocol

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")


def theta_with(**adm):
    return replace(TH, admission=replace(TH.admission, **adm))


# --------------------------------------------------------------------------
# N. Admission observes instead of blocking
# --------------------------------------------------------------------------
# A batch that clearly makes the book worse. Under the old code this was
# rejected and its factors never entered the pool, so nothing downstream could
# ever learn from it.
HARMFUL = [-0.40, -0.52, -0.38, -0.61, -0.45]
# Large zoo so the bootstrap branch (which admits unconditionally) is not what
# is being measured.
ZOO = 50

d_block = decide(HARMFUL, ZOO, theta_with(blocking=True, min_size=6))
assert d_block.admit is False, "N1: blocking=True must still reject a harmful batch"
assert d_block.verdict is Verdict.NET_HARMFUL, f"N1: expected NET_HARMFUL, got {d_block.verdict}"

d_obs = decide(HARMFUL, ZOO, theta_with(blocking=False, min_size=6))
assert d_obs.admit is True, "N2: blocking=False must admit so the pool accumulates"
assert d_obs.verdict is Verdict.NET_HARMFUL, (
    f"N2: the VERDICT must survive non-blocking mode -- it is the learning "
    f"signal and the refinement router. Got {d_obs.verdict}")

# The measurement itself must be identical; only the decision changes.
assert abs(d_obs.mean - d_block.mean) < 1e-12, "N3: mean must not change"
assert abs(d_obs.t_stat - d_block.t_stat) < 1e-12, "N3: t must not change"
assert d_obs.deltas == d_block.deltas, "N3: per-seed deltas must not change"

# A constant signal cannot be traded whatever the bar says.
d_path = decide(HARMFUL, ZOO,
                theta_with(blocking=False, min_size=6),
                metrics={"signal_std": 0.0})
assert d_path.admit is False, "N4: a pathology must block even when non-blocking"
assert d_path.verdict is Verdict.CONSTANT, f"N4: expected CONSTANT, got {d_path.verdict}"

print("N1-N4 PASS  admission observes, verdict survives, pathologies still block")


# --------------------------------------------------------------------------
# L. The LLM authors the diagnosis
# --------------------------------------------------------------------------
import quantaalpha.pipeline.evolution.llm_diagnosis as LD
from quantaalpha.pipeline.evolution.diagnosis import RefineTarget


@dataclass
class FakeParent:
    hypothesis: str = "Overnight order imbalance predicts next-day reversal."
    factors: list = field(default_factory=lambda: [
        {"name": "Overnight_Gap_5D",
         "expression": "RANK(TS_SUM(($open - DELAY($close,1)) / DELAY($close,1), 5))"}])
    backtest_metrics: dict = field(default_factory=dict)
    trajectory_id: str = "t-parent"


# L1: contract preserved -- nothing to diagnose without an objective vector.
assert LD.llm_diagnose(FakeParent(backtest_metrics={})) is None, \
    "L1: no U must still return None (the caller breeds orthogonally)"

metrics = {"U": 0.31, "rank_ic": 0.019, "turnover_book": 0.055, "cost_bps": 4.7,
           "delta_mean": -0.08, "delta_se": 0.23, "delta_t": -0.35,
           "admitted": False, "verdict": "marginal", "rho_max": 0.61,
           "e_turnover": 0.10, "e_effectiveness": 0.22}
parent = FakeParent(backtest_metrics=metrics)


class _Boom:
    def build_messages_and_create_chat_completion(self, **_):
        raise RuntimeError("provider down")


# L2: an LLM failure must never block a round.
import quantaalpha.llm.client as _client
_real = _client.APIBackend
_client.APIBackend = _Boom
try:
    d = LD.llm_diagnose(parent)
finally:
    _client.APIBackend = _real
assert d is not None, "L2: an LLM failure must fall back, not return None"
assert d.directive_text, "L2: the fallback must still carry a directive"
print("L1-L2 PASS  contract preserved; LLM failure falls back to the table")


# L3: a well-formed response becomes a directive.
GOOD = ('{"diagnosis": "Turnover 0.055/day costs 4.7bps against a RankIC of '
        '0.019. The five-day sum re-ranks almost every day.", '
        '"weakness": "cost", "layer": "expression", '
        '"directive": "The premise holds but the construction trades too fast '
        'for the edge it earns. Change how the signal is accumulated.", '
        '"exhausted": false}')


class _Good:
    def __init__(self): self.seen = None
    def build_messages_and_create_chat_completion(self, user_prompt=None, **_):
        self.seen = user_prompt
        return GOOD


spy = _Good()
_client.APIBackend = lambda: spy
try:
    d = LD.llm_diagnose(parent)
finally:
    _client.APIBackend = _real

assert d.refine_target is RefineTarget.EXPRESSION, f"L3: layer -> {d.refine_target}"
assert d.frozen_layers == ["hypothesis"], "L3: an expression-refine keeps the premise"
assert d.weakness_dimension == "cost", f"L3: weakness -> {d.weakness_dimension}"
assert "trades too fast" in d.directive_text, "L3: directive text must reach the child"
assert d.is_refinement() is True, "L3: this must route to REFINE"

# L4: the prompt must carry the MEASUREMENTS, not just a verdict label. The old
# table rendered one canned sentence per (verdict, category) and showed the
# model nothing else.
for token in ("0.0550", "RankIC", "Book turnover", "Max correlation vs library"):
    assert token in spy.seen, f"L4: prompt is missing {token!r}"
# And it must not smuggle in a prior about which market this is.
for banned in ("CSI", "China", "A-share", "reversal is", "momentum works"):
    assert banned not in spy.seen, f"L4: prompt leaks a market prior: {banned!r}"
print("L3-L4 PASS  LLM diagnosis parsed; prompt carries measurements, no market prior")

# L5: the diagnosis DIAGNOSES only -- it must not ask the model for a remedy. The
# diagnose-never-prescribe hard rule: the prompt states the measured problem and
# where it sits, never what to change, and closes with "yours to determine". The
# 0823 smoke showed the old system prompt said "2. PRESCRIBE. Say what to change"
# and the directive field asked for "what to change about it" -- the LLM obeyed and
# the directive prescribed ("drop the volume filter / consider smoothing"). A
# prescription that happens to be right is still a prescription.
for _banned in ("PRESCRIBE", "what to change about it", "Say what to change"):
    assert _banned not in LD._SYSTEM, f"L5: system prompt prescribes: {_banned!r}"
assert "yours to determine" in LD._SYSTEM, "L5: system prompt must close non-prescriptively"
assert "what to change" not in spy.seen, "L5: the user-prompt closer must not prescribe"
assert "yours to determine" in spy.seen, "L5: the user prompt must close non-prescriptively"
print("L5 PASS  diagnosis prompt diagnoses only (no remedy; closes 'yours to determine')")

# L6: the diagnosis frames an ADMITTED parent as "where is there room to push",
# not "what is wrong". ADMITTED-PUSH reuses this diagnosis path on a winner; the
# shared prompt must branch on the gate's decision (surfaced as `admitted` /
# `verdict` in the measurement block) so a push child gets a room-to-push
# directive instead of a fix-what's-broken one. Both branches locate only.
for _tok in ("ADMITTED factor is not broken", "room to be pushed",
             "REJECTED factor has a measured shortfall"):
    assert _tok in LD._SYSTEM, f"L6: framing paragraph missing {_tok!r}"
# The gate decision must reach the model: the gloss renders `admitted` / `verdict`.
assert "Admitted to the repository" in spy.seen, "L6: admission status not in prompt"
assert "Gate verdict" in spy.seen, "L6: verdict label not in prompt"
print("L6 PASS  diagnosis branches on the gate decision (admit -> room to push; reject -> shortfall)")


# --------------------------------------------------------------------------
# P. The diagnosis is shown the population
# --------------------------------------------------------------------------
prior = [FakeParent(backtest_metrics={"U": 0.2, "rank_ic": 0.031,
                                      "delta_mean": 0.02, "admitted": True},
                    hypothesis="Volume-conditioned reversal over 20 days.")]
spy2 = _Good()
_client.APIBackend = lambda: spy2
try:
    LD.llm_diagnose(parent, ancestors=None, population=prior)
finally:
    _client.APIBackend = _real

assert "Volume-conditioned reversal" in spy2.seen, \
    "P1: prior attempts must reach the prompt -- this channel did not exist"
assert "Attempts that came before" in spy2.seen, "P1: population section missing"
print("P1 PASS  prior attempts reach the diagnosis prompt")


# ---------------------------------------------------------------------------
# L7: the TABLE directives diagnose, they never prescribe -- including the
# ADMITTED-PUSH and FULL branches.
#
# Found in the 20260823 10-dir smoke log, not by reading code: the ADMITTED
# branch shipped "Refine the EXPRESSION to push the edge further: strengthen
# TS_MEAN / extend its horizon on AbnormalTurnover_5_60". That is two
# prescriptions -- it names the parent's OWN top operator as the thing to keep,
# and "extend its horizon" as the edit. Both admitted parents produced it 5x
# each, so whichever operator was admitted first got fed back as the operator to
# reuse. TS_MEAN holds 64-83% of the window-summarizer slot in every mine on
# record. The FULL branch had the same defect ("strengthen the signal or cut
# turnover").
#
# The diagnosis must SURVIVE: the directive still has to name WHERE the credit
# sits, or the child is told nothing.
# ---------------------------------------------------------------------------
from quantaalpha.pipeline.evolution.diagnosis import diagnose as _table_diagnose

_ADMIT = SimpleNamespace(
    factors=[{"name": "AbnormalTurnover_5_60",
              "expression": "TS_MEAN($volume, 5) / (TS_MEDIAN($volume, 60) + 1e-8)"}],
    backtest_metrics={"U": 1.2, "rank_ic": 0.0272, "verdict": "admitted",
                      "admitted": True, "e_effectiveness": 0.9, "e_turnover": 0.8,
                      "factor_attribution": {"AbnormalTurnover_5_60": {
                          "weight": 0.62, "rank_ic": 0.0272, "turnover_share": 0.03}}},
    hypothesis="abnormal turnover reverses", expected_ic_sign="negative",
    trajectory_id="ADM", refine_actions=[])
_FULL = SimpleNamespace(
    factors=[{"name": "F", "expression": "TS_MEAN($close, 20)"}],
    backtest_metrics={"U": 0.1, "verdict": "full", "admitted": False,
                      "reason": "repository at capacity", "e_effectiveness": 0.2},
    hypothesis="h", expected_ic_sign="positive", trajectory_id="FUL",
    refine_actions=[])

_PRESCRIPTIONS = ["strengthen ", "extend its horizon", "extend the horizon",
                  "lengthen", "shorten", "cut turnover", "simplify",
                  "blend linearly", "push the edge further"]
for _label, _p in (("ADMITTED", _ADMIT), ("FULL", _FULL)):
    _d = _table_diagnose(_p)
    assert _d is not None, f"L7: {_label} produced no directive"
    _low = _d.directive_text.lower()
    _leak = [t for t in _PRESCRIPTIONS if t in _low]
    assert not _leak, f"L7: the {_label} directive prescribes {_leak}: {_d.directive_text!r}"
    assert "yours to determine" in _low, (
        f"L7: the {_label} directive must hand the decision back")
# ...and the ADMITTED diagnosis must still LOCATE the credit (not go vague).
_d_adm = _table_diagnose(_ADMIT)
assert "TS_MEAN" in _d_adm.directive_text and "AbnormalTurnover_5_60" in _d_adm.directive_text, (
    "L7: removing the prescription must not remove the diagnosis -- the directive "
    "still has to name WHERE the combiner's credit sits")
print("L7 PASS  ADMITTED/FULL directives diagnose only; credit still located")

print("\nALL PASS")
