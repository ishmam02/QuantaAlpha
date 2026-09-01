"""Winsorization and multiple-testing control.

W1  a single extreme tick cannot move the whole cross-section's residual
W2  winsorization keeps genuine tails (it clips at 5 MAD, not to the median)
W3  a degenerate date is left alone rather than collapsed
F1  the FDR bar RISES as more factors are tested -- the search is itself a
    multiple-testing machine
F2  at one test the bar is inert (BH on n=1 is a no-op)
F3  a factor clearing |t|>=3 can still fail FDR once enough tests accumulate
"""
import numpy as np, pandas as pd
from quantaalpha.eval.neutralize import winsorize

rng = np.random.default_rng(0)
D, N = 30, 200
dates = pd.date_range("2020-01-01", periods=D)
cols = [f"s{i}" for i in range(N)]
X = pd.DataFrame(rng.normal(0, 1, (D, N)), index=dates, columns=cols)

# --- W1: one bad tick ---
dirty = X.copy()
dirty.iloc[5, 0] = 1e6                      # a single absurd print
w = winsorize(dirty, 5.0)
assert abs(w.iloc[5, 0]) < 100, f"W1: outlier survived winsorization: {w.iloc[5,0]}"
# and the rest of that date is untouched
assert np.allclose(w.iloc[5, 1:], dirty.iloc[5, 1:]), "W1: winsorization moved clean names"
print(f"W1 PASS  1e6 outlier clipped to {w.iloc[5,0]:.2f}; the other 199 names untouched")

# --- W2: genuine tails survive ---
kept = (w.abs() > 2.0).sum().sum()
raw = (X.abs() > 2.0).sum().sum()
assert kept >= 0.9 * raw, f"W2: winsorization ate real tails ({kept} vs {raw})"
print(f"W2 PASS  genuine tails preserved: {kept}/{raw} values beyond 2 sigma kept")

# --- W3: a constant date is not collapsed ---
flat = X.copy(); flat.iloc[7, :] = 3.0
wf = winsorize(flat, 5.0)
assert np.allclose(wf.iloc[7, :], 3.0), "W3: a zero-MAD date must be left alone"
print("W3 PASS  a zero-dispersion date is left alone, not zeroed")

# --- FDR ---
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as R

class Stub:
    _fdr_bar = R._fdr_bar
    def __init__(self, theta): self.theta = theta

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")
st = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))

# A realistic history: mostly noise, which is what a search actually produces.
# BH over a null-heavy history yields NO discovery at all -- correct, and the
# reason the history must include the failures (see F4).
rng2 = np.random.default_rng(7)
for _ in range(119): st._fdr_bar(abs(rng2.normal(0, 1)))
req_null, n_null, _ = st._fdr_bar(2.0)
assert req_null == float("inf"), f"F1: a null-heavy history should yield no discovery, got {req_null}"
print(f"F1 PASS  after {n_null} null trials nothing is discoverable (bar = inf)")

# A genuinely strong factor still rescues itself at rank 1 -- BH does not
# punish real evidence for the company it keeps.
req_hot, n_hot, _ = st._fdr_bar(4.8)
assert req_hot <= 4.8, f"F1b: a strong factor must clear its own BH rank-1 bar, needed {req_hot}"
print(f"F1b PASS  a |t|=4.8 factor clears at n={n_hot} (bar {req_hot:.2f}); real evidence survives a noisy run")

st2 = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))
req1, n1, _ = st2._fdr_bar(3.2)
assert n1 == 1, "F2: first call should register one test"
print(f"F2 PASS  at n=1 the bar is {req1:.2f} and the gate skips it (n_tests>1 required)")

st3 = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))
for _ in range(200): st3._fdr_bar(1.0)      # a pile of null factors
req3, n3, _ = st3._fdr_bar(3.2)
assert req3 > 3.0, f"F3: after {n3} mostly-null tests, 3.2 should not clear ({req3:.2f})"
print(f"F3 PASS  after {n3} tests a |t| of 3.2 no longer suffices (needs {req3:.2f})")

# --- F4: the truncation bug ---
# If only survivors of the fixed |t|>=3 bar entered the history, every p would
# be tiny, BH would admit them all, and the bar would collapse to ~3.0 -- an
# inert gate. Confirm a survivors-only history is exactly that degenerate.
st4 = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))
for t in [3.1, 3.3, 3.5, 4.0, 4.4]: req4, n4, _ = st4._fdr_bar(t)
assert req4 < 3.2, f"F4: a survivors-only history is degenerate as expected ({req4:.2f})"
print(f"F4 PASS  survivors-only history collapses the bar to {req4:.2f} (inert) -- "
      "which is why the runner now records every trial before the fixed bar")

import re, pathlib as _pl
src = _pl.Path("quantaalpha/factors/net_cost_runner.py").read_text()
i_rec, i_bar = src.index("t_req, n_tests, q = self._fdr_bar(t)"), src.index("if abs(t) < bar:")
assert i_rec < i_bar, "F5: _fdr_bar must be called BEFORE the fixed |t| bar"
print("F5 PASS  the runner records the trial before the fixed bar, so failures are counted")

# --- F6: replay_t_history reads EVERY trial (admitted + rejected), skips evictions ---
# The ledger persistence that makes the FDR gate rehydratable. The helper must
# collect |t_nw| from every batch's factor_tearsheets -- admitted and rejected --
# in evaluation order, and ignore eviction/transition records (no tearsheets)
# and pre-fix records (no tearsheets). FDR corrects over the full family (F4).
import os, tempfile
from quantaalpha.eval.ledger import Ledger, replay_t_history

with tempfile.TemporaryDirectory() as td:
    _p = os.path.join(td, "ledger.jsonl")
    _lg = Ledger(_p)
    # batch 1: two factors scored (one strong-admitted, one weak-rejected)
    _lg.append({"factor_exprs": ["e1"], "rejected_exprs": [], "admitted": True,
                "factor_tearsheets": {"e1": {"t_nw": 4.0}, "e2": {"t_nw": -2.0}}})
    # batch 2: a rejected factor
    _lg.append({"factor_exprs": [], "rejected_exprs": ["e3"], "admitted": False,
                "factor_tearsheets": {"e3": {"t_nw": 1.5}}})
    # an eviction record: NO tearsheets -- must contribute nothing
    _lg.append({"evicted_exprs": ["e1"], "n_factors": 0, "metrics": {}, "U": None})
    # a pre-fix-style record (no tearsheets) -- skipped, not crashed
    _lg.append({"factor_exprs": ["e9"], "admitted": True, "metrics": {}})
    # a factor that produced no usable IC has no t_nw -- skipped gracefully
    _lg.append({"factor_tearsheets": {"e10": {"rank_ic_neutral": 0.0}}})
    hist = replay_t_history(_p)
    assert hist == [4.0, 2.0, 1.5], f"F6: expected [4.0, 2.0, 1.5] (abs, in order), got {hist}"
print(f"F6 PASS  replay_t_history collected {len(hist)} trials (admitted+rejected); "
      "evictions, pre-fix records, and t_nw-less sheets skipped -- no double-count")

# --- F7: the fix -- rehydration makes the BH gate bind across runner instances ---
# THE BUG: _t_history is in-memory only, so a fresh runner (one per evolution
# task under parallel execution) starts every batch with n_tests=1 and the
# n_tests>1 guard skips the gate (F2). THE FIX: _zoo rehydrates _t_history from
# the ledger via replay_t_history, so a fresh runner inherits the run's prior
# trials and the bar tightens as the docstring promises.
with tempfile.TemporaryDirectory() as td:
    _p = os.path.join(td, "ledger.jsonl")
    _lg = Ledger(_p)
    for _i in range(120):                       # 120 prior null trials, |t|=1.0
        _lg.append({"factor_tearsheets": {f"n{_i}": {"t_nw": 1.0}}})
    assert len(replay_t_history(_p)) == 120, "F7: rehydration must recover all 120 prior trials"
    # WITHOUT rehydration (the bug): a fresh runner is inert
    st_bug = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))
    _, n_bug, _ = st_bug._fdr_bar(3.2)
    assert n_bug == 1, "F7: a fresh runner without rehydration sees n_tests=1 (inert)"
    # WITH rehydration (the fix): exactly what _zoo now does
    st_fix = Stub(replace(TH, admission=replace(TH.admission, fdr_q=0.10)))
    st_fix._t_history = replay_t_history(_p)    # _zoo: self._t_history = replay_t_history(path)
    req_fix, n_fix, _ = st_fix._fdr_bar(3.2)
    assert n_fix == 121, f"F7: rehydrated runner must see 121 tests (120 prior + this), got {n_fix}"
    assert req_fix > 3.2, f"F7: after 120 null trials |t|=3.2 must FAIL the BH bar (need {req_fix}), not pass"
print(f"F7 PASS  rehydration lifts n_tests {n_bug} -> {n_fix}; |t|=3.2 now needs "
      f"{req_fix} (it cleared at n=1) -- the gate binds across runner instances")

print("\nALL PASS")
