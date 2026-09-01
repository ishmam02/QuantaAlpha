#!/usr/bin/env python
"""replay_repository must merge the per-factor tear sheet into the rehydrated
metrics, or the replace duel is blind to the incumbent's |t_nw|.

Bug (measured 2026-08-24): the ledger persists batch ``metrics`` (rank_ic, icir,
rho_max, ...) and the per-factor ``factor_tearsheets`` (t_nw, rank_ic_neutral,
ic_breakeven_book, ...) as SEPARATE fields. ``replay_repository`` read only
``metrics``, so a fresh runner (one per evolution task) rehydrated the repository
with batch metrics alone -- every incumbent lacked ``t_nw``, ``_research_score``
returned -inf, and the replace duel let a WEAKER near-duplicate replace a
STRONGER incumbent (a |t| 8.92 WMA-smoothed factor replaced a |t| 11.15 one).
The admission path merges the tear sheet (net_cost_runner.py:540); rehydration
must do the same.

Usage::

    python tests/eval/test_replay_repository_tearsheet.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.ledger import replay_repository
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as R


class _Stub:
    """_research_score reads only its metrics arg (no self.theta)."""
    _research_score = R._research_score


def _ledger(records):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def test_replay_merges_tearsheet():
    rec = {"factor_exprs": ["E1"], "admitted": True, "evicted_exprs": [],
           "metrics": {"rank_ic": 0.03, "icir": 0.2, "rho_max": 0.0},  # batch only
           "factor_tearsheets": {"E1": {"t_nw": -11.15, "rank_ic_neutral": -0.06,
                                       "ic_breakeven_book": 0.087}}}
    path = _ledger([rec])
    try:
        repo = replay_repository(path)
        assert "E1" in repo and repo["E1"].get("t_nw") == -11.15, \
            "T1: rehydrated metrics must carry t_nw (merged from factor_tearsheets)"
        assert "rank_ic_neutral" in repo["E1"] and "ic_breakeven_book" in repo["E1"], \
            "T1: the economic-bar fields must be merged too"
    finally:
        os.unlink(path)
    print("T1 PASS  replay_repository merges factor_tearsheets -> t_nw present")


def test_research_score_reads_rehydrated_t_nw():
    stub = _Stub()
    good = {"t_nw": -11.15, "rank_ic_neutral": -0.06, "ic_breakeven_book": 0.087}
    assert stub._research_score(good) > 0, "T2: incumbent with t_nw scores >0"
    bad = {"rank_ic": 0.03}  # no t_nw -- the pre-fix rehydrated metrics
    assert stub._research_score(bad) == float("-inf"), \
        "T2: without t_nw the score is -inf (the bug, now fixed at the source)"
    print("T2 PASS  _research_score reads t_nw from rehydrated metrics (not -inf)")


def test_decisively_weaker_duplicate_does_not_replace():
    """A |t| 5.0 near-duplicate must NOT replace a |t| 11.15 incumbent.

    With the bug the incumbent scored -inf and the 5.0 'won' decisively
    (anti-learning). With the fix the incumbent scores ~9.26 (|t| 11.15 soft-gated
    by the economic bar), the 5.0 is decisively weaker (gap > margin), and the
    incumbent is retained.
    """
    stub = _Stub()
    inc_m = {"t_nw": -11.15, "rank_ic_neutral": -0.06, "ic_breakeven_book": 0.087}
    inc_score = stub._research_score(inc_m)          # ~9.26 after the soft gate
    cand_score = 5.0                                # a decisively weaker near-duplicate
    margin = 1.0
    gap = cand_score - inc_score
    cand_wins = (gap > 0) if abs(gap) > margin else False
    assert inc_score > 0, f"T3: incumbent must score >0, got {inc_score}"
    assert not cand_wins, (
        f"T3: weaker candidate ({cand_score}) must not replace stronger incumbent "
        f"({inc_score:.2f}); gap {gap:.2f}")
    # document the bug it fixes: pre-fix rehydrated metrics (no t_nw) -> -inf -> wins
    inc_buggy = stub._research_score({"rank_ic": 0.03})
    assert inc_buggy == float("-inf") and (cand_score - inc_buggy) > margin, \
        "T3: without the fix the incumbent scores -inf and the weaker candidate wins"
    print(f"T3 PASS  decisively weaker ({cand_score}) kept out; incumbent ({inc_score:.2f}) retained")


def test_source_merges_tearsheets():
    src = Path("quantaalpha/eval/ledger.py").read_text()
    assert "factor_tearsheets" in src and "m.update(sheet)" in src, \
        "T4: replay_repository must merge the per-factor tearsheet"
    print("T4 PASS  replay_repository source merges factor_tearsheets")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    test_replay_merges_tearsheet()
    test_research_score_reads_rehydrated_t_nw()
    test_decisively_weaker_duplicate_does_not_replace()
    test_source_merges_tearsheets()
    print("\nALL PASS")