#!/usr/bin/env python
"""The sign-match gate must derive its predicted sign from the mechanism TEXT,
not a freely-set field, so a refine cannot flip the sign token without rewriting
the stated direction.

Bug (measured 2026-08-24): a rejected "predicts HIGHER returns" factor was
re-proposed with ``sign_predicted`` flipped to negative but the mechanism text
byte-identical, and the gate admitted it -- the falsifiability was hollow and
the "rescue" was a sign-bit flip, not a mechanism correction. Fix:
``derive_sign_from_mechanism`` reads the stated direction from the text; the
gate uses that, ignoring the model's ``expected_ic_sign`` field.

Usage::

    python tests/eval/test_sign_from_mechanism.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.factors.net_cost_runner import derive_sign_from_mechanism

# Real mechanism texts from the 2026-08-24 mine (ledger_proofgate_fix).
ADMIT_LOWER = (
    "A high ratio of the past 5-day average volume to the past 60-day average "
    "volume predicts LOWER forward returns because sudden volume surges in a "
    "market with limited short-selling indicate attention-based herding and "
    "temporary overpricing that subsequently reverts.")
REJECT_HIGHER = (
    "A high 20-day rolling correlation between daily returns and daily volume "
    "predicts HIGHER forward returns because price moves confirmed by volume "
    "reflect genuine accumulation, whereas unconfirmed advances are fragile.")
CONVERSELY = (
    "A high 20-day rolling correlation between daily returns and log daily "
    "volume indicates that price moves are supported by proportional "
    "participation (conviction), so a HIGH factor value predicts HIGHER forward "
    "returns; conversely, a negative correlation (price rising on falling "
    "volume) signals demand exhaustion and predicts LOWER forward returns.")
GENUINE_REWRITE = (
    "A high 20-day rolling correlation between daily returns and daily volume "
    "predicts LOWER forward returns because confirmed price-volume co-movement "
    "reflects overextension that reverts.")


def test_reads_real_mechanisms():
    assert derive_sign_from_mechanism(ADMIT_LOWER) == "negative", "S1: 'predicts LOWER' -> negative"
    assert derive_sign_from_mechanism(REJECT_HIGHER) == "positive", "S2: 'predicts HIGHER' -> positive"
    print("S1/S2 PASS  derive_sign reads real 'predicts HIGHER/LOWER' mechanisms")


def test_conversely_uses_primary_clause():
    # the factor-HIGH clause (before "conversely") says HIGHER -> positive,
    # even though the contrast clause says LOWER
    assert derive_sign_from_mechanism(CONVERSELY) == "positive", \
        "S3: primary (factor-HIGH) clause 'predicts HIGHER' -> positive"
    print("S3 PASS  conversely pattern: primary clause (HIGH->HIGHER) -> positive")


def test_gaming_blocked_field_flip_ignored():
    """The TS_CORR child kept the mechanism text ('predicts HIGHER') and only
    flipped the sign_predicted FIELD to negative. Deriving from the text still
    yields positive -- a field flip cannot clear the gate. The text must change.
    """
    # child mechanism == parent mechanism (unchanged, still 'predicts HIGHER')
    assert derive_sign_from_mechanism(CONVERSELY) == "positive", \
        "S4: unchanged 'predicts HIGHER' text -> positive (field flip ignored)"
    # the genuine correction rewrites the text to 'predicts LOWER' -> negative
    assert derive_sign_from_mechanism(GENUINE_REWRITE) == "negative", \
        "S5: rewritten 'predicts LOWER' text -> negative (the real rescue)"
    print("S4/S5 PASS  gaming blocked: field flip can't change the derived sign; only a text rewrite can")


def test_unparseable_is_safe():
    assert derive_sign_from_mechanism("") == ""
    assert derive_sign_from_mechanism("an interesting factor") == ""
    assert derive_sign_from_mechanism(None) == ""
    print("S6 PASS  unparseable -> '' (gate's no-direction branch rejects; cannot be lied about)")


def test_source_gate_uses_derive_not_field():
    src = Path("quantaalpha/factors/net_cost_runner.py").read_text()
    assert "exp_sign = derive_sign_from_mechanism(mech_txt)" in src, \
        "S7: gate must derive exp_sign from the mechanism text"
    assert "exp_sign = (expected_sign or" not in src, \
        "S7: gate must NOT read the freely-set expected_sign field"
    print("S7 PASS  gate derives sign from mechanism text, ignores the free field")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    test_reads_real_mechanisms()
    test_conversely_uses_primary_clause()
    test_gaming_blocked_field_flip_ignored()
    test_unparseable_is_safe()
    test_source_gate_uses_derive_not_field()
    print("\nALL PASS")