"""The operator must do structural work, not edit a constant.

O1  a child differing only in numeric literals is REJECTED
O2  a swapped operator / changed signal argument is ACCEPTED
O3  the check is not fooled by whitespace or float-vs-int spelling
O4  a crossover child inheriting from one parent only is REJECTED
O5  a child carrying material distinctive to BOTH parents is ACCEPTED
O6  unparseable input PASSES -- this is not a second syntax checker
O7  enforce_* never empties a batch silently; it reports what it dropped
"""
from quantaalpha.pipeline.evolution.operator_contract import (
    check_refine, check_crossover, enforce_refine, enforce_crossover,
    ContractReport, rejection_note,
)

PARENT = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"

# --- O1: the measured failure mode -------------------------------------------
window_edit = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 20))"
r = check_refine(window_edit, PARENT)
assert not r.ok and r.reason == "literal_only", f"O1: {r.reason} -- {r.detail}"
# Several constants at once is still only constants.
many = "RANK(TS_SUM(($open - DELAY($close, 3)) / DELAY($close, 7), 60))"
assert not check_refine(many, PARENT).ok, "O1: multi-constant edit must still be rejected"
print("O1 PASS  literal-only edits rejected (one constant and many)")

# --- O2: real structural work -------------------------------------------------
op_swap  = "RANK(TS_MEAN(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"
arg_swap = "RANK(TS_SUM(($vwap - DELAY($close, 1)) / DELAY($close, 1), 5))"
wrapped  = "ZSCORE(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"
for name, e in (("operator swap", op_swap), ("signal arg", arg_swap), ("outer op", wrapped)):
    assert check_refine(e, PARENT).ok, f"O2: {name} must be accepted"
print("O2 PASS  operator swap / signal change / re-wrap all accepted")

# --- O3: not fooled by spelling ----------------------------------------------
spaced = "RANK(TS_SUM( ($open - DELAY($close,1)) / DELAY($close,1) , 5 ))"
assert not check_refine(spaced, PARENT).ok, "O3: whitespace is not a structural edit"
floaty = "RANK(TS_SUM(($open - DELAY($close, 1.0)) / DELAY($close, 1), 5.0))"
assert not check_refine(floaty, PARENT).ok, "O3: 1 vs 1.0 is not a structural edit"
print("O3 PASS  whitespace and int/float spelling do not count as structure")

# --- O4/O5: crossover must carry BOTH parents --------------------------------
A = ["RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 5))"]     # $open
B = ["RANK(TS_MEAN(ABS($return) / ($volume + 1), 20))"]                    # $volume, $return, ABS
one_parent = "RANK(TS_SUM(($open - DELAY($close, 1)) / DELAY($close, 1), 20))"
r = check_crossover(one_parent, A, B)
assert not r.ok and r.reason == "single_parent", f"O4: {r.reason} -- {r.detail}"
both = "RANK(TS_MEAN(($open - DELAY($close, 1)) / ($volume + 1), 20))"      # $open AND $volume
r = check_crossover(both, A, B)
assert r.ok, f"O5: a genuine recombination must pass -- {r.detail}"
print(f"O4-O5 PASS  single-parent rejected; recombination accepted ({r.detail})")

# --- O6: not a syntax checker -------------------------------------------------
assert check_refine("this is not an expression((", PARENT).ok, "O6: must not reject on parse failure"
assert check_crossover("((((", A, B).ok, "O6: must not reject on parse failure"
print("O6 PASS  unparseable input passes through (this is not a syntax checker)")

# --- O7: enforcement reports, and never silently empties ----------------------
rep = ContractReport()
ok, bad = enforce_refine([window_edit, op_swap], [PARENT], rep)
assert ok == [op_swap] and bad == [window_edit], f"O7: {ok} / {bad}"
assert rep.checked == 2 and rep.rejected == 1, f"O7: report {rep}"
rep2 = ContractReport()
ok2, bad2 = enforce_crossover([one_parent, both], A, B, rep2)
assert ok2 == [both] and bad2 == [one_parent], f"O7: {ok2} / {bad2}"
assert "only numeric constants" in rejection_note("literal_only")
assert "both" in rejection_note("single_parent")
print("O7 PASS  enforcement splits accepted/rejected and reports counts")

# The note names the defect and stops -- it must not prescribe the fix.
for reason in ("literal_only", "single_parent"):
    note = rejection_note(reason).lower()
    for banned in ("you should", "try ", "instead use", "prefer ", "lengthen", "shorten"):
        assert banned not in note, f"O8: rejection note prescribes ({banned!r}): {note}"
print("O8 PASS  rejection notes state the defect without prescribing a fix")

print("\nALL PASS")
