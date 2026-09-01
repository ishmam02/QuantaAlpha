"""The feedback must report the ECONOMIC bar, not only the statistical one.

Measured 2026-08-24 on the live run:

    |t| required by multiple-testing control : 2.18  (median)
    |t| the factors achieve                  : 4.77  (median)
    |IC| the factors achieve                 : 0.0216
    |IC| at which the book turns profitable  : 0.0728 (oracle ladder, measured)

The search was SUCCEEDING at the bar it was shown -- clearing |t| by more than
2x -- while the book it produced lost 3.57pp/yr to a no-alpha baseline. The
statistical bar also gets EASIER as data accumulates (|t|=3 needs |IC| 0.0173
on the 3-year valid window, 0.0094 on the 10-year test), while the economic bar
is fixed by cost. Nothing in the feedback said the second bar existed.

E1  the gap is stated with both numbers and the ratio
E2  it PRESCRIBES NOTHING (the hard rule: diagnose, never prescribe)
E3  a factor that clears the book bar is told so, not warned
E4  missing/garbage inputs degrade to silence, never to a fabricated bar
E5  the per-factor turnover bar is reported independently of the book bar
E9  the book bar is compared against the BOOK, never against one factor
"""
import re

from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback as F

# The real numbers this run produced.
# MEASURED, not estimated. 0.0728 comes from the oracle ladder
# (scripts/qa_ic_ladder.py, 10 rungs, 2026-08-24) and is bracketed by real
# rungs: IC 0.0920 -> net_IR +0.547, IC 0.0522 -> net_IR -0.589. A previous
# session carried 0.13 as an ESTIMATE; the measurement is 1.8x lower, and this
# fixture used to hold that estimate.
LIVE = {"rank_ic_neutral": -0.0216,
        "ic_breakeven_solo": 0.0021,
        "ic_breakeven_book": 0.0872,   # re-measured on VALID; the first was
                                       # calibrated on final_test (a leak)
        "rank_ic": -0.0504}            # the COMPOSITE the book actually trades


def render(sheet):
    return F._economic_gap(sheet)


def flat(lines):
    return " ".join(" ".join(lines).split())


# ---------------------------------------------------------------------------
# E1 -- both numbers and the ratio, in the model's own units.
# ---------------------------------------------------------------------------
lines = render(LIVE)
txt = flat(lines)
assert lines, "E1: the gap block is empty on live numbers"
assert "0.0216" in txt, f"E1: the achieved |IC| is not stated: {txt}"
assert "0.0872" in txt, f"E1: the book bar is not stated: {txt}"
# The bar is a BOOK-level number: the shortfall reported must be the
# BOOK's (0.0504 vs 0.0872 = 1.7x), never the single factor's (4.6x). The
# latter asks "could this factor be the whole book?", which it never had to be.
assert re.search(r"1\.7x SHORT", txt), f"E1: must report the BOOK's shortfall: {txt}"
assert "4.6x SHORT" not in txt and "4.0x SHORT" not in txt, (
    f"E1: reported the single factor's gap against a book-level bar: {txt}")
assert "0.0504" in txt, f"E1: the composite is not stated: {txt}"
print("E1 PASS  states measured |IC|, the bar, and the multiple short")

# ---------------------------------------------------------------------------
# E2 -- DIAGNOSE, NEVER PRESCRIBE. Naming a remedy here would hand the search a
# construction it did not reason to, and the named edit then gets selected for
# that reason alone.
# ---------------------------------------------------------------------------
low = txt.lower()
for bad in ("lengthen", "shorten", "simplify", "smooth", "widen", "blend",
            "you should", "try ", "use a", "consider ", "instead of",
            "increase the", "reduce the"):
    assert bad not in low, f"E2: prescription leaked: {bad!r} in {txt}"
assert "yours to determine" in low, (
    "E2: must hand the decision back to the model")
print("E2 PASS  no remedy named; closes 'yours to determine'")

# ---------------------------------------------------------------------------
# E3 -- a factor ABOVE the book bar must not be told it is short.
# ---------------------------------------------------------------------------
strong = dict(LIVE, rank_ic=-0.20)   # the BOOK clears the bar
t2 = flat(render(strong))
assert "SHORT" not in t2, f"E3: a factor above the bar was warned: {t2}"
assert "clears it" in t2, f"E3: should say it clears the bar: {t2}"
assert "yours to determine" not in t2.lower(), (
    "E3: no directive is owed to a factor that clears the bar")
print("E3 PASS  a factor above the book bar is told it clears it")

# ---------------------------------------------------------------------------
# E4 -- degrade to SILENCE, never to a fabricated bar. An invented threshold is
# worse than none: the search would optimise toward a number nobody measured.
# ---------------------------------------------------------------------------
assert render({}) == [], "E4: no IC -> no claim"
assert render({"rank_ic_neutral": float("nan")}) == [], "E4: NaN IC -> no claim"
assert render({"rank_ic_neutral": 0.0}) == [], "E4: zero IC -> no ratio to state"
assert render({"rank_ic_neutral": -0.02}) == [], (
    "E4: an IC with NO measured bar must produce nothing, not a guess")
assert render({"rank_ic_neutral": -0.02, "ic_breakeven_book": "n/a"}) == [], (
    "E4: an unparseable bar must be ignored, not rendered")
print("E4 PASS  missing/garbage input yields silence, not an invented bar")

# ---------------------------------------------------------------------------
# E5 -- the two bars are independent. The per-factor turnover bar is cheap and
# always available; the book bar needs the oracle ladder and may be absent.
# ---------------------------------------------------------------------------
solo_only = {"rank_ic_neutral": -0.0216, "ic_breakeven_solo": 0.0021}
t3 = flat(render(solo_only))
assert "own trading" in t3, f"E5: the solo bar is not reported alone: {t3}"
assert "book" not in t3.lower(), f"E5: a book bar was invented: {t3}"
# and the solo bar reports a factor that CLEARS it as clearing it
assert "10.3x the bar" in t3, f"E5: solo ratio wrong: {t3}"
print("E5 PASS  the turnover bar reports independently of the book bar")

# ---------------------------------------------------------------------------
# E6 -- BREADTH is reported as a distribution, not an extremum. rho_max names
# the closest single neighbour; it cannot say that 15 factors carry 10.6
# independent bets, and the search was rewarded for factor COUNT either way.
# ---------------------------------------------------------------------------
b = flat(F._breadth_note({"metrics": {"effective_rank": 10.6,
                                      "book_n_factors": 15}}))
assert "15 factors" in b and "10.6 independent bets" in b, f"E6: {b}"
assert "71%" in b, f"E6: the density ratio is not stated: {b}"
low_b = b.lower()
for bad in ("you should", "try ", "add more", "drop the", "instead"):
    assert bad not in low_b, f"E6: prescription leaked into the breadth note: {bad!r}"
print("E6 PASS  breadth stated as count + independent bets + density, no remedy")

# E7 -- degrade to silence rather than a fabricated breadth claim.
assert F._breadth_note({}) == [], "E7: no metrics -> no claim"
assert F._breadth_note({"metrics": {"book_n_factors": 1,
                                    "effective_rank": 1.0}}) == [], (
    "E7: a one-factor book has no breadth to report")
assert F._breadth_note({"metrics": {"effective_rank": float("nan"),
                                    "book_n_factors": 9}}) == [], (
    "E7: NaN rank -> no claim")
print("E7 PASS  breadth degrades to silence on missing/degenerate input")

# ---------------------------------------------------------------------------
# E8 -- breadth must be emitted where it ALWAYS runs.
#
# It was first placed inside _per_factor_lines, which returns early when
# `factor_tearsheets` is absent -- i.e. on every batch whose factors were
# rejected before a tear sheet was built. Measured on the gate2 run:
# effective_rank was present in the ledger for every batch and reached the
# model on NONE of them. It is now emitted from _format_metric_block, against
# the normalised (flat) metric dict.
# ---------------------------------------------------------------------------
import inspect
_src = inspect.getsource(F._format_metric_block)
assert "breadth_note(" in _src, (
    "E8: breadth is not emitted from the metric block, so it is dropped on "
    "every batch that has no tear sheets")
_pf = inspect.getsource(F._per_factor_lines)
assert "breadth_note(" not in _pf, (
    "E8: breadth is still inside the per-factor block, which returns early "
    "when tear sheets are absent")
# and it must read a FLAT metric dict, which is what that caller has
assert F._breadth_note({"effective_rank": 1.96, "book_n_factors": 2}), (
    "E8: breadth must accept the flat, normalised metric dict")
print("E8 PASS  breadth is emitted from the metric block and reads flat metrics")

# ---------------------------------------------------------------------------
# E9 -- the book bar is a BOOK-level number and must never be compared against
# one factor. The bar comes from an oracle ladder that priced a SINGLE signal,
# so "|IC| 0.0189 vs 0.0872 -- 4.6x SHORT" asks whether that factor could be
# the entire book. Under Grinold's law the composite is IC * sqrt(independent
# bets): a 0.02 factor inside a book of ten independent bets is doing its job.
# The measured truth is the BOOK is 1.7x short (0.0504 vs 0.0872).
# ---------------------------------------------------------------------------
t9 = flat(render(LIVE))
assert "BOOK, not this factor" in t9, f"E9: must name whose gap this is: {t9}"
assert "1.7x SHORT" in t9, f"E9: must report the book's 1.7x, not the factor's: {t9}"
assert "contributes" in t9, (
    "E9: the factor's own IC must be framed as a CONTRIBUTION to the composite")
assert "independent bets" in t9.lower(), (
    "E9: must say why a single factor is not expected to clear a book bar")

# without a composite it must NOT fall back to comparing the factor
t9b = flat(render({"rank_ic_neutral": -0.0189, "ic_breakeven_book": 0.0872}))
assert "SHORT" not in t9b, (
    f"E9: with no composite priced, no shortfall may be attributed: {t9b}")
assert "not priced this batch" in t9b, f"E9: must say the composite is absent: {t9b}"
print("E9 PASS  the book bar is compared to the BOOK; the factor is a contribution")

print("\nALL PASS")
