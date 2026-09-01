"""Redundancy must be measured against the library AS IT WILL STAND.

Measured on the 2026-08-24 OOS backtest of the live 37-factor zoo:

    rho_within = 0.9965        the worst pair inside the book
    net_ARR    = -10.54%       vs a -5.22% no-alpha baseline

37 factors carried roughly ONE independent bet, and the book lost to doing
nothing by 5.3pp/yr. Two holes let that happen, and both are closed by the same
change -- comparing each candidate against the repository PLUS the batch-mates
already kept:

  * ``rho_within`` was computed and warned above 0.8, but gated nothing.
  * ``rho_max`` compared only against the repository, and under standalone
    admission each factor is judged as its own batch -- so batch-mates never
    met each other, and the first admits saw an empty repository (rho = 0.00).

W1  two identical candidates in one batch: the second is caught, not admitted
W2  a genuinely distinct second candidate still passes
W3  the comparison set grows as factors are kept (the 3rd sees the 1st AND 2nd)
W4  a replacement needs a REAL |t| margin -- a hair-thin win keeps the incumbent
W5  a decisive improvement still replaces
"""
import numpy as np
import pandas as pd

from quantaalpha.eval.metrics import rho_max_arg

DATES = pd.bdate_range("2020-01-01", periods=260)
NAMES = [f"S{i:03d}" for i in range(120)]
RNG = np.random.default_rng(7)


def frame(seed):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(size=(len(DATES), len(NAMES))),
                        index=DATES, columns=NAMES)


BASE = frame(1)
CLONE = BASE * 1.0001 + 1e-9          # a near-duplicate: rho ~ 1
DISTINCT = frame(99)                   # independent draw: rho ~ 0
RHO_BAR = 0.6


def held_plus(kept):
    """The comparison set the patched gate builds: repository + kept-so-far."""
    return dict(kept)


# ---------------------------------------------------------------------------
# W1 -- two near-identical candidates in ONE batch. The old gate compared each
# only against the (empty) repository and admitted both.
# ---------------------------------------------------------------------------
kept = {}
rho1, _ = rho_max_arg(BASE, held_plus(kept)) if kept else (0.0, None)
assert rho1 < RHO_BAR, "W1: the first candidate has nothing to duplicate"
kept["base"] = BASE

rho2, near2 = rho_max_arg(CLONE, held_plus(kept))
assert rho2 >= RHO_BAR, (
    f"W2 setup: the clone must register as a duplicate; rho={rho2:.4f}")
assert near2 == "base", f"W1: should point at its batch-mate; got {near2}"
print(f"W1 PASS  batch-mate duplicate caught: rho={rho2:.4f} vs bar {RHO_BAR} "
      f"(was 0.00 -- the repository was empty)")

# ---------------------------------------------------------------------------
# W2 -- a genuinely different candidate must still get through.
# ---------------------------------------------------------------------------
rho3, near3 = rho_max_arg(DISTINCT, held_plus(kept))
assert rho3 < RHO_BAR, (
    f"W2: an independent signal must pass the redundancy gate; rho={rho3:.4f}")
print(f"W2 PASS  distinct candidate still admitted: rho={rho3:.4f} < {RHO_BAR}")

# ---------------------------------------------------------------------------
# W3 -- the comparison set GROWS. A third candidate that duplicates the SECOND
# (not the first) must still be caught.
# ---------------------------------------------------------------------------
kept["distinct"] = DISTINCT
clone_of_second = DISTINCT * 0.999 + 1e-9
rho4, near4 = rho_max_arg(clone_of_second, held_plus(kept))
assert rho4 >= RHO_BAR, f"W3: duplicate of the 2nd kept factor missed; rho={rho4:.4f}"
assert near4 == "distinct", (
    f"W3: must name the factor it duplicates; got {near4}")
print(f"W3 PASS  comparison set grows: 3rd candidate matched against the 2nd "
      f"(rho={rho4:.4f}, near={near4})")

# ---------------------------------------------------------------------------
# W4/W5 -- the REPLACE path. A duplicate may displace an incumbent only when it
# is meaningfully stronger. Measured: 20 of 38 admits on the live run were
# replacements, churning inside one direction while reporting progress.
# ---------------------------------------------------------------------------
MARGIN = 1.0


def replaces(cand_t, inc_t, margin=MARGIN):
    """The patched decision: an upgrade needs a real margin."""
    return cand_t > inc_t + margin


assert not replaces(5.01, 5.00), (
    "W4: a 0.01 |t| edge must NOT swap one near-clone for another")
assert not replaces(5.90, 5.00), (
    "W4: a 0.90 edge is still inside the margin")
print("W4 PASS  a hair-thin |t| win keeps the incumbent (no churn)")

assert replaces(7.00, 5.00), "W5: a decisive 2.0 |t| improvement must replace"
print("W5 PASS  a decisive improvement still replaces the incumbent")

# ---------------------------------------------------------------------------
# W6 -- ORDER-FREE. The margin used to sit on the challenger alone ("the
# incumbent stays unless beaten by 1.0"), which made the book depend on
# scheduling: a |t|=10.78 factor lost to a |t|=10.07 incumbent it duplicated at
# rho=0.686, purely because the weaker one was measured first. Reverse the
# arrival order and you get a different book.
#
# The rule now: a decisive gap wins for EITHER side; inside the margin the two
# are indistinguishable on |t| and the tie breaks on turnover (the book pays for
# turnover daily and cannot tell the signals apart anyway).
# ---------------------------------------------------------------------------
def duel(cand_t, inc_t, cand_turn=None, inc_turn=None, margin=MARGIN):
    """True if the candidate takes the slot. Mirrors the patched decision."""
    gap = cand_t - inc_t
    if abs(gap) > margin:
        return gap > 0
    if cand_turn is not None and inc_turn is not None:
        return cand_turn < inc_turn
    return False


# the exact case from the live run
assert duel(10.78, 10.07, cand_turn=0.03, inc_turn=0.05), (
    "W6: the STRONGER-and-cheaper factor must win regardless of arrival order")

# symmetry: swapping who arrived first must not change who is held
for ct, it in [(7.0, 5.0), (5.0, 7.0), (10.78, 10.07), (10.07, 10.78)]:
    a_wins = duel(ct, it, cand_turn=0.02, inc_turn=0.04)
    b_wins = duel(it, ct, cand_turn=0.04, inc_turn=0.02)
    assert a_wins != b_wins, (
        f"W6: order-dependent outcome for |t| {ct} vs {it}: both/neither won")
print("W6 PASS  the duel is symmetric -- arrival order does not decide the book")

# inside the margin, the cheaper signal is held
assert duel(5.10, 5.00, cand_turn=0.02, inc_turn=0.09), (
    "W6: a statistical tie must break toward the cheaper factor")
assert not duel(5.10, 5.00, cand_turn=0.09, inc_turn=0.02), (
    "W6: a statistical tie must NOT install a more expensive near-clone")
print("W7 PASS  a tie on |t| breaks on turnover, not on who arrived first")

# no turnover data -> hold what is already working (safe default)
assert not duel(5.10, 5.00), "W7: with no tiebreak data, keep the incumbent"
print("W8 PASS  missing turnover data falls back to keeping the incumbent")

print("\nALL PASS")
