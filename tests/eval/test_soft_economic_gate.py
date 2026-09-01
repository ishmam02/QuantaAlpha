"""The economic bar shapes the RANKING, not the verdict.

Measured 2026-08-24 on the gate2 run: three factors were admitted at |t| 3.93,
6.94 and 3.97 -- clearing every gate that exists (significance, FDR, IC sign,
monotonicity, turnover, capacity, redundancy) -- while sitting 2.5x to 3.9x
below the |IC| at which a book of them turns net-profitable. The bar was
reported to the generator and never consulted by the gate, so the feedback said
"3.9x SHORT" and the verdict said "admitted" about the same factor.

A HARD gate is the wrong fix: 0 of 11 factors cleared the bar, so it would
admit nothing, and mutation / crossover / admitted-push all consume ADMITTED
parents. The search would stop learning entirely.

So the bar is folded into ``_research_score`` -- the single number admission,
eviction and the replacement duel all rank on. Everything that clears the
statistical gate is still admitted; a factor further below the economic bar
sorts lower, so the library cap evicts it first and the zoo drifts toward
factors that can pay.

S1  a factor AT the bar is unpenalised (score == |t|)
S2  the penalty grows with the shortfall, monotonically
S3  ordering can flip: a weaker-|t| factor closer to the bar outranks a
    stronger-|t| factor far below it
S4  no measured bar -> rank on |t| alone (never invent a penalty)
S5  the score stays on the |t| scale, so admission/eviction/duel agree
S6  the two live admits rank in the order the bar implies
"""
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as R

BAR = 0.0728          # measured; see scripts/qa_ic_ladder.py


class _S:             # the method reads only `metrics`
    pass


def score(t=None, ic=None, bar=BAR):
    m = {}
    if t is not None:
        m["t_nw"] = t
    if ic is not None:
        m["rank_ic_neutral"] = ic
    if bar is not None:
        m["ic_breakeven_book"] = bar
    return R._research_score(_S(), m)


# ---------------------------------------------------------------------------
# S1 -- at or above the bar there is nothing to penalise.
# ---------------------------------------------------------------------------
assert abs(score(4.0, BAR) - 4.0) < 1e-9, "S1: a factor AT the bar must score |t|"
assert abs(score(4.0, 2 * BAR) - 4.0) < 1e-9, "S1: above the bar must not exceed |t|"
print("S1 PASS  at/above the bar the score is |t| (no bonus, no penalty)")

# ---------------------------------------------------------------------------
# S2 -- monotone in the shortfall.
# ---------------------------------------------------------------------------
ladder = [score(4.0, BAR / k) for k in (1, 2, 4, 8, 16)]
assert all(a > b for a, b in zip(ladder, ladder[1:])), (
    f"S2: penalty is not monotone in the shortfall: {ladder}")
assert all(x > 0 for x in ladder), "S2: the score must stay positive"
print(f"S2 PASS  penalty grows with the shortfall: "
      f"{' > '.join(f'{x:.2f}' for x in ladder)}")

# ---------------------------------------------------------------------------
# S3 -- the ORDER can flip. This is the point: |t| alone cannot see that one
# factor is closer to paying for itself than another.
# ---------------------------------------------------------------------------
weak_close = score(4.0, BAR * 0.9)     # modest |t|, nearly at the bar
strong_far = score(6.0, BAR / 12)      # big |t|, far below the bar
assert weak_close > strong_far, (
    f"S3: a factor near the bar must outrank one far below it "
    f"({weak_close:.3f} vs {strong_far:.3f})")
print(f"S3 PASS  |t|4.0 near the bar ({weak_close:.2f}) outranks |t|6.0 far "
      f"below it ({strong_far:.2f})")

# ---------------------------------------------------------------------------
# S4 -- never invent a penalty. Missing/garbage bar or IC falls back to |t|.
# ---------------------------------------------------------------------------
assert abs(score(4.0, 0.02, bar=None) - 4.0) < 1e-9, "S4: no bar -> |t|"
assert abs(score(4.0, None) - 4.0) < 1e-9, "S4: no IC -> |t|"
assert abs(score(4.0, 0.02, bar=float("nan")) - 4.0) < 1e-9, "S4: NaN bar -> |t|"
assert abs(score(4.0, 0.02, bar=0.0) - 4.0) < 1e-9, "S4: zero bar -> |t|"
assert abs(score(4.0, float("nan")) - 4.0) < 1e-9, "S4: NaN IC -> |t|"
assert score(None, 0.02) == float("-inf"), "S4: no |t| still sorts last"
print("S4 PASS  a missing/garbage bar falls back to |t|, never a made-up penalty")

# ---------------------------------------------------------------------------
# S5 -- the score stays on the |t| scale. Admission, eviction and the
# replacement duel all read this one number; an additive penalty in different
# units would make them disagree about what "stronger" means.
# ---------------------------------------------------------------------------
assert 0 < score(4.0, BAR / 4) <= 4.0, "S5: the penalised score left the |t| scale"
assert score(8.0, BAR / 4) > score(4.0, BAR / 4), (
    "S5: at equal shortfall the stronger |t| must still win")
print("S5 PASS  the penalised score stays within (0, |t|]; |t| still orders ties")

# ---------------------------------------------------------------------------
# S6 -- the two live admits from the gate2 run.
# ---------------------------------------------------------------------------
a = score(6.94, 0.0289)      # 2.5x short
b = score(3.93, 0.0189)      # 3.9x short
assert a > b, f"S6: the closer-to-paying factor must rank higher ({a:.3f} vs {b:.3f})"
print(f"S6 PASS  live admits rank {a:.2f} (2.5x short) > {b:.2f} (3.9x short) "
      "-- the further-short one is evicted first")

print("\nALL PASS")
