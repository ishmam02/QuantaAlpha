"""Two-best crossover selection (Eq. 7): select the two best-performing
parents, NOT a high-signal x low-turnover niche pair.

The old ``select_crossover_pairs`` scored pairs by ``directions*2 + phases +
avg_metric`` and added a ``NICHE_BONUS`` that nudged a high-signal parent toward
a low-turnover parent whenever a cost weakness was present -- a diversity
heuristic the paper does not ask for. The redesign ranks by shrunk fitness and
forms ``crossover_n`` disjoint TOP groups: group 1 = the two best, group 2 =
the next two, and so on. The only tie-break: among fitness-near-equal
candidates, prefer a different ``direction_id`` so a pair is not two
same-direction near-clones.
"""
from dataclasses import dataclass, field

from quantaalpha.pipeline.evolution.crossover import CrossoverOperator
from quantaalpha.pipeline.evolution.trajectory import RoundPhase as _RP


@dataclass
class T:
    trajectory_id: str
    direction_id: int = 0
    hypothesis: str = "h"
    factors: list = field(default_factory=list)
    backtest_metrics: dict = field(default_factory=dict)
    phase: object = _RP.ORIGINAL
    feedback: str = ""
    round_idx: int = 0
    parent_trajectory_ids: list = field(default_factory=list)
    expected_ic_sign: str = ""


op = CrossoverOperator()

# Six candidates. A is the best AND cost-weak (high turnover); E/F are the
# low-turnover niche parents the OLD NICHE_BONUS would have paired A with.
A = T("A", direction_id=0, backtest_metrics={"turnover_book": 0.50, "rank_ic": 0.10})
B = T("B", direction_id=1, backtest_metrics={"turnover_book": 0.04, "rank_ic": 0.09})
C = T("C", direction_id=2, backtest_metrics={"turnover_book": 0.04, "rank_ic": 0.08})
D = T("D", direction_id=3, backtest_metrics={"turnover_book": 0.04, "rank_ic": 0.07})
E = T("E", direction_id=4, backtest_metrics={"turnover_book": 0.01, "rank_ic": 0.02})
F = T("F", direction_id=5, backtest_metrics={"turnover_book": 0.01, "rank_ic": 0.01})

fitness = {"A": 1.5, "B": 1.4, "C": 1.3, "D": 1.2, "E": 0.2, "F": 0.1}
cands = [A, B, C, D, E, F]

# --------------------------------------------------------------------------
# S1: a cost-weak best still pairs with the SECOND-best, not the low-turnover
# niche parent. NICHE_BONUS is gone.
# --------------------------------------------------------------------------
groups = op.select_crossover_pairs(
    cands, crossover_size=2, crossover_n=3,
    prefer_diverse=True, fitness_of=fitness)
assert len(groups) == 3, f"S1: expected 3 groups, got {len(groups)}"
g1 = [t.trajectory_id for t in groups[0]]
assert g1 == ["A", "B"], (
    f"S1: top-2 must be [A,B] (the two best by fitness), not a highxlow niche "
    f"pair; got {g1}")
g2 = [t.trajectory_id for t in groups[1]]
assert g2 == ["C", "D"], f"S1: group 2 must be the next two best [C,D]; got {g2}"
g3 = [t.trajectory_id for t in groups[2]]
assert g3 == ["E", "F"], f"S1: group 3 must be [E,F]; got {g3}"
print("S1 PASS  cost-weak best pairs with second-best; NICHE_BONUS gone")

# --------------------------------------------------------------------------
# S2: tie-break prefers a different direction among fitness-near-equal
# candidates (avoid two same-direction near-clones).
# --------------------------------------------------------------------------
# A dir0 fit1.50; P dir0 fit1.49 (same dir, within band); Q dir1 fit1.48 (diff
# dir, within band). The tie-break should pair A with Q, not P.
P = T("P", direction_id=0)
Q = T("Q", direction_id=1)
fit2 = {"A": 1.50, "P": 1.49, "Q": 1.48, "D": 1.20, "E": 0.2, "F": 0.1}
groups2 = op.select_crossover_pairs(
    [A, P, Q, D, E, F], crossover_size=2, crossover_n=3,
    fitness_of=fit2)
g = [t.trajectory_id for t in groups2[0]]
assert g == ["A", "Q"], (
    f"S2: among near-equal candidates the tie-break must prefer the different "
    f"direction [A,Q], not the same-direction near-clone [A,P]; got {g}")
print("S2 PASS  tie-break prefers a different direction among near-equal")

# --------------------------------------------------------------------------
# S3: the tie-break never overrides a REAL quality gap (a candidate outside
# the band is picked strictly on fitness regardless of direction).
# --------------------------------------------------------------------------
# A dir0 fit1.50; R dir0 fit1.30 (same dir but OUTSIDE the band of A). The
# next-best after A is B (dir1 fit1.40). The tie-break does not reach R.
R = T("R", direction_id=0)
fit3 = {"A": 1.50, "B": 1.40, "R": 1.30, "D": 1.20, "E": 0.2, "F": 0.1}
groups3 = op.select_crossover_pairs(
    [A, B, R, D, E, F], crossover_size=2, crossover_n=3, fitness_of=fit3)
g = [t.trajectory_id for t in groups3[0]]
assert g == ["A", "B"], (
    f"S3: a real quality gap must beat the tie-break; got {g}")
print("S3 PASS  real quality gap beats the direction tie-break")

# --------------------------------------------------------------------------
# S4: graceful -- fewer candidates than a full group returns what fits / none.
# --------------------------------------------------------------------------
assert op.select_crossover_pairs([A], crossover_size=2, fitness_of=fitness) == [], (
    "S4: a single candidate cannot form a pair")
few = op.select_crossover_pairs([A, B, C], crossover_size=2, crossover_n=3,
                                fitness_of={"A": 1.5, "B": 1.4, "C": 1.3})
assert [t.trajectory_id for t in few[0]] == ["A", "B"], "S4: one group from three"
assert len(few) == 1, "S4: only one full group fits"
print("S4 PASS  graceful on too few candidates")

# --------------------------------------------------------------------------
# S5: fitness falls back to the primary metric when fitness_of is absent.
# --------------------------------------------------------------------------
def _primary(self):
    return self.backtest_metrics.get("rank_ic")
T.get_primary_metric = _primary
groups5 = op.select_crossover_pairs(cands, crossover_size=2, crossover_n=3)
g = [t.trajectory_id for t in groups5[0]]
assert g == ["A", "B"], f"S5: fitness fallback to primary metric; got {g}"
print("S5 PASS  fitness falls back to the primary metric")

print("\nALL PASS")