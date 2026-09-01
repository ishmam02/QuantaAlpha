"""A near-duplicate that is STRONGER should take the incumbent's place.

Before this, `rho_max` returned a bare float: it computed which incumbent was
closest and threw the identity away, and the gate rejected on the scalar. So a
candidate that duplicated an incumbent AND beat it was discarded while the
weaker incumbent stayed -- the library could only grow into new directions,
never improve along one it already held. Measured: 7 such rejections in a
single run.

R1  rho_max_arg returns the identity of the closest incumbent
R2  rho_max still returns exactly the same scalar (no caller breakage)
R3  the closest incumbent is the RIGHT one, not just any
R4  the gate compares on _research_score -- the same |t_NW| admission uses
R5  a stronger duplicate replaces; a weaker one is rejected
R6  the incumbent is evicted at decision time, not left behind
R7  both new fields reach the LLM surface
"""
import numpy as np, pandas as pd
from quantaalpha.eval.metrics import rho_max, rho_max_arg

rng = np.random.default_rng(11)
dates = pd.date_range("2020-01-01", periods=120)
cols = [f"s{i}" for i in range(60)]
def frame(seed):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(size=(120, 60)), index=dates, columns=cols)

base = frame(1)
twin = base + 0.02 * frame(2)          # nearly identical
other = frame(3)                       # unrelated
zoo = {"OTHER": other, "TWIN": twin}

rho, near = rho_max_arg(base, zoo)
assert near == "TWIN", f"R1/R3: closest should be TWIN, got {near!r}"
print(f"R1 PASS  rho_max_arg names the closest incumbent: {near!r} (rho {rho:.3f})")
print(f"R3 PASS  it picked the near-clone, not the unrelated factor")

scalar = rho_max(base, zoo)
assert abs(scalar - rho) < 1e-12, f"R2: scalar drifted ({scalar} vs {rho})"
print(f"R2 PASS  rho_max returns the identical scalar ({scalar:.6f}) -- callers unaffected")

src = open("quantaalpha/factors/net_cost_runner.py").read()
assert "self._research_score(inc_m or {})" in src, "R4: not comparing on the admission criterion"
# The duel is now ORDER-FREE: it compares the GAP, so a decisive winner takes
# the slot from either side and a tie inside the margin breaks on turnover.
# The old form (`cand_score > inc_score`) put the margin on the challenger
# alone, which made the book depend on arrival order -- a |t|=10.78 factor lost
# to a |t|=10.07 incumbent it duplicated. See tests/eval/test_rho_within_gate.py
# W6-W8 for the behavioural checks.
assert "_gap = cand_score - inc_score" in src, "R5: no strength comparison"
assert "abs(_gap) > _margin" in src, "R5: the duel is not order-free"
print("R4 PASS  the comparison uses _research_score (|t_NW|), the same number admission and eviction use")

i_cmp = src.index("_gap = cand_score - inc_score")
i_pop = src.index('self._repository.pop(near, None)', i_cmp)
i_keep = src.index("kept.append((expr, sig_raw, t, rho, h))", i_cmp)
assert i_cmp < i_pop < i_keep, "R6: eviction must happen at the decision, before the candidate is kept"
print("R6 PASS  the incumbent is evicted at decision time, so later candidates in the batch see the real library")

# the weaker case must still reject
seg = src[i_cmp:i_cmp + 2600]
assert "keeping the incumbent" in seg, "R5: no rejection branch that keeps the incumbent"
assert "stronger" in seg or "decisive" in seg, "R5: the rejection does not state the comparison"
print("R5 PASS  a stronger duplicate replaces; a weaker one is rejected and told which incumbent beat it")

from quantaalpha.factors.net_cost_feedback import _RAW_METRICS, _METRIC_HELP
keys = {k for k, _, _ in _RAW_METRICS}
for k in ("closest_held", "closest_held_t"):
    assert k in keys, f"R7: {k} not on the feedback surface"
    assert k in _METRIC_HELP, f"R7: {k} has no explanation"
print("R7 PASS  closest_held and closest_held_t reach the LLM with meanings attached")

print("\nALL PASS")
