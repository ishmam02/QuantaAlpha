"""The library cap must actually bind, and evict on the ADMISSION criterion.

C1  max_library was declared but INERT -- no code read it. It must now bind.
C2  eviction drops the weakest by |t_nw|, the same number admission used
C3  a stronger newcomer displaces a weaker incumbent (mining more helps)
C4  a weaker newcomer is the one dropped (mining more never dilutes)
C5  members with no research score sort last, not silently strong
C6  the eviction is recorded in the ledger with its rule and scores
"""
from dataclasses import replace
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.factors.net_cost_runner import NetCostFactorRunner as NetCostRunner

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")
CAP = 5


class Stub:
    """Only what _enforce_library_cap touches."""
    _enforce_library_cap = NetCostRunner._enforce_library_cap
    _research_score = NetCostRunner._research_score

    def __init__(self, theta, repo):
        self.theta = theta
        self._repository = repo
        self.ledger = []


def make(theta, scores):
    repo = {f"f{i}": (None, ({} if t is None else {"t_nw": t}))
            for i, t in enumerate(scores)}
    return Stub(theta, repo)


th = replace(TH, admission=replace(TH.admission, max_library=CAP))

# C1 / C2: over the cap -> the weakest go
st = make(th, [8.0, 3.1, 5.5, 3.0, 7.2, 4.4, 9.9])      # 7 members, cap 5
dropped = st._enforce_library_cap()
assert len(st._repository) == CAP, f"C1: cap did not bind; |zoo|={len(st._repository)}"
assert set(dropped) == {"f3", "f1"}, f"C2: wrong members evicted: {dropped}"
kept_t = sorted(abs(m["t_nw"]) for _, (_, m) in st._repository.items())
assert kept_t[0] >= 4.4, f"C2: a weak member survived: {kept_t}"
print(f"C1-C2 PASS  cap binds at {CAP}; evicted the two weakest |t_nw| (3.0, 3.1)")

# C3: a strong newcomer displaces a weaker incumbent
st = make(th, [8.0, 3.1, 5.5, 4.0, 7.2])                 # exactly at cap
st._repository["new"] = (None, {"t_nw": 6.6})            # a better factor arrives
dropped = st._enforce_library_cap()
assert dropped == ["f1"], f"C3: expected the weakest (3.1) out, got {dropped}"
assert "new" in st._repository, "C3: the stronger newcomer must be kept"
print("C3 PASS  a stronger newcomer displaces the weakest incumbent")

# C4: a weak newcomer is itself the one dropped -- mining more never dilutes
st = make(th, [8.0, 5.1, 5.5, 4.0, 7.2])
st._repository["weak"] = (None, {"t_nw": 3.2})
dropped = st._enforce_library_cap()
assert dropped == ["weak"], f"C4: the weak newcomer should be dropped, got {dropped}"
print("C4 PASS  a weaker newcomer is dropped; the library cannot degrade")

# C5: unscored members sort last
st = make(th, [8.0, 5.1, 5.5, 4.0, 7.2, None])
dropped = st._enforce_library_cap()
assert dropped == ["f5"], f"C5: the unscored member should go first, got {dropped}"
print("C5 PASS  members with no research score sort last, not silently strong")

# C6: auditable
st = make(th, [8.0, 3.1, 5.5, 3.0, 7.2, 4.4, 9.9])
st._enforce_library_cap()
assert len(st.ledger) == 1, "C6: eviction not recorded"
rec = st.ledger[0]
assert rec["eviction_rule"] == "library_cap" and rec["eviction_bar"] == CAP
assert set(rec["eviction_scores"]) == {"f1", "f3"}, f"C6: {rec['eviction_scores']}"
print("C6 PASS  eviction recorded with rule, bar and per-factor scores")

# Under the cap: nothing happens.
st = make(th, [8.0, 5.1])
assert st._enforce_library_cap() == [], "must not evict below the cap"
# Cap disabled: nothing happens.
st0 = make(replace(TH, admission=replace(TH.admission, max_library=0)), [1.0]*50)
assert st0._enforce_library_cap() == [], "max_library=0 must disable the cap"
print("C7 PASS  inert below the cap and when disabled")

print("\nALL PASS")
