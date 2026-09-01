"""No training label may be realised inside the window it is validated on.

P1  the purge gap is at least horizon + embargo_days, at every fold
P2  the gap SCALES with the horizon (a 20-day label leaks 20 days, not one)
P3  folds never reach into final_test
P4  folds expand from a common start (later regime, not less data)
P5  the extended window actually buys the regimes it was extended for
"""
import pandas as pd
from quantaalpha.eval.protocol import load_protocol

TH = load_protocol("quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml")
sp = TH.splits
EMB = int(sp.embargo_days)
assert EMB > 0, "embargo_days must be set; purging was entirely absent before this"

test_start = pd.Timestamp(sp.final_test[0])

gaps_by_h = {}
for h in (1, 5, 20):
    folds = sp.walk_forward_folds(3, horizon=h)
    assert len(folds) == 3, f"expected 3 folds at h={h}, got {len(folds)}"
    gaps = []
    for tr, va in folds:
        gap = (pd.Timestamp(va[0]) - pd.Timestamp(tr[1])).days
        gaps.append(gap)
        # P1
        assert gap >= EMB + h, (
            f"P1: h={h} fold {tr[1]} -> {va[0]} leaves only {gap}d, "
            f"needs >= {EMB + h} (embargo {EMB} + horizon {h})")
        # P3
        assert pd.Timestamp(va[1]) < test_start, (
            f"P3: fold valid {va} reaches into final_test ({sp.final_test[0]})")
    gaps_by_h[h] = min(gaps)
    # P4
    assert len({tr[0] for tr, _ in folds}) == 1, "P4: folds must expand, not slide"

print(f"P1 PASS  every fold leaves >= embargo({EMB}) + horizon; "
      f"min gaps {gaps_by_h}")
# P2
assert gaps_by_h[20] > gaps_by_h[1], (
    f"P2: the gap must grow with the horizon; got {gaps_by_h}")
print(f"P2 PASS  gap scales with horizon: h=1 -> {gaps_by_h[1]}d, h=20 -> {gaps_by_h[20]}d")
print("P3 PASS  final_test never touched")
print("P4 PASS  expanding, common start")

# P5: the extension must actually add the stress regimes it was justified by.
tr0 = pd.Timestamp(sp.train[0])
assert tr0 <= pd.Timestamp("2008-01-01"), (
    f"P5: train starts {sp.train[0]}; the extension was justified by the 2008 "
    f"crash and 2015 bubble, and must actually include them")
folds = sp.walk_forward_folds(3, horizon=1)
valid_years = {int(va[0][:4]) for _, va in folds}
assert any(y <= 2015 for y in valid_years), (
    f"P5: no fold validates on or before 2015; valid starts {sorted(valid_years)}")
print(f"P5 PASS  train from {sp.train[0]}; fold valid windows start "
      f"{sorted(valid_years)} (2008 crash and 2015 bubble now in-sample)")

# The purged train window must be strictly shorter than the raw one.
raw_end = pd.Timestamp(sp.train[1])
for h in (1, 20):
    pend = pd.Timestamp(sp.train_window_purged(h)[1])
    assert pend < raw_end, f"train_window_purged(h={h}) did not purge anything"
print("P6 PASS  train_window_purged trims the tail of the training window")

print("\nALL PASS")
