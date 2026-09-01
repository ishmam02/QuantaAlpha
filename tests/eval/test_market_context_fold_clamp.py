"""The disclosed research window must end BEFORE the earliest fold's validation.

``_market_context`` discloses ``Θ.splits.train`` to the planner so it knows which
market and era it is mining. That is leak-free at ``folds=1``, because fold 1's
validation window IS ``Θ.splits.valid``, which starts after ``train_end``.

It is NOT leak-free above that. ``Θ.splits.train`` is only the LAST fold's
training span; earlier folds validate on periods carved out of it. Measured on
protocol_csi300_meanvar_soft_linear at folds=3:

    fold 1  train 2005-01-01..2012-12-09   valid 2012-12-31..2015-12-31
    fold 2  train 2005-01-01..2015-12-10   valid 2016-01-01..2018-12-31
    fold 3  train 2005-01-01..2018-12-10   valid 2019-01-01..2021-12-31

The prompt said "Research window you may reason about: 2005-01-01 to
2018-12-31" while folds 1 and 2 select on 2012-2018 -- six years of the
selection window handed to the model as free research material, with an
explicit invitation to reason from it. A factor motivated by knowledge of
2015 then looks excellent on the fold that validates on 2015.

M1  folds=1 is byte-identical to the pre-clamp behaviour (strict generalisation)
M2  folds>1 clamps the disclosure to the day before the earliest validation day
M3  the STRICT KNOWLEDGE CUTOFF text uses the clamped date, not train_end
M4  no validation or holdout year is ever named, at any fold count
M5  a clamp failure fails CLOSED (no disclosure) rather than leaking the window
"""
import importlib
import os
import pathlib
import re
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTO = ROOT / "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml"
SRC = PROTO.read_text()


def _render(folds):
    """Render the market-context block with walk_forward.folds = `folds`."""
    y = yaml.safe_load(SRC)
    y["walk_forward"]["folds"] = folds
    y["walk_forward"]["enabled"] = True
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(y, fh)
    fh.close()
    os.environ["QA_PROTOCOL"] = fh.name
    import quantaalpha.pipeline.planning as P
    importlib.reload(P)
    return P._market_context()


def _window(ctx):
    line = [l for l in ctx.splitlines() if "Research window" in l][0]
    return re.findall(r"\d{4}-\d{2}-\d{2}", line)


# ---------------------------------------------------------------------------
# M1: folds=1 unchanged -- the fix must be a strict generalisation.
# ---------------------------------------------------------------------------
_cfg = yaml.safe_load(SRC)["splits"]
CFG_TRAIN = tuple(_cfg["train"])          # read from Θ, not hardcoded: the split
CFG_VALID = tuple(_cfg["valid"])          # is re-cut per study and this test must
CFG_TEST = tuple(_cfg["final_test"])      # keep testing the LEAK, not the dates.

ctx1 = _render(1)
start1, end1 = _window(ctx1)
assert (start1, end1) == CFG_TRAIN, (
    f"M1: folds=1 must disclose the configured train window {CFG_TRAIN}; "
    f"got {start1}..{end1}")
print(f"M1 PASS  folds=1 discloses the configured train window {start1}..{end1}")

# ---------------------------------------------------------------------------
# M2: folds=3 clamps to the day before the earliest fold's validation start.
# ---------------------------------------------------------------------------
import quantaalpha.pipeline.planning as _P  # noqa: E402  (reloaded by _render)
from quantaalpha.eval.protocol import load_protocol  # noqa: E402

ctx3 = _render(3)
start3, end3 = _window(ctx3)
theta = load_protocol(os.environ["QA_PROTOCOL"])
folds = theta.splits.walk_forward_folds(3)
import pandas as pd  # noqa: E402
earliest = min(pd.Timestamp(v[0]) for _, v in folds)
want = (earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
assert end3 == want, (
    f"M2: folds=3 must clamp to {want} (day before the earliest validation day "
    f"{earliest.date()}); got {end3}")
assert end3 < end1, "M2: the clamp must actually narrow the disclosure"
assert start3 == "2005-01-01", "M2: the start date must not move"
print(f"M2 PASS  folds=3 clamps {end1} -> {end3} (earliest valid {earliest.date()})")

# ---------------------------------------------------------------------------
# M3: the knowledge-cutoff prose must move with it. Disclosing a narrow window
# but telling the model its knowledge ends on train_end re-opens the leak in
# the very paragraph meant to close it.
# ---------------------------------------------------------------------------
cut = set(re.findall(r"knowledge ENDS on (\d{4}-\d{2}-\d{2})", ctx3))
assert cut == {end3}, f"M3: the cutoff date must be the clamped date {end3}; got {cut}"
assert end1 not in ctx3, (
    f"M3: the unclamped train end {end1} must not appear anywhere in the block")
print("M3 PASS  the STRICT KNOWLEDGE CUTOFF uses the clamped date")

# ---------------------------------------------------------------------------
# M4: no validation or holdout period is ever named, at any fold count.
# ---------------------------------------------------------------------------
v_start, v_end = theta.splits.valid
t_start, t_end = theta.splits.final_test
# Every year from the first validation year to the last test year is off limits.
_banned_years = {str(y) for y in range(int(v_start[:4]), int(t_end[:4]) + 1)}
for n, ctx in ((1, ctx1), (3, ctx3)):
    for banned in (v_start, v_end, t_start, t_end):
        assert banned not in ctx, f"M4: folds={n} names {banned} (valid/holdout)"
    for yr in sorted(_banned_years):
        assert yr not in ctx, f"M4: folds={n} names the year {yr} (valid/test)"
print("M4 PASS  no validation/holdout date or year is disclosed at any fold count")

# ---------------------------------------------------------------------------
# M5: a clamp failure must FAIL CLOSED. If the fold arithmetic raises, the
# block must be empty -- not fall back to disclosing the wider train window.
# ---------------------------------------------------------------------------
y = yaml.safe_load(SRC)
y["walk_forward"]["folds"] = 3
y["walk_forward"]["enabled"] = True
fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(y, fh)
fh.close()
os.environ["QA_PROTOCOL"] = fh.name
import quantaalpha.pipeline.planning as P  # noqa: E402
importlib.reload(P)

_orig = type(theta.splits).walk_forward_folds
try:
    def _boom(self, n, horizon=1):
        raise RuntimeError("fold arithmetic exploded")
    type(theta.splits).walk_forward_folds = _boom
    ctx_fail = P._market_context()
    assert ctx_fail == "", (
        "M5: a clamp failure must disclose NOTHING; it returned a block that "
        f"may carry the unclamped window: {ctx_fail[:120]!r}")
    print("M5 PASS  a clamp failure fails closed (no window disclosed)")
finally:
    type(theta.splits).walk_forward_folds = _orig
    os.environ.pop("QA_PROTOCOL", None)

print("\nALL PASS")
