"""The remine split must put the post-decay regime ENTIRELY in test.

The rule: "the date from the hard decay onwards will be the new test range and
all the old date range is the train and valid."

R1  test starts exactly on the hard-decay date and runs to the last day
R2  train + valid cover everything before it, contiguously, with no gap and no
    overlap -- "all the old date range", nothing discarded
R3  NO train or validation day is on or after the decay date (the whole point:
    a remine must not see the regime it is being asked to survive)
R4  valid is large enough that the gate can still admit on merit
R5  the fold arithmetic never reaches into test
R6  a decay date too early to leave a trainable history is refused, not silently
    shrunk into a useless split
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# The interpreter running the tests is by definition the one with the project's
# dependencies installed, so the subprocess reuses it. Naming a conda path here instead
# would pass on this machine and fail on any other.
PY = sys.executable


def run(*args):
    import json
    # The child inherits this process's environment with PYTHONPATH overridden, rather
    # than being handed a scrubbed dict. A minimal env omits HOME, and qlib resolves
    # Path("~/.cache/...").expanduser() at import, which raises "Could not determine
    # home directory" -- so the split script died before parsing an argument.
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([PY, str(ROOT / "scripts/qa_remine_split.py"), *args],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    if r.returncode != 0:
        return None, r.stderr
    # stdout is the json block
    txt = r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1]
    return json.loads(txt), r.stderr


HD = "2019-07-31"
s, err = run("--hard-decay", HD)
assert s is not None, f"remine split failed: {err[-500:]}"

# ---------------------------------------------------------------------------
# R1 -- test opens ON the decay date
# ---------------------------------------------------------------------------
assert s["final_test"][0] == HD, (
    f"R1: test must start on the hard-decay date {HD}; got {s['final_test'][0]}")
assert pd.Timestamp(s["final_test"][1]) >= pd.Timestamp("2026-01-01"), (
    f"R1: test must run to the end of the data; got {s['final_test'][1]}")
print(f"R1 PASS  test = {s['final_test'][0]}..{s['final_test'][1]} ({s['test_years']}y)")

# ---------------------------------------------------------------------------
# R2 -- train+valid tile everything before the decay date
# ---------------------------------------------------------------------------
tr0, tr1 = map(pd.Timestamp, s["train"])
va0, va1 = map(pd.Timestamp, s["valid"])
hd = pd.Timestamp(HD)
assert tr0 <= tr1 < va0 <= va1, f"R2: windows out of order: {s['train']} {s['valid']}"
gap = (va0 - tr1).days
assert gap <= 4, f"R2: {gap}d gap between train end and valid start (weekend at most)"
assert (hd - va1).days <= 4, (
    f"R2: valid ends {(hd - va1).days}d before the decay date -- that history is "
    "being thrown away, but the rule says ALL of it is train+valid")
print(f"R2 PASS  train {s['train'][0]}..{s['train'][1]} + valid {s['valid'][0]}.."
      f"{s['valid'][1]} tile everything before {HD}")

# ---------------------------------------------------------------------------
# R3 -- NOTHING fitted or selected on sees the post-decay regime
# ---------------------------------------------------------------------------
assert tr1 < hd and va1 < hd, (
    f"R3: train ends {tr1.date()}, valid ends {va1.date()} -- both must be "
    f"strictly before {HD}")
print("R3 PASS  no train/valid day is on or after the hard-decay date")

# ---------------------------------------------------------------------------
# R4 -- the gate can still admit on merit, not on window length
# ---------------------------------------------------------------------------
assert s["gate_can_admit"], (
    f"R4: valid is {s['valid_days']}d; a typical IC reaches only "
    f"t={s['valid_t_at_typical_ic']} against a k_sigma={s['k_sigma']} bar -- the "
    "gate would reject for window length, not for anything about the factor")
assert s["valid_t_at_typical_ic"] >= s["k_sigma"]
print(f"R4 PASS  valid {s['valid_days']}d -> t={s['valid_t_at_typical_ic']} "
      f">= k_sigma {s['k_sigma']} (min admittable IC {s['min_admittable_ic']})")

# ---------------------------------------------------------------------------
# R5 -- folds stay inside train+valid
# ---------------------------------------------------------------------------
import yaml, tempfile, os  # noqa: E402
sys.path.insert(0, str(ROOT))
y = yaml.safe_load((ROOT / "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml").read_text())
y["splits"]["train"], y["splits"]["valid"], y["splits"]["final_test"] = (
    s["train"], s["valid"], s["final_test"])
y["walk_forward"]["folds"] = s["folds"]
fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(y, fh); fh.close()
from quantaalpha.eval.protocol import load_protocol  # noqa: E402
th = load_protocol(fh.name)
folds = th.splits.walk_forward_folds(s["folds"])
assert len(folds) >= 1
last_valid = max(pd.Timestamp(v[1]) for _, v in folds)
assert last_valid < hd, (
    f"R5: a fold validates to {last_valid.date()}, at/after the decay date {HD}")
for tr, va in folds:
    assert pd.Timestamp(tr[1]) < hd and pd.Timestamp(va[1]) < hd, (
        f"R5: fold {tr}/{va} reaches into the test regime")
print(f"R5 PASS  all {len(folds)} folds end by {last_valid.date()}, before {HD}")

# ---------------------------------------------------------------------------
# R6 -- an impossibly early decay date is refused
# ---------------------------------------------------------------------------
bad, err2 = run("--hard-decay", "2005-06-01")
assert bad is None, "R6: a decay date with no usable history must be refused"
assert "too early" in err2 or "needs" in err2, f"R6: unclear refusal: {err2[-200:]}"
print("R6 PASS  a decay date too early to remine against is refused, not fudged")

print("\nALL PASS")
