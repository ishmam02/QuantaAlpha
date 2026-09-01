"""`factors_per_hypothesis` must actually govern generation.

It did not. The key was read ONLY by `expected_factor_count`, a budget
estimator with no callers, while the construction prompt hardcoded "2-3 Factors
per Generation". A live run on 2026-08-21 was measured emitting 3 sub-workspaces
per hypothesis with the config asking for 1, and the sent prompt (captured in
the LLM I/O log) contained the literal string "2-3 Factors per Generation".

F1  the prompt no longer hardcodes a count
F2  rendering at 1 asks for exactly 1, in the singular
F3  rendering at 3 asks for 3, in the plural
F4  the render site passes the variable (StrictUndefined would raise otherwise)
F5  the config value reaches the environment the render site reads
"""
import os, re, yaml
from pathlib import Path
from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[2]
tmpl = yaml.safe_load((ROOT / "quantaalpha/factors/prompts/prompts.yaml").read_text()
                      )["hypothesis2experiment"]["system_prompt"]

assert "2-3 Factors per Generation" not in tmpl, "F1: the hardcoded count is still there"
assert "factors_per_hypothesis" in tmpl, "F1: the prompt does not reference the config value"
print("F1 PASS  the prompt no longer hardcodes a factor count")

def render(n):
    return (Environment(undefined=StrictUndefined).from_string(tmpl)
            .render(targets="factors", scenario="s", experiment_output_format="f",
                    factors_per_hypothesis=n))

one = render(1)
assert "exactly 1 factor." in one, f"F2: singular form missing:\n{[l for l in one.splitlines() if 'Generation' in l or 'exactly' in l]}"
assert "1 Factor per Generation" in one, "F2: heading not singular"
assert "2-3" not in one, "F2: a 2-3 survived rendering"
print(f"F2 PASS  at 1: {[l.strip() for l in one.splitlines() if 'exactly' in l][0]!r}")

three = render(3)
assert "exactly 3 factors." in three, "F3: plural form wrong"
assert "3 Factors per Generation" in three, "F3: heading not plural"
print(f"F3 PASS  at 3: {[l.strip() for l in three.splitlines() if 'exactly' in l][0]!r}")

# F4: the live render site must pass the variable, or StrictUndefined raises
src = (ROOT / "quantaalpha/factors/proposal.py").read_text()
assert "factors_per_hypothesis=_factors_per_hypothesis()" in src, \
    "F4: the render site does not pass the variable -- StrictUndefined will raise at runtime"
print("F4 PASS  the live render site passes it (proposal.py)")

# F5: config -> environment -> helper
from quantaalpha.factors.proposal import _factors_per_hypothesis
for want in (1, 3, 7):
    os.environ["QA_FACTORS_PER_HYPOTHESIS"] = str(want)
    got = _factors_per_hypothesis()
    assert got == want, f"F5: env {want} -> helper {got}"
os.environ.pop("QA_FACTORS_PER_HYPOTHESIS", None)
assert _factors_per_hypothesis() == 1, "F5: default must be 1, not the old 2-3 behaviour"

mining = (ROOT / "quantaalpha/pipeline/factor_mining.py").read_text()
assert 'os.environ["QA_FACTORS_PER_HYPOTHESIS"]' in mining, \
    "F5: factor_mining does not export the config value"
assert 'factor_cfg.get("factors_per_hypothesis")' in mining, \
    "F5: factor_mining does not read the config key"
print("F5 PASS  config -> env -> helper round-trips (1/3/7), default 1")

print("\nALL PASS")

# ---------------------------------------------------------------------------
# The EXAMPLE must agree with the instruction. A two-factor worked example next
# to "produce exactly 1 factor" is a contradiction the model resolves in favour
# of the example -- which is what the 2026-08-21 run did.
fmt = yaml.safe_load((ROOT / "quantaalpha/factors/prompts/prompts.yaml").read_text()
                     )["factor_experiment_output_format"]

def render_fmt(n):
    return (Environment(undefined=StrictUndefined).from_string(fmt)
            .render(factors_per_hypothesis=n))

f1 = render_fmt(1)
assert '"factor name 2"' not in f1, "F6: schema still shows a second slot at n=1"
assert "Volume_Range_Correlation_Factor_20D" not in f1, "F6: example still shows 2 factors at n=1"
assert "Normalized_Intraday_Range_Factor_10D" in f1, "F6: the single example vanished"
assert "EXACTLY 1 top-level key," in f1, "F6: the explicit count is missing"
print("F6 PASS  at n=1 the schema and the worked example both show exactly one factor")

f2 = render_fmt(2)
assert '"factor name 2"' in f2 and "Volume_Range_Correlation_Factor_20D" in f2, \
    "F7: at n=2 both slots must return"
assert "EXACTLY 2 top-level keys," in f2
print("F7 PASS  at n=2 both slots and both examples return")

src2 = (ROOT / "quantaalpha/factors/proposal.py").read_text()
assert "render(factors_per_hypothesis=_factors_per_hypothesis())" in src2, \
    "F8: the output format is not rendered with the count"
print("F8 PASS  the live site renders the schema with the configured count")

# The JSON example must stay parseable-looking: balanced braces.
for n, txt in ((1, f1), (2, f2)):
    assert txt.count("{") == txt.count("}"), f"F9: unbalanced braces at n={n}"
print("F9 PASS  brace counts balance at n=1 and n=2")

# The worked example must itself be valid JSON. It was not: the template omitted
# the comma after every `variables` object (4 of them), so the model was shown
# malformed JSON and asked to reply in strict JSON.
import json as _json
for n in (1, 2):
    ex = render_fmt(n)
    ex = ex[ex.index("Here is an example:") + len("Here is an example:"):].strip()
    cleaned = re.sub(r",(\s*[}\]])", r"\1", ex)   # drop illustrative trailing commas
    parsed = _json.loads(cleaned)                 # raises if malformed
    assert len(parsed) == n, f"F10: example shows {len(parsed)} factors at n={n}"
print("F10 PASS  the worked example parses as JSON and holds exactly n factors")
