"""Every prompt key the run looks up must exist and must render.

Caught the hard way on 2026-08-21: an edit that rewrote the
`factor_experiment_output_format` block computed its end offset as "the next
top-level key I happened to list" and so deleted `factor_feedback_generation`
along the way. The run launched fine, mined factors fine, and then failed
EVERY batch with `KeyError: 'factor_feedback_generation'` -- the loop caught it
per-task and carried on, so the process stayed alive and healthy-looking while
the entire feedback channel was dead. Nothing in the feedback loop reached the
LLM for the whole run.

P1  every key referenced as `<dict>["<key>"]` in the factor/pipeline source
    exists in the YAML it is read from
P2  every prompt template parses as Jinja (an unbalanced tag renders nothing)
P3  the specific lookups on the live feedback and construction paths resolve
P4  no gate-feedback block PRESCRIBES a remedy or names an operator as the way
    to satisfy it (diagnose-never-prescribe)
"""
import re, yaml
from pathlib import Path
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "prompts": ROOT / "quantaalpha/factors/prompts/prompts.yaml",
    "proposal": ROOT / "quantaalpha/factors/prompts/proposal.yaml",
}
loaded = {n: yaml.safe_load(p.read_text()) for n, p in FILES.items() if p.exists()}
all_keys = set()
for d in loaded.values():
    all_keys |= set(d or {})

# --- P1: static lookups in source must resolve ---
SRC = list((ROOT / "quantaalpha/factors").rglob("*.py")) + \
      list((ROOT / "quantaalpha/pipeline").rglob("*.py"))
pat = re.compile(r'(?:qa_prompt_dict|base_prompt_dict|qa_feedback_prompts|prompt_dict)'
                 r'\[\s*"([A-Za-z0-9_]+)"\s*\]')
referenced = {}
for f in SRC:
    for m in pat.finditer(f.read_text()):
        referenced.setdefault(m.group(1), []).append(f.name)

missing = {k: v for k, v in referenced.items() if k not in all_keys}
assert not missing, (
    "prompt keys referenced in source but ABSENT from the YAML "
    f"(this is exactly the failure that killed the 04:19 run): {missing}")
print(f"P1 PASS  {len(referenced)} referenced prompt keys all resolve "
      f"({len(all_keys)} defined)")

# --- P2: every template is syntactically valid Jinja ---
env = Environment(undefined=StrictUndefined)
bad = []
def walk(node, path):
    if isinstance(node, str):
        try: env.parse(node)
        except TemplateSyntaxError as e: bad.append((path, str(e)))
    elif isinstance(node, dict):
        for k, v in node.items(): walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node): walk(v, f"{path}[{i}]")
for name, d in loaded.items(): walk(d, name)
assert not bad, f"P2: templates that do not parse: {bad}"
print(f"P2 PASS  every prompt template in {len(loaded)} file(s) parses as Jinja")

# --- P3: the two lookups that were live when this broke ---
from quantaalpha.factors.feedback import qa_feedback_prompts
t = qa_feedback_prompts["factor_feedback_generation"]["system"]
assert isinstance(t, str) and len(t) > 500, "P3: feedback system prompt is empty"
from quantaalpha.factors.proposal import qa_prompt_dict
for k in ("hypothesis2experiment", "factor_experiment_output_format",
          "hypothesis_output_format"):
    assert qa_prompt_dict[k], f"P3: {k} missing on the construction path"
print("P3 PASS  the feedback and construction lookups both resolve at import time")

# ---------------------------------------------------------------------------
# P4 -- gate feedback DIAGNOSES, it never PRESCRIBES.
#
# `hypothesis2experiment.user_prompt` used to close its duplication alert with
# "Replace raw variables with transformed variants ... such as using
# `$close/TS_MEAN($close, 10)`". Naming an operator as the way to be novel hands
# the search a construction instead of letting it reason to one, and the named
# operator then gets selected for that reason alone. Measured across 277
# expressions from 5 mines: TS_MEAN took 59% of expressions and 64-83% of the
# window-summarizer slot in every single run.
#
# The check is on the RENDERED prompt with the conditional branch FIRED -- a
# prescription hidden inside `{% if %}` is still shipped to the model, and this
# one fired in the 20260823 push smoke.
# ---------------------------------------------------------------------------
_env = Environment(undefined=StrictUndefined)
_rendered = _env.from_string(
    loaded["prompts"]["hypothesis2experiment"]["user_prompt"]
).render(
    targets="factors", target_hypothesis="h", hypothesis_and_feedback="f",
    function_lib_description="ops", target_list="[]", RAG=None,
    expression_duplication="- Proposed Expression: RANK($close)",
)
_low = re.sub(r"\s+", " ", _rendered.lower())

# A remedy stated as an instruction.
_PRESCRIPTIONS = [
    r"\bsuch as using\b", r"\bplease reduce\b", r"\bexperiment with a mix\b",
    r"\btry using\b", r"\byou should\b", r"\bwe recommend\b",
]
_leaks = [p for p in _PRESCRIPTIONS if re.search(p, _low)]
assert not _leaks, f"P4: the duplication feedback prescribes a remedy: {_leaks}"

# An operator named as the way to satisfy the gate. The neutral glossary and the
# JSON output-format example legitimately contain operator names, so this checks
# only the duplication-alert region -- the part that reacts to a failed gate.
_i = _rendered.index("Alert: Duplication")
_alert = _rendered[_i:]
_named = re.findall(r"\b(TS_[A-Z_]+|CSRANK|CSZSCORE)\s*\(", _alert)
assert not _named, (
    f"P4: the duplication alert names operator(s) {sorted(set(_named))} as the "
    "way to be novel; the search must reason to a construction, not be handed one")
assert "yours to determine" in _alert.lower() or "your judgement" in _alert.lower(), (
    "P4: the duplication alert must hand the decision back to the model")
print("P4 PASS  gate feedback diagnoses only: no remedy, no operator named as the fix")

print("\nALL PASS")
