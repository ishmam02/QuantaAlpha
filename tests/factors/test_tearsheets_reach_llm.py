"""The per-factor tear sheets must survive the whole path to the prompt.

Measured on the 2026-08-21 run: 243 LLM calls, ZERO carrying "Neutralized
RankIC", "Direction your hypothesis committed to", or any other field the
standalone gate judges on. Both ends were correct and the middle silently
dropped them:

    _decide_standalone  -> res["factor_tearsheets"]        (published)
    _to_series(res)     -> pd.Series of an ALLOWLIST       <-- dropped here
    _as_dict(result)    -> payload
    _per_factor_lines   -> the rendered block              (consumed)

`_to_series` is an explicit scalar allowlist, so anything not named in it
vanishes one step before the summarizer. The generator was scored on the
research gate and told only batch aggregates plus a reason string -- precisely
the "one uninformative scalar" the diagnosis layer exists to replace.

T1  `_to_series` names factor_tearsheets and admitted_exprs
T2  a dict survives the Series round-trip that `_as_dict` performs
T3  the rendered block contains the falsification verdict
T4  every metric the gate can write is renderable (no silent drops)
"""
import re

import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

src = (ROOT / "quantaalpha/factors/net_cost_runner.py").read_text()
i_ser = src.index("def _to_series")
# The body ends at the next method, not at a fixed character count. A magic slice silently
# turns into a false failure the moment the method outgrows it -- which is what happened
# here: _to_series passed 9000 chars and the key it names last fell outside the window.
_next = re.search(r"\n    def ", src[i_ser + 10:])
body = src[i_ser: i_ser + 10 + _next.start()] if _next else src[i_ser:]
for key in ('"factor_tearsheets"', '"admitted_exprs"'):
    assert key in body, f"T1: {key} is not named in _to_series -- it will be dropped"
print("T1 PASS  _to_series names factor_tearsheets and admitted_exprs")

from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback as FB
sheets = {"Sub($close,Mean($close,20))": {
    "rank_ic_neutral": -0.0312, "t_nw": -3.65, "best_horizon": 5,
    "ic_pos_frac": 0.54, "monotonicity": 0.61, "rho_max": 0.42,
    "sign_predicted": "positive", "sign_realized": "negative",
    "mechanism_validated": False, "fdr_t_required": 3.40, "fdr_n_tests": 40,
    "capacity_cny": 4.2e8, "turnover_solo": 0.18, "exposure_size": -0.31}}
ser = pd.Series({"U": 0.55, "factor_tearsheets": sheets, "admitted_exprs": []})
payload = FB._as_dict(ser)
got = payload.get("factor_tearsheets")
assert isinstance(got, dict) and got, "T2: the sheets did not survive Series -> _as_dict"
print("T2 PASS  a nested dict survives the Series round-trip _as_dict performs")

class Stub:
    _fmt = staticmethod(FB._fmt)
    _per_factor_lines = FB._per_factor_lines
block = "\n".join(Stub()._per_factor_lines(payload))
for probe in ("Direction your hypothesis committed to: positive",
              "Direction the measurement actually produced: negative",
              "Did the measurement confirm your stated mechanism: NO"):
    assert probe in block, f"T3: missing from the rendered block: {probe!r}"
print("T3 PASS  the falsification verdict renders into the block the LLM receives")

# T4: nothing the gate writes may silently vanish
from quantaalpha.factors.net_cost_feedback import _RAW_METRICS
renderable = {k for k, _, _ in _RAW_METRICS}
written = set().union(*(set(v) for v in sheets.values()))
missing = [k for k in written if k not in renderable]
assert not missing, f"T4: gate writes these but the surface cannot render them: {missing}"
print(f"T4 PASS  all {len(written)} gate-written metrics are renderable")

print("\nALL PASS")
