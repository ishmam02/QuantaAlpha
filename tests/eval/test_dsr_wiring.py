"""Deflated Sharpe wired into the run, not just built and tested.

`defense.py` had DSR and PBO implemented and tested, and was imported by
NOTHING -- the correction existed for the final claim and never for the run.

D1  a null strategy found by a large search does NOT clear the DSR bar
D2  the same Sharpe is worth less as the trial count grows
D3  the runner charges the HONEST trial count (every factor scored)
D4  the return series is POPPED -- it must never reach the LLM or the ledger
D5  DSR is computed only where a book exists (not on skipped batches)
D6  DSR reaches both LLM surfaces
"""
import numpy as np
from quantaalpha.eval.defense import deflated_sharpe_ratio

rng = np.random.default_rng(3)

# --- D1: a lucky null ---
paths = [rng.normal(0, 0.01, 750) for _ in range(200)]
best = max(paths, key=lambda r: r.mean() / r.std(ddof=1))
sr = best.mean() / best.std(ddof=1) * np.sqrt(252)
d = deflated_sharpe_ratio(best, n_trials=200)
assert d.dsr < 0.95, f"D1: a null found by 200 trials cleared the bar (dsr={d.dsr:.3f})"
print(f"D1 PASS  best-of-200 pure noise: annualized Sharpe {sr:+.2f}, DSR {d.dsr:.3f} < 0.95")

# --- D2: monotone in trials ---
d1 = deflated_sharpe_ratio(best, n_trials=2).dsr
d2 = deflated_sharpe_ratio(best, n_trials=2000).dsr
assert d2 < d1, f"D2: more trials must deflate further ({d1:.3f} -> {d2:.3f})"
print(f"D2 PASS  same path, 2 trials DSR {d1:.3f} -> 2000 trials DSR {d2:.3f}")

# --- D3/D4/D5: the wiring in the runner ---
src = open("quantaalpha/factors/net_cost_runner.py").read()
assert "from quantaalpha.eval.defense import deflated_sharpe_ratio" in src, "D3: not imported"
assert 'n_trials = max(len(getattr(self, "_t_history", []) or []), 1)' in src, \
    "D3: the trial count is not the number of factors actually scored"
print("D3 PASS  the runner charges DSR the number of factors it actually scored")

assert 'res.pop("_net_return_series", None)' in src, "D4: the series is not popped"
i_pop = src.index('res.pop("_net_return_series"')
i_use = src.index("deflated_sharpe_ratio(series", i_pop)
assert i_pop < i_use, "D4: popped after use"
from quantaalpha.factors.net_cost_feedback import _RAW_METRICS
keys = {k for k, _, _ in _RAW_METRICS}
assert "_net_return_series" not in keys, "D4: the raw series is on the LLM surface"
print("D4 PASS  the series is popped before use and is not on any metric surface")

assert "if series is not None and not skip_book:" in src, "D5: DSR runs on skipped batches"
print("D5 PASS  DSR is computed only where a book was actually priced")

assert "dsr" in keys and "dsr_n_trials" in keys, "D6: DSR missing from feedback"
from quantaalpha.pipeline.evolution.llm_diagnosis import _METRIC_GLOSS
assert "dsr" in {k for k, _, _, _ in _METRIC_GLOSS}, "D6: DSR missing from the gloss"
print("D6 PASS  DSR reaches both LLM surfaces")

# The operator must actually attach it, or D3-D5 guard nothing.
op = open("quantaalpha/eval/operator.py").read()
assert 'metrics["_net_return_series"] = r_net' in op, "D7: the operator never attaches the series"
print("D7 PASS  the operator attaches the return path the DSR needs")

print("\nALL PASS")

# D8: the series must survive `evaluate`'s `m_` prefixing under the exact key
# the runner pops. Left unhandled it becomes `m__net_return_series`, the pop
# misses, and a full return series rides into the ledger JSON.
op = open("quantaalpha/eval/operator.py").read()
i_lift = op.index('_ret_series = metrics.pop("_net_return_series"')
i_prefix = op.index('result: dict = {f"m_{k}": v for k, v in metrics.items()}')
assert i_lift < i_prefix, "D8: the series is prefixed before it is lifted out"
assert 'result["_net_return_series"] = _ret_series' in op, "D8: not re-attached unprefixed"
assert 'result: dict = {f"m__net_return_series' not in op
print("D8 PASS  the series is lifted out before m_-prefixing, under the key the runner pops")
