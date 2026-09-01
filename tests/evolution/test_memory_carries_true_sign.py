"""The failure memory must show the FACTOR'S sign, not the book's.

Measured on `mine_20260821_0438`: the memory block rendered `rank_ic`, which
`_to_series` fills from `m_rank_ic` -- the COMPOSITE book's IC. Across the run's
prompts, 43 of 43 values shown were POSITIVE (median +0.0283) while the factors
themselves realized NEGATIVE 71% of the time (median -0.0154). Directly below
them the prompt says "Read the RankIC SIGN as evidence".

So the single directional signal in the cross-batch memory was inverted, under
an explicit instruction to trust it. Sign calibration came out at 54% against a
71% base rate -- worse than always answering with the majority.

This also settles which of the two candidate explanations was right: memory
REACHED the prompt (89/106 carried prior rejections; 0/106 were told it was
round one), so the loop was not amnesiac. It was misinformed.

M1  _to_series carries the factor's own signed IC under a distinct name
M2  the memory block prefers it and never falls back to the book's number
M3  the aggregate direction is stated once, from measurement
M4  the aggregate is DIAGNOSTIC, not prescriptive (the standing rule)
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

runner = (ROOT / "quantaalpha/factors/net_cost_runner.py").read_text()
assert '"rank_ic_own"' in runner, "M1: the factor's own signed IC is not carried"
i_own = runner.index('"rank_ic_own"')
assert "rank_ic_neutral" in runner[i_own:i_own + 400], "M1: not sourced from the neutralized tear sheet"
print("M1 PASS  _to_series carries rank_ic_own, the factor's own signed neutralized IC")

ctl = (ROOT / "quantaalpha/pipeline/evolution/controller.py").read_text()
i_ric = ctl.index('ric = m.get("rank_ic_own")')
seg = ctl[i_ric:i_ric + 260]
assert 'm.get("rank_ic")' not in seg, "M2: still falls back to the composite book IC"
assert "rank_ic_neutral" in seg, "M2: no per-factor fallback"
print("M2 PASS  the memory block reads the factor's own IC and never the book's")

assert "Direction, measured across this run" in ctl, "M3: the aggregate is never stated"
i_agg = ctl.index("Direction, measured across this run")
# Only the PROMPT TEXT, not the surrounding code comments -- the comments
# legitimately cite the measured 71% that motivated the fix, but nothing
# market-specific may reach the model.
agg = ctl[i_agg:ctl.index("These were measured and REJECTED", i_agg)]
guard = ctl[i_agg - 1200:i_agg]          # the code around it, for the M3 checks
assert "realized a " in agg and "NEGATIVE" in agg, "M3: the aggregate states no direction"
assert "len(signed) >= 8" in guard, "M3: no minimum sample before asserting a base rate"
print("M3 PASS  the aggregate direction is stated once, computed from this run's own measurements")

BANNED = ["you should", "try ", "instead use", "lengthen", "shorten", "prefer ",
          "we recommend", "make sure to", "always use"]
low = agg.lower()
hits = [b for b in BANNED if b in low]
assert not hits, f"M4: prescriptive language in the aggregate block: {hits}"
assert "yours to determine" in agg, "M4: does not leave the action open"
assert "not an assumption" in agg, "M4: does not mark it as measured rather than assumed"
print("M4 PASS  the aggregate diagnoses without prescribing, and leaves the action open")

# it must be computed, never hardcoded -- no market prior may be baked in
for lit in ("71%", "CSI300", "reversal-dominated", "China"):
    assert lit not in agg, f"M4b: a market-specific prior is hardcoded: {lit!r}"
print("M4b PASS  no hardcoded market prior -- the number is computed from the run")

print("\nALL PASS")
