#!/usr/bin/env python
"""Benchmark reference returns for the report's ARR tables.

Three conventions appear in the report:
  csi300_price      SH000300 price return -- what Backtest v2 subtracts (dividends EXCLUDED)
  csi300_total      SH000300 price + estimated dividend yield
  universe_ew_total equal-weight universe, total return -- what the corrected
                    full-cost path subtracts

IMPORTANT: the equal-weight series is constructed from the universe mask over the
range it is LOADED for, so loading 2016-2026 and slicing to 2022-2025 gives a
different composition (+4.89%/yr) than loading 2022-2025 directly (+1.49%/yr) over
the identical 966 days. Every window below is therefore loaded on its own range, so
each figure matches the evaluation that used that window.
"""
from __future__ import annotations
import json, sys, warnings
from dataclasses import replace
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantaalpha.eval.protocol import load_protocol
from quantaalpha.eval.data import load_benchmark

TD = 243
def ann(r):
    r = pd.Series(r).astype(float).dropna()
    return float((1 + r).prod() ** (TD / len(r)) - 1) if len(r) else float("nan")

base = load_protocol("quantaalpha/eval/protocol_csi300.yaml")
VARIANTS = [("price", "index", "csi300_price"),
            ("estimated_total", "index", "csi300_total"),
            ("estimated_total", "equal", "universe_ew_total")]
WINDOWS = {"2022_2025": ("2022-01-01", "2025-12-26"),
           "2016_2026": ("2016-01-01", "2026-01-09")}

out = {"note": "each window loaded on its own range; the equal-weight series depends "
               "on the load range via the universe mask"}
for basis, constr, key in VARIANTS:
    th = replace(base, benchmark_basis=basis, benchmark_construction=constr)
    rec = {}
    for wname, (a, b) in WINDOWS.items():
        s = load_benchmark(th, a, b)
        s.index = pd.to_datetime(s.index)
        rec[wname] = ann(s)
        if wname == "2016_2026":                       # per-year off the long load
            rec["per_year"] = {str(y): ann(s[s.index.year == y]) for y in range(2016, 2027)}
            rec["2017_2025"] = ann(s["2017-01-01":"2025-12-31"])
    out[key] = rec
    py = rec["per_year"]
    print(f"{key:<20} " + " ".join(f"{y}:{100*py[str(y)]:+.1f}%" for y in range(2016, 2026)))
    print(f"{'':<20} 2022-25 {100*rec['2022_2025']:+.2f}% (own load) | "
          f"2017-25 {100*rec['2017_2025']:+.2f}% | 2016-26 {100*rec['2016_2026']:+.2f}%")
Path("data/results/report_benchmark_ref.json").write_text(json.dumps(out, indent=2))
print("wrote data/results/report_benchmark_ref.json")
