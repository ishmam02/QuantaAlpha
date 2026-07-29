#!/usr/bin/env python
"""The paired Arm B − Arm A difference across generation seeds.

This is the only figure in the study that can support a claim about the
*objective* rather than about one lucky trajectory.

The reason is measured, not assumed. An unchanged control arm re-run under an
identical configuration moved 4.05pp of annualised return and 0.0128 of IC --
roughly nine and a hundred times the combiner-seed spread that the per-run
tables report as their error bars. Any single-arm number therefore carries
uncertainty two orders of magnitude larger than its own ± suggests.

That variance is common-mode: between the same two runs the two arms moved
almost identically (IC −0.0128 and −0.0125), so it cancels under differencing.
The paired difference reproduced to 0.39pp on return and 0.0003 on IC. So the
paired difference is the estimator, and its spread across generation seeds is
its uncertainty -- which is what this script reports.

Usage::

    python scripts/qa_paired_summary.py --stamps <run>/stamps.txt
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# "| Arm A (mined) | +0.0524 ± 0.0001 | ... | +7.91% ± 0.47% | ..."
ROW = re.compile(r"^\|\s*(Arm [AB]) \(mined\)\s*\|(.+)$")
NUM = re.compile(r"([+-]?\d+\.?\d*)\s*%?")


def parse_flat(path: Path) -> dict[str, dict[str, float]]:
    """Pull each arm's headline row out of a flat-fee comparison table."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|") if c.strip()]
        vals = []
        for c in cells[:7]:
            g = NUM.search(c)
            v = float(g.group(1)) if g else float("nan")
            vals.append(v / 100.0 if "%" in c else v)
        if len(vals) >= 6:
            out[m.group(1)] = {"ic": vals[0], "rank_ic": vals[1], "arr": vals[4], "ir": vals[5]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamps_path = Path(args.stamps)
    if not stamps_path.exists():
        print(f"no stamps file at {stamps_path}; nothing to summarise")
        return 1

    pairs: list[tuple[str, dict]] = []
    for line in stamps_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        seed, stamp = parts
        flat = parse_flat(ROOT / f"data/results/backtest_v2_comparison_{stamp}.md")
        if "Arm A" in flat and "Arm B" in flat:
            pairs.append((seed, {k: flat["Arm B"][k] - flat["Arm A"][k] for k in flat["Arm A"]}))

    if not pairs:
        print("no complete pairs found — was the flat-fee comparison produced for each seed?")
        return 1

    lines = ["# Paired difference (Arm B − Arm A) across generation seeds", "",
             f"{len(pairs)} paired invocation(s). Each pair mined BOTH arms in one run, "
             "which is what makes run-to-run generation variance cancel.", "",
             "| Seed | Δ IC | Δ Rank IC | Δ Ann. return | Δ Info ratio |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for seed, d in pairs:
        lines.append(f"| {seed} | {d['ic']:+.4f} | {d['rank_ic']:+.4f} | "
                     f"{100*d['arr']:+.2f}% | {d['ir']:+.4f} |")

    lines += ["", "## Is the effect real?", ""]
    for label, key, pct in (("IC", "ic", False), ("Rank IC", "rank_ic", False),
                            ("Annualised return", "arr", True), ("Information ratio", "ir", False)):
        vals = [d[key] for _, d in pairs if d[key] == d[key]]
        if not vals:
            continue
        mean = st.fmean(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else float("nan")
        fmt = (lambda v: f"{100*v:+.2f}%") if pct else (lambda v: f"{v:+.4f}")
        if len(vals) < 2:
            verdict = "single pair — no uncertainty estimate"
        elif sd == 0:
            verdict = "identical across seeds"
        elif abs(mean) > 2 * sd:
            verdict = f"**{abs(mean)/sd:.1f}× its own spread — survives**"
        else:
            verdict = f"**{abs(mean)/sd:.1f}× its own spread — NOT resolvable**"
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        lines.append(f"- **{label}**: mean {fmt(mean)}"
                     + (f", sd {fmt(sd).lstrip('+')}" if len(vals) > 1 else "")
                     + f" — {verdict}"
                     + (f"; all {len(vals)} pairs agree in sign" if same_sign and len(vals) > 1 else ""))

    if len(pairs) < 3:
        lines += ["", f"**Only {len(pairs)} pair(s).** Two pairs agreeing in sign is what one "
                  "coin flip in four produces by chance; the informative part is whether the "
                  "MAGNITUDE is consistent. Three to five pairs make the spread meaningful."]

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
