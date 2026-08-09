#!/usr/bin/env python
"""Admission rate by round -- the direct test of whether U's feedback teaches.

The claim under test is that rejecting a batch on ``U`` and telling the
generator *why* raises the quality of what it proposes next. That predicts one
thing specifically: **the admission rate should rise across rounds.** If it is
flat, the feedback is not teaching, whatever else the arm achieves.

Reading this honestly needs one caveat that the numbers cannot supply. Feedback
can only reach a batch generated *after* the verdict, so batches produced
concurrently inside a round cannot learn from each other. Under parallel
evolution a flat admission rate is uninformative rather than negative evidence
-- it means the mechanism never had a chance. The report says so when it
detects that batches share a repository snapshot, which is the signature of
concurrent generation.

Usage::

    python scripts/qa_analyze_ledger.py data/results/ledger_treatment_<stamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(Path(args.ledger))
    evals = [r for r in rows if "admitted" in r]
    evictions = [r for r in rows if r.get("evicted_exprs")]
    if not evals:
        print("No evaluations with an admission decision in this ledger.")
        print("(Ledgers written before admission gating carry no 'admitted' field.)")
        return 1

    lines = [f"# Admission trajectory — `{Path(args.ledger).name}`", ""]

    # Concurrency detection. Equal zoo sizes are NOT evidence on their own: a
    # rejected batch leaves the repository unchanged, so the next batch
    # legitimately sees the same size under strictly sequential execution.
    # Simulate instead -- track the size each batch SHOULD have seen given the
    # verdicts before it, and flag only those that saw less, since that means
    # they were evaluated before an earlier admission had landed.
    expected, concurrent = 0, 0
    for r in evals:
        if r.get("zoo_size", 0) < expected:
            concurrent += 1
        if r.get("admitted"):
            expected += int(r.get("n_factors", 0) or 0)

    lines += [
        f"{len(evals)} evaluated batch(es), "
        f"{sum(1 for r in evals if r['admitted'])} admitted, "
        f"{sum(1 for r in evals if not r['admitted'])} rejected, "
        f"{len(evictions)} eviction event(s).",
        "",
        "| # | zoo before | U | verdict | net ARR |",
        "| ---: | ---: | ---: | :--- | ---: |",
    ]
    for i, r in enumerate(evals, 1):
        m = r.get("metrics") or {}
        arr = m.get("net_arr")
        arr_s = f"{100 * float(arr):+.2f}%" if isinstance(arr, (int, float)) else "—"
        u = r.get("U")
        u_s = f"{float(u):.4f}" if isinstance(u, (int, float)) else "—"
        lines.append(f"| {i} | {r.get('zoo_size', '—')} | {u_s} | "
                     f"{'ADMIT' if r['admitted'] else '**reject**'} | {arr_s} |")

    # --- THE learning test, when the ledger carries it ---------------------
    deltas = [(i, r.get("delta_mean")) for i, r in enumerate(evals)
              if isinstance(r.get("delta_mean"), (int, float))
              and r.get("delta_mean") == r.get("delta_mean")]
    if len(deltas) >= 6:
        xs = [i for i, _ in deltas]
        ys = [d for _, d in deltas]
        n = len(ys)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
        se = ((sum(r * r for r in resid) / (n - 2)) ** 0.5 / sxx ** 0.5) if n > 2 and sxx else float("inf")
        t = b / se if se not in (0.0, float("inf")) else 0.0
        h = n // 2
        lines += [
            "", "## Does the generator improve? (the actual test)", "",
            "Marginal contribution per batch -- what the candidates themselves add "
            "to the book, independent of how large the repository has grown. This is "
            "the statistic that answers the question; the admission rate below does "
            "not, and has been actively misleading.",
            "",
            f"- First half: **{sum(ys[:h])/max(h,1):+.5f}** mean contribution",
            f"- Second half: **{sum(ys[h:])/max(n-h,1):+.5f}**",
            f"- Slope: **{b:+.6f}** per batch, t = **{t:+.2f}** over {n} measured batches",
            "",
        ]
        if t > 2:
            lines.append("**The search is learning.** Later batches contribute more, and "
                         "the trend is larger than its own standard error.")
        elif t < -2:
            lines.append("**The search is degrading.** Later batches contribute *less*.")
        else:
            lines.append("**No resolvable trend.** The generator is proposing at a "
                         "constant quality: selection is working on a stationary "
                         "distribution rather than an improving one. That is the "
                         "state measured before, at t=+0.42 over 79 batches.")
    elif any("delta_mean" in r for r in evals):
        lines += ["", "## Does the generator improve?", "",
                  f"Only {len(deltas)} batch(es) carry a measured contribution -- too few "
                  "for a trend. Bootstrap batches record NaN by design."]
    else:
        lines += ["", "## Does the generator improve?", "",
                  "**This ledger predates the marginal-contribution record.** It carries "
                  "no `delta_mean`, so the learning question cannot be answered from it "
                  "-- only the admission rate below, which is not the same question."]

    # --- is the REPOSITORY improving, not just the candidates? -------------
    # delta_mean above asks whether each new batch is good. This asks whether
    # the thing being built is getting better, which is not the same question:
    # a stream of mildly-positive batches raises the book, and a stream of
    # rejected ones leaves it exactly where it was. base_net_ir is the book
    # WITHOUT the candidates -- the repository's own quality at that moment.
    books = [(i, (r.get("metrics") or {}).get("base_net_ir"))
             for i, r in enumerate(evals)]
    books = [(i, float(v)) for i, v in books
             if isinstance(v, (int, float)) and v == v]
    if len(books) >= 4:
        ys = [v for _, v in books]
        h = len(ys) // 2
        first_b, last_b = ys[0], ys[-1]
        lines += [
            "", "## Is the repository itself improving?", "",
            "`base_net_ir` is the book built from the repository ALONE, so this "
            "tracks the asset being accumulated rather than the quality of any "
            "one proposal.",
            "",
            f"- First measured: **{first_b:+.4f}**   latest: **{last_b:+.4f}**   "
            f"change **{last_b - first_b:+.4f}**",
            f"- First half mean **{sum(ys[:h])/max(h,1):+.4f}**, "
            f"second half **{sum(ys[h:])/max(len(ys)-h,1):+.4f}**",
            f"- Range over the run: {min(ys):+.4f} .. {max(ys):+.4f}",
        ]
        if last_b > first_b:
            lines.append("")
            lines.append("The book is better than it started. Accumulation is working.")
        elif abs(last_b - first_b) < 1e-9:
            lines += ["", "**The book has not moved.** Either nothing has been "
                      "admitted since the first measurement, or admissions are "
                      "cancelling out."]
        else:
            lines += ["", "**The book is WORSE than it started.** Admitted batches "
                      "are not accumulating into a better repository -- check "
                      "whether bootstrap admissions (which are untested by design) "
                      "are still sitting in it."]

    # --- is eviction doing its job? ----------------------------------------
    lines += ["", "## Is eviction removing bad factors?", ""]
    if not evictions:
        adm_batches = sum(1 for r in evals if r["admitted"])
        lines += [
            f"No eviction has fired yet. `_prune` runs only on an ADMITTED batch, "
            f"and {adm_batches} have been admitted so far.",
            "",
            "If `evict_every` is 0 this is not a cadence matter -- eviction is off "
            "entirely and the repository can only grow.",
        ]
    else:
        total_out = sum(len(r.get("evicted_exprs") or []) for r in evictions)
        lines += [f"{len(evictions)} eviction event(s), {total_out} factor(s) removed.", ""]
        wrong = []
        for n, r in enumerate(evictions, 1):
            scores = r.get("eviction_scores") or {}
            allc = r.get("repository_contributions") or {}
            bar = r.get("eviction_bar")
            vals = [v for v in scores.values() if isinstance(v, (int, float))]
            lines.append(
                f"**Event {n}** — rule `{r.get('eviction_rule', '?')}`, bar "
                f"{bar}, removed {len(r.get('evicted_exprs') or [])}"
                + (f", contributions {min(vals):+.5f} .. {max(vals):+.5f}" if vals else ""))
            # The check that matters: everything removed should be BELOW the bar.
            if isinstance(bar, (int, float)):
                bad = [v for v in vals if v >= bar]
                if bad:
                    wrong.append((n, bad))
            if allc:
                neg = sum(1 for v in allc.values()
                          if isinstance(v, (int, float)) and v < 0)
                lines.append(
                    f"    repository at that moment: {len(allc)} members, {neg} with "
                    f"negative contribution, {neg - len(vals)} of those KEPT by the "
                    f"min_size floor." if neg >= len(vals) else
                    f"    repository at that moment: {len(allc)} members, {neg} negative.")
        lines.append("")
        if wrong:
            lines.append(f"**Eviction removed {sum(len(b) for _, b in wrong)} factor(s) "
                         f"at or ABOVE the bar** — it is not removing what it claims to.")
        else:
            lines.append("Every evicted factor scored below the bar: eviction is "
                         "removing measured-harmful members, not arbitrary ones.")

    # --- is the search stuck? ----------------------------------------------
    streak, longest, cur = 0, 0, 0
    for r in evals:
        cur = 0 if r["admitted"] else cur + 1
        longest = max(longest, cur)
    streak = cur
    lines += ["", "## Is the search stuck?", "",
              f"- Current consecutive rejections: **{streak}**",
              f"- Longest streak this run: **{longest}**"]
    ts = [r.get("ts") for r in evals if r.get("ts")]
    if ts:
        import datetime as _dt
        last_adm = [r.get("ts") for r in evals if r["admitted"] and r.get("ts")]
        if last_adm:
            gap = (_dt.datetime.fromisoformat(ts[-1])
                   - _dt.datetime.fromisoformat(last_adm[-1])).total_seconds() / 60
            lines.append(f"- Since the last admission: **{gap:.0f} min** of evaluated batches")
    if streak >= 15:
        lines += ["", "**STALLED.** The recorded stall ran 33 consecutive rejections "
                  "on a bootstrap-luck ratchet. Under marginal-contribution admission "
                  "the bar cannot ratchet, so a streak this long means the generator "
                  "is genuinely not producing batches that improve the book."]
    elif streak >= 8:
        lines += ["", "Watch this. Not yet a stall, but the generator has not "
                  "improved the book in a while."]
    else:
        lines += ["", "Not stuck."]

    # --- the admission rate: reported, but NOT the learning test ---
    half = max(len(evals) // 2, 1)
    rate = lambda chunk: 100.0 * sum(1 for r in chunk if r["admitted"]) / max(len(chunk), 1)
    first, second = rate(evals[:half]), rate(evals[half:])
    lines += ["", "## Admission rate (context, not evidence)", "",
              "_Read this only alongside the contribution trend above._ On the "
              "mean_variance run it climbed 54% -> 78% while marginal contribution "
              "stayed flat, because `U` rises with repository size (corr +0.41) as "
              "the zoo fills with mediocre members and the bar softens. A rising "
              "admission rate is as easily a falling standard as an improving search.",
              "",
              f"- First half: **{first:.0f}%** admitted ({half} batches)",
              f"- Second half: **{second:.0f}%** admitted ({len(evals) - half} batches)",
              f"- Change: **{second - first:+.0f}pp**"]

    if concurrent:
        lines += [
            "",
            f"**Not interpretable.** {concurrent} batch(es) were evaluated against a "
            "repository of the same size as another, meaning they were generated "
            "concurrently and could not have seen each other's verdicts. Feedback "
            "only reaches batches produced after it, so a flat rate here says "
            "nothing about whether the feedback teaches. Re-run with "
            "`QA_SEQUENTIAL_EVOLUTION=true` for a test that can actually fail.",
        ]
    elif len(evals) < 4:
        lines += ["", "**Too few batches to read a trend.** Treat as descriptive."]
    elif second > first:
        lines += ["", "Consistent with the feedback teaching: later batches clear the "
                  "bar more often, even though the bar itself rises with the repository."]
    else:
        lines += ["", "**Not consistent with the feedback teaching.** Later batches do "
                  "not clear the bar more often. Since the bar rises as the repository "
                  "improves, a flat rate is ambiguous, but a falling one is evidence "
                  "against the mechanism."]

    us = [float(r["U"]) for r in evals if isinstance(r.get("U"), (int, float))]
    if len(us) > 1:
        lines += ["", f"U across batches: min {min(us):.4f}, max {max(us):.4f}, "
                  f"sd {st.pstdev(us):.4f}."]
        if st.pstdev(us) < 1e-9:
            lines.append("**U is constant — the objective is not discriminating at all.**")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
