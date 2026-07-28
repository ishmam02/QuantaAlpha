#!/usr/bin/env python
"""Generate the LaTeX results tables for the A/B evaluation report.

Every number in reports/net_cost_ab_evaluation.tex comes from here, built from
the raw per-seed artifacts the comparison scripts write:

  data/results/backtest_v2_raw.json          flat-fee runs, all configs x seeds,
                                             including the per-year breakdown
  data/results/arm_comparison_<stamp>.raw.json   E_Theta runs, all configs x seeds

Hand-transcribing ~200 figures into a document is how a report ends up
disagreeing with its own measurements. Emitting reports/results_tables.tex and
\\input-ing it makes that impossible: rerun the comparisons, rerun this, and the
document is correct by construction.

Usage::

    python scripts/qa_make_report.py --stamp 20260728_151227
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Published figures for the IC-maximizing baseline, for reference rows.
PAPER = {"ic": 0.0472, "rank_ic": 0.0459, "arr": 0.0468, "ir": 0.6453, "mdd": -0.1180}

FLAT_ORDER = ["alpha158_20", "base features", "Arm A (mined)", "Arm B (mined)"]
FLAT_TEX = {"alpha158_20": r"\texttt{alpha158\_20}",
            "base features": "Base features (4, hand-written)",
            "Arm A (mined)": "Arm A --- RankIC",
            "Arm B (mined)": "Arm B --- net-of-cost $\\mathcal{U}$"}


def agg(rows, key):
    vals = []
    for r in rows or []:
        try:
            v = float(r.get(key))
        except (TypeError, ValueError):
            continue
        if v == v:
            vals.append(v)
    if not vals:
        return None, None
    return st.fmean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def cell(mean, sd, spec="{:+.4f}", pct=False):
    """A mean +/- sd cell. `--` when the metric never materialised."""
    if mean is None:
        return "--"
    fmt = (lambda v: f"{100*v:+.2f}\\%") if pct else (lambda v: spec.format(v))
    if not sd:
        return f"${fmt(mean)}$"
    sd_txt = (f"{100*sd:.2f}" if pct else spec.format(sd).lstrip("+"))
    return f"${fmt(mean)} \\pm {sd_txt}$"


def group_flat(raw: dict) -> dict[str, list[dict]]:
    """Group cached flat-fee runs by configuration.

    Rows carry their own ``__label__`` (qa_backtest_all writes it), which is the
    only reliable key: the cache is indexed by a content fingerprint, so a run
    that mines new arm libraries leaves the previous ones cached alongside them.
    Grouping by anything inferred from the metrics would silently mix runs.
    """
    out: dict[str, list[dict]] = {}
    for key, row in raw.items():
        if key == "__schema__":
            continue
        label = row.get("__label__")
        if label is None:
            raise SystemExit(
                "backtest_v2_raw.json holds unlabelled rows from an older cache "
                "schema. Delete data/results/backtest_v2_raw.json and re-run "
                "scripts/qa_backtest_all.py so every row carries its label."
            )
        out.setdefault(label, []).append(row)
    return out


def tbl_headline(flat: dict) -> str:
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Configuration & IC & Rank IC & Ann.\ return & Info.\ ratio & Max DD \\",
         r"\midrule"]
    for key in FLAT_ORDER:
        if key not in flat:
            continue
        r = flat[key]
        L.append(" & ".join([
            FLAT_TEX[key],
            cell(*agg(r, "ic_oos")), cell(*agg(r, "ric_oos")),
            cell(*agg(r, "annualized_return"), pct=True),
            cell(*agg(r, "information_ratio")),
            cell(*agg(r, "max_drawdown"), pct=True)]) + r" \\")
    L += [r"\midrule",
          f"\\emph{{Published (GPT-5.2)}} & ${PAPER['ic']:+.4f}$ & ${PAPER['rank_ic']:+.4f}$ "
          f"& ${100*PAPER['arr']:+.2f}\\%$ & ${PAPER['ir']:+.4f}$ & ${100*PAPER['mdd']:+.2f}\\%$ \\\\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_contamination(flat: dict) -> str:
    L = [r"\begin{tabular}{lrrl}", r"\toprule",
         r"Configuration & IC as reported & IC on test split & Span scored \\",
         r"\midrule"]
    for key in FLAT_ORDER:
        if key not in flat:
            continue
        r = flat[key]
        full, oos = r[0].get("ic_full_span_n"), r[0].get("ic_oos_n")
        span = (f"{oos} d (test only)" if full == oos
                else f"\\textbf{{{full} d (2016--2025, contaminated)}}")
        L.append(" & ".join([FLAT_TEX[key], cell(*agg(r, "IC")),
                             cell(*agg(r, "ic_oos")), span]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_yearly(flat: dict, prefix: str, pct: bool) -> str:
    years = sorted({int(k.rsplit("_", 1)[-1]) for r in flat.values() for k in r[0]
                    if k.startswith(prefix + "_") and k.rsplit("_", 1)[-1].isdigit()})
    L = [r"\begin{tabular}{l" + "r" * (len(years) + 1) + "}", r"\toprule",
         "Configuration & " + " & ".join(str(y) for y in years) + r" & Full period \\",
         r"\midrule"]
    overall = "annualized_return" if prefix == "xarr" else f"{prefix}_oos"
    for key in FLAT_ORDER:
        if key not in flat:
            continue
        r = flat[key]
        cells = [cell(*agg(r, f"{prefix}_{y}"), pct=pct) for y in years]
        cells.append(cell(*agg(r, overall), pct=pct))
        L.append(FLAT_TEX[key] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_etheta(raw: dict) -> str:
    res = raw["results"]
    keys = list(res)                      # baseline, Arm A, Arm B (insertion order)
    short = ["Baseline (no mining)", "Arm A --- RankIC", "Arm B --- net-of-cost $\\mathcal{U}$"]
    L = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
         "Metric & " + " & ".join(short[:len(keys)]) + r" \\", r"\midrule"]
    for label, k, pct in (("IC", "ic", False), ("Rank IC", "rank_ic", False),
                          ("ICIR", "icir", False), ("Rank ICIR", "rank_icir", False),
                          (r"\textbf{Net IR}", "net_ir", False),
                          (r"\textbf{Net ARR}", "net_arr", True),
                          ("Max drawdown", "mdd", True),
                          ("Turnover (book)", "turnover_book", False),
                          ("Cost (bps/day)", "cost_bps", False),
                          ("IS$\\rightarrow$OOS gap", "is_oos_gap", False)):
        spec = "{:.4f}" if k in ("turnover_book", "cost_bps") else "{:+.4f}"
        L.append(label + " & " + " & ".join(
            cell(*agg(res[key], k), spec, pct=pct) for key in keys) + r" \\")
    n = raw.get("n_factors", {})
    L += [r"\midrule",
          f"Factor count & 0 & {n.get('arm_a','--')} & {n.get('arm_b','--')} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_resolvability(flat: dict, etheta: dict) -> str:
    L = [r"\begin{tabular}{llrrl}", r"\toprule",
         r"Cost model & Metric & Span & Seed sd & Verdict \\", r"\midrule"]

    def block(name, groups, metrics):
        for label, k, pct in metrics:
            stats = [agg(v, k) for v in groups.values()]
            stats = [(m, s) for m, s in stats if m is not None]
            if len(stats) < 2:
                continue
            span = max(m for m, _ in stats) - min(m for m, _ in stats)
            noise = max((s or 0.0) for _, s in stats)
            fmt = (lambda v: f"{100*v:.2f}\\%") if pct else (lambda v: f"{v:.4f}")
            if not noise:
                verdict = "seed-invariant"
            elif span < 2 * noise:
                verdict = f"\\textbf{{{span/noise:.1f}$\\times$ --- not resolvable}}"
            else:
                verdict = f"{span/noise:.1f}$\\times$ --- resolvable"
            L.append(f"{name} & {label} & ${fmt(span)}$ & ${fmt(noise)}$ & {verdict} \\\\")
            name = ""

    block("Flat fee", flat, [("IC (test split)", "ic_oos", False),
                             ("Rank IC (test split)", "ric_oos", False),
                             ("Ann.\\ return", "annualized_return", True),
                             ("Info.\\ ratio", "information_ratio", False)])
    L.append(r"\midrule")
    block("Full ($E_\\Theta$)", etheta["results"],
          [("Rank IC", "rank_ic", False), ("Net IR", "net_ir", False),
           ("Net ARR", "net_arr", True)])
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_runcompare(flat: dict, etheta: dict, run1: dict) -> str:
    """Run 1 (utility inert) against run 2 (utility active), side by side.

    This is the comparison the two runs exist to support. Run 1's repository
    never accumulated, so its utility was constant and only the *feedback*
    channel was live; run 2 has both. Holding the generation process fixed
    across the pair, the difference isolates utility-based selection from
    cost-aware feedback.
    """
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Cost model & Metric & Run 1 ($\util$ inert) & Run 2 ($\util$ active) & Change \\",
         r"\midrule"]

    def line(model, label, r1, r2, pct):
        if r1 is None or r2 is None:
            return
        f = (lambda v: f"{100*v:+.2f}\\%") if pct else (lambda v: f"{v:+.4f}")
        d = r2 - r1
        L.append(f"{model} & {label} & ${f(r1)}$ & ${f(r2)}$ & ${f(d)}$ \\\\")

    model = "Flat fee"
    for arm in ("Arm A (mined)", "Arm B (mined)"):
        if arm not in flat:
            continue
        short = "Arm A" if "A (" in arm else "Arm B"
        for label, key, r1key, pct in (("IC", "ic_oos", "ic_oos", False),
                                       ("Ann.\\ return", "annualized_return", "arr", True)):
            line(model, f"{short} {label}", run1["flat_fee"][arm].get(r1key),
                 agg(flat[arm], key)[0], pct)
            model = ""
    L.append(r"\midrule")
    model = "Full ($E_\\Theta$)"
    keys = list(etheta["results"])
    for r1name, k in zip(("Arm A", "Arm B"), keys[1:]):
        for label, key, pct in (("Rank IC", "rank_ic", False), ("Net ARR", "net_arr", True)):
            line(model, f"{r1name} {label}", run1["etheta"][r1name].get(key),
                 agg(etheta["results"][k], key)[0], pct)
            model = ""
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_pairing(flat: dict, run1: dict) -> str:
    """Generation noise, measured from the Arm A replicate, against the effect.

    Arm A's code path was untouched by the repository fix, so running it twice
    under identical configuration and seed yields an accidental replicate --- the
    only direct measurement of run-to-run generation variance available here.
    That variance turns out to be large, and largely COMMON to both arms, so it
    cancels in the paired difference. Reporting the effect against unpaired noise
    would understate the design; reporting it against the paired spread is what
    the back-to-back construction was for.
    """
    rows = [("Ann.\\ return", "annualized_return", "arr", True),
            ("Test-split IC", "ic_oos", "ic_oos", False)]
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"& Ann.\ return & Test-split IC \\", r"\midrule"]

    def num(v, pct):
        return f"${100*v:+.2f}\\%$" if pct else f"${v:+.4f}$"

    vals = {}
    for label, k2, k1, pct in rows:
        vals[label] = {
            "a1": run1["flat_fee"]["Arm A (mined)"][k1],
            "a2": agg(flat["Arm A (mined)"], k2)[0],
            "b1": run1["flat_fee"]["Arm B (mined)"][k1],
            "b2": agg(flat["Arm B (mined)"], k2)[0],
            "sd": agg(flat["Arm A (mined)"], k2)[1], "pct": pct}

    def row(name, fn):
        cells = []
        for label in vals:
            v = vals[label]
            cells.append(num(fn(v), v["pct"]))
        L.append(name + " & " + " & ".join(cells) + r" \\")

    row(r"Arm A, run 1", lambda v: v["a1"])
    row(r"Arm A, run 2 \emph{(code unchanged)}", lambda v: v["a2"])
    row(r"\quad $\Rightarrow$ generation noise", lambda v: abs(v["a2"] - v["a1"]))
    L.append(r"\midrule")
    row(r"Combiner-seed noise (sd)", lambda v: v["sd"])
    L.append(r"\midrule")
    row(r"Arm B $-$ Arm A, run 1", lambda v: v["b1"] - v["a1"])
    row(r"Arm B $-$ Arm A, run 2", lambda v: v["b2"] - v["a2"])
    row(r"\quad $\Rightarrow$ spread of the paired effect",
        lambda v: abs((v["b2"] - v["a2"]) - (v["b1"] - v["a1"])))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--out", default="reports/results_tables.tex")
    # Path overrides exist so the document toolchain can be exercised against
    # fixtures without touching (or waiting on) a live run's artifacts.
    ap.add_argument("--flat-raw", default="data/results/backtest_v2_raw.json")
    ap.add_argument("--etheta-raw", default=None)
    ap.add_argument("--run1", default="reports/run1_reference.json")
    args = ap.parse_args()

    flat_raw = json.loads((ROOT / args.flat_raw).read_text())
    et_path = ROOT / (args.etheta_raw or
                      f"data/results/arm_comparison_{args.stamp}.raw.json")
    etheta = json.loads(et_path.read_text())
    flat = group_flat(flat_raw)
    run1 = json.loads((ROOT / args.run1).read_text())

    macros = {
        "ThetaHash": etheta["theta_hash"],
        "EvalWindow": f"{etheta['window'][0]}--{etheta['window'][1]}",
        "Seeds": ", ".join(str(s) for s in etheta["seeds"]),
        "NFactorsA": str(etheta["n_factors"]["arm_a"]),
        "NFactorsB": str(etheta["n_factors"]["arm_b"]),
        "KappaZero": str(etheta["costs"]["kappa0"]),
        "KappaOne": str(etheta["costs"]["kappa1"]),
        "KappaTwo": str(etheta["costs"]["kappa2"]),
        "TopK": str(etheta["portfolio"]["topk"]),
        "NDrop": str(etheta["portfolio"]["n_drop"]),
        "RunStamp": args.stamp.replace("_", r"\_"),
    }

    out = ["% GENERATED by scripts/qa_make_report.py -- do not edit by hand.",
           f"% source: backtest_v2_raw.json + arm_comparison_{args.stamp}.raw.json", ""]
    for k, v in macros.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.append("")
    for name, body in (("TblHeadline", tbl_headline(flat)),
                       ("TblContamination", tbl_contamination(flat)),
                       ("TblYearlyIC", tbl_yearly(flat, "ic", False)),
                       ("TblYearlyRankIC", tbl_yearly(flat, "ric", False)),
                       ("TblYearlyARR", tbl_yearly(flat, "xarr", True)),
                       ("TblEtheta", tbl_etheta(etheta)),
                       ("TblResolvability", tbl_resolvability(flat, etheta)),
                       ("TblRunCompare", tbl_runcompare(flat, etheta, run1)),
                       ("TblPairing", tbl_pairing(flat, run1))):
        out += [f"\\newcommand{{\\{name}}}{{%", body, "}", ""]

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")
    for k, v in flat.items():
        print(f"  flat-fee: {k:<16} {len(v)} seed run(s)")
    for k, v in etheta["results"].items():
        print(f"  E_theta : {k:<40} {len(v)} seed run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
