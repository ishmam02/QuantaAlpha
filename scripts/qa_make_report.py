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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--out", default="reports/results_tables.tex")
    # Path overrides exist so the document toolchain can be exercised against
    # fixtures without touching (or waiting on) a live run's artifacts.
    ap.add_argument("--flat-raw", default="data/results/backtest_v2_raw.json")
    ap.add_argument("--etheta-raw", default=None)
    args = ap.parse_args()

    flat_raw = json.loads((ROOT / args.flat_raw).read_text())
    et_path = ROOT / (args.etheta_raw or
                      f"data/results/arm_comparison_{args.stamp}.raw.json")
    etheta = json.loads(et_path.read_text())
    flat = group_flat(flat_raw)

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
                       ("TblResolvability", tbl_resolvability(flat, etheta))):
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
