#!/usr/bin/env python
"""Run backtest v2 on every configuration and collect one comparison table.

This is the OLD cost model -- Qlib's flat commission from configs/backtest.yaml
(open_cost 0.0005 + close_cost 0.0015), with no slippage, impact or borrow. It
is deliberately the paper-comparable path, and it will read considerably better
than E_Theta's net-of-cost figures. Both numbers are worth having; only this one
compares to published results.

Four configurations:

  alpha158_20     Qlib's own 20-factor Alpha158 subset -- an external reference
                  that owes nothing to this pipeline
  base            the four hand-written base features the combiner always had
                  (translated into the mining DSL, since --factor-source custom
                  routes everything through expr_parser rather than Qlib)
  arm-a           Arm A's mined factors
  arm-b           Arm B's mined factors

Note `--factor-source custom` uses ONLY the library's factors; it does not add
the base features. So `base` and the two arms are disjoint feature sets, and the
comparison is like-for-like.

**Seeds.** configs/backtest.yaml pins seed 42, so any single run is
reproducible -- but reproducible is not the same as resolvable. The first
measurement here put alpha158_20 at +4.22% ARR, the base features at +4.42% and
the paper at +4.68%: a 0.46pp span, against a combiner seed sd measured at
~1.2pp elsewhere in this repo. A one-seed table cannot tell those three apart.
Pass --seeds to sweep, and the table reports mean +/- sd with an explicit
verdict on whether any gap survives the noise.

Usage::

    python scripts/qa_backtest_all.py --arm-a <libA.json> --arm-b <libB.json>
    python scripts/qa_backtest_all.py --arm-a ... --arm-b ... --seeds 42,1,7
    python scripts/qa_backtest_all.py --skip alpha158_20
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_backtest_all")

# The combiner's four base features, translated from Qlib syntax into the mining
# DSL. --factor-source custom sends every expression through expr_parser +
# function_lib, so Qlib's Mean()/Ref() must become TS_MEAN()/DELAY().
#   ($close-$open)/$open            -> unchanged (plain arithmetic)
#   $volume/Mean($volume, 20)       -> $volume / TS_MEAN($volume, 20)
#   ($high-$low)/Ref($close, 1)     -> ($high - $low) / DELAY($close, 1)
#   $close/Ref($close, 1)-1         -> $close / DELAY($close, 1) - 1
BASE_FEATURES = {
    "BASE_OPEN_RET": "($close - $open) / $open",
    "BASE_VOL_RATIO": "$volume / TS_MEAN($volume, 20)",
    "BASE_RANGE_RET": "($high - $low) / DELAY($close, 1)",
    "BASE_CLOSE_RET": "$close / DELAY($close, 1) - 1",
}

PAPER_ROW = ("*paper (GPT-5.2)*", "+0.0472", "+0.0459", "—", "—", "+4.68%", "0.6453", "-11.80%")

# Bump when the per-run metric payload gains fields, so a stale cache written by
# an older version is discarded rather than silently rendering blank columns.
CACHE_SCHEMA = 2


def test_window(config: str) -> tuple[str, str]:
    """The test split from the config -- the only window an OOS figure may use."""
    import yaml

    cfg = yaml.safe_load((ROOT / config).read_text())

    def find(node):
        if isinstance(node, dict):
            if "segments" in node and isinstance(node["segments"], dict):
                seg = node["segments"].get("test")
                if seg:
                    return str(seg[0]), str(seg[1])
            for v in node.values():
                got = find(v)
                if got:
                    return got
        return None

    return find(cfg) or ("2022-01-01", "2025-12-26")


def write_base_library(path: Path) -> Path:
    """Emit the base-feature set as a factor library the backtest can consume."""
    import hashlib

    factors = {}
    for name, expr in BASE_FEATURES.items():
        fid = hashlib.md5(f"{name}_{expr}".encode()).hexdigest()[:16]
        factors[fid] = {
            "factor_id": fid,
            "factor_name": name,
            "factor_expression": expr,
            "factor_description": "Combiner base feature (translated to the mining DSL)",
            "factor_formulation": "",
            "cache_location": {},
            "metadata": {"experiment_id": "base_features"},
            "backtest_results": {},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"metadata": {"total_factors": len(factors), "subset": "base_features_only"},
         "factors": factors}, indent=2))
    logger.info("wrote base-feature library: %s (%d factors)", path, len(factors))
    return path


def seeded_config(config: str, seed: int, tmpdir: Path) -> str:
    """Copy the backtest config with every seed field set to `seed`.

    Patches the top-level `random_seed` and the LightGBM `model.params.seed` /
    `random_state`, leaving everything else -- splits, costs, topk/n_drop --
    exactly as the tracked config has it. Sweeping the seed must vary the seed
    and nothing else, or the spread stops being a noise measurement.
    """
    import yaml

    src = ROOT / config
    cfg = yaml.safe_load(src.read_text())
    patched = copy.deepcopy(cfg)
    patched["random_seed"] = seed
    params = patched.setdefault("model", {}).setdefault("params", {})
    for key in ("seed", "random_state"):
        if key in params:
            params[key] = seed
    out = tmpdir / f"{src.stem}_seed{seed}.yaml"
    out.write_text(yaml.safe_dump(patched, sort_keys=False))
    return str(out)


def fingerprint(factor_json: Path | None, config: str) -> str:
    """Identify a (feature set, protocol) pair so cached results can't be stale.

    Keyed on the factor library's *contents* rather than its path, because the
    arm libraries are rewritten in place between runs under the same filename.
    """
    import hashlib

    h = hashlib.sha256()
    h.update((ROOT / config).read_bytes())
    h.update(factor_json.read_bytes() if factor_json else b"alpha158_20")
    return h.hexdigest()[:16]


def load_cache(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.pop("__schema__", None) != CACHE_SCHEMA:
        logger.info("cache written by an older version; recomputing")
        return {}
    return payload


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"__schema__": CACHE_SCHEMA, **cache}, indent=2))


def read_metrics(out_dir: Path, factor_json: Path | None, started: float) -> dict | None:
    """Locate the metrics file this run just wrote.

    Two traps. Without --factor-json the runner has no output_name and falls
    back to the config's plain `backtest_metrics.json`, which a
    "*_backtest_metrics.json" glob does not match. And because the filename
    depends on the factor library rather than the seed, a seed sweep overwrites
    the same path every time -- so a stale file from a *different* config would
    otherwise be read as this run's result. Require the mtime to postdate the
    run's start.
    """
    stem = factor_json.stem if factor_json else None
    exact = out_dir / (f"{stem}_backtest_metrics.json" if stem else "backtest_metrics.json")
    candidates = [exact] + sorted(
        out_dir.glob("*backtest_metrics.json"), key=lambda p: -p.stat().st_mtime
    )
    for c in candidates:
        if c.exists() and c.stat().st_mtime >= started:
            payload = json.loads(c.read_text())
            return {"__file__": c.name, **payload.get("metrics", payload)}
    return None


def newest_since(paths, started: float):
    """The most recent of `paths`, provided this run actually wrote it."""
    fresh = [p for p in paths if p.exists() and p.stat().st_mtime >= started]
    return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None


def yearly_breakdown(factor_json: Path | None, config: str, started: float) -> dict:
    """Per-year IC, Rank IC and excess annualised return over the test split.

    Two things this repairs as a side effect.

    **The reported IC is not always out-of-sample.** SigAnaRecord scores whatever
    span the prediction covers. Under `--factor-source alpha158_20` that is the
    966-day test window; under `--factor-source custom` it is the full 2427-day
    2016-2025 span, so the headline IC is a blend of train, validation and test.
    Measured here: the four base features report IC +0.0389 full-span but
    +0.0481 on 2022-2025 alone. Slicing to the test window makes the two sources
    comparable and is the only IC worth quoting.

    **Annualisation.** Qlib's risk_analysis scales daily returns by 238, not 252,
    and accumulates arithmetically. Rather than reimplement that, each year's
    slice is handed to risk_analysis itself, so the yearly excess returns compose
    consistently with the headline figure.
    """
    import pandas as pd
    from qlib.contrib.evaluate import risk_analysis

    lo, hi = test_window(config)
    out: dict = {"test_window": f"{lo}..{hi}"}

    ic_p = newest_since(list((ROOT / "mlruns").rglob("sig_analysis/ic.pkl")), started)
    ric_p = newest_since(list((ROOT / "mlruns").rglob("sig_analysis/ric.pkl")), started)
    for key, path in (("ic", ic_p), ("ric", ric_p)):
        if path is None:
            continue
        s = pd.read_pickle(path)
        if not isinstance(s, pd.Series) or s.empty:
            continue
        full = s
        oos = s[(s.index >= lo) & (s.index <= hi)]
        out[f"{key}_full_span_n"] = int(len(full))
        out[f"{key}_oos_n"] = int(len(oos))
        if len(oos):
            out[f"{key}_oos"] = float(oos.mean())
            out[f"{key}_oos_ir"] = float(oos.mean() / oos.std()) if oos.std() > 0 else 0.0
            for year, v in oos.groupby(oos.index.year):
                out[f"{key}_{year}"] = float(v.mean())

    stem = factor_json.stem if factor_json else None
    csv_dir = ROOT / "data/results/backtest_v2_results"
    csv_p = newest_since(
        [csv_dir / f"{stem}_cumulative_excess.csv"] if stem else list(
            csv_dir.glob("*_cumulative_excess.csv")), started
    ) or newest_since(list(csv_dir.glob("*_cumulative_excess.csv")), started)
    if csv_p is not None:
        df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
        r = df["daily_excess_return"].dropna()
        r = r[(r.index >= lo) & (r.index <= hi)]
        for year, v in r.groupby(r.index.year):
            if len(v) < 20:      # a stub year cannot be annualised meaningfully
                continue
            a = risk_analysis(v)
            a = a["risk"] if isinstance(a, pd.DataFrame) and "risk" in a.columns else (
                a.iloc[:, 0] if isinstance(a, pd.DataFrame) else a)
            out[f"xarr_{year}"] = float(a.get("annualized_return", float("nan")))
            out[f"xir_{year}"] = float(a.get("information_ratio", float("nan")))
    return out


def run_backtest(label, source, factor_json, config, seed, tmpdir) -> dict | None:
    cfg = seeded_config(config, seed, tmpdir) if seed is not None else config
    cmd = [sys.executable, "-m", "quantaalpha.backtest.run_backtest",
           "-c", cfg, "--factor-source", source]
    if factor_json:
        cmd += ["--factor-json", str(factor_json)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("[%s seed=%s] FAILED (rc=%s)\n%s", label, seed, proc.returncode,
                     proc.stdout[-1500:])
        return None

    m = read_metrics(ROOT / "data/results/backtest_v2_results", factor_json, t0)
    if m is None:
        logger.error("[%s seed=%s] no fresh metrics file produced", label, seed)
        return None
    try:
        m.update(yearly_breakdown(factor_json, config, t0))
    except Exception as exc:                       # never lose a completed run
        logger.warning("[%s seed=%s] yearly breakdown unavailable: %s", label, seed, exc)
    logger.info("[%-14s seed=%-3s] %.0fs  IC=%+.4f (oos %+.4f, n=%s/%s)  ARR=%+.2f%%",
                label, seed, time.time() - t0, _f(m, "IC"), _f(m, "ic_oos"),
                m.get("ic_oos_n", "?"), m.get("ic_full_span_n", "?"),
                100 * _f(m, "annualized_return"))
    return m


def _f(m, key):
    try:
        return float((m or {}).get(key))
    except (TypeError, ValueError):
        return float("nan")


def agg(rows, key):
    vals = [_f(r, key) for r in rows or []]
    vals = [v for v in vals if v == v]
    if not vals:
        return None, None
    return st.fmean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(mean, sd, spec="{:+.4f}"):
    if mean is None:
        return "—"
    if not sd:
        return spec.format(mean)
    return f"{spec.format(mean)} ± {spec.format(sd).lstrip('+')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a")
    ap.add_argument("--arm-b")
    ap.add_argument("--config", default="configs/backtest.yaml")
    ap.add_argument("--out", default="data/results/backtest_v2_comparison.md")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated model seeds; >1 gives a noise band")
    ap.add_argument("--skip", nargs="*", default=[], help="labels to skip")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute every (configuration, seed) even if cached")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base_lib = write_base_library(ROOT / "data/factorlib/base_features_library.json")

    plan = [("alpha158_20", "alpha158_20", None),
            ("base features", "custom", base_lib)]
    if args.arm_a:
        plan.append(("Arm A (mined)", "custom", Path(args.arm_a)))
    if args.arm_b:
        plan.append(("Arm B (mined)", "custom", Path(args.arm_b)))

    cache_path = ROOT / "data/results/backtest_v2_raw.json"
    cache = {} if args.no_cache else load_cache(cache_path)

    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="qa_bt_") as td:
        tmpdir = Path(td)
        for label, source, fj in plan:
            if label in args.skip:
                logger.info("skipping %s", label)
                continue
            if fj and not Path(fj).exists():
                logger.warning("%s: %s not found, skipping", label, fj)
                continue
            fp = fingerprint(fj, args.config)
            rows = []
            for seed in seeds:
                key = f"{fp}:{seed}"
                if key in cache:
                    logger.info("[%-14s seed=%-3s] cached", label, seed)
                    rows.append(cache[key])
                    continue
                m = run_backtest(label, source, fj, args.config, seed, tmpdir)
                if m:
                    cache[key] = m
                    rows.append(m)
                    save_cache(cache_path, cache)
            results[label] = rows

    n_ok = sum(1 for r in results.values() if r)
    lines = [
        "# Backtest v2 — flat-fee cost model (paper-comparable)",
        "",
        "Qlib's own cost model from `configs/backtest.yaml`: `open_cost 0.0005` + "
        "`close_cost 0.0015`, **no slippage, impact or borrow**. This is the path "
        "the published numbers use, so it is the only column comparable to them — "
        "and it reads considerably better than `E_Θ`'s net-of-cost figures for "
        "exactly that reason.",
        "",
        "`--factor-source custom` uses only the library's own factors, so these "
        "feature sets are disjoint: **base features** and each arm's **mined "
        "factors** are separate models, not nested ones.",
        "",
        f"Seeds swept: `{seeds}`."
        + ("" if len(seeds) > 1 else
           "  **One seed — the differences below are not resolvable.** "
           "Re-run with `--seeds 42,1,7` for a noise band."),
        "",
        "| Config | IC | Rank IC | ICIR | Rank ICIR | Ann. Return | Info Ratio | Max DD |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, rows in results.items():
        if not rows:
            lines.append(f"| {label} | FAILED | | | | | | |")
            continue
        cells = [fmt(*agg(rows, k)) for k in ("IC", "Rank IC", "ICIR", "Rank ICIR")]
        cells.append(fmt(*agg(rows, "annualized_return"), "{:+.2%}"))
        cells.append(fmt(*agg(rows, "information_ratio")))
        cells.append(fmt(*agg(rows, "max_drawdown"), "{:+.2%}"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| " + " | ".join(PAPER_ROW) + " |")

    # --- out-of-sample correction ---------------------------------------
    # The IC column above is whatever SigAnaRecord scored, and that span is not
    # the same for every factor source. Restate it on the test split alone.
    win = next((r[0].get("test_window") for r in results.values() if r), None)
    contaminated = [lab for lab, r in results.items()
                    if r and r[0].get("ic_full_span_n", 0) > r[0].get("ic_oos_n", 0)]
    if any(r and "ic_oos" in r[0] for r in results.values()):
        lines += [
            "", f"## Out-of-sample only ({win})", "",
            "The IC column above is whatever `SigAnaRecord` happened to score, and "
            "**that span differs by factor source**: `alpha158_20` is scored on the "
            "966-day test window, while `--factor-source custom` is scored on the "
            "full 2427-day 2016–2025 span — training data included. Those two "
            "numbers were never comparable. Restated on the test split alone:",
            "",
            "| Config | IC (as reported) | IC (test split) | Rank IC (test split) | ICIR (test split) | Scored span |",
            "| :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
        for label, rows_ in results.items():
            if not rows_:
                continue
            span = ("full 2016–2025 — **contaminated**"
                    if label in contaminated else "test split only ✓")
            lines.append(
                f"| {label} | {fmt(*agg(rows_, 'IC'))} | {fmt(*agg(rows_, 'ic_oos'))} | "
                f"{fmt(*agg(rows_, 'ric_oos'))} | {fmt(*agg(rows_, 'ic_oos_ir'))} | {span} |"
            )
        lines.append("| *paper (GPT-5.2)* | +0.0472 | — | — | — | not stated |")
        if contaminated:
            lines += [
                "",
                f"**{', '.join(contaminated)}** {'was' if len(contaminated) == 1 else 'were'} "
                "scored over the training period as well. Here that *understates* the "
                "out-of-sample IC, because 2016–2020 was a weaker IC regime than "
                "2022–2025 — but the direction is incidental, and the blend is not a "
                "quantity anyone should report. Use the test-split column.",
                "",
                "It is also worth knowing which convention the published +0.0472 "
                "follows before setting anything against it.",
            ]

    # --- yearly breakdown ------------------------------------------------
    years = sorted({int(k.split("_")[-1]) for r in results.values() if r
                    for k in r[0] if k.startswith("ic_") and k.split("_")[-1].isdigit()})
    if years:
        lines += [
            "", "## Year by year (CSI 300)", "",
            "Annual IC and Rank IC are the mean daily cross-sectional correlation "
            "within each calendar year. Excess ARR is annualised from that year's "
            "daily excess-return series by Qlib's own `risk_analysis` — which scales "
            "by **238**, not 252, and accumulates arithmetically — so the yearly "
            "figures compose consistently with the headline. Excess is over the "
            "CSI 300 benchmark and net of the flat commission.",
            "",
        ]
        for metric_label, prefix, spec in (("Annual IC", "ic", "{:+.4f}"),
                                           ("Annual Rank IC", "ric", "{:+.4f}"),
                                           ("Excess ARR", "xarr", "{:+.2%}")):
            lines += [f"### {metric_label}", "",
                      "| Config | " + " | ".join(str(y) for y in years) + " | Full period |",
                      "| :--- | " + " | ".join("---:" for _ in years) + " | ---: |"]
            for label, rows_ in results.items():
                if not rows_:
                    continue
                cells = [fmt(*agg(rows_, f"{prefix}_{y}"), spec) for y in years]
                overall = ("ic_oos" if prefix == "ic" else
                           "ric_oos" if prefix == "ric" else "annualized_return")
                cells.append(fmt(*agg(rows_, overall), spec))
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")

    # --- resolvability, per metric --------------------------------------
    # Not every metric is equally noisy, and the difference is the single most
    # useful thing a seed sweep tells you. Measured here: the base features
    # return IC = +0.0389 on all three seeds while their annualised return
    # ranges 3.68%-4.95%. Identical signal, 1.27pp of return spread -- that
    # spread is the top-k dropout construction's path dependence through a
    # 50-name book, not model quality. So a metric-by-metric verdict decides
    # which column an arm comparison is allowed to be settled on.
    lines += ["", "## Which metrics can actually separate these?", "",
              "Span = best minus worst across configurations. Noise = the largest "
              "seed sd within any one configuration. A span under 2x the noise "
              "cannot support a ranking.", "",
              "| Metric | Span | Seed noise | Verdict |", "| :--- | ---: | ---: | :--- |"]
    if len(results) < 2 or not any(results.values()):
        lines.append("| — | — | — | not enough configurations completed |")
    elif len(seeds) < 2:
        lines.append("| — | — | — | single seed; noise not measured |")
        lines += ["", "Re-run with `--seeds 42,1,7`. For reference the combiner seed sd "
                  "measured under `E_Θ` is ~1.2pp on annualised return, larger than the "
                  "whole span of this table."]
    else:
        # Use the test-split IC, not the reported one: the reported figure is
        # scored over different spans by different factor sources, so its span
        # would be measuring that inconsistency as much as any real difference.
        for label, key, spec in (("IC (test split)", "ic_oos", "{:.4f}"),
                                 ("Rank IC (test split)", "ric_oos", "{:.4f}"),
                                 ("ICIR (test split)", "ic_oos_ir", "{:.4f}"),
                                 ("Ann. return", "annualized_return", "{:.2%}"),
                                 ("Info ratio", "information_ratio", "{:.4f}")):
            stats = [agg(v, key) for v in results.values() if v]
            stats = [(m, s) for m, s in stats if m is not None]
            if len(stats) < 2:
                continue
            span = max(m for m, _ in stats) - min(m for m, _ in stats)
            noise = max((s or 0.0) for _, s in stats)
            if not noise:
                verdict = "seed-invariant — any gap is real"
            elif span < 2 * noise:
                verdict = f"**{span / noise:.1f}× noise — NOT resolvable**"
            else:
                verdict = f"{span / noise:.1f}× noise — resolvable"
            lines.append(f"| {label} | {spec.format(span)} | {spec.format(noise)} | {verdict} |")
        lines += [
            "",
            "Where annualised return is not resolvable but IC is, the two are "
            "measuring different things: IC is an average over thousands of daily "
            "cross-sections, while annualised return is one equity curve whose path "
            "depends on which names the top-k dropout rule happens to hold. Report "
            "the resolvable column.",
        ]

    lines += [
        "",
        "## Reading this",
        "",
        "- **alpha158_20** owes nothing to this pipeline. An arm that cannot beat it "
        "has not shown that LLM mining adds anything over a standard factor set.",
        "- **base features** is the four hand-written expressions the combiner always "
        "had, with no mining whatsoever. It is the floor: mining has to clear it to "
        "have contributed anything.",
        "- Arms are **not** nested with the base features here, so their columns are "
        "not 'base + mined' — each is its own model on its own features.",
        "- **Annualised return is the wrong headline if the span is inside the noise.** "
        "Prefer IC / ICIR, which are estimated from thousands of daily "
        "cross-sections rather than one equity curve, and are correspondingly "
        "steadier across seeds.",
        "- Under the full cost model (`scripts/qa_compare_arms.py`) the same factor "
        "sets score materially worse. The gap between the two tables is the "
        "slippage and impact this backtest does not charge, and it is the point of "
        "the whole exercise.",
    ]

    table = "\n".join(lines)
    print()
    print(table)
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(table + "\n")
    logger.info("wrote %s (%d/%d configurations succeeded)", args.out, n_ok, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
