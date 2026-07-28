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
    logger.info("[%-14s seed=%-3s] %.0fs  IC=%+.4f  ARR=%+.2f%%  IR=%+.4f",
                label, seed, time.time() - t0, _f(m, "IC"),
                100 * _f(m, "annualized_return"), _f(m, "information_ratio"))
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
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base_lib = write_base_library(ROOT / "data/factorlib/base_features_library.json")

    plan = [("alpha158_20", "alpha158_20", None),
            ("base features", "custom", base_lib)]
    if args.arm_a:
        plan.append(("Arm A (mined)", "custom", Path(args.arm_a)))
    if args.arm_b:
        plan.append(("Arm B (mined)", "custom", Path(args.arm_b)))

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
            rows = []
            for seed in seeds:
                m = run_backtest(label, source, fj, args.config, seed, tmpdir)
                if m:
                    rows.append(m)
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

    # --- resolvability -------------------------------------------------
    lines += ["", "## Is anything here resolvable?", ""]
    arr = {k: agg(v, "annualized_return") for k, v in results.items() if v}
    if len(arr) < 2:
        lines.append("- Not enough configurations completed to compare.")
    else:
        spread = max(m for m, _ in arr.values()) - min(m for m, _ in arr.values())
        noise = max((s or 0.0) for _, s in arr.values())
        lines.append(f"- Span across configurations (annualised return): **{spread:.2%}**")
        if len(seeds) > 1:
            lines.append(f"- Largest seed sd within a configuration: **{noise:.2%}**")
            if noise and spread < 2 * noise:
                lines.append(
                    f"- **{spread / noise:.1f}× the noise — NOT resolvable.** Every "
                    "configuration in this table, including the published number, sits "
                    "inside the band a different random seed would move it by. No "
                    "ordering here should be reported as one approach beating another."
                )
            elif noise:
                lines.append(f"- **{spread / noise:.1f}× the noise** — the ordering "
                             "survives seed variance at this sample size.")
        else:
            lines.append(
                "- Seed sd not measured (single seed). For reference, the combiner "
                "seed sd measured under `E_Θ` is ~1.2pp on annualised return — larger "
                "than the span above, which is why the sweep matters."
            )

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
