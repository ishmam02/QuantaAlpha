#!/usr/bin/env python
"""TRANSFER test: score the CSI300-mined factor zoo on S&P 500 data, no re-mining.

The decisive method-vs-market test. The CSI300-mined factor expressions are pure
functions of the 8 cache fields ($open,$close,$high,$low,$volume,$factor,$return,
$vwap), so they can be recomputed on a US price/volume panel and backtested under a
US protocol with NO re-mining. If the formulas still rank stocks out-of-sample on
the US, the edge is a property of the METHOD; if they collapse, it is local to the
CSI300 market.

This mirrors scripts/qa_eval_oneshot.py EXCEPT that it does NOT read the md5-cached
CSI signal (qa_eval_oneshot.load_aligned_signal, which returns the CSI300 signal).
Instead it recomputes each expression on the US ``daily_pv.h5`` cache via the SAME
path the miner uses (expr_parser.parse_expression + parse_symbol + function_lib),
then aligns the recomputed signal onto the US eval panel and hands the whole set to
``EvaluationOperator.evaluate`` as one book.

Usage::

    conda run -n quantaalpha python scripts/qa_transfer_us.py \\
        --library data/factorlib/all_factors_library_meanvar_20260828_194432.json \\
        --protocol quantaalpha/eval/protocol_sp500_meanvar_soft_linear.yaml \\
        --cache data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \\
        --qlib-dir data/qlib/us_data --report

Run qa_eval_oneshot.py with the CSI300 protocol for the matched baseline, then
compare the two blocks -- same zoo, same split years, different market.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_transfer_us")

DEFAULT_CACHE = "data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5"
DEFAULT_PROTOCOL = "quantaalpha/eval/protocol_sp500_meanvar_soft_linear.yaml"
DEFAULT_LIBRARY = "data/factorlib/all_factors_library_meanvar_20260828_194432.json"


def load_library(path: str) -> list[tuple[str, str]]:
    """[(name, expression)] for every factor in the library (mirrors qa_eval_oneshot)."""
    payload = json.loads(Path(path).read_text())
    factors = payload.get("factors", payload)
    items = factors.values() if isinstance(factors, dict) else factors
    out: list[tuple[str, str]] = []
    for entry in items:
        expr = entry.get("factor_expression") or entry.get("expression") or ""
        name = entry.get("factor_name") or entry.get("name") or "unnamed"
        if expr:
            out.append((name, expr))
    return out


# --------------------------------------------------------------------------- #
# Factor recompute -- the miner's path, mirrored verbatim.
# --------------------------------------------------------------------------- #
def recompute_factor(expression_str: str, df: "pd.DataFrame",
                     factor_name: str = "factor") -> "pd.Series":
    """Evaluate a mined factor expression on a (datetime, instrument)-indexed PV
    dataframe. Returns a (datetime, instrument) MultiIndex float64 Series -- the
    shape quantaalpha.eval.data.align_signal expects.

    Mirrors quantaalpha/backtest/custom_factor_calculator.py:calculate_factor and
    factors/coder/template.jinjia2: parse_symbol -> parse_expression (stdout
    suppressed) -> replace each bare field with df['$field'] -> eval with every
    operator in function_lib in scope.
    """
    import numpy as np
    import pandas as pd
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
    import quantaalpha.factors.coder.function_lib as func_lib

    # $return is derived in the cache (generate.py:69). Only copy+derive when it is
    # absent; skipping the copy when present avoids 150 x ~90MB allocations (the
    # recompute's main cost + memory churn), and the eval below does not mutate df.
    if "$return" not in df.columns:
        df = df.copy()
        df["$return"] = (
            df.groupby(level="instrument")["$close"]
              .pct_change(fill_method="ffill").fillna(0)
        )

    # 1) strip '$' and normalize TRUE/FALSE/NAN (expr_parser.parse_symbol)
    expr = parse_symbol(expression_str, df.columns)
    # 2) rewrite +/-/*/</>/?:/&& into ADD/SUBTRACT/.../WHERE. parse_expression prints
    #    to stdout -- suppress it (custom_factor_calculator.py does the same).
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        expr = parse_expression(expr)
    finally:
        sys.stdout = old_stdout
    # 3) swap each bare field name back to df['$field']  (template.jinjia2)
    for col in df.columns:
        if isinstance(col, str) and col.startswith("$"):
            expr = expr.replace(col[1:], f"df['{col}']")
    # 4) eval with df + every public operator from function_lib in scope
    exec_globals = {"df": df, "np": np, "pd": pd}
    for name in dir(func_lib):
        if not name.startswith("_"):
            obj = getattr(func_lib, name)
            if callable(obj):
                exec_globals[name] = obj
    result = eval(expr, exec_globals)
    # 5) normalize to a (datetime, instrument) Series
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)
    result = result.astype(np.float64)
    result.name = factor_name
    if not result.index.equals(df.index):
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep="last")]
        result = result.reindex(df.index)
    return result


def _f(v, pct: bool = False) -> str:
    try:
        if v is None or v != v:
            return "n/a"
        return f"{100 * float(v):+.4f}%" if pct else f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    p.add_argument("--protocol-flat", default=None,
                   help="second protocol (e.g. flat cost) evaluated on the SAME recomputed signals")
    p.add_argument("--cache", default=DEFAULT_CACHE, help="US daily_pv.h5 (8-col cache)")
    p.add_argument("--qlib-dir", default="data/qlib/us_data", help="us_data Qlib dir for the eval panel")
    p.add_argument("--region", default="us")
    p.add_argument("--ledger", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report", action="store_true", help="score on final_test (else search_oos)")
    p.add_argument("--cache-signals", default=None,
                   help="pickle path to cache/load the recomputed+aligned signals (skip the "
                        "~13-min recompute on retries / across cost models). Keyed to the "
                        "panel grid -- delete it if the window/qlib-dir/market changes.")
    args = p.parse_args()

    # Point the eval engine at the US Qlib dir BEFORE any qlib-using call. _init_qlib
    # reads QLIB_PROVIDER_URI at first use and is idempotent, so setting it here in a
    # fresh process makes load_panel / equal_weight_benchmark read us_data.
    os.environ["QLIB_PROVIDER_URI"] = str(Path(args.qlib_dir).resolve())
    os.environ["QLIB_REGION"] = args.region

    from quantaalpha.eval.data import align_signal  # noqa: E402
    from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
    from quantaalpha.eval.protocol import load_protocol  # noqa: E402
    import pandas as pd  # noqa: E402

    theta = load_protocol(args.protocol)
    factors = load_library(args.library)
    if args.limit:
        factors = factors[: args.limit]
    if not factors:
        logger.error("no factors in %s", args.library)
        return 2

    logger.info("theta=%s | market=%s | factors=%d | window=%s | cache=%s",
                theta.hash, theta.market, len(factors),
                "final_test" if args.report else "search_oos", args.cache)

    # US factor cache (the 8-col panel the miner computes against), (datetime, instr).
    df = pd.read_hdf(args.cache, key="data")
    logger.info("US cache: %s rows x %d cols (%s..%s, %d instruments)",
                df.shape[0], df.shape[1],
                df.index.get_level_values("datetime").min().date(),
                df.index.get_level_values("datetime").max().date(),
                df.index.get_level_values("instrument").nunique())

    op = EvaluationOperator(theta)
    p_start, p_end, _ = op._windows(args.report)
    panel = op._panel(p_start, p_end)
    logger.info("US eval panel: %s..%s, %d instruments", p_start, p_end, len(panel.instruments))

    # Recompute every factor on the US cache and align onto the US panel. Cache the
    # aligned frames to a pickle so the ~13-min recompute is paid once across the
    # full/flat backtests and any retries (the cache is keyed to the panel grid:
    # window / qlib-dir / market -- delete it if any of those change).
    import pickle
    cs = args.cache_signals
    if cs and Path(cs).exists():
        logger.info("loading cached recomputed+aligned signals from %s", cs)
        with open(cs, "rb") as fh:
            candidates, ok, fail = pickle.load(fh)
        logger.info("loaded %d cached signals (ok=%d fail=%d)", len(candidates), ok, fail)
    else:
        t0 = time.time()
        candidates: dict[str, object] = {}
        ok = fail = 0
        for name, expr in factors:
            try:
                sig = recompute_factor(expr, df, factor_name=name)
                candidates[expr] = align_signal(sig, panel)
                ok += 1
            except Exception as exc:
                fail += 1
                logger.warning("recompute FAILED for %s (%s): %s", name, expr[:50], exc)
        logger.info("recomputed %d/%d factors in %.1fs (%d failed)",
                    ok, len(factors), time.time() - t0, fail)
        if cs:
            try:
                with open(cs, "wb") as fh:
                    pickle.dump((candidates, ok, fail), fh)
                logger.info("cached %d signals to %s", len(candidates), cs)
            except Exception as exc:
                logger.warning("could not cache signals (%s)", exc)
    if not candidates:
        logger.error("no factors recomputed successfully")
        return 2

    def _report(tag: str, th, res) -> None:
        def g(k):
            return res.get(f"m_{k}")
        print()
        print("=" * 78)
        print(f"US TRANSFER [{tag}]  |  protocol {res.get('theta_hash')}  |  market {th.market}  "
              f"|  window {res.get('eval_window')}")
        print(f"  combiner={th.combiner.model}  construction={th.portfolio.construction}  "
              f"topk={th.portfolio.topk}  |  costs kappa0={th.costs.kappa0} "
              f"kappa1={th.costs.kappa1} kappa2={th.costs.kappa2}")
        print(f"  factors in book: {res.get('n_factors')}   (zoo_size={res.get('zoo_size')})")
        print("=" * 78)
        print(f"  net_ir      : {_f(g('net_ir'))}")
        print(f"  net_arr     : {_f(g('net_arr'), pct=True)}   (raw {g('net_arr')})")
        print(f"  rank_ic     : {_f(g('rank_ic'))}   (of the COMBINED prediction)")
        print(f"  cost_bps    : {_f(g('cost_bps'))}  bps/day  "
              f"(~{250 * float(g('cost_bps') or 0) / 100:.2f}%/yr)")
        print(f"  turnover    : {_f(g('turnover_book'))}")
        print(f"  rho_max     : {_f(g('rho_max'))}   (vs empty zoo)")
        print(f"  rho_within  : {_f(g('rho_within'))}   (worst pair inside the book)")
        print(f"  cx          : {_f(g('cx'))}")
        print(f"  mdd         : {_f(g('mdd'))}")
        print(f"  TC          : {_f(g('transfer_coefficient'))}")
        print(f"  U           : {_f(res.get('U'))}   (repository-relative; degenerate w/ empty zoo)")
        print("-" * 78)
        print(f"  baseline (empty zoo): net_ir={_f(g('base_net_ir'))}  "
              f"net_arr={_f(g('base_net_arr'), pct=True)}")
        print(f"  delta vs baseline  : net_ir={_f(g('delta_net_ir'))}  "
              f"net_arr={_f(g('delta_net_arr'), pct=True)}")
        if g("failed_gates") is not None:
            print(f"  failed_gates        : {g('failed_gates')}")
        print("=" * 78)
        if args.ledger:
            from quantaalpha.eval.ledger import Ledger  # noqa: E402
            Ledger(args.ledger).append({
                "mode": "us_transfer", "cost_model": tag, "n_factors": res.get("n_factors"),
                "theta_hash": res.get("theta_hash"), "market": th.market,
                "combiner": th.combiner.model, "construction": th.portfolio.construction,
                "zoo_size": res.get("zoo_size"),
                "net_ir": g("net_ir"), "net_arr": g("net_arr"), "rank_ic": g("rank_ic"),
                "cost_bps": g("cost_bps"), "turnover_book": g("turnover_book"),
                "base_net_ir": g("base_net_ir"), "base_net_arr": g("base_net_arr"),
                "delta_net_ir": g("delta_net_ir"), "delta_net_arr": g("delta_net_arr"),
                "U": res.get("U"), "eval_window": res.get("eval_window"),
            })

    # Full cost model (LightGBM + topk). The recomputed candidates are reused for
    # the flat-cost run below -- the signals are protocol-independent; only the
    # cost model + (here identical) construction differ.
    t1 = time.time()
    res = op.evaluate(candidates, zoo_signals={}, zoo_metrics=[], report=args.report)
    logger.info("evaluate [full]: %.1fs", time.time() - t1)
    _report("FULL COST", theta, res)

    # Flat-fee backtest_v2: same recomputed candidates, different cost protocol.
    if args.protocol_flat:
        theta_flat = load_protocol(args.protocol_flat)
        op_flat = EvaluationOperator(theta_flat)
        t2 = time.time()
        res_flat = op_flat.evaluate(candidates, zoo_signals={}, zoo_metrics=[], report=args.report)
        logger.info("evaluate [flat]: %.1fs", time.time() - t2)
        _report("FLAT COST (backtest_v2)", theta_flat, res_flat)

    print("\nCompare against qa_eval_oneshot.py on the CSI300 protocol (same zoo, same "
          "split years) to answer general-vs-local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())