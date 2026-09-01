#!/usr/bin/env python
"""Score a whole factor library as ONE book on a single ``E_Θ`` pass -- no grow.

Unlike ``qa_eval_probe`` (which grows the zoo one factor at a time and re-prices
the book at every step -- O(n) full-book evaluations, impractical past ~20
factors), this loads every cached factor, hands them ALL to
``EvaluationOperator.evaluate`` as the candidate set in one call, and reports
the resulting book. ``evaluate`` is documented to accept "every factor the
experiment produced, evaluated **together**" -- this is its primary use; the
probe walks one at a time only to expose the marginal-contribution trajectory.

Use for: "what does the full N-factor book score on the test period with the
full cost model?" when N is large enough that the grow is impractical (135
factors ≈ hours grown, minutes one-shot). The empty-zoo baseline is computed
alongside, so the delta-vs-doing-nothing is reported too.

Usage::

    python scripts/qa_eval_oneshot.py \
        --library data/factorlib/all_factors_library_mine_20260821_0438.json \
        --protocol quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml \
        --report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantaalpha.eval.data import load_aligned_signal, load_factor_signal  # noqa: E402
from quantaalpha.eval.ledger import Ledger  # noqa: E402
from quantaalpha.eval.operator import EvaluationOperator  # noqa: E402
from quantaalpha.eval.protocol import default_protocol_path, load_protocol  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_eval_oneshot")


def load_zoo(ledger_path: str) -> list[tuple[str, str]]:
    """The ADMITTED book (the zoo), replayed from a run's ledger.

    The library holds everything the search produced, admitted or not. The zoo
    is the subset the gate accepted -- that is the book the system would
    actually deploy, and the thing "is the new zoo better?" is asking about.
    """
    from quantaalpha.eval.ledger import replay_repository
    return [(f"zoo_{i}", expr)
            for i, expr in enumerate(replay_repository(ledger_path))]


def load_library(path: str) -> list[tuple[str, str]]:
    """``[(name, expression)]`` for every factor in the library (cache-checked by the caller)."""
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


def _f(v, pct: bool = False) -> str:
    try:
        if v is None or v != v:
            return "n/a"
        return f"{100 * float(v):+.4f}%" if pct else f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--library", help="score every factor the search produced")
    src.add_argument("--zoo", metavar="LEDGER",
                     help="score the ADMITTED book replayed from this ledger")
    p.add_argument("--protocol", default=None)
    p.add_argument("--ledger", default=None, help="write one JSONL summary row here")
    p.add_argument("--limit", type=int, default=None, help="only use the first N factors")
    p.add_argument("--report", action="store_true",
                   help="score on final_test (else the search_oos proxy)")
    args = p.parse_args()

    theta = load_protocol(args.protocol or default_protocol_path())
    factors = load_zoo(args.zoo) if args.zoo else load_library(args.library)
    source = f"zoo({args.zoo})" if args.zoo else f"library({args.library})"
    if args.limit:
        factors = factors[: args.limit]
    if not factors:
        logger.error("no factors in %s", args.library)
        return 2

    logger.info("theta=%s | factors=%d | window=%s | mode=ONE-SHOT | source=%s",
                theta.hash, len(factors),
                "final_test" if args.report else "search_oos", source)

    op_for_panel = EvaluationOperator(theta)
    p_start, p_end, _ = op_for_panel._windows(args.report)
    panel = op_for_panel._panel(p_start, p_end)

    def _free_gb() -> float:
        """Free + inactive pages, in GB. Inactive counts: macOS reclaims it."""
        import subprocess
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            free = inact = 0
            for line in out.splitlines():
                if "Pages free" in line:
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    inact = int(line.split(":")[1].strip().rstrip("."))
            return (free + inact) * 4096 / 1024 ** 3
        except Exception:
            return float("inf")      # unknown -> do not block the run

    t0 = time.time()
    candidates: dict[str, object] = {}
    hits = 0
    # Memory guard. Each aligned frame on the report panel is ~36 MB resident
    # (measured: 5045 x 933 float64), so a 38-factor book is ~1.3 GB on top of
    # the panel and the fit. This box has 16 GB with swap already near full
    # while the mine runs, and an OOM here would take the mine down with it.
    # Bail out with a clear message rather than thrash.
    MIN_FREE_GB = 0.4
    for name, expr in factors:
        if _free_gb() < MIN_FREE_GB:
            logger.error(
                "aborting: only %.2f GB free after loading %d/%d factors. "
                "Re-run when the machine is quieter (the aligned frames already "
                "written are cached, so the retry is much cheaper).",
                _free_gb(), len(candidates), len(factors))
            return 3
        try:
            # ALIGNED cache: the same 13.3MB frames the mine writes. Aligning
            # here costs 3.57s/factor (a full unstack of a 14.2M-row MultiIndex)
            # and this report loads the whole book at once, so on a 40-factor
            # zoo that is ~2.4 minutes of pure re-derivation per run.
            before = time.time()
            candidates[expr] = load_aligned_signal(expr, panel)
            if time.time() - before < 0.5:
                hits += 1
        except Exception as exc:  # a missing/bad signal must not sink the book
            logger.warning("could not load signal for %s (%s): %s", name, expr[:40], exc)
    logger.info("loaded %d/%d signals in %.1fs (%d from the aligned cache)",
                len(candidates), len(factors), time.time() - t0, hits)
    if not candidates:
        logger.error("no loadable signals")
        return 2

    op = op_for_panel
    t1 = time.time()
    # evaluate is documented to accept the whole factor SET as one book.
    res = op.evaluate(candidates, zoo_signals={}, zoo_metrics=[], report=args.report)
    logger.info("one-shot evaluate: %.1fs", time.time() - t1)

    def g(k):
        return res.get(f"m_{k}")

    print()
    print("=" * 78)
    print(f"ONE-SHOT E_Θ  |  protocol {res.get('theta_hash')}  |  window {res.get('eval_window')}")
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
        Ledger(args.ledger).append({
            "mode": "oneshot", "n_factors": res.get("n_factors"),
            "theta_hash": res.get("theta_hash"), "zoo_size": res.get("zoo_size"),
            "net_ir": g("net_ir"), "net_arr": g("net_arr"), "rank_ic": g("rank_ic"),
            "cost_bps": g("cost_bps"), "turnover_book": g("turnover_book"),
            "base_net_ir": g("base_net_ir"), "base_net_arr": g("base_net_arr"),
            "delta_net_ir": g("delta_net_ir"), "delta_net_arr": g("delta_net_arr"),
            "U": res.get("U"), "eval_window": res.get("eval_window"),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())