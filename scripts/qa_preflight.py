#!/usr/bin/env python
"""Pre-flight checks before committing days of compute to a paper-scale run.

Every check here corresponds to something that actually went wrong in this
project and cost a full run to discover:

* a comparison step that died on ``No module named 'qlib'`` **after** both arms
  had finished mining;
* an evaluation whose repository never accumulated, leaving the objective
  constant and the run scientifically empty;
* `_baseline` and `_strategy_batch` calling each other, which surfaced as a
  RecursionError deep inside numpy;
* an admission gate that silently never engaged across process boundaries;
* a protocol whose splits let the search see the test window.

Exits non-zero if anything would invalidate the run. Runs in seconds.

Usage::

    python scripts/qa_preflight.py --config configs/experiment_paper.yaml
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BAD = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:                       # noqa: BLE001 - report, don't raise
        print(f"[{BAD}] {name}: {type(exc).__name__}: {exc}")
        _failures.append(name)
        return
    if detail is False:
        print(f"[{BAD}] {name}")
        _failures.append(name)
    else:
        print(f"[{OK}] {name}" + (f" — {detail}" if isinstance(detail, str) else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_paper.yaml")
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    args = ap.parse_args()

    print(f"Pre-flight for {args.config}\n" + "=" * 68)

    # --- environment -----------------------------------------------------
    def imports():
        for mod in ("qlib", "lightgbm", "pandas", "numpy", "yaml"):
            importlib.import_module(mod)
        import qlib
        return f"qlib {qlib.__version__}, python {sys.version.split()[0]}"
    check("interpreter can import qlib and lightgbm", imports)

    def disk():
        free = shutil.disk_usage(ROOT).free / 1e9
        if free < args.min_free_gb:
            raise RuntimeError(f"only {free:.0f} GB free, want >= {args.min_free_gb:.0f} GB")
        return f"{free:.0f} GB free"
    check(f"disk has >= {args.min_free_gb:.0f} GB free", disk)

    def env():
        missing = [k for k in ("CHAT_MODEL",) if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"unset: {', '.join(missing)} (source .env)")
        temp = os.environ.get("CHAT_TEMPERATURE", "")
        if temp not in ("0", "0.0"):
            raise RuntimeError(f"CHAT_TEMPERATURE={temp!r}; arms would diverge by sampling noise")
        if not os.environ.get("CHAT_SEED"):
            raise RuntimeError("CHAT_SEED unset; the run would not be reproducible")
        return f"model={os.environ['CHAT_MODEL']} temp={temp} seed={os.environ['CHAT_SEED']}"
    check("LLM env pinned (temperature 0, seed set)", env)

    # --- protocol --------------------------------------------------------
    from quantaalpha.eval.protocol import default_protocol_path, load_protocol
    theta = load_protocol(default_protocol_path())

    check("protocol loads and hashes", lambda: f"theta={theta.hash}")

    def property3():
        s = theta.splits
        for name in (s.search_is, s.search_oos):
            if s.window(name)[1] >= s.final_test[0]:
                raise RuntimeError(f"split {name} reaches the test window")
        if theta.walk_forward.enabled:
            for tr, va in s.walk_forward_folds(int(theta.walk_forward.folds)):
                if va[1] >= s.final_test[0]:
                    raise RuntimeError(f"walk-forward fold {va} reaches the test window")
        return f"test window {s.final_test[0]}..{s.final_test[1]} is unreachable by the search"
    check("Property 3: search cannot see the test window", property3)

    def admission():
        a = theta.admission
        if not a.enabled:
            return "gating disabled (zoo will equal total mined)"
        if not 0 <= a.tau_evict <= a.tau_admit <= 1:
            raise RuntimeError("tau_evict must be <= tau_admit")
        if a.tau_evict >= a.tau_admit:
            raise RuntimeError("no hysteresis: the repository will oscillate")
        return f"admit>={a.tau_admit} evict<{a.tau_evict} min_size={a.min_size}"
    check("admission thresholds have hysteresis", admission)

    # --- the failure modes that cost whole runs --------------------------
    def no_recursion():
        import inspect

        from quantaalpha.eval.operator import EvaluationOperator
        src = inspect.getsource(EvaluationOperator._baseline)
        if "_strategy_batch" in src:
            raise RuntimeError("_baseline calls _strategy_batch — infinite recursion")
        return "_baseline and _strategy_batch do not call each other"
    check("no evaluation recursion", no_recursion)

    def gate_engages():
        """The gate must see the rehydrated repository, not an empty in-process one."""
        import inspect

        from quantaalpha.factors.net_cost_runner import NetCostFactorRunner
        zoo_src = inspect.getsource(NetCostFactorRunner._zoo)
        if "self._repository[expr]" not in zoo_src:
            raise RuntimeError("_zoo does not populate self._repository; the gate cannot engage")
        dev = inspect.getsource(NetCostFactorRunner.develop)
        if dev.index("self._admit") > dev.index("self.ledger.append"):
            raise RuntimeError("ledger is written before the admission decision")
        return "rehydration populates the repository; admission precedes the ledger write"
    check("admission gate engages across processes", gate_engages)

    def utility_discriminates():
        from quantaalpha.eval.scoring import rank, utility
        if rank(1.0, []) != 0.5:
            raise RuntimeError("empty-repository rank is not neutral; batch 1 would dominate")
        # Respect each dimension's direction: turnover and rho_max are
        # lower-is-better, so negating them would make the "bad" batch better.
        good = {"delta_net_ir": 1.0, "net_arr": .2, "rank_icir": 1.0, "turnover_book": .02,
                "rho_max": .1, "decay_slope": 0., "cx": 5, "is_oos_gap": .01}
        bad = {"delta_net_ir": -1.0, "net_arr": -.2, "rank_icir": -1.0, "turnover_book": .5,
               "rho_max": .95, "decay_slope": -.1, "cx": 60, "is_oos_gap": .9}
        zoo = [bad] * 3
        if utility(good, zoo, theta) <= utility(bad, zoo, theta):
            raise RuntimeError("U does not rank a good batch above a bad one")
        return "U is neutral on an empty repository and discriminates on a populated one"
    check("utility is well-behaved", utility_discriminates)

    def constraints():
        c = theta.constraints
        if not c.enabled:
            return "constraints disabled — results will not be A-share realistic"
        return (f"price_limits={c.enforce_price_limits} suspension={c.enforce_suspension} "
                f"t1={c.enforce_t1} (t1 cannot bind at daily frequency)")
    check("A-share constraints configured", constraints)

    def turnover_can_move():
        if not theta.portfolio.cost_aware_dropout:
            raise RuntimeError(
                "cost_aware_dropout is off: turnover is pinned at n_drop/topk = "
                f"{theta.portfolio.n_drop / theta.portfolio.topk:.4f} and the cost model "
                "cannot influence the book"
            )
        return f"cost-aware dropout on, hurdle={theta.portfolio.swap_hurdle}"
    check("turnover is a decision, not a constant", turnover_can_move)

    # --- data ------------------------------------------------------------
    def data_ok():
        import yaml
        cfg = yaml.safe_load((ROOT / args.config).read_text())
        provider = Path(os.path.expanduser(
            cfg.get("data", {}).get("provider_uri", "~/.qlib/qlib_data/cn_data")))
        local = ROOT / "data/qlib/cn_data"
        root = local if local.exists() else provider
        if not root.exists():
            raise RuntimeError(f"qlib data not found at {root}")
        return str(root)
    check("qlib data present", data_ok)

    def expected_count():
        import yaml

        from quantaalpha.pipeline.evolution.controller import expected_factor_count
        cfg = yaml.safe_load((ROOT / args.config).read_text())
        ev = cfg.get("evolution", {})
        n = expected_factor_count(cfg.get("planning", {}).get("num_directions", 2),
                                  ev.get("crossover_n", 2), ev.get("max_rounds", 3),
                                  cfg.get("factor", {}).get("factors_per_hypothesis", 1))
        if n <= 0:
            raise RuntimeError("config would mine no factors")
        return f"{n} factors expected per arm (Arm B mines until it ADMITS this many)"
    check("config produces a sane factor target", expected_count)

    print("=" * 68)
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED: {', '.join(_failures)}")
        print("Fix these before starting the run.")
        return 1
    print("All checks passed. Safe to start the paper-scale run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
