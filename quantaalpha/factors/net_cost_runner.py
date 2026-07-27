"""Treatment-arm runner: evaluates factors with ``E_Θ`` instead of Qlib.

Swapped in through the existing plugin mechanism
(``QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner``),
so the loop, the DSL, CoSTEER and the mutation/crossover operators are all
untouched — the generation process is held fixed and only the *evaluation*
changes, which is the controlled-A/B condition the whole design rests on.

Two integration hazards are handled here:

* **Pickle-cache collision.** ``CachedRunner.get_cache_key`` hashes task-info
  strings only — no objective, no protocol — so without the override below the
  treatment arm would silently serve control-arm results from cache and the
  A/B would compare nothing at all. The protocol hash goes into the key.
* **``_extract_metrics`` is pandas-only.** ``controller.py`` reads metrics off a
  ``pd.Series``/``pd.DataFrame``; a plain dict yields all-``None`` metrics and
  therefore zero successful trajectories, silently. ``exp.result`` is a
  ``pd.Series`` carrying the canonical Qlib metric names *and* the new ones.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from quantaalpha.components.runner import CachedRunner
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.eval.ledger import DEFAULT_LEDGER_PATH, Ledger
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import default_protocol_path, load_protocol
from quantaalpha.factors.experiment import QlibFactorExperiment
from quantaalpha.factors.runner import QlibFactorRunner
from quantaalpha.llm.client import md5_hash

logger = logging.getLogger(__name__)

# Canonical names `controller._extract_metrics` already knows how to read, so
# the control and treatment arms stay mutually reportable.
_CANONICAL = ("IC", "ICIR", "RankIC", "RankICIR", "annualized_return", "information_ratio", "max_drawdown")


def _net_cost_cache_key(self: "NetCostFactorRunner", exp: Any, **kwargs: Any) -> str:
    """Module-level indirection so the override dispatches on the instance."""
    return self.get_cache_key(exp, **kwargs)


class NetCostFactorRunner(QlibFactorRunner):
    """``QlibFactorRunner`` with ``E_Θ`` in place of the Qlib backtest."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.theta = load_protocol(default_protocol_path())
        self.op = EvaluationOperator(self.theta)
        self.ledger = Ledger(os.environ.get("QA_LEDGER", DEFAULT_LEDGER_PATH))
        self._zoo_metrics: list[dict] = []
        logger.info(
            "NetCostFactorRunner active: theta=%s protocol=%s ledger=%s",
            self.theta.hash, default_protocol_path(), self.ledger.path,
        )

    # ------------------------------------------------------------------
    def get_cache_key(self, exp: Any, **kwargs: Any) -> str:
        """Task info **plus the protocol hash** (Hazard 2).

        Two arms with different objectives must not share a cache entry; the
        base key cannot tell them apart.
        """
        base = super().get_cache_key(exp, **kwargs)
        return md5_hash(f"{base}|{self.theta.hash}")

    # ------------------------------------------------------------------
    @cache_with_pickle(_net_cost_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1], use_local=use_local)

        new_factors = self.process_factor_data(exp)
        if new_factors is None or new_factors.empty:
            # Preserve the skip-loop semantics at workflow.py:116.
            raise FactorEmptyError("No valid factor data found to merge.")

        zoo_signals = self._zoo(exp)
        expressions = self._expressions(exp, new_factors.columns)

        results: list[dict] = []
        for column in new_factors.columns:
            expr = expressions.get(column, str(column))
            signal = new_factors[column].dropna()
            if signal.empty:
                logger.warning("candidate %s produced an empty signal; skipping", column)
                continue
            try:
                res = self.op.evaluate(signal, expr, zoo_signals, self._zoo_metrics)
            except Exception:
                logger.exception("E_theta evaluation failed for %s", column)
                continue
            res["factor_name"] = str(column)
            results.append(res)
            self.ledger.append(
                {
                    "factor_id": md5_hash(f"{column}_{expr}")[:16],
                    "factor_name": str(column),
                    "factor_expr": expr,
                    "theta_hash": res["theta_hash"],
                    "zoo_hash": res["zoo_hash"],
                    "zoo_size": res["zoo_size"],
                    "metrics": {k[2:]: v for k, v in res.items() if k.startswith("m_")},
                    "e": {k[2:]: v for k, v in res.items() if k.startswith("e_")},
                    "U": res["U"],
                    "feasible": res["feasible"],
                    "failed_gates": res["failed_gates"],
                }
            )

        if not results:
            raise FactorEmptyError("E_theta produced no evaluable factors.")

        best = self._best(results)
        # Incumbents for the next candidate's repository-relative ranking.
        self._zoo_metrics.extend({k[2:]: v for k, v in r.items() if k.startswith("m_")} for r in results)

        exp.result = self._to_series(best)
        return exp

    # ------------------------------------------------------------------
    @staticmethod
    def _best(results: list[dict]) -> dict:
        """The trajectory's fitness is its best factor, feasible ones first."""
        feasible = [r for r in results if r.get("feasible")]
        pool = feasible or results
        return max(pool, key=lambda r: (r.get("U") if r.get("U") == r.get("U") else float("-inf")))

    def _zoo(self, exp: QlibFactorExperiment) -> dict[str, pd.Series]:
        """Repository signals — ``runner.py:89`` already builds exactly this."""
        if not exp.based_experiments:
            return {}
        try:
            frame = self.process_factor_data(exp.based_experiments)
        except FactorEmptyError:
            logger.warning("no SOTA factor data available; scoring against an empty zoo")
            return {}
        if frame is None or frame.empty:
            return {}
        names = self._expressions(exp.based_experiments, frame.columns)
        return {names.get(c, str(c)): frame[c].dropna() for c in frame.columns}

    @staticmethod
    def _expressions(exp_or_list: Any, columns: Any) -> dict[str, str]:
        """Map factor column name → factor expression, via the sub-tasks."""
        experiments = exp_or_list if isinstance(exp_or_list, list) else [exp_or_list]
        out: dict[str, str] = {}
        for experiment in experiments:
            for task in getattr(experiment, "sub_tasks", []) or []:
                name = getattr(task, "factor_name", None)
                expr = getattr(task, "factor_expression", "")
                if name and expr:
                    out[str(name)] = expr
        return out

    def _to_series(self, res: dict) -> pd.Series:
        """Flatten one evaluation into the Series ``controller`` can read.

        ``annualized_return``/``information_ratio`` are mapped from the **net**
        figures deliberately: the headline numbers the rest of the system
        reports are then the net-of-cost ones, which is the point.
        """
        payload: dict[str, Any] = {
            "IC": res.get("m_ic"),
            "ICIR": res.get("m_icir"),
            "RankIC": res.get("m_rank_ic"),
            "RankICIR": res.get("m_rank_icir"),
            "annualized_return": res.get("m_net_arr"),
            "information_ratio": res.get("m_net_ir"),
            "max_drawdown": res.get("m_mdd"),
            # Canonical snake_case duplicates so gates.describe_failures and the
            # feedback block can read the same names the ledger records.
            "rank_ic": res.get("m_rank_ic"),
            "rank_icir": res.get("m_rank_icir"),
            "net_ir": res.get("m_net_ir"),
            "net_arr": res.get("m_net_arr"),
            "mdd": res.get("m_mdd"),
            "U": res.get("U"),
            "feasible": res.get("feasible"),
            "theta_hash": res.get("theta_hash"),
            "zoo_hash": res.get("zoo_hash"),
            "zoo_size": res.get("zoo_size"),
            "rho_max": res.get("m_rho_max"),
            "turnover_book": res.get("m_turnover_book"),
            "turnover_solo": res.get("m_turnover_solo"),
            "cx": res.get("m_cx"),
            "cost_bps": res.get("m_cost_bps"),
            "failed_gates": ",".join(res.get("failed_gates") or []),
        }
        payload.update({k: v for k, v in res.items() if k.startswith("e_")})
        return pd.Series(payload)


__all__ = ["NetCostFactorRunner"]
