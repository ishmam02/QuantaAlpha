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
from quantaalpha.eval.data import load_factor_signal
from quantaalpha.eval.ledger import DEFAULT_LEDGER_PATH, Ledger
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import default_protocol_path, load_protocol
from quantaalpha.eval.scoring import utility
from quantaalpha.factors.experiment import QlibFactorExperiment
from quantaalpha.factors.runner import QlibFactorRunner
from quantaalpha.llm.client import md5_hash

logger = logging.getLogger(__name__)


def _bridge_eval_logging() -> None:
    """Route ``quantaalpha.eval.*`` stdlib records into the repo's loguru sink.

    The eval package uses stdlib ``logging`` so it stays independent of
    QuantaAlpha, but nothing in this repo ever configures stdlib logging -- the
    root logger has no handlers and defaults to WARNING. Every ``logger.info``
    in the engine was therefore discarded, including the per-evaluation
    "E_theta eval:" line that is supposed to be the primary signal that the new
    objective is live. Without this bridge the only evidence a run produces is
    the ledger file.
    """
    from quantaalpha.log import logger as qa_logger

    # RDAgentLog exposes only info/warning/error, each taking a single message.
    class _ToLoguru(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                sink = {
                    "WARNING": qa_logger.warning,
                    "ERROR": qa_logger.error,
                    "CRITICAL": qa_logger.error,
                }.get(record.levelname, qa_logger.info)
                sink(self.format(record))
            except Exception:
                pass

    eval_logger = logging.getLogger("quantaalpha.eval")
    if not any(isinstance(h, _ToLoguru) for h in eval_logger.handlers):
        eval_logger.addHandler(_ToLoguru())
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = False

    own = logging.getLogger(__name__)
    if not any(isinstance(h, _ToLoguru) for h in own.handlers):
        own.addHandler(_ToLoguru())
        own.setLevel(logging.INFO)
        own.propagate = False

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
        _bridge_eval_logging()
        self.theta = load_protocol(default_protocol_path())
        self.op = EvaluationOperator(self.theta)
        self.ledger = Ledger(os.environ.get("QA_LEDGER", DEFAULT_LEDGER_PATH))
        # THE effective-alpha repository. Ordered {expression -> (signal, metrics)},
        # appended to only when a factor is admissible. Deriving both the
        # combination inputs and the ranking incumbents from this one object is
        # what keeps Eq. 2 and Eq. 11 synchronized on the same repository state:
        # two separate accumulators would silently drift apart.
        self._repository: dict[str, tuple[Any, dict]] = {}
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

        # A based/container experiment carries accumulated state but no factors
        # of its own. The recursion above hands us exactly that, and raising
        # FactorEmptyError here aborts the whole develop before the real
        # candidate -- which does have sub-workspaces -- is ever evaluated.
        # QlibFactorRunner never hits this because it guards its entire
        # factor-loading block behind `if exp.based_experiments:`, so a based
        # experiment skips straight to qrun.
        if not (getattr(exp, "sub_workspace_list", None) or []):
            logger.info(
                "Experiment has no sub-workspaces (based/container experiment); "
                "nothing for E_theta to evaluate here"
            )
            if getattr(exp, "result", None) is None:
                exp.result = pd.Series(dtype=float)
            return exp

        # Shares the control arm's recovery path: a missing result.h5 is re-run
        # rather than treated as a dead loop. Calling process_factor_data
        # directly here made a single missing output skip the whole loop before
        # E_theta was ever reached.
        new_factors = self.acquire_new_factors(exp)
        if new_factors is None or new_factors.empty:
            # Preserve the skip-loop semantics at workflow.py:116.
            raise FactorEmptyError("No valid factor data found to merge.")

        # One repository state, read once, used for BOTH the combiner refit and
        # the relative ranking (Eq. 2 and Eq. 11 synchronized at the evaluation).
        # Held fixed across every candidate in this experiment: admitting a
        # sibling mid-loop would make the score depend on column ordering.
        zoo_signals, zoo_metrics = self._zoo(exp)
        expressions = self._expressions(exp, new_factors.columns)

        # ONE evaluation of the whole batch, matching how the control arm's qrun
        # scores a round: all of the round's factors enter the combined model
        # together and the book is measured once. Evaluating them individually
        # made the arms structurally incomparable.
        candidates = {}
        for column in new_factors.columns:
            expr = expressions.get(column, str(column))
            signal = new_factors[column].dropna()
            if signal.empty:
                logger.warning("candidate %s produced an empty signal; skipping", column)
                continue
            candidates[expr] = signal

        if not candidates:
            raise FactorEmptyError("E_theta produced no evaluable factors.")

        try:
            res = self.op.evaluate(candidates, zoo_signals=zoo_signals, zoo_metrics=zoo_metrics)
        except Exception:
            logger.exception("E_theta evaluation failed for this experiment")
            raise FactorEmptyError("E_theta evaluation failed.")

        names = [str(c) for c in new_factors.columns]
        res["factor_name"] = ", ".join(names)
        self.ledger.append(
            {
                "factor_ids": [md5_hash(f"{n}_{e}")[:16] for n, e in zip(names, candidates)],
                "factor_names": names,
                "factor_exprs": list(candidates),
                "n_factors": len(candidates),
                "theta_hash": res["theta_hash"],
                "zoo_hash": res["zoo_hash"],
                "zoo_size": res["zoo_size"],
                "metrics": {k[2:]: v for k, v in res.items() if k.startswith("m_")},
                "e": {k[2:]: v for k, v in res.items() if k.startswith("e_")},
                "U": res["U"],
            }
        )

        batch_metrics = {k[2:]: v for k, v in res.items() if k.startswith("m_")}
        admitted = self._admit(float(res["U"]), candidates, batch_metrics)
        res["admitted"] = admitted
        if admitted:
            evicted = self._prune()
            res["evicted"] = len(evicted)

        exp.result = self._to_series(res)
        return exp

    # ------------------------------------------------------------------
    def _admit(self, u: float, candidates: dict, batch_metrics: dict) -> bool:
        """Gate the batch on ``U`` (Eq. 12) against Θ's admission bar.

        ``U`` is a weighted mean of percentile complements, so ``tau_admit=0.5``
        means "better than the median incumbent". That is the adaptive bar of
        §3.4: the number is constant while the standard it encodes rises as the
        repository improves.

        The first ``min_size`` factors are admitted unconditionally. They are
        scored against an empty or near-empty repository, where the rank carries
        no information (``scoring.rank`` returns a neutral 0.5), so gating on it
        would be arbitrary.
        """
        adm = self.theta.admission
        n = len(self._repository)
        if not adm.enabled or n < adm.min_size:
            reason = "gating disabled" if not adm.enabled else (
                f"bootstrapping (|zoo|={n} < min_size={adm.min_size})")
            for expr, signal in candidates.items():
                self._repository[expr] = (signal, batch_metrics)
            logger.info("repository: +%d factor(s) admitted unconditionally, %s; |zoo| = %d",
                        len(candidates), reason, len(self._repository))
            return True

        if u < adm.tau_admit:
            logger.info(
                "repository: REJECTED %d factor(s), U=%.4f < tau_admit=%.2f; |zoo| = %d",
                len(candidates), u, adm.tau_admit, n,
            )
            return False

        for expr, signal in candidates.items():
            self._repository[expr] = (signal, batch_metrics)
        logger.info("repository: +%d factor(s), U=%.4f >= tau_admit=%.2f; |zoo| = %d",
                    len(candidates), u, adm.tau_admit, len(self._repository))
        return True

    def _prune(self) -> list[str]:
        """Evict incumbents that have fallen behind the repository they are in.

        Each member is re-scored against the repository *excluding itself*, so
        the comparison is against its peers rather than against a set containing
        it. A member drops out when that score falls below ``tau_evict``.

        Eviction uses a lower bar than admission on purpose. ``U`` is a
        percentile, so about half of any repository sits below the median at any
        moment; evicting at ``tau_admit`` would drop half the members every
        round and cascade toward empty. The gap is hysteresis -- a factor must
        fall clearly behind, not merely below average.

        Granularity note: factors admitted in the same batch share one metric
        vector, so they are ranked identically and leave together. Eviction is
        therefore effectively per-batch, which is the finest granularity the
        batch evaluation supports.
        """
        adm = self.theta.admission
        if not adm.enabled or len(self._repository) <= adm.min_size:
            return []

        entries = list(self._repository.items())
        scored = []
        for expr, (signal, metrics) in entries:
            peers = [m for other, (_, m) in entries if other != expr]
            scored.append((utility(metrics, peers, self.theta), expr, signal, metrics))

        # Never shrink below min_size: if too many fall behind, keep the best.
        scored.sort(key=lambda row: row[0], reverse=True)
        keep_floor = {row[1] for row in scored[: adm.min_size]}
        evicted = [
            expr for u, expr, _, _ in scored
            if u < adm.tau_evict and expr not in keep_floor
        ]
        for expr in evicted:
            del self._repository[expr]
        if evicted:
            logger.info(
                "repository: EVICTED %d factor(s) with U < tau_evict=%.2f; |zoo| = %d",
                len(evicted), adm.tau_evict, len(self._repository),
            )
        return evicted

    # ------------------------------------------------------------------
    def _zoo(self, exp: QlibFactorExperiment) -> tuple[dict[str, Any], list[dict]]:
        """The effective-alpha repository: admissible factors only.

        Returns ``(signals, metrics)`` drawn from the *same* ordered store, so
        the combiner refit (Eq. 2) and the relative ranking (Eq. 11) always see
        an identical repository state.

        Note this deliberately does **not** use ``exp.based_experiments``
        wholesale the way ``runner.py:89`` does. That chain carries every factor
        the loop has produced, admissible or not; the repository is by
        definition the subset that passed evaluation.

        **Rehydrated from the ledger, because the in-memory store does not
        survive.** Under ``parallel_execution``/``parallel_enabled`` every
        experiment gets its own process, hence its own runner and its own empty
        ``_repository``. Measured on the 20260728_103159 run: all six batches
        evaluated against ``zoo_size=0`` and scored an identical ``U=0.9500``
        despite net ARR spanning -6.72% to +9.61% -- with nothing to rank
        against, the repository-relative scoring collapses and the objective
        stops discriminating at all. The ledger is the fix because it is written
        synchronously inside this runner, immediately after each evaluation, so
        a sibling process can always see it.

        Metrics and signals recover independently, and the distinction matters:

        * **Metrics** drive Eq. 11's relative ranking and come straight from the
          ledger, so they are always recoverable. This is what restores ``U``.
        * **Signals** drive the combiner refit and ``rho_max``, and are loaded
          from the md5 cache that ``library.py:232`` writes. That write happens
          when the pipeline saves the library, which may lag a sibling process,
          so signal recovery is best-effort and is logged.

        A partially recovered zoo still ranks correctly; it just refits the
        combiner on fewer incumbents.
        """
        signals = {expr: signal for expr, (signal, _) in self._repository.items()}
        metrics = [m for _, (_, m) in self._repository.items()]

        recovered_metrics = recovered_signals = 0
        for record in self.ledger.read():
            batch_metrics = record.get("metrics") or {}
            for expr in record.get("factor_exprs") or []:
                if not expr or expr in signals or expr in self._repository:
                    continue
                metrics.append(batch_metrics)
                recovered_metrics += 1
                try:
                    signals[expr] = load_factor_signal(expr)
                    recovered_signals += 1
                except (FileNotFoundError, OSError, ValueError):
                    # Not yet cached by a sibling process. The factor still
                    # counts as a ranking incumbent; it just cannot join the
                    # combiner refit this time round.
                    continue

        if recovered_metrics:
            logger.info(
                "repository rehydrated from ledger: +%d incumbent(s) for ranking, "
                "%d with signals for the refit (in-process held %d)",
                recovered_metrics, recovered_signals, len(self._repository),
            )
        if not metrics:
            logger.info(
                "repository empty (%d based experiment(s) present, ledger has no "
                "prior records); scoring this batch against an empty zoo -- U "
                "cannot discriminate until the repository has incumbents",
                len(exp.based_experiments or []),
            )
        return signals, metrics

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
            "n_factors": res.get("n_factors"),
            # The real admission decision, not a constant. This is what the
            # library reads to emit the zoo subset, so a rejected batch is
            # absent from <library>_zoo.json while still appearing in the full
            # library -- which is the whole point of keeping both files.
            "in_zoo": bool(res.get("admitted", True)),
            "theta_hash": res.get("theta_hash"),
            "zoo_hash": res.get("zoo_hash"),
            "zoo_size": res.get("zoo_size"),
            "rho_max": res.get("m_rho_max"),
            "turnover_book": res.get("m_turnover_book"),
            "turnover_solo": res.get("m_turnover_solo"),
            "cx": res.get("m_cx"),
            "cost_bps": res.get("m_cost_bps"),
        }
        payload.update({k: v for k, v in res.items() if k.startswith("e_")})
        return pd.Series(payload)


__all__ = ["NetCostFactorRunner"]
