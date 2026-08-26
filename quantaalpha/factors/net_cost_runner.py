"""The net-cost runner: evaluates factors with ``E_Θ`` instead of Qlib.

Swapped in through the existing plugin mechanism
(``QLIB_FACTOR_RUNNER=quantaalpha.factors.net_cost_runner.NetCostFactorRunner``),
so the loop, the DSL, CoSTEER and the mutation/crossover operators are all
untouched — the generation process is held fixed and only the *evaluation*
changes.

Two integration hazards are handled here:

* **Pickle-cache collision.** ``CachedRunner.get_cache_key`` hashes task-info
  strings only — no objective, no protocol — so without the override below a
  stale Qlib result would silently be served from cache. The protocol hash
  goes into the key.
* **``_extract_metrics`` is pandas-only.** ``controller.py`` reads metrics off a
  ``pd.Series``/``pd.DataFrame``; a plain dict yields all-``None`` metrics and
  therefore zero successful trajectories, silently. ``exp.result`` is a
  ``pd.Series`` carrying the canonical Qlib metric names *and* the new ones.
"""

from __future__ import annotations

import logging
from functools import lru_cache
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace as _replace
from typing import Any

import pandas as pd

from quantaalpha.components.runner import CachedRunner
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.eval.admission import PopulationStats
from quantaalpha.eval.combiner import clear_cache as _clear_combiner_cache
from quantaalpha.eval.data import (align_signal, load_aligned_signal,
                                   load_factor_signal)
from quantaalpha.eval.ledger import (
    DEFAULT_LEDGER_PATH, Ledger, replay_repository, replay_t_history,
)
from quantaalpha.eval.operator import EvaluationOperator
from quantaalpha.eval.protocol import default_protocol_path, load_protocol
from quantaalpha.eval.scoring import utility
from quantaalpha.factors.experiment import QlibFactorExperiment
from quantaalpha.factors.runner import QlibFactorRunner
from quantaalpha.llm.client import md5_hash

logger = logging.getLogger(__name__)


def derive_sign_from_mechanism(mechanism: str) -> str:
    """Derive the predicted IC sign from the mechanism's STATED direction.

    The sign-match gate (``require_sign_match``) used to read a freely-set
    ``expected_ic_sign`` field, so a refine could flip the sign token without
    rewriting the mechanism -- the gate then passed (``sign_predicted ==
    sign_realized``) on the FLIPPED field while the mechanism text still stated
    the opposite direction. Measured 2026-08-24: a rejected "predicts HIGHER
    returns" factor was re-proposed with ``sign_predicted`` flipped to negative
    but the mechanism text byte-identical, and admitted -- the falsifiability
    was hollow and the "rescue" was a sign-bit flip, not a mechanism correction.

    Deriving the sign from the mechanism text closes that: to change the sign
    the model must rewrite the stated direction. The primary claim is the clause
    before any "conversely ..." contrast (the factor-HIGH direction). Returns
    'positive' / 'negative' / '' (unparseable -> the gate's no-direction branch
    rejects, which is safe: a direction the system cannot read cannot be lied
    about).
    """
    t = (mechanism or "").lower()
    if not t:
        return ""
    # primary clause = before any contrasting clause ("conversely ...")
    cut = len(t)
    for _sep in ("conversely", "whereas", "on the other hand", "; however"):
        _i = t.find(_sep)
        if 0 <= _i < cut:
            cut = _i
    primary = t[:cut]
    pos = any(_k in primary for _k in (
        "predicts higher", "higher forward return", "higher future return",
        "higher subsequent return", "predicts a higher", "higher return",
        "outperform", "positive return", "returns rise", "returns increase",
        "increasing return"))
    neg = any(_k in primary for _k in (
        "predicts lower", "lower forward return", "lower future return",
        "lower subsequent return", "predicts a lower", "lower return",
        "underperform", "negative return", "returns fall", "returns decrease",
        "decreasing return"))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return ""


# ------------------------------------------------------------------
# Parallel seed-evaluation (fork ProcessPool).
#
# _decide_marginal scores the batch across the admission test_seeds; each seed
# is an independent E_theta evaluation (its own combiner refit + book
# construction), so they are embarrassingly parallel. Done sequentially this is
# the 5x multiplier on the per-batch cost; a ProcessPool runs them concurrently.
#
# fork (not spawn) is safe here because the ICIR linear combiner never imports
# LightGBM/OpenMP -- the only reason the replay scripts ever used spawn was to
# dodge a fork-after-OpenMP deadlock, and that constraint is gone. With fork the
# workers inherit the main process's memory copy-on-write: the panel, the trade
# mask, the benchmark and the pre-seeded per-seed operators are shared, not
# reloaded per worker (one cold start, not five). Per-batch candidate/zoo
# signals are stashed in _MP_STATE before the pool is created so the forked
# workers inherit them COW too -- no large-DataFrame pickling across the pipe.
#
# A fresh pool per batch keeps the model simple and loses nothing: the only
# cross-batch reusable cache (the panel) is pre-seeded into self._seed_ops from
# self.op in the MAIN process, so workers inherit it COW; the baseline cache is
# keyed by zoo_hash, which changes every batch, so it is never reused anyway.
# ------------------------------------------------------------------
_MP_STATE: dict[str, Any] = {}
_SEED_OPS: dict[int, EvaluationOperator] = {}


def _seed_worker_eval(seed: int) -> dict:
    """Evaluate one seed's marginal contribution in a forked worker.

    Reads the batch's candidates/zoo and the pre-seeded per-seed operator from
    module globals (inherited copy-on-write at fork). Returns the E_theta
    metrics dict, or a stub with ``m_delta_net_ir=None`` on failure so one bad
    seed does not kill the decision.
    """
    try:
        op = _SEED_OPS[int(seed)]
        return op.evaluate(
            _MP_STATE["candidates"],
            zoo_signals=_MP_STATE["zoo_signals"],
            zoo_metrics=_MP_STATE["zoo_metrics"],
        )
    except Exception as ex:  # noqa: BLE001 -- one bad seed must not kill the run
        logger.exception("marginal contribution failed on seed %s", seed)
        return {"m_delta_net_ir": None, "_error": f"{type(ex).__name__}: {ex}"}


_EVICT_STATE: dict[str, Any] = {}


def _evict_worker_eval(held_out: str) -> tuple[str, float | None]:
    """Re-price the repository WITHOUT ``held_out``, in a forked worker.

    The leave-one-out eviction loop is the single most expensive thing the
    runner does -- one full evaluation per repository member, and the
    repository is the thing that grows (at |zoo|=150 that is 151 sequential
    evaluations, hours of wall-clock, on one core while the rest of the box
    idles). The members are independent re-pricings of the same fixed
    repository, so they parallelize exactly like the seed evaluations do.

    Reads the entries + operator from module globals inherited copy-on-write at
    fork, so the panel and the signals are shared, not pickled per task; only
    the held-out expression string crosses the pipe. Returns
    ``(expr, net_ir_without)``, with ``None`` when the re-pricing fails, so one
    bad member cannot kill the eviction round.
    """
    try:
        op = _EVICT_STATE["op"]
        entries = _EVICT_STATE["entries"]
        others = {e: s for e, (s, _) in entries if e != held_out}
        metrics = [m for e, (_, m) in entries if e != held_out]
        without = op.evaluate({}, zoo_signals=others, zoo_metrics=metrics)
        w_ir = without.get("m_net_ir")
        if w_ir is None or w_ir != w_ir:
            return held_out, None
        return held_out, float(w_ir)
    except Exception:  # noqa: BLE001 -- one bad member must not kill the round
        logger.exception("eviction: could not re-price %s", held_out[:60])
        return held_out, None


def _short(expr: str, n: int = 40) -> str:
    """A factor expression trimmed for a one-line log/reason string."""
    e = " ".join(str(expr).split())
    return e if len(e) <= n else e[: n - 1] + "…"


def _label_horizons(theta) -> list[int]:
    """Forward-return horizons (in days) admission scores a factor against.

    ``QA_LABEL_HORIZONS=1,5,20`` overrides; otherwise ``[1]``, which reproduces
    the single-horizon behaviour exactly. Always includes 1 so the protocol's
    own ``label_expr`` horizon is never dropped, and is deduplicated/sorted so
    the reason string is stable.

    Kept OUT of Θ deliberately: the horizons change which factors are ADMITTED,
    not how the book is priced -- the portfolio still fills open[t+1] and
    rebalances daily at every setting.
    """
    raw = os.environ.get("QA_LABEL_HORIZONS")
    if not raw:
        # Theta decides; the env var is only an override for experiments.
        cfg = getattr(getattr(theta, "admission", None), "label_horizons", None)
        if cfg:
            return sorted({max(1, int(h)) for h in cfg})
        return [1]
    out = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        try:
            h = int(part)
        except ValueError:
            logger.warning("QA_LABEL_HORIZONS: ignoring %r", part)
            continue
        if h >= 1:
            out.add(h)
    out.add(1)
    return sorted(out)


def _seed_workers() -> int:
    """How many parallel seed-eval processes to run.

    Defaults to the number of admission test_seeds (one worker per seed).
    ``QA_SEED_WORKERS`` caps it (e.g. ``QA_SEED_WORKERS=2`` on a tight box), and
    ``=1`` or ``=0`` falls back to the original sequential loop. Never more than
    the seed count -- extra workers would just idle.
    """
    raw = os.environ.get("QA_SEED_WORKERS")
    if raw is None:
        return 5
    try:
        n = int(raw)
    except ValueError:
        return 5
    return max(0, n)


_BOOK_BREAKEVEN_CACHE: dict = {}


def _book_breakeven(theta) -> float | None:
    """The |IC| at which a book turns net-profitable under this protocol.

    Read, never derived: an override, else the ladder artifact keyed by the
    protocol hash, else ``None``. Cached because it is read once per factor and
    the answer cannot change inside a run.
    """
    # Keyed on the protocol HASH, not the object: Protocol carries dict fields
    # and is unhashable, so @lru_cache on it raised TypeError for every factor.
    _h = getattr(theta, "hash", None)
    _h = _h() if callable(_h) else _h
    if _h in _BOOK_BREAKEVEN_CACHE:
        return _BOOK_BREAKEVEN_CACHE[_h]

    def _remember(v):
        _BOOK_BREAKEVEN_CACHE[_h] = v
        return v

    raw = os.environ.get("QA_IC_BREAKEVEN_BOOK")
    if raw:
        try:
            return _remember(float(raw))
        except ValueError:
            logger.warning(f"QA_IC_BREAKEVEN_BOOK={raw!r} is not a number; ignoring")
    try:
        import json as _json
        from pathlib import Path as _Path
        h = getattr(theta, "hash", None)
        h = h() if callable(h) else h
        p = _Path("data/results") / f"ic_breakeven_{h}.json"
        if p.exists():
            v = _json.loads(p.read_text()).get("breakeven_rank_ic")
            return _remember(float(v) if v is not None else None)
    except Exception as exc:
        logger.warning(f"could not read the measured book break-even: {exc}")
    return _remember(None)


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
# trajectories stay mutually reportable with canonical Qlib names.
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
        # Every batch's utility, admitted or not -- the reference the generator
        # is told its percentile against. Deliberately NOT the repository, which
        # is a winners-only sample.
        self._population = PopulationStats()
        # Last measured marginal contribution per admitted expression, so a full
        # repository knows which incumbent a candidate would have to displace.
        self._contributions: dict[str, float] = {}
        # Cadence counter for contribution-based eviction.
        self._rounds_since_evict = 0
        # Across-batch cache for the marginal-er gate's repo-repo |Spearman|
        # block (the O(zoo**2) term). The block is a pure function of the held
        # signals on the eval panel, so it is keyed by the repo COMPOSITION
        # plus the (start, end) window the panel was built on. A reject batch
        # leaves the zoo unchanged -> the key matches -> the block is reused
        # instead of rebuilt (O(zoo**2) -> O(1) on ~60% of batches). An
        # admit/replace/evict changes the composition -> key mismatch ->
        # rebuild once, then cached. n_factors is 1 (one candidate per batch),
        # which is what made the old per-call local cache a build-and-discard
        # no-op and the instance cache the actual win. One entry only.
        self._mer_cache = None
        # One EvaluationOperator per admission test seed, reused for the
        # whole run so the panel and baseline caches survive across batches.
        self._seed_ops: dict[int, EvaluationOperator] = {}
        logger.info(
            "NetCostFactorRunner active: theta=%s protocol=%s ledger=%s",
            self.theta.hash, default_protocol_path(), self.ledger.path,
        )
        # Fail-loud guard against the silent frictionless fallback. This runner
        # IS the net-of-cost objective, so its feedback PROSE must come from
        # NetCostFactorFeedback -- that is where the U / e_j / turnover / cost
        # block the generator reads is rendered. If the summarizer was not
        # pinned (a direct ``cli.py mine`` launch without run.sh's
        # QLIB_FACTOR_SUMMARIZER export), the setting falls back to the
        # frictionless AlphaAgent...Feedback and that prose degrades. The numeric
        # metrics still flow either way (they come from this runner via
        # _extract_net_cost_metrics, not from the summarizer), so this is an
        # ERROR, not a hard raise -- the run is still mostly functional, but the
        # misconfiguration is unmissable at startup. The default launch path
        # (run.sh) pins the summarizer; a direct launch must export it too.
        try:
            from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
            _summ = getattr(ALPHA_AGENT_FACTOR_PROP_SETTING, "summarizer", "") or ""
        except Exception:
            _summ = ""
        if "NetCost" not in _summ:
            logger.error(
                "NetCostFactorRunner is active but the factor-mining summarizer "
                "is %r, not NetCostFactorFeedback. The feedback prose will be "
                "frictionless (no U / e_j / turnover / cost block). Set "
                "QLIB_FACTOR_SUMMARIZER=quantaalpha.factors.net_cost_feedback."
                "NetCostFactorFeedback -- run.sh does this; a direct cli launch "
                "must export it too.",
                _summ or "<unset>",
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

        # Shares the Qlib runner's recovery path: a missing result.h5 is re-run
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

        # ONE evaluation of the whole batch, matching how qrun
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

        # Phase 6: the book is a PERIODIC CHECK, not a per-candidate gate.
        # Under `standalone` the factor is admitted on its own neutralized
        # significance, so the full three-fold priced book is reporting rather
        # than selection. Priced every `book_eval_every` batches; 1 keeps the
        # old behaviour exactly.
        every = int(getattr(self.theta.admission, "book_eval_every", 1) or 1)
        self._batches_seen = getattr(self, "_batches_seen", 0) + 1
        skip_book = (self.theta.admission.mode == "standalone" and every > 1
                     and (self._batches_seen % every) != 0)
        try:
            res = self.op.evaluate(candidates, zoo_signals=zoo_signals,
                                   zoo_metrics=zoo_metrics, skip_book=skip_book)
        except Exception:
            logger.exception("E_theta evaluation failed for this experiment")
            raise FactorEmptyError("E_theta evaluation failed.")
        # DEFLATED SHARPE, on the batches where a book actually exists.
        #
        # Benjamini-Hochberg (in `_decide_standalone`) corrects the SELECTION of
        # individual factors; it says nothing about the strategy those factors
        # compose. The deflated Sharpe is the claim-side correction: it asks
        # whether this book's Sharpe survives having been found by a search of
        # this size, pricing the skew and kurtosis of its return path.
        #
        # They are not interchangeable and applying both to the same statistic
        # would double-count the multiplicity -- FDR gates the factor, DSR
        # reports on the book. Running it DURING the mine rather than only at
        # the end means a search that is merely getting luckier says so while
        # there is still time to notice.
        series = res.pop("_net_return_series", None) if isinstance(res, dict) else None
        if series is not None and not skip_book:
            try:
                from quantaalpha.eval.defense import deflated_sharpe_ratio
                n_trials = max(len(getattr(self, "_t_history", []) or []), 1)
                d = deflated_sharpe_ratio(series, n_trials=n_trials)
                res["dsr"] = d.dsr
                res["dsr_n_trials"] = n_trials
                res["dsr_sharpe"] = d.sharpe
                logger.info("DSR %.3f (Sharpe %.3f deflated for %d trials); "
                            "0.95 is the conventional bar",
                            d.dsr, d.sharpe, n_trials)
            except Exception as exc:
                logger.warning("deflated Sharpe unavailable: %s", type(exc).__name__)

        # Record the (|IC|, net_ir) pair from any batch whose book WAS priced.
        # This is what lets `_live_ic_breakeven` interpolate the crossing from
        # the run's own data instead of relying on an offline ladder.
        if not skip_book:
            try:
                _ic = res.get("m_rank_ic")
                _ir = res.get("m_net_ir")
                if (_ic is not None and _ir is not None
                        and _ic == _ic and _ir == _ir):
                    if not hasattr(self, "_ic_ir_pairs"):
                        self._ic_ir_pairs = []
                    self._ic_ir_pairs.append((abs(float(_ic)), float(_ir)))
            except (TypeError, ValueError):
                pass

        if skip_book:
            logger.info("book not priced this batch (%d of every %d); the factor "
                        "gate does not need it", self._batches_seen % every, every)

        names = [str(c) for c in new_factors.columns]
        res["factor_name"] = ", ".join(names)

        # Decide BEFORE writing the ledger. The ledger is what sibling processes
        # rehydrate the repository from, so a record written ahead of the
        # decision re-admits a rejected batch through the back door -- which is
        # exactly what happened on run 20260729_075350, where two batches logged
        # as REJECTED were still recorded with their factor_exprs.
        batch_metrics = {k[2:]: v for k, v in res.items() if k.startswith("m_")}

        # Feedback reference: EVERY batch, admitted or not. The repository is a
        # winners-only sample and a poor yardstick for how generation is going;
        # this one is the distribution the generator actually produces. It gates
        # nothing, which is why it can safely include the failures.
        # U is book-derived, so it is absent on a batch where the book was not
        # priced. Observing NaN would silently poison the running distribution
        # that every later percentile is measured against.
        _u = res.get("U", float("nan"))
        try:
            _u = float(_u)
        except (TypeError, ValueError):
            _u = float("nan")
        if _u == _u:
            self._population.observe(_u)
            res["population"] = self._population.summary(_u)
        else:
            res["population"] = "book not priced this batch"

        if self.theta.admission.mode == "standalone":
            # Per-FACTOR admission on the factor's own significance. Unlike the
            # marginal branch below this admits a SUBSET of the batch, so a
            # hypothesis that produced one good factor and two near-clones
            # contributes the good one rather than being judged whole.
            hyp_obj = getattr(exp, "hypothesis", None)
            mech = getattr(hyp_obj, "hypothesis", None) or (
                hyp_obj if isinstance(hyp_obj, str) else None)
            # FALL BACK TO THE FACTOR'S OWN STATED RATIONALE.
            #
            # Attaching the hypothesis in `_convert_with_history_limit` fixed the
            # ORIGINAL path, and the run then broke again 75 minutes in: the
            # mutation and crossover operators build their experiments elsewhere,
            # so `exp.hypothesis` was empty for them and the gate rejected 8
            # consecutive batches for "no mechanism".
            #
            # Chasing every construction path is the fragile fix. Every
            # FactorTask already carries `factor_description` and
            # `factor_formulation` -- the model's own account of WHY the factor
            # should work, written per factor rather than per batch. That is the
            # mechanism, and it exists no matter which operator produced the
            # task.
            if not (isinstance(mech, str) and mech.strip()):
                parts = []
                for task in getattr(exp, "sub_tasks", []) or []:
                    d = (getattr(task, "factor_description", "") or "").strip()
                    f = (getattr(task, "factor_formulation", "") or "").strip()
                    if d:
                        parts.append(f"{getattr(task, 'factor_name', '?')}: {d}"
                                     + (f" [{f}]" if f else ""))
                if parts:
                    mech = " | ".join(parts)
                    logger.info("mechanism taken from the factor descriptions "
                                "(no hypothesis on this experiment: %s operator)",
                                getattr(exp, "_qa_operator", "mutation/crossover"))
            decision, keep = self._decide_standalone(
                candidates, zoo_signals, batch_metrics,
                mechanism=(mech or "").strip() if isinstance(mech, str) else "",
                expected_sign=str(getattr(hyp_obj, "expected_ic_sign", "") or "").lower())
            admitted = decision.admit
            for expr, signal in keep:
                # Store the factor's OWN tear sheet alongside the batch metrics.
                # Admission is per-factor on `t_nw` / monotonicity, so eviction
                # must rank on the same numbers -- otherwise a factor admitted
                # for its neutralized IC gets evicted for its book contribution,
                # and the library churns on a criterion nothing was selected by.
                m = dict(batch_metrics)
                m.update(getattr(self, "_last_tearsheets", {}).get(expr, {}))
                # Record the MECHANISM the factor was proposed under. A factor
                # without a plausible economic story is a likely false discovery
                # regardless of its t-statistic, and this system already has the
                # story -- the model wrote one to propose the factor. Keeping it
                # with the measurement is what makes that checkable later
                # instead of lost.
                hyp = getattr(getattr(exp, "hypothesis", None), "hypothesis", None) \
                    or getattr(exp, "hypothesis", None)
                if isinstance(hyp, str) and hyp.strip():
                    m["mechanism"] = hyp.strip()[:600]
                elif bool(getattr(self.theta.admission, "require_mechanism", False)):
                    # A factor with no stated mechanism is a likely false
                    # discovery whatever its t-statistic. The model already
                    # writes one to propose the factor, so an empty mechanism
                    # means the channel broke, not that the idea lacks one.
                    logger.warning("admitted factor carries NO stated mechanism: %s",
                                   _short(expr))
                    m["mechanism"] = ""
                    m["mechanism_missing"] = True
                self._repository[expr] = (self._compact(signal), m)
            dropped = self._enforce_library_cap()
            if dropped:
                res["cap_evicted"] = dropped
            # Publish the PER-FACTOR tear sheets, admitted or not. `batch_metrics`
            # is assembled from the `m_` keys before this point and carries only
            # book aggregates, so without this the model is scored on `t_nw`,
            # neutralized RankIC, monotonicity and size exposure and told none of
            # them. The REJECTED sheets matter most -- that is where the lesson is.
            if getattr(decision, "replaced_pairs", None):
                res["replaced_pairs"] = decision.replaced_pairs
            res["factor_tearsheets"] = dict(getattr(self, "_last_tearsheets", {}))
            res["admitted_exprs"] = [e for e, _ in keep]
            logger.info("repository: %s -- %s; |zoo| = %d  [%s]",
                        "ADMIT" if admitted else "REJECT", decision.reason,
                        len(self._repository), res["population"])
            res.update(decision.as_record())
        elif self.theta.admission.mode == "marginal_contribution":
            decision = self._decide_marginal(candidates, zoo_signals, zoo_metrics,
                                             batch_metrics, main_res=res)
            admitted = decision.admit
            if admitted:
                if decision.displaced:
                    self._repository.pop(decision.displaced, None)
                for expr, signal in candidates.items():
                    self._repository[expr] = (self._compact(signal), batch_metrics)
            logger.info("repository: %s -- %s; |zoo| = %d  [%s]",
                        "ADMIT" if admitted else "REJECT", decision.reason,
                        len(self._repository), res["population"])
            res.update(decision.as_record())
        else:
            admitted = self._admit(float(res["U"]), candidates, batch_metrics)
        res["admitted"] = admitted
        if admitted:
            res["evicted"] = len(self._prune())

        self.ledger.append(
            {
                "factor_ids": [md5_hash(f"{n}_{e}")[:16] for n, e in zip(names, candidates)],
                "factor_names": names,
                # Only an ADMITTED batch contributes incumbents. A rejected one
                # is still recorded -- the ledger is the audit trail of every
                # evaluation -- but under a key rehydration ignores.
                "factor_exprs": list(candidates) if admitted else [],
                "rejected_exprs": [] if admitted else list(candidates),
                "admitted": admitted,
                "n_factors": len(candidates),
                "theta_hash": res["theta_hash"],
                "zoo_hash": res["zoo_hash"],
                "zoo_size": res["zoo_size"],
                "metrics": batch_metrics,
                "e": {k[2:]: v for k, v in res.items() if k.startswith("e_")},
                "U": res["U"],
                # PER-FACTOR measurement, admitted or rejected. Without this the
                # ledger is blind to t_nw / sign / monotonicity / capacity, and
                # the FDR trial history (replay_t_history) has nothing to
                # rehydrate from -- so a fresh runner (one per evolution task
                # under parallel execution) starts every batch with n_tests=1
                # and the Benjamini-Hochberg gate's n_tests>1 guard never fires.
                # The rejected sheets matter most: FDR corrects over the full
                # family of tests, not the survivors (test_winsor_fdr F4).
                "factor_tearsheets": dict(getattr(self, "_last_tearsheets", {})),
                # WHY the verdict fell as it did. The ledger built this dict from
                # named keys, so decision.as_record() -- which is merged into
                # `res` above -- never reached disk: every record carried
                # admitted=True/False and nothing to explain it. That silently
                # removed the only measurement that says whether the search is
                # learning, since the delta slope is computed from these.
                **{k: v for k, v in res.items()
                   if k in ("reason", "delta_mean", "delta_se", "delta_t",
                            "delta_per_seed", "displaced", "pathology",
                            "verdict", "population")},
            }
        )

        # The ICIR prediction cache is keyed by (zoo_hash, candidate_key,
        # theta.hash) -- and every seed/eval this batch has a distinct key, so
        # it is never reused across batches; it only accumulates. Left unbounded
        # it holds ~1500 wide (T x N) frames by the end of a 150-batch run and
        # tips the box into swap. Clear it at the batch boundary; nothing
        # downstream of the decision reads it.
        _clear_combiner_cache()

        exp.result = self._to_series(res)
        return exp

    # ------------------------------------------------------------------
    def _decide_standalone(self, candidates, zoo_signals, batch_metrics,
                           mechanism: str = "", expected_sign: str = ""):
        """Admit PER FACTOR on its own significance -- the desk criterion.

        ``marginal_contribution`` asks "does this batch improve the current
        book". That is the right question for curating a book and the wrong one
        for building a library: it is self-limiting, because every admission
        raises the bar for the next, so the rate falls toward zero by
        construction and the repository stalls (measured: 9 admits in 37
        decisions, then nothing).

        This asks the question a research desk asks instead:

          1. **Is the factor real?**  |t| of its own mean per-date rank IC
             against ``admission.k_sigma``. At 3.0 that is the Harvey-Liu-Zhu
             bar for a newly discovered factor, which already carries the
             multiple-testing correction the search needs.
          2. **Is it new?**  ``rho_max`` against the repository, in RANK space
             (what the combiner fits on), against ``gates.rho_bar``.

        Redundancy is then handled where it belongs -- the ICIR combiner shrinks
        correlated factors' weights rather than the gate refusing them.

        Judged one factor at a time, not per batch. A hypothesis that yields one
        good factor and two near-clones of it contributes the good one instead of
        being accepted or rejected whole; that is exactly the case ``rho_within``
        measures and the batch-level gate could not act on.

        Returns ``(Decision, admitted_exprs)``.
        """
        from quantaalpha.eval.admission import Decision
        from quantaalpha.core.verdict import Verdict
        from quantaalpha.eval.metrics import (
            _cross_sectional_corr, _slice, label_frame_at, rho_max, rho_max_arg,
            newey_west_t, quantile_metrics, effective_rank,
            effective_rank_cached, spearman_abs_matrix,
        )

        adm = self.theta.admission
        bar = float(adm.k_sigma)
        rho_bar = getattr(self.theta.gates, "rho_bar", None)
        # Marginal-effective_rank gate (opt-in, env-driven so the frozen
        # protocol hash does not move -- a Θ field would, since hash =
        # sha256(asdict(theta))). 0.0 disables, which is the byte-identical
        # default. Catches a candidate that passes rho_max individually but
        # adds few INDEPENDENT directions (the gap rho_max, a pairwise test,
        # leaves). See operator.py:210-243 for the metric.
        _min_mer = float(os.environ.get("QA_MIN_MARGINAL_ER", "0.0") or "0.0")
        horizons = _label_horizons(self.theta)

        # ECONOMIC MECHANISM, as a GATE rather than a note. Paleologo's point:
        # a factor with no mechanism is a likely false discovery whatever its
        # t-statistic, because a search over enough expressions will always
        # surface something that fits. This used to be recorded at repository-
        # insert time -- AFTER admission -- and only logged a warning, so it
        # could not reject anything.
        #
        # The model already writes a mechanism to propose the factor, so an
        # empty one means the channel broke rather than that the idea lacks a
        # story; either way it is not admissible evidence.
        want_mech = bool(getattr(adm, "require_mechanism", False))
        want_sign = bool(getattr(adm, "require_sign_match", False))
        mech_txt = (mechanism or "").strip()
        # Derive the predicted sign from the mechanism TEXT, not the freely-set
        # expected_sign field. A refine cannot then flip the sign token without
        # rewriting the stated direction -- the sign-match gate's falsifiability
        # is real, not gameable (measured 2026-08-24: a "predicts HIGHER returns"
        # factor was re-admitted with sign_predicted flipped to negative but the
        # mechanism text byte-identical, so the gate was hollow). The ``expected_sign``
        # arg (the model's free field) is ignored at the gate.
        exp_sign = derive_sign_from_mechanism(mech_txt)
        if want_mech and not mech_txt:
            notes = []
            for expr in candidates:
                notes.append(f"{_short(expr)}: no economic mechanism was stated, so "
                             f"there is nothing for the measurement to confirm or "
                             f"contradict")
            return Decision(False, "; ".join(notes[:4]), (), float("nan"),
                            float("nan"), float("nan"),
                            verdict=Verdict.NO_MECHANISM), []
        # A mechanism that predicts no direction is unfalsifiable, and an
        # unfalsifiable claim cannot be evidence for anything.
        if want_sign and not exp_sign:
            notes = []
            for expr in candidates:
                notes.append(f"{_short(expr)}: the mechanism names no direction "
                             f"(expected_ic_sign absent or hedged), so no "
                             f"measurement could contradict it")
            return Decision(False, "; ".join(notes[:4]), (), float("nan"),
                            float("nan"), float("nan"),
                            verdict=Verdict.NO_MECHANISM), []

        start, end, win = self.op._windows(False)

        # China permitted margin trading and short selling only from March 2010,
        # so any dollar-neutral long/short statistic computed before then
        # describes a book that could not have been held. `research_start` is
        # the floor for the RESEARCH gate; the long-only book is unaffected and
        # still uses the full history from 2005.
        rs = getattr(self.theta.splits, "research_start", None)
        if rs and win[0] < rs:
            logger.info("research gate: window %s..%s clamped to start %s "
                        "(short selling unavailable before then)", win[0], win[1], rs)
            win = (rs, win[1])

        panel = self.op._panel(start, end)
        labels = {h: label_frame_at(panel, self.theta, h) for h in horizons}

        # NEUTRALIZE FIRST. Raw IC cannot distinguish alpha from a repackaged
        # size bet, and on this market that distinction is the whole game:
        # equal-weight versus cap-weighted CSI300 reproduces the search's exact
        # per-fold sign pattern (-9.8pp 2019, -13.1pp 2020, +6.0pp 2021) with
        # zero alpha involved. A gate on raw IC therefore rewards size exposure.
        # Falls back to the raw signal when the reference data is absent, so a
        # missing parquet degrades the gate rather than killing the run.
        try:
            from quantaalpha.eval.neutralize import residualize, exposure_report
            _neutralize_ok = True
        except Exception:
            _neutralize_ok = False

        mono_bar = float(getattr(adm, "min_monotonicity", 0.0) or 0.0)
        pos_bar = float(getattr(adm, "min_ic_pos_frac", 0.0) or 0.0)

        kept, notes, replaced = [], [], []
        self._last_tearsheets = {}
        # Marginal-er gate cache is instance-level now (self._mer_cache, set up
        # in __init__): the repo-repo |Spearman| block survives across batches,
        # keyed by repo composition + eval window, so a reject batch reuses the
        # prior block instead of rebuilding it every batch.

        for expr, signal in candidates.items():
            sig_raw = align_signal(signal, panel)
            sig = sig_raw
            sheet: dict = {}
            if _neutralize_ok:
                try:
                    sig = residualize(sig_raw, panel, self.theta)
                    sheet.update(exposure_report(sig_raw, panel))
                except Exception as exc:
                    logger.warning("neutralization failed for %s (%s); scoring raw",
                                   _short(expr), type(exc).__name__)
                    sig = sig_raw
            sliced = _slice(sig, win)

            # Best horizon wins. A factor predictive at 20 days is a different
            # alpha from one predictive at 1 day, not a worse version of it --
            # with a single horizon they all approximate the same target, which
            # is most of why the search saturates.
            best = None
            for h, lab in labels.items():
                ic = _cross_sectional_corr(sliced, lab, "spearman").dropna()
                n = len(ic)
                if n < 100:
                    continue
                mean, sd = float(ic.mean()), float(ic.std())
                if sd <= 0 or sd != sd:
                    continue
                # OVERLAP CORRECTION. A horizon-h label sampled daily produces
                # IC observations that share h-1 of their h days, so consecutive
                # values are strongly autocorrelated and the naive t-stat is
                # inflated by roughly sqrt(h). The effective number of
                # independent observations is n/h, not n.
                n_eff = max(n / float(h), 2.0)
                t_overlap = mean / (sd / (n_eff ** 0.5))
                # Newey-West generalises that to ordinary autocorrelation (a
                # persistent signal autocorrelates its IC even at h=1). Take the
                # more conservative of the two rather than the flattering one.
                t_nw = newey_west_t(ic)
                t = t_nw if (t_nw == t_nw and abs(t_nw) < abs(t_overlap)) else t_overlap
                pos = float((ic > 0).mean()) if mean >= 0 else float((ic < 0).mean())
                if best is None or abs(t) > abs(best[1]):
                    best = (h, t, n, mean, pos, sd)
            if best is None:
                notes.append(f"{_short(expr)}: no horizon produced a usable IC series")
                continue
            h, t, n, ric, pos, sd = best
            # ICIR = mean(IC)/sd(IC): the stability metric. mean and sd are
            # already in hand from the winning horizon's IC series (sd > 0
            # here, the loop `continue`s otherwise); storing it costs nothing.
            # On the NEUTRALIZED signal, so it is the stability of the alpha
            # the gate scored, not of a repackaged risk exposure.
            _icir = (ric / sd) if (sd and sd > 0) else float("nan")
            # Raw RankIC (before neutralization) at the same winning horizon,
            # so the feedback can state the raw-vs-neutralized gap that
            # exposure_size already implies (a large gap = the raw edge was
            # mostly risk, not alpha). The gloss has always listed this key;
            # without it the entry was dead.
            try:
                _raw_ic = _cross_sectional_corr(_slice(sig_raw, win),
                                                labels[h], "spearman").dropna()
                _raw_rank_ic = (float(_raw_ic.mean())
                                if len(_raw_ic) >= 100 else float("nan"))
            except Exception:
                _raw_rank_ic = float("nan")
            sheet.update({"rank_ic_neutral": ric, "t_nw": t, "best_horizon": h,
                          "ic_pos_frac": pos, "n_obs": n,
                          "rank_icir": _icir, "rank_ic": _raw_rank_ic})


            # MULTIPLE TESTING across the whole search, not just this factor.
            # Recorded BEFORE the fixed bar: every factor scored is a trial
            # whether or not it survives, and a correction computed only over
            # the survivors is a correction computed on a sample truncated to
            # its own winners -- it would never bind.
            t_req, n_tests, q = self._fdr_bar(t)
            sheet["fdr_t_required"] = t_req
            sheet["fdr_n_tests"] = n_tests

            if abs(t) < bar:
                notes.append(f"{_short(expr)}: best |t| {abs(t):.2f} (h={h}d) < {bar:g}")
                self._last_tearsheets[expr] = sheet
                continue
            if q > 0 and n_tests > 1 and abs(t) < t_req:
                need = ("no |t| among them clears it" if t_req == float("inf")
                        else f"needs {t_req:.2f}")
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} clears the fixed bar "
                             f"but not Benjamini-Hochberg at q={q:g} over {n_tests} "
                             f"tests so far ({need})")
                self._last_tearsheets[expr] = sheet
                continue

            # ORDER MATTERS: this runs AFTER the significance bars, not before.
            # Placed first, it tested the direction of factors whose effect is
            # indistinguishable from zero -- the sign of a |t|=0.3 series is a
            # coin flip, so a coin flip was rejecting candidates before anything
            # asked whether they predicted at all. Measured over 70 scored
            # factors: 54% sign agreement, against a market that realizes
            # negative 71% of the time. A mechanism claim is only meaningful for
            # an effect that exists; establish the effect first, then ask whether
            # the stated story explains it.
            # THE FALSIFICATION TEST. The hypothesis pre-registered a direction;
            # this is the measurement it committed to. Comparing them needs no
            # market prior and no LLM judgment -- it only asks whether the story
            # the factor was proposed under survived contact with the data.
            realized = "positive" if ric > 0 else "negative" if ric < 0 else ""
            sheet["mechanism"] = mech_txt[:600]
            sheet["sign_predicted"] = exp_sign
            sheet["sign_realized"] = realized
            validated = bool(exp_sign) and bool(realized) and exp_sign == realized
            sheet["mechanism_validated"] = validated
            if want_sign and not validated:
                notes.append(
                    f"{_short(expr)}: the mechanism predicted a {exp_sign or 'n/a'} "
                    f"IC and the measurement came back {realized or 'flat'} "
                    f"(RankIC {ric:+.4f}, |t| {abs(t):.2f}). The factor may still "
                    f"carry signal, but the stated mechanism does not explain it, "
                    f"so what remains is an unexplained fit.")
                self._last_tearsheets[expr] = sheet
                continue

            # A factor carried by a handful of outlier days is not a factor.
            if pos_bar and pos < pos_bar:
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} OK but IC positive on "
                             f"only {pos:.0%} of days < {pos_bar:.0%}")
                self._last_tearsheets[expr] = sheet
                continue

            # Quantile shape. A signal whose extremes differ sharply but whose
            # middle is noise scores a wide spread and a monotonicity near zero,
            # and it dies under any position cap -- the book holds ~34 of 300
            # names and cannot take a bet that only exists in the far tail.
            try:
                qm = quantile_metrics(sig, panel, self.theta, win, horizon=h)
                sheet.update({k: v for k, v in qm.items() if k != "q_means"})
            except Exception:
                qm = {}
            mono = abs(float(qm.get("monotonicity", float("nan"))))
            if mono_bar and mono == mono and mono < mono_bar:
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} OK but decile "
                             f"monotonicity {mono:.2f} < {mono_bar:.2f} (tails-only)")
                self._last_tearsheets[expr] = sheet
                continue

            # IMPLEMENTABILITY. Measured on the factor alone -- no book needed.
            # A gate on IC alone admits signals that cannot be traded at size,
            # and the transfer coefficient then quietly destroys them.
            try:
                from quantaalpha.eval.metrics import solo_turnover as _solo
                from quantaalpha.eval.costs import trailing_adv as _adv
                turn = float(_solo(sig, panel, self.theta))
                sheet["turnover_solo"] = turn
                med_adv = float(_adv(panel, self.theta).stack().median())
                part = float(getattr(adm, "capacity_participation", 0.05) or 0.05)
                # Capacity: a book of N names trades NAV*turn/N per name per
                # day, so the ceiling is
                #
                #     NAV_max = participation * ADV * N / turnover
                #
                # BREADTH MATTERS. Omitting N treats the whole book's turnover
                # as concentrated in a single name and understates capacity by a
                # factor of N -- at 50% turnover that reported 0.6e8 CNY, below
                # the 1e8 NAV, and would have rejected a perfectly tradeable
                # factor on false capacity grounds.
                n_names = int(panel.universe.sum(axis=1).median()) or 1
                sheet["capacity_cny"] = (med_adv * n_names * part / turn) if turn > 0 else float("inf")

                # THE ECONOMIC BAR, alongside the statistical one.
                #
                # |t| >= 3 asks "is this signal REAL?" and its IC threshold
                # falls as the window grows: measured on this protocol, |t|=3
                # needs IC 0.0173 over the 3-year valid window and only 0.0094
                # over the 10-year test. The economic bar does not move -- it is
                # set by what the book pays to trade.
                #
                # Measured 2026-08-24: the mined factors carry median |IC|
                # 0.0216 and clear |t| 4.77 against a required 2.18, so the
                # search is SUCCEEDING at the bar it is shown -- while the book
                # they form loses 3.57pp/yr to a no-alpha baseline. The
                # generator was never told the second number exists.
                #
                # The bar is the IC at which this factor's own trading pays for
                # itself: cost per unit turnover, converted to IC units through
                # the protocol's own realised alpha-per-IC scale.
                _k0 = float(getattr(self.theta.costs, "kappa0", 0.0) or 0.0)
                _apic = float(os.environ.get("QA_ALPHA_PER_IC", "0.0207"))
                if turn == turn and turn > 0 and _apic > 0:
                    sheet["ic_breakeven_solo"] = (_k0 * turn) / _apic
                # The BOOK-level bar, MEASURED by the oracle ladder
                # (scripts/qa_ic_ladder.py), never assumed.
                #
                # Resolution order, all of it traceable to a measurement:
                #   1. QA_IC_BREAKEVEN_BOOK -- an explicit override;
                #   2. the ladder artifact for THIS protocol hash, if one has
                #      been run (data/results/ic_breakeven_<theta>.json);
                #   3. nothing.
                #
                # (3) is deliberate. A live run cannot compute this itself: the
                # ladder needs a look-ahead oracle signal, which is not
                # something a mine may build. So the bar is either measured
                # offline for this protocol or it is absent -- and absent is
                # correct, because a guessed threshold would have the search
                # optimising toward a number nobody measured. Measured on this
                # protocol 2026-08-24: 0.0728, against a 0.13 ESTIMATE that a
                # previous session carried -- the estimate was 1.8x too high.
                # (0) LIVE, when this run has bracketed the crossing itself.
                # Every priced book contributes an (|IC|, net_ir) pair; once
                # there are pairs on both sides of zero the crossing is this
                # run's OWN evidence and outranks an offline calibration.
                _live = self._live_ic_breakeven()
                sheet["ic_breakeven_book"] = (
                    _live if _live is not None else _book_breakeven(self.theta))
            except Exception:
                turn = float("nan")

            max_turn = float(getattr(adm, "max_solo_turnover", 0.0) or 0.0)
            if max_turn and turn == turn and turn > max_turn:
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} OK but signal turnover "
                             f"{turn:.1%}/day > {max_turn:.0%} -- pays the spread "
                             f"repeatedly for one edge")
                self._last_tearsheets[expr] = sheet
                continue
            min_cap = float(getattr(adm, "min_capacity_cny", 0.0) or 0.0)
            cap_cny = sheet.get("capacity_cny", float("inf"))
            if min_cap and cap_cny == cap_cny and cap_cny < min_cap:
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} OK but capacity "
                             f"{cap_cny/1e8:.2f}e8 CNY < {min_cap/1e8:.2f}e8")
                self._last_tearsheets[expr] = sheet
                continue

            # Redundancy is measured against the repository AND against the
            # factors already kept in THIS batch.
            #
            # Measured 2026-08-24 on the 37-factor OOS book: rho_within = 0.9965
            # -- the worst pair inside the zoo was 99.65% correlated, so 37
            # factors carried roughly one independent bet, and the book lost to
            # a no-alpha baseline by 5.3pp/yr. ``rho_within`` existed as a metric
            # and logged a warning above 0.8, but it gated NOTHING, and the
            # repository comparison could not see it: under standalone admission
            # each factor is judged as its own batch, so batch-mates never met
            # each other, and a factor admitted at zoo_size=0 was compared
            # against an empty repository (rho_max = 0.00 for the first admits).
            #
            # Including the kept-so-far list closes both holes with one check:
            # candidates within a batch now measure against each other, and the
            # comparison set is the library as it will actually stand.
            _held = {e: s for e, (s, _) in self._repository.items()}
            for _e, _s, _t, _r, _h in kept:
                _held.setdefault(_e, _s)
            rho, near = rho_max_arg(sig, _held) if _held else (0.0, None)
            sheet["rho_max"] = rho
            sheet["closest_held"] = near or ""
            self._last_tearsheets[expr] = sheet
            if rho_bar is not None and rho >= float(rho_bar):
                # A DUPLICATE IS NOT AUTOMATICALLY THE LOSER.
                #
                # This used to drop the candidate unconditionally, so a factor
                # that duplicated an incumbent AND was stronger than it was
                # discarded while the weaker incumbent stayed -- the library
                # could never improve along a direction it already held, only
                # add new ones. Measured: 7 such rejections in one run, any of
                # which could have been an upgrade.
                #
                # Compare on `_research_score` -- the same |t_NW| admission and
                # eviction use -- so all three decisions rank on one number and
                # cannot disagree.
                inc_m = (self._repository.get(near) or (None, {}))[1] if near else {}
                inc_score = self._research_score(inc_m or {})
                cand_score = abs(float(t)) if t == t else float("-inf")
                sheet["closest_held_t"] = inc_score if inc_score > float("-inf") else None
                # Between two near-duplicates, KEEP THE STRONGER ONE -- and
                # decide that on the scores alone, not on arrival order.
                #
                # A margin on the challenger alone is not order-free: it says
                # "the incumbent stays unless beaten by 1.0", so a |t|=10.78
                # factor loses to a |t|=10.07 incumbent it duplicates at
                # rho=0.686, purely because the weaker one was measured first.
                # Reverse the arrival order and the book changes -- which means
                # the book is partly an artifact of scheduling.
                #
                # The margin belongs on the DECISION, not on one side of it:
                # * a clear winner (|gap| > margin) always takes the slot,
                #   whichever side it is on;
                # * inside the margin the two are statistically indistinguishable
                #   on |t|, so the tie is broken by a property of the FACTOR --
                #   lower turnover, since the book pays for turnover every day
                #   and cannot tell the two signals apart anyway.
                # Order never enters either branch.
                _margin = float(os.environ.get("QA_REPLACE_MARGIN", "1.0"))
                _gap = cand_score - inc_score
                if abs(_gap) > _margin:
                    _cand_wins = _gap > 0
                    _why = (f"|t| {cand_score:.2f} vs {inc_score:.2f}, "
                            f"a decisive {abs(_gap):.2f} gap")
                else:
                    # Tie on significance -> prefer the cheaper signal to hold.
                    # The field is `turnover_solo` (written at the
                    # implementability check above), NOT `turnover` -- reading
                    # the wrong key silently returns NaN and the tiebreak
                    # degrades to "always keep the incumbent", i.e. back to the
                    # order-dependence this branch exists to remove.
                    _inc_turn = float((inc_m or {}).get(
                        "turnover_solo", (inc_m or {}).get("turnover", float("nan"))))
                    _cand_turn = float(sheet.get("turnover_solo", float("nan")))
                    if _cand_turn == _cand_turn and _inc_turn == _inc_turn:
                        _cand_wins = _cand_turn < _inc_turn
                        _why = (f"|t| within {_margin:.1f} ({cand_score:.2f} vs "
                                f"{inc_score:.2f}); broken on turnover "
                                f"{_cand_turn:.3f} vs {_inc_turn:.3f}")
                    else:
                        _cand_wins = False      # no tiebreak data: hold what works
                        _why = (f"|t| within {_margin:.1f} and no turnover to "
                                "compare -- keeping the incumbent")
                if near is not None and not _cand_wins:
                    notes.append(
                        f"{_short(expr)}: duplicates {_short(near)} (rho {rho:.3f}); "
                        f"{_why} -- keeping the incumbent")
                    self._last_tearsheets[expr] = sheet
                    continue
                if near is not None and _cand_wins:
                    # Evict here, not later: a subsequent candidate in this same
                    # batch must measure its rho against the library as it will
                    # actually stand, otherwise two near-clones can both "beat"
                    # an incumbent that only one of them replaces.
                    self._repository.pop(near, None)
                    # a row/column left the repo -> the block is stale. The
                    # key check at the next candidate would catch this too (the
                    # composition changed); drop it now defensively so the
                    # rebuild happens on the post-pop zoo, not the cached one.
                    self._mer_cache = None
                    replaced.append((near, expr, cand_score, inc_score, rho))
                    logger.info("repository: REPLACE %s (|t| %.2f) with %s "
                                "(|t| %.2f), rho %.3f",
                                _short(near), inc_score, _short(expr),
                                cand_score, rho)
                    notes.append(
                        f"{_short(expr)}: duplicates {_short(near)} (rho {rho:.3f}) "
                        f"but is STRONGER (|t| {cand_score:.2f} vs {inc_score:.2f}) "
                        f"-- replacing the incumbent")
                    kept.append((expr, sig_raw, t, rho, h))
                    continue
                notes.append(f"{_short(expr)}: |t| {abs(t):.2f} OK but rho_max "
                             f"{rho:.3f} >= {rho_bar} -- duplicates "
                             f"{_short(near) if near else 'an incumbent'}, which is "
                             f"stronger (|t| {inc_score:.2f})"
                             if inc_score > float("-inf") else
                             f"{_short(expr)}: |t| {abs(t):.2f} OK but rho_max "
                             f"{rho:.3f} >= {rho_bar} (duplicates "
                             f"{_short(near) if near else 'an incumbent'})")
                continue
            # Marginal-effective_rank gate (novel path only). rho_max is a
            # PAIRWISE test: it blocks a candidate duplicating one incumbent
            # (rho >= rho_bar) but lets through a candidate that is < rho_bar
            # to each of several held factors individually and so adds few
            # independent directions. effective_rank measures the book's true
            # breadth; this makes "does this factor WIDEN the book?" a gate,
            # on the novel path only. A REPLACE (the branch above) is a
            # like-for-like swap on an existing direction by construction, so
            # it is not gated here -- gating it would block every upgrade.
            # Default-off (QA_MIN_MARGINAL_ER=0): the block is skipped and the
            # frozen admission path is byte-identical.
            if _min_mer and _held:
                try:
                    # Reuse the cached repo-repo |Spearman| block; only the
                    # kept batch-mates and this candidate are fresh. The block
                    # is keyed by the repo COMPOSITION + the eval window the
                    # panel was built on -- a pure function of those -- so a
                    # matching key means the cached R_repo is valid. A reject
                    # batch (zoo unchanged) hits and skips the O(zoo**2) build;
                    # an admit/replace/evict changes the composition and misses.
                    # Within a batch the repo changes only on a replace (the
                    # pop below); across batches only on admit/replace/evict,
                    # so the key is the exact invalidation signal. Eigenvalues
                    # are permutation-invariant, so the cached path is
                    # bit-identical to the uncached one (verdict unchanged).
                    _mer_key = (tuple(sorted(self._repository)), start, end)
                    if self._mer_cache is None or self._mer_cache[0] != _mer_key:
                        _repo_sigs = {
                            e: s for e, (s, _) in self._repository.items()}
                        self._mer_cache = (
                            _mer_key, spearman_abs_matrix(_repo_sigs),
                            _repo_sigs)
                    _R_repo, _repo_sigs = self._mer_cache[1], self._mer_cache[2]
                    _kept_sigs = {e: s for e, s, _t, _r, _h in kept}
                    _er_held = effective_rank_cached(
                        _R_repo, _repo_sigs, _kept_sigs)
                    _er_all = effective_rank_cached(
                        _R_repo, _repo_sigs, {**_kept_sigs, expr: sig})
                    _marginal = _er_all - _er_held
                    sheet["marginal_er"] = _marginal
                except Exception:
                    _marginal = float("inf")      # never reject on a metric error
                if _marginal < _min_mer:
                    notes.append(
                        f"{_short(expr)}: |t| {abs(t):.2f} OK and rho_max "
                        f"{rho:.2f} < {rho_bar}, but it adds only "
                        f"{_marginal:.2f} independent directions (< {_min_mer}) "
                        f"-- redundant at the margin")
                    self._last_tearsheets[expr] = sheet
                    continue
            kept.append((expr, sig_raw, t, rho, h))

        if kept:
            reason = "; ".join(
                f"{_short(e)} |t|={abs(t):.2f}@{h}d rho={r:.2f}"
                for e, _, t, r, h in kept)
            why = (f"admitted {len(kept)}/{len(candidates)} on standalone "
                   f"significance: {reason}")
            v = Verdict.ADMITTED
        else:
            why = (f"no factor cleared |t| >= {bar:g} (horizons {horizons}) with "
                   f"rho_max < {rho_bar}: " + "; ".join(notes[:3]))
            v = Verdict.MARGINAL
        best_t = max((abs(t) for _, _, t, _, _ in kept), default=float("nan"))
        if replaced:
            v = Verdict.REPLACED
            why = (f"replaced {len(replaced)} weaker duplicate(s): "
                   + "; ".join(f"{_short(o)} -> {_short(n)} "
                               f"(|t| {ns:.2f} > {os_:.2f}, rho {r:.3f})"
                               for o, n, ns, os_, r in replaced[:3])
                   + ((" | " + why) if why else ""))
        dec = Decision(bool(kept), why, (), float("nan"), float("nan"),
                       best_t, verdict=v)
        try:
            dec.replaced_pairs = [(o, n) for o, n, _, _, _ in replaced]
        except Exception:
            pass
        return dec, [(e, s) for e, s, _, _, _ in kept]

    def _decide_marginal(self, candidates, zoo_signals, zoo_metrics,
                         batch_metrics, main_res: dict | None = None) -> "Decision":
        """Measure the batch's marginal contribution once per seed, then judge.

        The delta is what ``E_Theta`` already computes -- the book with these
        factors minus the book without them -- but measured across seeds rather
        than once, because as a point estimate it flipped 3 of 4 verdicts. The
        combiner averages its seed ensemble into a single prediction, so the
        spread has to come from separate evaluations; that costs roughly one
        extra fit per seed and buys a bar that means something.

        The per-seed evaluations are independent, so they run in a fork
        ProcessPool (see ``_seed_worker_eval``) -- the 5x sequential multiplier
        on the per-batch cost becomes concurrency. ``QA_SEED_WORKERS`` sizes the
        pool (default = number of test seeds); ``=1`` falls back to the original
        sequential loop.

        ``main_res`` is the batch evaluation ``develop`` already ran. When a test
        seed's Θ hashes equal to the main Θ -- which is exactly the case for the
        seed that matches ``combiner.seeds``, e.g. seed 42 under
        ``seeds: [42]`` -- that seed's evaluation is the *same computation* the
        main eval just performed, so it is reused rather than recomputed. The
        delta is identical by construction (equal Θ hash, equal zoo, equal
        candidates); this drops one full evaluation per batch and frees a core.
        """
        from quantaalpha.eval.admission import decide

        test_seeds = [int(s) for s in self.theta.admission.test_seeds]

        # The seed whose Θ is byte-identical to the main Θ (if any): its result
        # is already in hand. Identity is decided by the protocol hash, not by
        # comparing seed ints, so this stays correct if combiner.seeds changes.
        reuse: dict[int, float | None] = {}
        if main_res is not None:
            main_hash = self.theta.hash
            for seed in test_seeds:
                th = _replace(self.theta,
                              combiner=_replace(self.theta.combiner, seeds=(seed,)))
                if th.hash == main_hash:
                    reuse[seed] = main_res.get("m_delta_net_ir")
                    logger.info(
                        "seed %d shares the main Θ hash (%s); reusing the batch "
                        "evaluation instead of recomputing it", seed, main_hash)
                    break

        # One operator per seed for the whole run, pre-seeded with the main op's
        # seed-independent caches (panel, trade mask, benchmark). The seed
        # replacement only changes combiner.seeds, so _windows/_panel/_trade_mask
        # return the same keys and the pre-seeded entries hit on the first worker
        # call -- the panel is never reloaded per seed. Held on self._seed_ops,
        # the panel cache then survives across batches (fork workers inherit it
        # copy-on-write).
        # Only the seeds that still have to be computed (the reused one is
        # already in hand).
        todo = [s for s in test_seeds if s not in reuse]
        for seed in todo:
            if seed not in self._seed_ops:
                th = _replace(self.theta,
                              combiner=_replace(self.theta.combiner, seeds=(seed,)))
                sop = EvaluationOperator(th)
                sop._panels.update(self.op._panels)
                sop._masks.update(self.op._masks)
                sop._benchmarks.update(self.op._benchmarks)
                self._seed_ops[seed] = sop

        computed: dict[int, float | None] = {}
        nw = _seed_workers()
        # fork-in-fork guard: if develop is itself running inside a forked
        # worker, spawning another ProcessPool here can deadlock on inherited
        # locks. Sequential evolution runs develop in the main process, so this
        # is normally off; the guard keeps it safe if that ever changes.
        in_child = mp.parent_process() is not None
        if nw >= 2 and len(todo) >= 2 and not in_child:
            global _MP_STATE, _SEED_OPS
            # Publish the batch's signals + the pre-seeded ops for the forked
            # workers. The pool forks at creation (after these binds), so the
            # workers inherit them copy-on-write -- no large-DataFrame pickling
            # across the pipe, only the seed int is sent per task.
            _MP_STATE = {"candidates": candidates, "zoo_signals": zoo_signals,
                         "zoo_metrics": zoo_metrics}
            _SEED_OPS = self._seed_ops
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(nw, len(todo)),
                                     mp_context=ctx) as ex:
                fut_map = {ex.submit(_seed_worker_eval, s): s for s in todo}
                for fut, seed in fut_map.items():
                    r = fut.result()
                    if r is None:
                        r = {"m_delta_net_ir": None}
                    computed[seed] = r.get("m_delta_net_ir")
        else:
            # Sequential fallback: 1 seed, workers disabled, or already in a
            # forked child. Same evaluation, no pool overhead.
            for seed in todo:
                op = self._seed_ops[seed]
                try:
                    r = op.evaluate(candidates, zoo_signals=zoo_signals,
                                    zoo_metrics=zoo_metrics)
                    computed[seed] = r.get("m_delta_net_ir")
                except Exception:
                    logger.exception("marginal contribution failed on seed %s", seed)
                    computed[seed] = None

        # Reassemble in test_seeds order so the per-seed record is stable
        # regardless of which seed was reused or how the pool completed.
        deltas: list = [reuse.get(s, computed.get(s)) for s in test_seeds]

        weakest = None
        if self.theta.admission.capacity and self._contributions:
            name, value = min(self._contributions.items(), key=lambda kv: kv[1])
            weakest = (name, value)

        d = decide(deltas, len(self._repository), self.theta,
                   metrics=batch_metrics, weakest=weakest)
        if d.admit and d.mean == d.mean:
            for expr in candidates:
                self._contributions[expr] = d.mean
        if d.displaced:
            self._contributions.pop(d.displaced, None)
        return d

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
            self._repository[expr] = (self._compact(signal), batch_metrics)
        logger.info("repository: +%d factor(s), U=%.4f >= tau_admit=%.2f; |zoo| = %d",
                    len(candidates), u, adm.tau_admit, len(self._repository))
        return True

    def _compact(self, signal):
        """Store a signal on the evaluation grid, not on the full raw cache."""
        try:
            start, end, _ = self.op._windows(False)
            return align_signal(signal, self.op._panel(start, end))
        except Exception:                       # never lose a factor to this
            return signal

    @staticmethod
    def _weakest(res: dict, k: int = 2) -> str:
        """The k dimensions this batch scored worst on, named for the prompt.

        e_j is a percentile complement, so a low value means "most of the
        repository beats you here". Naming the two weakest turns a rejection
        into an instruction.
        """
        scores = [(v, key[2:]) for key, v in res.items()
                  if key.startswith("e_") and isinstance(v, (int, float))]
        if not scores:
            return ""
        scores.sort()
        return ", ".join(f"{name} (e={value:.2f})" for value, name in scores[:k])

    def _fdr_bar(self, t_new: float) -> tuple[float, int, float]:
        """Benjamini-Hochberg bar over EVERY factor scored so far this run.

        ``|t| >= 3`` is a per-factor threshold, but the search is itself a
        multiple-testing machine: 150 mined factors give noise 150 chances to
        clear any fixed bar. Harvey-Liu-Zhu recommend FDR over Bonferroni for
        exactly this -- Bonferroni at 150 trials demands ~3.6 sigma and starves
        the repository, while BH controls the expected FALSE-DISCOVERY share.

        Accumulated across the run rather than within a batch, because at one
        factor per hypothesis a batch is a single test and BH on n=1 is a no-op.
        The bar therefore TIGHTENS as the search proceeds, which is the honest
        behaviour: the more you look, the better the evidence has to be.

        Returns ``(t_required, n_tests, q)``.
        """
        from math import erfc, sqrt

        q = float(getattr(self.theta.admission, "fdr_q", 0.0) or 0.0)
        if q <= 0:
            return 0.0, 0, 0.0
        if not hasattr(self, "_t_history"):
            self._t_history: list[float] = []
        if t_new == t_new:
            self._t_history.append(abs(float(t_new)))

        ts = sorted(self._t_history, reverse=True)          # strongest first
        m = len(ts)
        if m == 0:
            return 0.0, 0, q
        # two-sided normal p-value; erfc is accurate in the tail where this lives
        ps = [erfc(t / sqrt(2.0)) for t in ts]              # ascending with rank
        crit = 0.0
        for k in range(m, 0, -1):                           # largest k that holds
            if ps[k - 1] <= (k / m) * q:
                crit = ps[k - 1]
                break
        if crit <= 0:                                       # nothing passes yet
            return float("inf"), m, q
        # invert the p back to a |t| so the reason string is readable
        lo, hi = 0.0, 12.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if erfc(mid / sqrt(2.0)) > crit:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), m, q

    def _live_ic_breakeven(self):
        """Interpolate the book's break-even |IC| from THIS run's priced books.

        Every batch whose book was priced contributes an ``(|IC|, net_ir)``
        pair. With pairs on both sides of ``net_ir = 0`` the crossing is a
        linear interpolation between the bracketing points -- the same
        construction the offline ladder uses, on live data instead of an oracle.

        Returns ``None`` until the run has actually bracketed the crossing.
        Extrapolating from same-signed points would INVENT a bar, and a bar
        nobody measured is worse than no bar: the search would optimise toward
        it.
        """
        pts = [(ic, ir) for ic, ir in getattr(self, "_ic_ir_pairs", [])
               if ic == ic and ir == ir and ic > 0]
        if len(pts) < 2:
            return None
        neg = [q for q in pts if q[1] <= 0]
        pos = [q for q in pts if q[1] > 0]
        if not neg or not pos:
            return None                    # not bracketed: no honest crossing
        lo = max(neg, key=lambda q: q[0])  # highest IC that still loses
        hi = min(pos, key=lambda q: q[0])  # lowest IC that already wins
        if hi[0] <= lo[0] or hi[1] == lo[1]:
            return None
        w = (0.0 - lo[1]) / (hi[1] - lo[1])
        return float(lo[0] + w * (hi[0] - lo[0]))

    def _research_score(self, metrics: dict) -> float:
        """Rank a library member by the criterion it was ADMITTED on.

        ``|t_nw|`` of the factor's own neutralized rank IC. Deliberately the
        same number the gate uses, so admission and eviction cannot disagree.
        Members predating the research gate carry no ``t_nw`` and sort last
        rather than being silently treated as strong.
        """
        t = metrics.get("t_nw")
        try:
            t = abs(float(t))
        except (TypeError, ValueError):
            return float("-inf")
        if t != t:
            return float("-inf")

        # SOFT ECONOMIC GATE.
        #
        # |t| answers "is this signal REAL?". It does not answer "is it big
        # enough to PAY?", and the two bars are far apart: the mined factors
        # clear |t| 3.9-6.9 while sitting 2.2x-10.8x below the |IC| at which a
        # book of them turns net-profitable. A HARD gate on the book bar would
        # admit nothing (0 of 11 clear it), leaving mutation, crossover and
        # admitted-push with no parents at all -- the search would stop
        # learning entirely. So the bar shapes the RANKING instead of the
        # verdict: everything that clears the statistical gate is still
        # admitted, but a factor further below the economic bar sorts lower, so
        # the library cap and the replacement duel evict it first and the zoo
        # drifts toward factors that can pay for themselves.
        #
        # Multiplicative, not additive: the penalty has to keep the score on the
        # |t| scale that admission, eviction and the duel all share, or the
        # three decisions start disagreeing about what "stronger" means.
        bar = metrics.get("ic_breakeven_book")
        ic = metrics.get("rank_ic_neutral")
        try:
            bar, ic = float(bar), abs(float(ic))
        except (TypeError, ValueError):
            return t                       # no measured bar -> rank on |t| alone
        if bar != bar or ic != ic or bar <= 0 or ic <= 0:
            return t
        # ratio in (0, 1]: 1.0 at or above the bar, falling as the shortfall
        # grows. Sqrt so a 4x shortfall halves the score rather than quartering
        # it -- the bar is a book-level property and a single factor below it is
        # not worthless, only worth less.
        return t * min(1.0, ic / bar) ** 0.5

    def _enforce_library_cap(self) -> list[str]:
        """Optionally hold the library at a size cap by dropping the weakest.

        ``QA_MAX_LIBRARY`` overrides ``admission.max_library`` (read here at
        call time); ``0`` disables eviction entirely via the ``cap <= 0``
        early-return below. Default (env unset) = ``admission.max_library``
        (40), byte-identical to the frozen protocol.

        The cap is OPTIONAL, not load-bearing. The admission gates
        (|delta_t|, monotonicity, mechanism+sign, FDR, rho_max, rho_within,
        marginal_er) are the quality filter; this cap is only a count guard,
        and it evicts by |t_nw| -- a DIFFERENT criterion than admission
        (delta_t), so it can remove factors the gate admitted for marginal
        contribution. Under the live ICIR combiner's ``shrinkage=0.5`` net_ir
        GROWS with zoo size with diminishing returns (measured +10.5% |IC| at
        15 factors -> +6.1% at 37); it does not dilute as 1/N. The gates
        self-limit (rho_max and marginal_er tighten as the zoo fills the
        direction space), so an uncapped zoo plateaus where non-redundant
        strong factors run out, not at infinity. The earlier "1/N dilution /
        calibrated to a 34.8 effective-rank ceiling" rationale is refuted -- a
        98-factor mined library measures er=27.5 and er grows with factor count
        (no ~14/34.8 ceiling; see qa-ohlcv-ceiling-14-refuted). Pair an
        uncap/raise with ``QA_RESTORE_CAP_EVICTED`` (ledger.replay_repository)
        so factors removed only by the count cap re-enter the zoo on rehydrate.
        """
        adm = self.theta.admission
        _mlib = os.environ.get("QA_MAX_LIBRARY")
        cap = int(_mlib) if _mlib not in (None, "") else int(getattr(adm, "max_library", 0) or 0)
        if cap <= 0 or len(self._repository) <= cap:
            return []

        ranked = sorted(self._repository.items(),
                        key=lambda kv: self._research_score(kv[1][1]))
        n_drop = len(self._repository) - cap
        dropped = [expr for expr, _ in ranked[:n_drop]]
        for expr in dropped:
            del self._repository[expr]

        self.ledger.append({
            "evicted_exprs": dropped, "n_factors": 0, "metrics": {}, "U": None,
            "eviction_rule": "library_cap",
            "eviction_bar": cap,
            "eviction_scores": {e: self._research_score(m)
                                for e, (_, m) in ranked[:n_drop]},
        })
        logger.info("repository: CAP %d reached -- evicted %d weakest by |t_nw| "
                    "(%s); |zoo| = %d", cap, len(dropped),
                    ", ".join(f"{_short(e)}" for e in dropped[:3]),
                    len(self._repository))
        return dropped

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

        if adm.mode == "marginal_contribution":
            return self._prune_by_contribution()

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
            # Persist the decision, or a sibling process undoes it.
            self.ledger.append({
                "evicted_exprs": evicted, "n_factors": 0, "metrics": {}, "U": None,
                # Same omission the admission record had: which factors left was
                # recorded and why was not, so an eviction could never be
                # audited or second-guessed after the fact.
                "eviction_rule": "utility_rank",
                "eviction_bar": adm.tau_evict,
                "eviction_scores": {e: u for u, e, _, _ in scored if e in set(evicted)},
            })
            logger.info(
                "repository: EVICTED %d factor(s) with U < tau_evict=%.2f; |zoo| = %d",
                len(evicted), adm.tau_evict, len(self._repository),
            )
        return evicted

    # ------------------------------------------------------------------
    def _prune_by_contribution(self) -> list[str]:
        """Evict on leave-one-out contribution, matching how admission decides.

        Admission asks whether the book improves when a batch is added; this
        asks whether it worsens when a member is removed. Both are marginal
        contribution measured against the CURRENT repository, so the two agree
        about what "behind" means -- the percentile ``_prune`` used does not,
        and running them together meant admitting on one criterion and evicting
        on another.

        **Re-measurement is the whole point and it is not free.** A factor that
        was additive at |zoo| = 12 can be redundant at 150, and nothing in its
        own stored metrics changes to say so; the contribution recorded at
        admission is exactly the stale number that cannot detect this. So each
        surviving member is re-priced without itself, which costs one
        evaluation per member. ``evict_every`` is the cadence, and 0 means
        never -- with it off this returns nothing rather than falling back to
        a stale figure or to a percentile that means something else.
        """
        adm = self.theta.admission
        self._rounds_since_evict += 1
        if not adm.evict_every or self._rounds_since_evict < adm.evict_every:
            return []
        self._rounds_since_evict = 0

        from quantaalpha.eval.admission import should_evict

        entries = list(self._repository.items())
        base = self.op.evaluate({}, zoo_signals={e: s for e, (s, _) in entries},
                                zoo_metrics=[m for _, (_, m) in entries])
        base_ir = base.get("m_net_ir")
        if base_ir is None or base_ir != base_ir:
            logger.warning("eviction: repository has no net_ir; skipping this round")
            return []

        # The leave-one-out re-pricings are independent, so they run in the same
        # fork pool the seed evaluations use. At |zoo|=150 this is the
        # difference between ~2.8h and ~0.6h for one eviction round; sequential
        # it was the single largest block of wall-clock in a long mine.
        contributions: dict[str, float] = {}
        nw = _seed_workers()
        in_child = mp.parent_process() is not None
        if nw >= 2 and len(entries) >= 2 and not in_child:
            global _EVICT_STATE
            # Bind BEFORE the pool is created so the workers inherit the
            # entries and the operator copy-on-write (same contract as
            # _MP_STATE); only the held-out expression is sent per task.
            _EVICT_STATE = {"op": self.op, "entries": entries}
            try:
                ctx = mp.get_context("fork")
                with ProcessPoolExecutor(max_workers=min(nw, len(entries)),
                                         mp_context=ctx) as ex:
                    for expr, w_ir in ex.map(_evict_worker_eval,
                                             [e for e, _ in entries]):
                        if w_ir is not None:
                            contributions[expr] = float(base_ir) - float(w_ir)
            finally:
                _EVICT_STATE = {}
        else:
            for expr, _ in entries:
                others = {e: s for e, (s, _) in entries if e != expr}
                try:
                    without = self.op.evaluate(
                        {}, zoo_signals=others,
                        zoo_metrics=[m for e, (_, m) in entries if e != expr])
                    w_ir = without.get("m_net_ir")
                    if w_ir is not None and w_ir == w_ir:
                        contributions[expr] = float(base_ir) - float(w_ir)
                except Exception:
                    logger.exception("eviction: could not re-price %s", expr[:60])

        ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        keep_floor = {e for e, _ in ranked[: adm.min_size]}
        evicted = [e for e, c in ranked if should_evict(c, self.theta) and e not in keep_floor]
        for expr in evicted:
            self._repository.pop(expr, None)
            self._contributions.pop(expr, None)
        self._contributions.update(contributions)
        if evicted:
            self.ledger.append({
                "evicted_exprs": evicted, "n_factors": 0, "metrics": {}, "U": None,
                "eviction_rule": "marginal_contribution",
                "eviction_bar": adm.evict_below,
                # The leave-one-out contributions this round measured. They cost
                # one evaluation per member to obtain and were being thrown
                # away; kept, they are also the record of how the repository's
                # value is distributed at this size.
                "eviction_scores": {e: contributions[e] for e in evicted
                                    if e in contributions},
                "repository_contributions": dict(ranked),
            })
            logger.info("repository: EVICTED %d factor(s) contributing < %.4f "
                        "on re-measurement; |zoo| = %d",
                        len(evicted), adm.evict_below, len(self._repository))
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
        # Rehydrate INTO self._repository, not into locals. _admit and _prune
        # both reason about `len(self._repository)`, so leaving the recovered
        # incumbents in local variables left every fresh process believing the
        # repository was empty -- it would bootstrap-admit unconditionally and
        # could never evict anything a sibling had contributed.
        # Replay the ledger IN ORDER: admissions add, evictions remove. An
        # eviction that is not replayed is no eviction at all -- the next
        # process would rehydrate the dropped factor straight back from the
        # admission record that precedes it.
        recovered = signals_found = 0
        # Align on load, and store the ALIGNED frame. A cached signal is a
        # (datetime, instrument) Series over the full 2008-2026 x 5982-name
        # cache -- 227 MB on disk and more in memory. Holding raw signals for a
        # 150-factor repository is tens of gigabytes resident, which is fine at
        # the 18 factors this ran at during development and fatal at paper
        # scale. Aligned to the evaluation panel each one is a couple of
        # megabytes. align_signal short-circuits on an already-wide frame, so
        # the operator's own alignment stays correct and costs nothing.
        panel = None
        for expr, batch_metrics in replay_repository(self.ledger.path).items():
            if expr in self._repository:
                continue
            try:
                if panel is None:
                    p_start, p_end, _ = self.op._windows(False)
                    panel = self.op._panel(p_start, p_end)
                # Aligned frames are CACHED (load_aligned_signal). Recomputing
                # them meant unstacking a 14.2M-row MultiIndex per incumbent per
                # batch -- 3.43s of the 3.57s, measured -- which is what made
                # backtest time 100s + 5.0s*|zoo| and the search slower the
                # better it did. The cache key includes the panel grid, so a
                # re-split misses rather than returning stale dates.
                signal = load_aligned_signal(expr, panel)
                signals_found += 1
            except (FileNotFoundError, OSError, ValueError):
                # Not yet cached by a sibling process. Still a ranking
                # incumbent; it just cannot join the combiner refit.
                signal = None
            self._repository[expr] = (signal, batch_metrics)
            recovered += 1

        if recovered:
            logger.info(
                "repository rehydrated from ledger: +%d incumbent(s), %d with signals; "
                "|zoo| = %d", recovered, signals_found, len(self._repository),
            )

        # Rehydrate the FDR trial history from the same ledger. _t_history is
        # otherwise in-memory only, and a fresh runner (one per evolution task
        # under parallel execution) starts every batch with m=1, so the
        # n_tests>1 guard at the BH gate never fires and the multiple-testing
        # defence the gate's docstring promises is inert. The ledger is
        # per-run (run.sh:224 gives each run its own ledger_<EXPERIMENT_ID>.jsonl),
        # so this is the current run's trials only -- no cross-run accumulation.
        # Overwrite, not merge: at this point in develop() the current batch has
        # not yet been gated or appended, so in-memory _t_history holds at most
        # prior batches -- exactly what the ledger has -- making the overwrite
        # idempotent in sequential mode and the correct base in parallel mode.
        # _fdr_bar then appends this batch's factors on top as they are gated.
        history = replay_t_history(self.ledger.path)
        if history:
            self._t_history = history
            logger.info(
                "FDR trial history rehydrated from ledger: %d prior test(s); "
                "first factor this batch faces n_tests=%d", len(history), len(history),
            )

        signals = {e: s for e, (s, _) in self._repository.items() if s is not None}
        metrics = [m for _, (_, m) in self._repository.items()]
        if not metrics:
            logger.info(
                "repository empty (%d based experiment(s), ledger has no prior "
                "records); scoring this batch against an empty zoo",
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
            # Carried into the evolution prompts so a rejected parent tells the
            # generator what to fix, not merely that it failed. Without these
            # the gate only removes factors; with them it steers the search,
            # which is the point of a repository-relative objective.
            "admitted": bool(res.get("admitted", True)),
            "tau_admit": self.theta.admission.tau_admit,
            "weakest_dimensions": self._weakest(res),
            "theta_hash": res.get("theta_hash"),
            "zoo_hash": res.get("zoo_hash"),
            "zoo_size": res.get("zoo_size"),
            "rho_max": res.get("m_rho_max"),
            # Redundancy WITHIN this batch, i.e. how much the factors of a single
            # hypothesis duplicate EACH OTHER. rho_max only ever measured overlap
            # with the repository, so three mutually identical factors scored as
            # novel. Threading it here is what lets the generator be TOLD its
            # three factors tested one direction instead of three.
            "rho_within": res.get("m_rho_within"),
            "turnover_book": res.get("m_turnover_book"),
            "turnover_solo": res.get("m_turnover_solo"),
            "cx": res.get("m_cx"),
            "cost_bps": res.get("m_cost_bps"),
            # Marginal contribution of this batch over the repository alone.
            # The evolution controller selects parents on _PRIMARY_METRIC, and
            # until now this key never reached it -- so "best" could only ever
            # mean best by U, which drifts with repository size (corr +0.41)
            # rather than tracking what the batch added (corr +0.04).
            "delta_net_ir": res.get("m_delta_net_ir"),
            "delta_net_arr": res.get("m_delta_net_arr"),
            "base_net_ir": res.get("m_base_net_ir"),
            # Seed-averaged marginal contribution and its standard error, from
            # decide() over admission.test_seeds (merged into res by
            # decision.as_record()). Selection ranks on the shrunk form of
            # delta_mean rather than the single-seed m_delta_net_ir point
            # estimate, which flipped 3 of 4 verdicts and made ranking chase
            # noise. Absent outside marginal_contribution mode -> None.
            "delta_mean": res.get("delta_mean"),
            "delta_se": res.get("delta_se"),
            # The t-statistic decide() computed (mean/se). Lets the diagnosis
            # distinguish a *resolvably negative* contribution (large |t|, the
            # book gets worse) from an *unresolved* one (small |t|), which the
            # reason string already encodes in prose but not in a machine-
            # readable scalar.
            "delta_t": res.get("delta_t"),
            # The verdict and its explanation, from decision.as_record(). These
            # were written to the ledger (see the **{...} merge above) but
            # dropped here, so the most diagnostic string in the pipeline --
            # "contribution is resolvably NEGATIVE ..." / "rho_max ... (duplicate)"
            # / "coverage ... (too sparse to price)" -- never reached
            # backtest_metrics, format_objective_note, or the mutation prompt.
            # Threading them through is what lets the diagnose-and-refine
            # operator turn a rejection into a *directional* refinement
            # instruction instead of a bare "you failed".  Absent (None) under
            # non-marginal-contribution mode, so the result stays
            # byte-identical to before.
            # THE PER-FACTOR TEAR SHEETS. `_to_series` is an explicit scalar
            # allowlist, so everything the standalone gate measures --
            # neutralized RankIC, t_NW, best horizon, monotonicity, size
            # exposure, the FDR bar, capacity, and the pre-registered-vs-
            # realized sign -- was published into `res` and then dropped right
            # here, one step before the summarizer.
            #
            # Measured on the 2026-08-21 run: 243 LLM calls, ZERO carrying any
            # of those fields. The generator was scored on the research gate and
            # told only the batch aggregates and a reason string, which is
            # exactly the "one uninformative scalar" this layer exists to
            # replace. A dict rides in a Series as an object; `_as_dict` calls
            # `.to_dict()` and hands it straight to `_per_factor_lines`.
            # The batch's best research score, available on EVERY batch because
            # it needs no book. Parent selection ranks on `QA_PRIMARY_METRIC`,
            # which is a book metric (`delta_net_ir`) and therefore absent
            # whenever the book is skipped -- `get_primary_metric()` then falls
            # back to 0.0 and every trajectory ties. With `book_eval_every: 5`
            # that was already true of 4 batches in 5. This is the same |t_NW|
            # admission judges on, so ranking on it makes breeding consistent
            # with selection and independent of how often the book is priced.
            # THE FACTOR'S OWN SIGNED IC -- distinct from `rank_ic`, which is the
            # COMPOSITE book's. The failure-memory block renders whatever this
            # trajectory carries under the instruction "Read the RankIC SIGN as
            # evidence", and it was rendering the book number: measured across
            # this run's prompts, 43 of 43 values shown were POSITIVE (median
            # +0.0283) while the factors themselves realized NEGATIVE 71% of the
            # time (median -0.0154). The one directional signal in the memory
            # channel was not merely uninformative, it was inverted, under an
            # explicit instruction to trust its sign.
            "rank_ic_own": next(
                (float(v["rank_ic_neutral"]) for v in
                 (res.get("factor_tearsheets") or {}).values()
                 if isinstance(v, dict) and v.get("rank_ic_neutral") is not None
                 and float(v["rank_ic_neutral"]) == float(v["rank_ic_neutral"])),
                None),
            "best_t_nw": max(
                (abs(float(v.get("t_nw"))) for v in
                 (res.get("factor_tearsheets") or {}).values()
                 if isinstance(v, dict) and v.get("t_nw") is not None
                 and float(v["t_nw"]) == float(v["t_nw"])),
                default=None),
            "factor_tearsheets": res.get("factor_tearsheets") or {},
            "admitted_exprs": res.get("admitted_exprs") or [],
            # Book-level deflation, present only on batches that priced a book.
            "dsr": res.get("dsr"),
            "dsr_n_trials": res.get("dsr_n_trials"),
            "reason": res.get("reason"),
            "pathology": res.get("pathology"),
            # The structured verdict (T1): the branch in decide() already knows
            # which case it is, so this is the authoritative verdict -- not a
            # string classify_verdict has to re-parse from ``reason``. diagnosis
            # reads it directly and only falls back to substring matching (with a
            # warning) for records written before this field existed.
            "verdict": res.get("verdict"),
            # The incumbent a replacement admission evicted, or None. Carried so
            # the diagnosis can tell "repository full, did not beat the weakest"
            # from "resolvably negative".
            "displaced": res.get("displaced"),
            # Per-seed marginal contributions (list of floats). The diagnosis
            # itself uses delta_mean/delta_se/reason; the raw per-seed vector is
            # kept for auditability -- it is the reproducible evidence behind
            # the verdict, and it already lives in the ledger.
            "delta_per_seed": res.get("delta_per_seed"),
            # T4: per-factor combiner credit (ICIR path only). Keyed by factor
            # expression -> {weight, weight_raw, weight_stability, ic_mean,
            # ic_std, rank_ic, turnover_share}. segment_profiling (T2) folds
            # these into each SegmentProfile so the expression-aware diagnosis
            # (T3) names the strongest/costliest sub-pattern by ACTUAL credit,
            # and the crossover (T7) splices high-credit segments across parents.
            # Empty dict on the LightGBM path (no per-factor weights) and on a
            # cache miss -- _NET_COST_DICT_KEYS carries it through to metrics.
            "factor_attribution": res.get("factor_attribution") or {},
        }
        payload.update({k: v for k, v in res.items() if k.startswith("e_")})
        return pd.Series(payload)


__all__ = ["NetCostFactorRunner"]
