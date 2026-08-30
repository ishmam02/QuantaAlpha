"""``E_Θ`` — the evaluation operator (Eq. 13).

The entry point the rest of the system talks to. A pure function of
``(f, zoo, Θ)``: identical inputs give byte-identical output, which is
Property 2 and is what the ledger's reproducibility claim rests on.

A whole factor SET is evaluated at once, exactly as ``qrun``
evaluates a round's factors together. Pipeline, in order:

1. hash the repository state (``zoo_hash``);
2. refit the combiner on ``zoo ∪ candidates``, map the composite prediction
   through ``g``, price Eq. 7, and measure the resulting book;
3. IC statistics **of that combined prediction** -- the same quantity Qlib's
   ``SigAnaRecord`` reports, which keeps the numbers comparable to Qlib;
4. ``ρ_max`` and ``cx`` of the new batch against the repository;
5. score each dimension by relative rank within the repository and scalarize
   to ``U``.

There are no absolute floors: admissibility gates were removed after
measurement showed the criteria were either anti-correlated with contribution
or irreproducible across combiner seeds. Selection is by ``U`` -- a *dynamic*,
repository-relative criterion whose bar rises as the repository improves.

Strategy-level metrics are measured on the **OOS proxy window**, never on the
window the combiner was fitted to — an in-sample net IR would be meaningless.
The final test window is reachable only through the explicit ``report=True``
mode, which exists solely for the end-of-run head-to-head.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from quantaalpha.eval import combiner as combiner_mod
from quantaalpha.eval import costs as costs_mod
from quantaalpha.eval.data import PanelBundle, align_signal, load_benchmark, load_panel
from quantaalpha.eval.execution import fill_prices, grinold_alpha, prediction_scale, realized_return
from quantaalpha.eval.metrics import (
    _abs_spearman,
    _cross_sectional_corr,
    cx,
    effective_rank,
    prediction_metrics,
    spearman_block_cached,
    strategy_metrics,
)
from quantaalpha.eval.portfolio import build_book
from quantaalpha.eval.protocol import Protocol
from quantaalpha.eval.tradability import trade_mask
from quantaalpha.eval.scoring import dimension_scores, utility

logger = logging.getLogger(__name__)


def _transfer_coefficient(alpha: pd.DataFrame, w: pd.DataFrame) -> tuple[float, int]:
    """Clarke-de Silva-Thorley (2002) transfer coefficient: the mean per-rebalance
    cross-sectional ``corr(α, w*)`` -- the fraction of the alpha vector the
    portfolio constraints (long-only, position cap, κ-inertia) let through.

    ``alpha`` and ``w`` are wide T×N frames aligned on the evaluation window.
    Returns ``(mean_tc, n_dates)``; ``mean_tc`` is NaN if no date had enough
    paired names. ``IR = TC·IC·√N``, so a low TC flags the position cap as the
    binding constraint rather than factor quality. Pure function -- no Θ -- so
    the diagnostic is unit-testable without standing up Qlib.
    """
    tc_vals = []
    for d in w.index:
        av, wv = alpha.loc[d].to_numpy(), w.loc[d].to_numpy()
        m = np.isfinite(av) & np.isfinite(wv)
        if m.sum() >= 5:
            av, wv = av[m] - av[m].mean(), wv[m] - wv[m].mean()
            denom = np.sqrt((av**2).sum() * (wv**2).sum())
            if denom > 0:
                tc_vals.append(float((av * wv).sum() / denom))
    mean_tc = float(np.mean(tc_vals)) if tc_vals else float("nan")
    return mean_tc, len(tc_vals)


class EvaluationOperator:
    """``E_Θ``. Construct once per run; Θ is frozen for its lifetime."""

    def __init__(self, theta: Protocol) -> None:
        self.theta = theta
        self._panels: dict[tuple[str, str], PanelBundle] = {}
        self._benchmarks: dict[tuple[str, str], pd.Series] = {}
        # Baseline books keyed by (zoo_hash, window) -- one fit per repository
        # state, not per candidate.
        self._baselines: dict[tuple, dict] = {}
        # The zoo x zoo |Spearman| block for the effective_rank metric lives in
        # the SHARED ``spearman_block_cached`` cache (metrics.py), keyed by the
        # held-zoo signal set + panel grid -- so the operator's effective_rank
        # metric and the runner's marginal-er gate (same held zoo, same panel)
        # share one O(zoo**2) build per batch instead of two.
        self._masks: dict[tuple, object] = {}

    # ------------------------------------------------------------------
    def _trade_mask(self, panel: PanelBundle):
        """Feasibility masks for a panel, built once and reused."""
        key = (panel.dates[0], panel.dates[-1], len(panel.instruments))
        if key not in self._masks:
            self._masks[key] = trade_mask(panel, self.theta)
        return self._masks[key]

    def _panel(self, start: str, end: str) -> PanelBundle:
        key = (start, end)
        if key not in self._panels:
            self._panels[key] = load_panel(self.theta, start, end)
        return self._panels[key]

    def _benchmark(self, start: str, end: str) -> pd.Series:
        key = (start, end)
        if key not in self._benchmarks:
            self._benchmarks[key] = load_benchmark(self.theta, start, end)
        return self._benchmarks[key]

    def _windows(self, report: bool) -> tuple[str, str, tuple[str, str]]:
        """Panel span and the window strategy metrics are measured on.

        Both come from Θ. ``report`` selects a *named mode*, not a raw window,
        so the test split cannot be reached by passing a date.
        """
        splits = self.theta.splits
        if report:
            train_start = splits.window(self.theta.combiner.fit_split)[0]
            eval_window = splits.final_test
            return train_start, eval_window[1], eval_window
        train_start = splits.window(self.theta.combiner.fit_split)[0]
        eval_window = splits.oos_window
        return train_start, eval_window[1], eval_window

    # ------------------------------------------------------------------
    def evaluate(
        self,
        candidates,
        candidate_expr: str | None = None,
        zoo_signals: dict[str, object] | None = None,
        zoo_metrics: list[dict] | None = None,
        *,
        report: bool = False,
        skip_book: bool = False,
    ) -> dict:
        """Score a factor SET as one book, the way the baseline's qrun does.

        ``candidates`` is ``{expression: signal}`` -- every factor the experiment
        produced, evaluated **together**, not one at a time. This mirrors the
        qrun: Qlib fits the combined model on all of the round's factors
        and reports one set of metrics for the resulting book. The only
        differences here are the cost model (kappa0+kappa1+kappa2 instead of a
        flat fee) and the repository-relative scoring layered on top.

        Evaluating factors one at a time, as this used to, was inconsistent with that
        and forced a per-factor admissibility verdict
        that measurement showed was not reproducible across combiner seeds.

        A bare signal is still accepted, for probes and ad-hoc scoring.
        """
        if not isinstance(candidates, dict):
            candidates = {candidate_expr or "CANDIDATE": candidates}
        zoo_signals = dict(zoo_signals or {})
        zoo_metrics = list(zoo_metrics or [])

        panel_start, panel_end, eval_window = self._windows(report)
        panel = self._panel(panel_start, panel_end)

        aligned_cands = {e: align_signal(s, panel) for e, s in candidates.items()}
        aligned_zoo = {expr: align_signal(sig, panel) for expr, sig in zoo_signals.items()}
        zoo_hash = combiner_mod.zoo_hash(zoo_signals)

        # --- the book: zoo + ALL new factors, one combiner refit per fold ---
        book = self._strategy_batch(aligned_cands, aligned_zoo, panel, eval_window,
                                    report=report, skip_book=skip_book)
        metrics = dict(book["metrics"])

        # --- predictive quality OF THE COMBINED PREDICTION ---
        # Qlib's SigAnaRecord reports IC of the combined model's output, not of
        # any single factor, so this is the directly comparable quantity.
        metrics.update(prediction_metrics(book["prediction"], panel, self.theta, eval_window))

        # --- properties of the new batch itself ---
        # rho_max, rho_within AND the effective_rank tail all need the SAME
        # candidate-involving |Spearman| entries -- candidate-vs-zoo (K x zoo)
        # and candidate-vs-candidate (K x K). Pre-Fix #3 each rebuilt its own
        # copy (rho_max's 250 + rho_within's 10 + the effective_rank tail's
        # 260, at zoo=50); building them once and reading all three metrics off
        # the one matrix cuts the per-batch spearman cost ~71s with no change to
        # any metric -- the entries are the same ``_abs_spearman`` primitive the
        # three paths already shared, assembled in the same dict-iteration
        # order. The zoo x zoo block is NOT rebuilt here: it is the shared
        # ``spearman_block_cached`` cache, slotted in only for effective_rank.
        _n_zoo = len(aligned_zoo)
        _n_cand = len(aligned_cands)
        _cand_keys = list(aligned_cands)
        _cand_sigs = list(aligned_cands.values())
        _zoo_sigs = list(aligned_zoo.values())
        _cz = np.zeros((_n_cand, _n_zoo))   # candidate i vs zoo j
        _cc = np.eye(_n_cand)               # candidate i vs candidate j, diag 1.0
        for _i in range(_n_cand):
            _si = _cand_sigs[_i]
            for _j in range(_n_zoo):
                _cz[_i, _j] = _abs_spearman(_si, _zoo_sigs[_j])
            for _j in range(_i):
                _v = _abs_spearman(_si, _cand_sigs[_j])
                _cc[_i, _j] = _cc[_j, _i] = _v

        # rho_max (Eq. 9): max |Spearman| of any candidate against any zoo
        # incumbent. ``rho_max_arg`` starts at 0.0 and skips empty corrs, so the
        # max over the candidate-vs-zoo block (whose ``_abs_spearman`` entries
        # are 0.0 for empty pairs) reproduces it bit-for-bit; an empty zoo or
        # empty batch -> 0.0 (trivially novel / nothing to compare).
        metrics["rho_max"] = float(_cz.max()) if (_n_zoo and _n_cand) else 0.0

        # WITHIN-batch redundancy. ``rho_max`` above compares each candidate to
        # the ZOO only, so a batch whose members are copies of EACH OTHER passes
        # the redundancy gate untouched. Measured 2026-08-15 on the live 9-factor
        # zoo: every one of the three admitted batches contained a near-duplicate
        # pair (rho 0.85, 1.00, 0.89), leaving 9 factors with an effective rank of
        # 5.0. ``factors_per_hypothesis: 3`` means a batch nominally adds three
        # directions; in practice it was adding about one. Read off the
        # candidate-vs-candidate block built above (spearman, for the same reason
        # as rho_max: the combiner rank-normalises features, so X and RANK(X) are
        # one column).
        _within = 0.0
        _worst_pair = None
        if _n_cand >= 2:
            _iu = np.triu_indices(_n_cand, k=1)
            if _iu[0].size:
                _flat = _cc[_iu]
                _idx = int(np.argmax(_flat))
                _within = float(_flat[_idx])
                # argmax returns the first maximal pair in (i<j) order, matching
                # the strict-``>`` running-max loop it replaces (ties keep the
                # first); name the pair only when the max is non-trivial.
                if _within > 0.0:
                    _worst_pair = (
                        _cand_keys[int(_iu[0][_idx])],
                        _cand_keys[int(_iu[1][_idx])],
                    )
        metrics["rho_within"] = float(_within)
        if _worst_pair is not None and _within > 0.8:
            logger.warning(
                "batch self-duplication: rho_within=%.3f between %.60s AND %.60s "
                "-- this batch adds fewer independent directions than it has factors",
                _within, _worst_pair[0], _worst_pair[1],
            )
        # EFFECTIVE RANK of the book: how many INDEPENDENT bets it carries.
        #
        # rho_max and rho_within are extrema -- they catch a duplicate pair and
        # say nothing about the book as a whole. Measured 2026-08-24: a
        # 37-factor zoo whose worst pair was 0.997 still carried 12.0
        # independent directions, and after the redundancy gate 15 factors
        # carried 10.6. Those are the numbers that describe breadth, and
        # neither extremum shows them.
        #
        # exp(entropy of the normalised eigenvalue spectrum) -- the standard
        # participation-ratio form. Computed over candidates AND zoo, since the
        # book is what gets traded. The candidate-involving tail is already
        # built above (shared with rho_max / rho_within); the zoo x zoo block is
        # the shared ``spearman_block_cached`` cache, so the only per-batch work
        # left here is one eigendecomposition of the assembled matrix.
        try:
            _all = {**aligned_zoo, **aligned_cands}
            _keys = list(_all)
            if len(_keys) >= 2:
                _n = len(_keys)
                if not (aligned_zoo.keys() & aligned_cands.keys()):
                    # No candidate duplicates a zoo expression: assemble the
                    # book's |Spearman| matrix from the cached zoo x zoo block
                    # and the pre-built candidate tail. Bit-identical to the
                    # ``effective_rank_cached`` path it replaces -- same
                    # ``_abs_spearman`` entries, same dict-iteration order, and
                    # eigenvalues are invariant to the block slotting.
                    _R_zoo = spearman_block_cached(
                        aligned_zoo, (panel_start, panel_end))
                    _R = np.eye(_n_zoo + _n_cand)
                    _R[:_n_zoo, :_n_zoo] = _R_zoo
                    _R[:_n_zoo, _n_zoo:] = _cz.T
                    _R[_n_zoo:, :_n_zoo] = _cz
                    _R[_n_zoo:, _n_zoo:] = _cc
                    _er = effective_rank(_R)
                else:
                    # A candidate expression duplicates a zoo one: the book
                    # dedupes the duplicate via the dict merge ({**zoo, **cands}
                    # keeps one), so the pre-built full tail does not apply --
                    # rebuild the deduped matrix inline (always bit-identical).
                    _R = np.eye(_n)
                    for _i in range(_n):
                        for _j in range(_i + 1, _n):
                            _c = _cross_sectional_corr(
                                _all[_keys[_i]], _all[_keys[_j]], "spearman")
                            _v = abs(float(_c.mean())) if not _c.empty else 0.0
                            _R[_i, _j] = _R[_j, _i] = _v
                    _ev = np.linalg.eigvalsh(_R)[::-1]
                    _ev = _ev[_ev > 1e-12]
                    _pp = _ev / _ev.sum()
                    _er = float(np.exp(-(_pp * np.log(_pp)).sum()))
                metrics["effective_rank"] = _er
                metrics["rank_density"] = _er / _n
                metrics["book_n_factors"] = _n
        except Exception:                      # never lose a batch to a metric
            pass

        metrics["cx"] = max((cx(e) for e in aligned_cands), default=0)
        metrics["n_factors"] = len(aligned_cands)

        # --- dynamic (repository-relative) scoring only; no absolute floors ---
        scores = dimension_scores(metrics, zoo_metrics, self.theta)
        u = utility(metrics, zoo_metrics, self.theta)

        # Lift the return PATH out before the `m_` prefixing. Left in, it would
        # become `m__net_return_series` -- which the caller's pop would miss, so
        # a full return series would ride into the ledger JSON and the metric
        # surfaces instead of being consumed and dropped.
        _ret_series = metrics.pop("_net_return_series", None)

        result: dict = {f"m_{k}": v for k, v in metrics.items()}
        if _ret_series is not None:
            result["_net_return_series"] = _ret_series
        result.update({f"e_{k}": v for k, v in scores.items()})
        result.update(
            U=float(u),
            theta_hash=self.theta.hash,
            zoo_hash=zoo_hash,
            zoo_size=len(zoo_signals),
            n_factors=len(aligned_cands),
            eval_window=f"{eval_window[0]}..{eval_window[1]}",
        )

        # T4: per-factor combiner credit (ICIR path only). The diagnosis (T3)
        # reads this to name the strongest/costliest sub-pattern by ACTUAL
        # combiner weight + turnover share, and the crossover (T7) splices the
        # high-credit segments across parents. LightGBM -> attribution None ->
        # empty dict (no per-factor weights on that path).
        result["factor_attribution"] = self._factor_attribution(
            book.get("attribution"), aligned_cands, aligned_zoo,
        )

        # The primary whole-system signal: if this line is absent from a run's
        # logs, the new engine is not being called at all.
        logger.info(
            "E_theta eval: %d factor(s) theta=%s zoo=%s U=%.4f net_ir=%.4f net_arr=%.2f%%",
            len(aligned_cands), self.theta.hash, zoo_hash, float(u),
            metrics.get("net_ir", float("nan")), 100 * metrics.get("net_arr", float("nan")),
        )
        return result

    # ------------------------------------------------------------------
    def _folds(self, report: bool) -> list[tuple]:
        """(theta, eval_window) pairs the in-loop score averages over.

        One entry unless walk-forward is on. The final-test evaluation always
        uses the configured split: progressive retraining is a property of how
        the *search* selects factors, and changing the reporting window would
        make the arms incomparable to each other and to published numbers.
        """
        wf = self.theta.walk_forward
        if report or not wf.enabled or wf.folds <= 1:
            return [(self.theta, None)]
        out = []
        # Pass the label horizon so the purge gap scales with it: a 20-day
        # label leaks 20 days across a boundary, a 1-day label leaks one.
        _h = int(getattr(self.theta.execution, 'label_horizon', 1) or 1)
        for train, valid in self.theta.splits.walk_forward_folds(int(wf.folds), horizon=_h):
            splits = replace(self.theta.splits, train=train, valid=valid)
            out.append((replace(self.theta, splits=splits), valid))
        return out

    def _signal_only(self, zoo_signals, candidate_signals, panel, eval_window):
        """Fit the combiner per fold, but never build a book.

        The expensive half of an evaluation is the portfolio: a capped-simplex
        projection per date across every fold, plus the cost model and the
        zoo-alone baseline. None of it is needed to score a factor on its own
        significance, which is what admission now does.

        Returns the LAST fold's prediction, exactly as ``_fit_and_price`` does,
        and computes NO prediction metrics of its own -- ``evaluate`` derives
        those from the returned prediction. Both properties matter: fit on a
        single whole-window pass, or average metrics across folds here, and the
        same factor reports a different RankIC depending on whether the book
        happened to be priced that batch -- a difference with nothing to do with
        the factor.
        """
        from quantaalpha.eval import combiner as combiner_mod

        prediction, attribution = None, None
        for theta_fold, _fold_window in self._folds(False):
            prediction, attribution = combiner_mod.fit_predict(
                zoo_signals, None, panel, theta_fold,
                candidate_signals=candidate_signals)
        return {"book_priced": False}, prediction, attribution

    def _fit_and_price(
        self,
        zoo_signals: dict[str, pd.DataFrame],
        candidate_signals: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
        report: bool,
    ) -> tuple[dict, pd.DataFrame, list[dict] | None]:
        """Refit and price one factor set, averaged over walk-forward folds.

        Shared by the candidate book and the baseline book. Keeping it separate
        from :meth:`_strategy_batch` is not cosmetic: the batch computes a
        marginal contribution against the baseline, so if the baseline were
        built by calling the batch the two would recurse without end.

        T4: the combiner now returns ``(prediction, attribution)``; the
        per-factor credit is carried alongside the prediction. Only the last
        fold's attribution is returned (the configured split, comparable with a
        non-walk-forward run's IC statistics, exactly as the prediction is) --
        there is nothing to average across folds, the weights are a property of
        the fit, not the window priced over. ``None`` on the LightGBM path.
        """
        folds = self._folds(report)
        per_fold, predictions, attributions, windows = [], [], [], []
        for theta_fold, fold_window in folds:
            pred, attribution = combiner_mod.fit_predict(
                zoo_signals, None, panel, theta_fold,
                candidate_signals=candidate_signals)
            win = fold_window or eval_window
            per_fold.append(self._book(pred, panel, win, theta_fold))
            predictions.append(pred)
            attributions.append(attribution)
            windows.append(f"{win[0]}..{win[1]}")

        if len(per_fold) == 1:
            return dict(per_fold[0]), predictions[0], attributions[0]

        keys = {k for m in per_fold for k in m}
        metrics: dict = {}
        for k in keys:
            vals = [float(m[k]) for m in per_fold
                    if isinstance(m.get(k), (int, float)) and m[k] == m[k]]
            metrics[k] = float(np.mean(vals)) if vals else np.nan
        metrics["n_folds"] = len(per_fold)

        # ---- KEEP THE PER-FOLD VECTOR -------------------------------------
        # The mean alone throws away the reason for running folds. Fold net_irs
        # of [+0.5, +0.5, +0.5] and [+1.5, 0.0, 0.0] average identically and are
        # completely different propositions: one is an edge that holds across
        # regimes, the other is an edge that existed in one year. The recorded
        # behaviour of this search is the second -- every batch loses 2019/2020
        # and wins 2021 -- and with folds=1 scored on 2021 alone that was
        # invisible, so an in-loop positive meant "worked in 2021", not "works".
        #
        # These per-fold entries are also the honest variance for the admission
        # test. The 5 test_seeds only resample the combiner, so they measure
        # model noise; dispersion ACROSS FOLDS measures whether the edge
        # survives a change of regime, which is the thing actually in question.
        metrics["fold_windows"] = list(windows)
        for key in ("net_ir", "net_arr", "rank_ic", "turnover_book", "cost_bps"):
            series = [float(m[key]) for m in per_fold
                      if isinstance(m.get(key), (int, float)) and m[key] == m[key]]
            metrics[f"fold_{key}"] = series
            if len(series) > 1:
                metrics[f"{key}_fold_std"] = float(np.std(series, ddof=1))
                metrics[f"{key}_fold_min"] = float(np.min(series))
                metrics[f"{key}_fold_max"] = float(np.max(series))

        ir_series = metrics.get("fold_net_ir") or []
        metrics["folds_positive"] = int(sum(1 for v in ir_series if v > 0))
        # Sign consistency: an allocator asks "did this work in every regime",
        # not "what is the pooled t-statistic".
        metrics["fold_sign_consistent"] = bool(
            ir_series and all(v > 0 for v in ir_series))

        worst = (min(range(len(ir_series)), key=lambda i: ir_series[i])
                 if ir_series else None)
        logger.info(
            "walk-forward: %d folds | net_ir per fold = %s | positive %d/%d%s",
            len(per_fold),
            ", ".join(f"{w}: {v:+.4f}" for w, v in zip(windows, ir_series)),
            metrics["folds_positive"], len(ir_series),
            f" | WORST {windows[worst]} at {ir_series[worst]:+.4f}"
            if worst is not None else "",
        )
        # The last fold is the configured split, so its prediction is the one
        # comparable with a non-walk-forward run's IC statistics.
        return metrics, predictions[-1], attributions[-1]

    def _strategy_batch(
        self,
        aligned_cands: dict[str, pd.DataFrame],
        aligned_zoo: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
        report: bool = False,
        skip_book: bool = False,
    ) -> dict:
        """The book from ``zoo ∪ candidates``, plus its gain over the zoo alone."""
        if skip_book:
            # Price the SIGNAL, not the book. Everything structural (hashes,
            # zoo size, window, prediction metrics) is still produced; only the
            # optimizer, the cost model and the baseline are skipped. Book
            # metrics are left ABSENT rather than filled with zeros, so a
            # consumer that needs them fails loudly instead of averaging in a
            # number nobody measured.
            metrics, prediction, attribution = self._signal_only(
                aligned_zoo, aligned_cands, panel, eval_window)
        else:
            metrics, prediction, attribution = self._fit_and_price(
                aligned_zoo, aligned_cands, panel, eval_window, report)

        # Marginal contribution of the whole batch over the repository alone.
        baseline = ({} if skip_book
                    else self._baseline(aligned_zoo, panel, eval_window, report=report))
        for key in ("net_ir", "net_arr"):
            base_v, full_v = baseline.get(key), metrics.get(key)
            metrics[f"base_{key}"] = base_v
            metrics[f"delta_{key}"] = (
                full_v - base_v
                if base_v is not None and full_v is not None
                and base_v == base_v and full_v == full_v
                else np.nan
            )
        return {"metrics": metrics, "prediction": prediction, "attribution": attribution}

    # ------------------------------------------------------------------
    def _factor_attribution(
        self,
        attribution: list[dict] | None,
        aligned_cands: dict[str, pd.DataFrame],
        aligned_zoo: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        """Turn per-factor combiner weights + signals into the credit dict T3/T7 read.

        Layers the **turnover share** (the fraction of the book's daily
        rebalance turnover attributable to each factor's signal movement) on
        top of the ICIR combiner's per-factor weights, so the expression-aware
        diagnosis (T3) and the AST-splice crossover (T7) can see not just WHICH
        factor carries the edge (``weight``) but which one drives the cost
        (``turnover_share``)::

            turnover_share_i = mean_t( |w_i · Δx_i| / Σ_j |w_j · Δx_j| )

        where ``w_i`` is the signed mean ICIR weight (``weight_raw``) and ``x_i``
        is the factor's aligned signal panel (held by the operator alongside the
        weights after the T4 widening). ``attribution`` is ``None`` on the
        LightGBM path or a cache miss -> empty dict, so downstream code degrades
        to the no-credit path rather than crashing.
        """
        if not attribution:
            return {}
        panels = {**aligned_zoo, **aligned_cands}
        # Per-date gross rebalance contribution |w_i·Δx_i|, summed across names.
        per_factor: dict[str, pd.Series] = {}
        for rec in attribution:
            expr = rec.get("expr")
            w = rec.get("weight_raw")
            sig = panels.get(expr) if expr is not None else None
            if w is None or w != w or sig is None or getattr(sig, "empty", True):
                continue
            dx = sig.diff()
            contrib = (float(w) * dx).abs().sum(axis=1)   # one value per date
            per_factor[expr] = contrib
        if per_factor:
            frame = pd.DataFrame(per_factor)
            total = frame.sum(axis=1)
            valid = total > 0
        else:
            valid = None
        out: dict[str, dict] = {}
        for rec in attribution:
            expr = rec.get("expr")
            share = None
            if expr in per_factor and valid is not None and valid.any():
                s = per_factor[expr]
                share = float((s[valid] / total[valid]).mean())
            out[expr] = {
                "weight": rec.get("weight"),
                "weight_raw": rec.get("weight_raw"),
                "weight_stability": rec.get("weight_stability"),
                "ic_mean": rec.get("ic_mean"),
                "ic_std": rec.get("ic_std"),
                "rank_ic": rec.get("rank_ic"),
                "turnover_share": share,
            }
        return out

    # ------------------------------------------------------------------
    def _baseline(
        self,
        aligned_zoo: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
        report: bool = False,
    ) -> dict:
        """The book built from ``zoo`` alone -- what f's contribution is measured against.

        Depends on ``(zoo, Θ, window)`` but **not** on the candidate, and the
        runner holds the zoo fixed across every candidate in an experiment, so
        this costs one extra combiner fit per experiment rather than per factor.
        With an empty zoo it is the null model: base features only.
        """
        key = (combiner_mod.zoo_hash(aligned_zoo), eval_window, report)
        if key not in self._baselines:
            self._baselines[key] = self._fit_and_price(
                aligned_zoo, {}, panel, eval_window, report)[0]
            logger.info(
                "baseline book for zoo=%s (|zoo|=%d): net_ir=%.4f net_arr=%.4f",
                key[0], len(aligned_zoo),
                self._baselines[key].get("net_ir", float("nan")),
                self._baselines[key].get("net_arr", float("nan")),
            )
        return self._baselines[key]

    def _book(
        self,
        prediction: pd.DataFrame,
        panel: PanelBundle,
        eval_window: tuple[str, str],
        theta: Protocol | None = None,
    ) -> dict:
        """Price one composite prediction through g and the cost model."""
        theta = theta or self.theta
        # Exposure-neutral construction: the book optimises a prediction with
        # size/industry/beta removed, so its active exposure to those is ~0 by
        # construction rather than by constraint. Degrades to the raw prediction
        # if the reference data is missing -- loudly, because a silently raw
        # book is exactly the bet this is meant to remove.
        if bool(getattr(theta.portfolio, "neutralize_prediction", False)):
            try:
                from quantaalpha.eval.neutralize import residualize
                prediction = residualize(prediction, panel, theta)
            except Exception as exc:
                logger.warning("prediction neutralization failed (%s: %s); the book "
                               "will carry its raw risk exposures",
                               type(exc).__name__, exc)
        start, end = eval_window
        window_pred = prediction.loc[str(start) : str(end)]
        if window_pred.empty:
            return {"net_ir": np.nan, "net_arr": np.nan, "mdd": np.nan,
                    "turnover_book": np.nan, "cost_bps": np.nan}

        y_tilde = realized_return(fill_prices(panel, theta))
        universe = panel.universe.loc[window_pred.index]

        sigma = costs_mod.trailing_vol(panel.close, theta.costs.vol_window)
        adv = costs_mod.trailing_adv(panel, theta)

        # Hard A-share feasibility (price limits, suspensions, T+1). Cached per
        # panel: the masks depend only on prices and Theta, never on the
        # prediction, so rebuilding them per candidate would be pure waste.
        mask = self._trade_mask(panel)

        # Put the prediction into expected-return units before the trading rule
        # compares it with a cost. Never the evaluation window -- that would leak
        # realised returns into the trading rule -- but WHICH earlier window
        # matters: see Portfolio.scale_split for why the fit split flatters it.
        model = getattr(theta.combiner, "model", "lightgbm")
        beta = 1.0
        if theta.portfolio.cost_aware_dropout or theta.portfolio.construction == "mean_variance":
            scale_window = theta.splits.window(
                getattr(theta.portfolio, "scale_split", None) or theta.combiner.fit_split
            )
            if model == "icir":
                # Grinold (1994) structural α = IC_c · σ_i · s_i: sign-guaranteed
                # and per-name vol-scaled, replacing the OLS β (prediction_scale)
                # that falls back to 1.0 when the in-sample slope is ≤0. IC_c is
                # the mean per-date cross-sectional corr on the scale window
                # (valid, not train). α is already in return units -> pred_scale=1.0;
                # its ~6e-4 magnitude matches empirical β·μ so λ=25 is preserved.
                prediction = grinold_alpha(prediction, y_tilde, sigma, scale_window)
                window_pred = prediction.loc[str(start) : str(end)]
                beta = 1.0
            else:
                beta = prediction_scale(prediction, y_tilde, scale_window)

        # The FULL close history, not the evaluation slice: the risk model
        # estimates from a trailing window that reaches back before the window
        # it prices, and truncating it would silently shorten that lookback.
        w, w_drift = build_book(
            window_pred, theta, y_tilde=y_tilde, universe=universe,
            mask=mask, sigma=sigma, pred_scale=beta, close=panel.close, adv=adv,
        )

        # Transfer coefficient (Clarke-de Silva-Thorley 2002): corr(α, w*) per
        # rebalance -- the fraction of the alpha the long-only + 3% cap + κ
        # constraints let through. IR = TC·IC·√N, so a low TC flags the position
        # cap as the binding constraint (a max_weight decision, not a factor
        # one). Gated to the icir path; the LightGBM path is byte-identical
        # without it.
        mean_tc = float("nan")
        if model == "icir":
            mean_tc, n_tc = _transfer_coefficient(window_pred, w)
            logger.info("transfer_coefficient mean=%.3f n=%d", mean_tc, n_tc)

        charges = pd.Series(
            [
                costs_mod.cost(
                    w.loc[date],
                    w_drift.loc[date],
                    sigma.loc[date] if date in sigma.index else pd.Series(dtype=float),
                    adv.loc[date] if date in adv.index else pd.Series(dtype=float),
                    theta,
                )
                for date in w.index
            ],
            index=w.index,
            name="cost",
        )

        bench = self._benchmark(str(start), str(end))
        r_net = costs_mod.net_return(w, y_tilde, bench, charges,
                                     delta=int(theta.execution.delta))
        metrics = strategy_metrics(w, w_drift, r_net, charges, theta)
        if model == "icir":
            metrics["transfer_coefficient"] = mean_tc
        # Carry the net return SERIES, not just its summary statistics. The
        # deflated Sharpe ratio needs the path -- it prices skew and kurtosis,
        # which no scalar in `metrics` preserves. Underscored and popped by the
        # caller so it never reaches the metric surfaces.
        metrics["_net_return_series"] = r_net
        return metrics


__all__ = ["EvaluationOperator"]
