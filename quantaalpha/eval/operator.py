"""``E_Θ`` — the evaluation operator (Eq. 13).

The entry point the rest of the system talks to. A pure function of
``(f, zoo, Θ)``: identical inputs give byte-identical output, which is
Property 2 and is what the ledger's reproducibility claim rests on.

A whole factor SET is evaluated at once, exactly as the control arm's ``qrun``
evaluates a round's factors together. Pipeline, in order:

1. hash the repository state (``zoo_hash``);
2. refit the combiner on ``zoo ∪ candidates``, map the composite prediction
   through ``g``, price Eq. 7, and measure the resulting book;
3. IC statistics **of that combined prediction** -- the same quantity Arm A's
   SigAnaRecord reports, which is what makes the arms comparable;
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

import numpy as np
import pandas as pd

from quantaalpha.eval import combiner as combiner_mod
from quantaalpha.eval import costs as costs_mod
from quantaalpha.eval.data import PanelBundle, align_signal, load_benchmark, load_panel
from quantaalpha.eval.execution import fill_prices, prediction_scale, realized_return
from quantaalpha.eval.metrics import (
    cx,
    prediction_metrics,
    rho_max,
    strategy_metrics,
)
from quantaalpha.eval.portfolio import topk_dropout
from quantaalpha.eval.protocol import Protocol
from quantaalpha.eval.tradability import trade_mask
from quantaalpha.eval.scoring import dimension_scores, utility

logger = logging.getLogger(__name__)


class EvaluationOperator:
    """``E_Θ``. Construct once per run; Θ is frozen for its lifetime."""

    def __init__(self, theta: Protocol) -> None:
        self.theta = theta
        self._panels: dict[tuple[str, str], PanelBundle] = {}
        self._benchmarks: dict[tuple[str, str], pd.Series] = {}
        # Baseline books keyed by (zoo_hash, window) -- one fit per repository
        # state, not per candidate.
        self._baselines: dict[tuple, dict] = {}
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
    ) -> dict:
        """Score a factor SET as one book, the way the baseline's qrun does.

        ``candidates`` is ``{expression: signal}`` -- every factor the experiment
        produced, evaluated **together**, not one at a time. This mirrors the
        control arm: Qlib fits the combined model on all of the round's factors
        and reports one set of metrics for the resulting book. The only
        differences here are the cost model (kappa0+kappa1+kappa2 instead of a
        flat fee) and the repository-relative scoring layered on top.

        Evaluating factors one at a time, as this used to, made the two arms
        structurally incomparable and forced a per-factor admissibility verdict
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

        # --- the book: zoo + ALL new factors, one combiner refit ---
        book = self._strategy_batch(aligned_cands, aligned_zoo, panel, eval_window)
        metrics = dict(book["metrics"])

        # --- predictive quality OF THE COMBINED PREDICTION ---
        # Arm A's SigAnaRecord reports IC of the combined model's output, not of
        # any single factor, so this is the directly comparable quantity.
        metrics.update(prediction_metrics(book["prediction"], panel, self.theta, eval_window))

        # --- properties of the new batch itself ---
        metrics["rho_max"] = max(
            (rho_max(sig, aligned_zoo) for sig in aligned_cands.values()), default=0.0
        )
        metrics["cx"] = max((cx(e) for e in aligned_cands), default=0)
        metrics["n_factors"] = len(aligned_cands)

        # --- dynamic (repository-relative) scoring only; no absolute floors ---
        scores = dimension_scores(metrics, zoo_metrics, self.theta)
        u = utility(metrics, zoo_metrics, self.theta)

        result: dict = {f"m_{k}": v for k, v in metrics.items()}
        result.update({f"e_{k}": v for k, v in scores.items()})
        result.update(
            U=float(u),
            theta_hash=self.theta.hash,
            zoo_hash=zoo_hash,
            zoo_size=len(zoo_signals),
            n_factors=len(aligned_cands),
            eval_window=f"{eval_window[0]}..{eval_window[1]}",
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
    def _strategy_batch(
        self,
        aligned_cands: dict[str, pd.DataFrame],
        aligned_zoo: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
    ) -> dict:
        """One combiner refit on ``zoo ∪ candidates`` → one priced book."""
        combined = {**aligned_zoo, **aligned_cands}
        prediction = combiner_mod.fit_predict(combined, None, panel, self.theta)
        metrics = self._book(prediction, panel, eval_window)

        # Marginal contribution of the whole batch over the repository alone.
        baseline = self._baseline(aligned_zoo, panel, eval_window)
        for key in ("net_ir", "net_arr"):
            base_v, full_v = baseline.get(key), metrics.get(key)
            metrics[f"base_{key}"] = base_v
            metrics[f"delta_{key}"] = (
                full_v - base_v
                if base_v is not None and full_v is not None
                and base_v == base_v and full_v == full_v
                else np.nan
            )
        return {"metrics": metrics, "prediction": prediction}

    # ------------------------------------------------------------------
    def _baseline(
        self,
        aligned_zoo: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
    ) -> dict:
        """The book built from ``zoo`` alone -- what f's contribution is measured against.

        Depends on ``(zoo, Θ, window)`` but **not** on the candidate, and the
        runner holds the zoo fixed across every candidate in an experiment, so
        this costs one extra combiner fit per experiment rather than per factor.
        With an empty zoo it is the null model: base features only.
        """
        key = (combiner_mod.zoo_hash(aligned_zoo), eval_window)
        if key not in self._baselines:
            pred = combiner_mod.fit_predict(aligned_zoo, None, panel, self.theta)
            self._baselines[key] = self._book(pred, panel, eval_window)
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
    ) -> dict:
        """Price one composite prediction through g and the cost model."""
        start, end = eval_window
        window_pred = prediction.loc[str(start) : str(end)]
        if window_pred.empty:
            return {"net_ir": np.nan, "net_arr": np.nan, "mdd": np.nan,
                    "turnover_book": np.nan, "cost_bps": np.nan}

        y_tilde = realized_return(fill_prices(panel, self.theta))
        universe = panel.universe.loc[window_pred.index]

        sigma = costs_mod.trailing_vol(panel.close, self.theta.costs.vol_window)
        adv = costs_mod.trailing_adv(panel, self.theta)

        # Hard A-share feasibility (price limits, suspensions, T+1). Cached per
        # panel: the masks depend only on prices and Theta, never on the
        # prediction, so rebuilding them per candidate would be pure waste.
        mask = self._trade_mask(panel)

        # Put the prediction into expected-return units before the trading rule
        # compares it with a cost. Fitted on the TRAINING split only, so the
        # rule never sees realised returns from the window it is evaluated on.
        beta = 1.0
        if self.theta.portfolio.cost_aware_dropout:
            beta = prediction_scale(
                prediction, y_tilde, self.theta.splits.window(self.theta.combiner.fit_split)
            )

        w, w_drift = topk_dropout(
            window_pred, self.theta, y_tilde=y_tilde, universe=universe,
            mask=mask, sigma=sigma, pred_scale=beta,
        )

        charges = pd.Series(
            [
                costs_mod.cost(
                    w.loc[date],
                    w_drift.loc[date],
                    sigma.loc[date] if date in sigma.index else pd.Series(dtype=float),
                    adv.loc[date] if date in adv.index else pd.Series(dtype=float),
                    self.theta,
                )
                for date in w.index
            ],
            index=w.index,
            name="cost",
        )

        bench = self._benchmark(str(start), str(end))
        r_net = costs_mod.net_return(w, y_tilde, bench, charges)
        return strategy_metrics(w, w_drift, r_net, charges, self.theta)


__all__ = ["EvaluationOperator"]
