"""``E_Θ`` — the evaluation operator (Eq. 13).

The entry point the rest of the system talks to. A pure function of
``(f, zoo, Θ)``: identical inputs give byte-identical output, which is
Property 2 and is what the ledger's reproducibility claim rests on.

Pipeline, in order:

1. hash the repository state (``zoo_hash``);
2. per-factor metrics on the IS window and the OOS proxy, both from Θ;
3. ``ρ_max`` against every incumbent signal;
4. refit the combiner on ``zoo ∪ {f}``, map the composite prediction through
   ``g``, price Eq. 7, and measure the resulting book;
5. apply ``F_Θ``;
6. score each dimension against the repository and scalarize to ``U``.

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
from quantaalpha.eval.execution import fill_prices, realized_return
from quantaalpha.eval.gates import feasible
from quantaalpha.eval.metrics import (
    per_factor_metrics,
    rho_max,
    solo_turnover,
    strategy_metrics,
)
from quantaalpha.eval.portfolio import topk_dropout
from quantaalpha.eval.protocol import Protocol
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

    # ------------------------------------------------------------------
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
        candidate_signal,
        candidate_expr: str,
        zoo_signals: dict[str, object] | None = None,
        zoo_metrics: list[dict] | None = None,
        *,
        report: bool = False,
    ) -> dict:
        """Score one candidate against the current repository."""
        zoo_signals = dict(zoo_signals or {})
        zoo_metrics = list(zoo_metrics or [])

        panel_start, panel_end, eval_window = self._windows(report)
        panel = self._panel(panel_start, panel_end)

        candidate = align_signal(candidate_signal, panel)
        aligned_zoo = {expr: align_signal(sig, panel) for expr, sig in zoo_signals.items()}
        zoo_hash = combiner_mod.zoo_hash(zoo_signals)

        # --- per-factor (independent of the cost model and of the zoo) ---
        metrics = per_factor_metrics(candidate, candidate_expr, panel, self.theta)
        metrics["turnover_solo"] = solo_turnover(candidate, panel, self.theta)
        metrics["rho_max"] = rho_max(candidate, aligned_zoo)

        # --- strategy level (one combiner refit) ---
        metrics.update(self._strategy(candidate, candidate_expr, aligned_zoo, panel, eval_window))

        # --- admissibility, then scoring ---
        is_feasible, failed_gates = feasible(metrics, self.theta)
        scores = dimension_scores(metrics, zoo_metrics, self.theta)
        u = utility(metrics, zoo_metrics, self.theta)

        result: dict = {f"m_{k}": v for k, v in metrics.items()}
        result.update({f"e_{k}": v for k, v in scores.items()})
        result.update(
            U=float(u),
            feasible=bool(is_feasible),
            failed_gates=failed_gates,
            theta_hash=self.theta.hash,
            zoo_hash=zoo_hash,
            zoo_size=len(zoo_signals),
            eval_window=f"{eval_window[0]}..{eval_window[1]}",
        )

        # The primary whole-system signal: if this line is absent from a run's
        # logs, the new engine is not being called at all.
        logger.info(
            "E_theta eval: factor=%s theta=%s zoo=%s U=%.4f feasible=%s",
            candidate_expr[:40], self.theta.hash, zoo_hash, float(u), is_feasible,
        )
        return result

    # ------------------------------------------------------------------
    def _strategy(
        self,
        candidate: pd.DataFrame,
        candidate_expr: str,
        aligned_zoo: dict[str, pd.DataFrame],
        panel: PanelBundle,
        eval_window: tuple[str, str],
    ) -> dict:
        """Combiner refit → g → Eq. 7 → Eq. 8 → book metrics, plus Δ vs baseline.

        The absolute figures describe the book *containing* f. What actually
        determines whether f is worth having is its **marginal contribution**:
        the same book built from ``zoo`` alone, subtracted off. Measured on the
        pilot's factors, stand-alone RankIC and marginal contribution came apart
        badly -- all 18 failed the RankIC gate, yet collectively they were worth
        +7pp of annual return over the base-features-only null model.
        """
        prediction = combiner_mod.fit_predict(
            aligned_zoo, candidate, panel, self.theta, candidate_expr=candidate_expr
        )
        metrics = self._book(prediction, panel, eval_window)

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
        return metrics

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

        w, w_drift = topk_dropout(
            window_pred, self.theta, y_tilde=y_tilde, universe=universe
        )

        sigma = costs_mod.trailing_vol(panel.close, self.theta.costs.vol_window)
        adv = costs_mod.trailing_adv(panel, self.theta)

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
