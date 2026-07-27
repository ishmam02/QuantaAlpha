"""Treatment-arm summarizer: shows the generator net-of-cost, per-dimension feedback.

This is the highest-leverage change in the whole design. Under the control
objective the generator is shown four ``excess_return_without_cost.*`` numbers
(``feedback.FRICTIONLESS_METRICS``) — so the feedback that shapes the next
hypothesis never mentions transaction cost at all. No amount of scoring
machinery downstream matters if the model proposing the factors cannot see
what it is being scored on.

What replaces it:

* ``U`` and every ``e_j`` **named and ordered worst-first**, so the model sees
  *what to improve* rather than a single opaque scalar;
* the raw ``m(f)`` for net IR, net ARR, MDD, turnover, cost in bps, ρ_max, cx;
* ``feasible``, and when false the **failed gates with the threshold missed** —
  "RankICIR 0.1100 < gamma_ir 0.2" is far more actionable than a score;
* ``theta_hash``, so the transcript records which protocol produced the
  feedback.

Wired in via ``QLIB_FACTOR_SUMMARIZER``; no prompt bodies are edited, which is
what keeps "the generation process is held fixed" true.
"""

from __future__ import annotations

import logging

import pandas as pd

from quantaalpha.eval.protocol import default_protocol_path, load_protocol
from quantaalpha.factors.feedback import AlphaAgentQlibFactorHypothesisExperiment2Feedback

logger = logging.getLogger(__name__)

# Presentation names and formatting for the raw metric vector.
_RAW_METRICS: tuple[tuple[str, str, str], ...] = (
    ("net_ir", "Net IR (after cost)", "{:+.4f}"),
    ("net_arr", "Net annualized return", "{:+.2%}"),
    ("mdd", "Max drawdown", "{:+.2%}"),
    ("turnover_book", "Book turnover (daily, one-way)", "{:.4f}"),
    ("cost_bps", "Transaction cost", "{:.2f} bps/day"),
    ("rho_max", "Max correlation vs repository", "{:.4f}"),
    ("cx", "Complexity (symbol length)", "{:.0f}"),
    ("rank_ic", "RankIC", "{:+.4f}"),
    ("rank_icir", "RankICIR", "{:+.4f}"),
)

_DIMENSION_HELP = {
    "effectiveness": "net risk-adjusted return of the book containing this factor",
    "arr": "net annualized return",
    "stability": "consistency of the rank correlation (RankICIR)",
    "turnover": "trading volume implied by the signal (lower is better)",
    "diversity": "orthogonality to factors already in the repository (lower correlation is better)",
    "overfit": "structural parsimony and how well the in-sample edge survived out of sample",
    "decay": "whether the edge is fading across the out-of-sample window",
}


class NetCostFactorFeedback(AlphaAgentQlibFactorHypothesisExperiment2Feedback):
    """Summarizer that surfaces the ``E_Θ`` metric vector to the generator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.theta = load_protocol(default_protocol_path())
        except Exception:
            logger.exception("could not load protocol for feedback; gate thresholds will be omitted")
            self.theta = None

    # ------------------------------------------------------------------
    def _build_combined_result(self, current_result, sota_result) -> str:
        """Render the net-of-cost block, falling back to the base behaviour."""
        payload = self._as_dict(current_result)
        if payload is None or "U" not in payload:
            # Not an E_theta result (e.g. a cached Qlib DataFrame) -- keep the
            # inherited rendering rather than emitting an empty block.
            return super()._build_combined_result(current_result, sota_result)

        block = self._format_metric_block(payload)
        sota = self._as_dict(sota_result)
        if sota and "U" in sota:
            block += f"\n\nPrevious best (SOTA) U: {self._fmt(sota.get('U'), '{:.4f}')}"
        return block

    # ------------------------------------------------------------------
    @staticmethod
    def _as_dict(result) -> dict | None:
        if result is None:
            return None
        if isinstance(result, pd.Series):
            return result.to_dict()
        if isinstance(result, pd.DataFrame):
            if result.empty or len(result.columns) == 0:
                return None
            return result.iloc[:, 0].to_dict()
        if isinstance(result, dict):
            return dict(result)
        return None

    @staticmethod
    def _fmt(value, spec: str) -> str:
        try:
            if value is None or (isinstance(value, float) and value != value):
                return "n/a"
            return spec.format(float(value))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _normalize(m: dict) -> dict:
        """Accept both ``net_ir`` and ``m_net_ir`` spellings.

        The operator returns ``m_``-prefixed keys; the runner's Series flattens
        them. Tolerating both keeps this renderable straight from either.
        """
        out = dict(m)
        for key, value in m.items():
            if key.startswith("m_") and key[2:] not in out:
                out[key[2:]] = value
        return out

    def _format_metric_block(self, raw: dict) -> str:
        """The text the model actually reads.

        Ordered so the weakest dimensions come first: the model should spend
        its attention on what to fix.
        """
        m = self._normalize(raw)
        lines: list[str] = []

        u = self._fmt(m.get("U"), "{:.4f}")
        feasible = bool(m.get("feasible", False))
        lines.append("=== NET-OF-COST EVALUATION (E_theta) ===")
        lines.append(f"Utility U = {u}   (0 = worst in repository, 1 = best)")
        lines.append(f"Admissible: {'YES' if feasible else 'NO'}")

        if not feasible:
            failed = m.get("failed_gates") or ""
            names = [f.strip() for f in (failed.split(",") if isinstance(failed, str) else failed) if f and str(f).strip()]
            if names:
                lines.append("")
                lines.append("REJECTED by hard feasibility gates -- fix these first:")
                for line in self._gate_lines(m, names):
                    lines.append(f"  - {line}")

        dims = sorted(
            ((k[2:], v) for k, v in m.items() if k.startswith("e_") and v is not None and v == v),
            key=lambda kv: kv[1],
        )
        if dims:
            lines.append("")
            lines.append("Per-dimension scores, WORST FIRST (each 0-1, ranked against the repository):")
            for name, score in dims:
                help_text = _DIMENSION_HELP.get(name, "")
                suffix = f" -- {help_text}" if help_text else ""
                lines.append(f"  {score:.3f}  {name}{suffix}")

        lines.append("")
        lines.append("Underlying measurements:")
        for key, label, spec in _RAW_METRICS:
            if key in m:
                lines.append(f"  {label}: {self._fmt(m.get(key), spec)}")

        if m.get("theta_hash"):
            lines.append("")
            lines.append(
                f"[protocol {m['theta_hash']} | repository size {m.get('zoo_size', 'n/a')}]"
            )

        lines.append("")
        lines.append(
            "NOTE: performance above is NET of transaction cost (commission, "
            "volatility-scaled slippage, and super-linear market impact) and is "
            "measured on the book built from the combined model prediction, not "
            "on this factor in isolation. A factor with a strong raw correlation "
            "but heavy turnover or high correlation to existing factors will "
            "score poorly here. Prefer signals that are cheap to trade and "
            "orthogonal to the repository."
        )
        return "\n".join(lines)

    def _gate_lines(self, m: dict, names: list[str]) -> list[str]:
        """"value vs threshold" for each failed gate."""
        if self.theta is None:
            return names
        from quantaalpha.eval.gates import describe_failures

        metrics = {k: v for k, v in m.items()}
        described = describe_failures(metrics, self.theta, names)
        return described or names


__all__ = ["NetCostFactorFeedback"]
