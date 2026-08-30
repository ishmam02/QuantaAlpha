"""LLM-authored diagnosis — the AlphaEvolve move.

``diagnosis.diagnose`` maps a parent to one of **ten hardcoded strings** keyed by
``(verdict, dimension_category)``. That table is the reason the search does not
compound. Three measured problems with it:

1. **It covers 2 of 10 verdicts.** Only ``NET_HARMFUL`` and ``MARGINAL`` have
   rows; the other eight fall through to a generic fallback.
2. **It collapses the diagnosis.** ``_DIMENSION_CATEGORY`` maps ``effectiveness``,
   ``arr`` and ``stability`` all to ``"signal"``, so "the IC is unstable" and
   "the book does not make money" receive the identical instruction.
3. **It invents a weakness.** ``weakest_dimensions`` falls back to "always at
   least the weakest", and every ``e_j`` is a repository-relative percentile, so
   about half of every batch is below median *by construction*. A factor with
   nothing wrong with it still gets told what is wrong with it.

Here the model is handed the parent, everything measured about it, and what the
population has already tried -- and writes the diagnosis itself.

**No priors are injected.** The prompt states measurements and nothing else: no
index name, no expected IC, no continuation/reversal claim, no remedy. The
system describes; the model reasons. That keeps the hard constraint (the system
must run on any market) while still letting the diagnosis be about the market,
because every number it reasons from is measured on the market it is running on.

Falls back to the deterministic table on any failure, so a diagnosis never
blocks a round.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from quantaalpha.core.verdict import Verdict
from quantaalpha.pipeline.evolution.diagnosis import (
    RefinementDirective,
    RefineTarget,
    classify_verdict,
    diagnose as _table_diagnose,
)

logger = logging.getLogger(__name__)

# On by default. Set QA_LLM_DIAGNOSIS=0 to fall back to the lookup table.
def _enabled() -> bool:
    return os.environ.get("QA_LLM_DIAGNOSIS", "1").strip().lower() not in ("0", "false", "no")


# How many prior attempts to show. Enough for the model to see a pattern,
# few enough to leave room for the parent's own detail.
_POPULATION_K = int(os.environ.get("QA_DIAGNOSIS_POPULATION_K", "8"))


# Measurement -> (label, direction, what it means). Direction is stated so the
# model does not have to guess a sign; the VALUE is never editorialised.
_METRIC_GLOSS: tuple[tuple[str, str, str, str], ...] = (
    # --- the gate's decision (frames the whole diagnosis) -------------------
    ("admitted", "Admitted to the repository", "passed the gate or not",
     "whether this factor cleared the significance + multiple-testing + redundancy gate. An ADMITTED factor is not broken -- it is a winner being pushed further; a rejected factor has a measured shortfall to locate"),
    ("verdict", "Gate verdict", "the structured decision string",
     "the admission verdict (ADMITTED / REJECTED / MARGINAL / ...); pairs with 'admitted' above to frame the diagnosis"),
    # --- what admission actually judges -------------------------------------
    ("rank_ic", "Raw RankIC (before neutralization)", "context only, not the bar",
     "rank correlation with the forward return BEFORE removing risk exposures. Compare it with the neutralized figure below: a large gap means the raw edge was mostly risk exposure, not alpha"),
    ("rank_ic_neutral", "Neutralized RankIC", "higher magnitude is stronger",
     "rank correlation with the forward return AFTER removing size, industry and beta. This is what admission scores"),
    ("t_nw", "t-statistic (Newey-West)", "|t| >= 3 clears the bar",
     "how many standard errors the correlation sits from zero, with autocorrelation priced in"),
    ("best_horizon", "Strongest horizon (days)", "measured, not assumed",
     "the forecast horizon at which the edge was largest, chosen across 1/5/20 days"),
    ("ic_pos_frac", "Days the IC keeps its sign", "higher is better",
     "near 50% means the edge came from a few days rather than persistently"),
    ("ic_crash", "IC on the worst 20% of days (by cross-sectional return)", "compare with ic_rally and the overall rank_ic",
     "this factor's rank IC averaged over the 20% of days with the worst cross-sectional return (the equal-weight mean of the forward return across the universe). An overall IC that turns negative on these days is an edge that did not hold when the cross-section fell; a gap between this and ic_rally means the edge is regime-concentrated"),
    ("ic_rally", "IC on the best 20% of days (by cross-sectional return)", "compare with ic_crash and the overall rank_ic",
     "this factor's rank IC averaged over the 20% of days with the best cross-sectional return. Read alongside ic_crash: an edge present only on these days is regime-conditional"),
    ("monotonicity", "Decile monotonicity", "1 = clean gradient, 0 = tails only",
     "whether return rises steadily from bottom decile to top; a tails-only signal cannot be held by a book owning a few dozen of a few hundred names"),
    ("q_spread", "Top minus bottom decile return", "higher is better",
     "average return of the top decile minus the bottom decile"),
    ("ls_sharpe", "Long/short Sharpe", "higher is better",
     "risk-adjusted return of a dollar-neutral top-minus-bottom book, before costs"),
    ("exposure_size", "Correlation with SIZE", "closer to zero is cleaner",
     "a large magnitude means most of the RAW signal was company size; the neutralized RankIC above is what survived removing it"),
    ("rho_max", "Max correlation vs library", "lower is better",
     "highest correlation with any factor already held"),
    ("dsr", "[book] Deflated Sharpe", "higher is better, 0.95 is the bar",
     "the probability the book's Sharpe survives being discounted for the size of the search that found it"),
    ("sign_predicted", "Direction the hypothesis committed to", "pre-registered, not descriptive",
     "the IC sign claimed BEFORE measurement"),
    ("sign_realized", "Direction the measurement produced", "the outcome",
     "the IC sign actually observed"),
    ("mechanism_validated", "Did the measurement confirm the stated mechanism", "yes is required",
     "when it does not, the factor may still carry signal but the stated mechanism does not explain it -- an unexplained fit is what a false discovery looks like"),
    ("fdr_t_required", "|t| required after multiple-testing control", "the bar to beat",
     "testing many ideas gives noise many chances to clear a fixed threshold, so this requirement RISES as the run proceeds; a factor can clear |t|>=3 and still fail here"),
    ("fdr_n_tests", "Factors tested so far this run", "context for the bar above",
     "how many candidates have been scored; the requirement is derived from this count"),
    ("capacity_cny", "Capacity in CNY", "higher is better",
     "the NAV this factor could carry at 5% of each name's daily volume before its own trading moved the market; falls as turnover rises"),
    ("turnover_solo", "Signal turnover", "lower is better",
     "how much of the ranking changes per day; higher means more cost to capture the same edge"),
    ("cx", "Expression complexity", "lower is better", "symbol count of the factor expression"),
    # --- the objective vector + per-dimension scores (all present in
    #     backtest_metrics but never in the gloss, so the diagnosis never saw
    #     what selection actually ranks on or WHICH dimension is weak) --------
    ("U", "Repository-relative utility", "higher is better",
     "the composite objective percentile (Eq. 10); what selection actually ranks on. Absent => no objective vector, nothing to diagnose"),
    ("RankICIR", "Rank IC information ratio", "higher is better",
     "rank IC mean / std -- the stability dimension e_stability ranks on"),
    ("cost_bps", "Cost in bps/day", "lower is better",
     "roughly daily turnover times the cost model; the drag the net figures subtract"),
    ("delta_mean", "Marginal contribution to book net IR", "higher is better",
     "seed-averaged estimate of how much this factor added to the book versus an empty zoo; the fitness selection shrinks"),
    ("delta_se", "Std error of marginal contribution", "context for delta_mean",
     "the standard error across combiner seeds behind the verdict"),
    ("delta_t", "t-stat of marginal contribution", "|t| behind the verdict",
     "delta_mean / delta_se -- distinguishes a resolvably-negative contribution from an unresolved one"),
    ("e_effectiveness", "Score: effectiveness", "repository percentile, 1=best",
     "where this book net IR ranks vs the repository"),
    ("e_arr", "Score: net annualised return", "repository percentile, 1=best",
     "where this book net ARR ranks vs the repository"),
    ("e_stability", "Score: rank ICIR", "repository percentile, 1=best",
     "where this rank ICIR ranks vs the repository"),
    ("e_turnover", "Score: turnover", "repository percentile, 1=best",
     "where this book turnover ranks vs the repository (lower turnover scores higher)"),
    ("e_diversity", "Score: diversity", "repository percentile, 1=best",
     "where the max correlation vs the library ranks (less correlated scores higher)"),
    ("e_decay", "Score: IC decay", "repository percentile, 1=best",
     "where the IC decay slope ranks vs the repository"),
    ("e_overfit", "Score: overfit risk", "lower is better",
     "parsimony (complexity) plus the in-sample vs out-of-sample gap; NOT repository-ranked"),
    ("weakest_dimensions", "Weakest scored dimensions", "context",
     "the k=2 lowest e_j scores, named -- what the deterministic router would target"),
    ("factor_attribution", "Combiner credit per factor", "context",
     "per-factor: weight in the book, the combiner's rank_ic for it, and its turnover share"),
    # --- downstream of selection, kept for context ---------------------------
    ("net_ir", "[book] Net information ratio", "higher is better",
     "annualised risk-adjusted excess return of the BOOK after cost -- downstream of selection, not what admitted this factor"),
    ("net_arr", "[book] Net annualised return", "higher is better",
     "annualised excess return after cost"),
    ("turnover_book", "[book] Book turnover", "lower is better",
     "fraction of the book replaced per day, one way"),
)


def _fmt(v: Any) -> str:
    if v is None:
        return "not measured"
    if isinstance(v, float):
        if v != v:
            return "not measured"
        return f"{v:+.4f}"
    return str(v)


# Gloss keys are lowercase; backtest_metrics carries CamelCase for a few
# (notably RankIC). Resolve a gloss key to its value, trying the exact key then
# the known CamelCase alias, so _measurement_block / _population_block / lineage
# never silently drop a measurement over casing. (diagnosis.py already does
# this ad hoc at its rank_ic read sites; centralising it here.)
_GLOSS_ALIASES = {"rank_ic": "RankIC"}


def _get(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    alias = _GLOSS_ALIASES.get(key)
    if alias is not None and alias in metrics:
        return metrics[alias]
    return None


def _measurement_block(metrics: dict[str, Any]) -> str:
    """Every measurement, with its direction. No interpretation."""
    lines = []
    for key, label, direction, meaning in _METRIC_GLOSS:
        if not (key in metrics or _GLOSS_ALIASES.get(key) in metrics):
            continue
        val = _get(metrics, key)
        if isinstance(val, dict) and val:
            # factor_attribution: per-factor combiner credit, keyed by expr.
            rows = []
            for rec in val.values():
                if isinstance(rec, dict):
                    rows.append(f"weight {_fmt(rec.get('weight'))}, "
                                f"rank_ic {_fmt(rec.get('rank_ic'))}, "
                                f"turnover_share {_fmt(rec.get('turnover_share'))}")
            lines.append(f"  {label}: {'; '.join(rows) if rows else 'empty'}   "
                         f"[{direction}] -- {meaning}")
        else:
            lines.append(f"  {label}: {_fmt(val)}   [{direction}] -- {meaning}")
    return "\n".join(lines) if lines else "  (nothing measured)"


def _population_block(population: list | None) -> str:
    """What has already been tried, and how it scored.

    This is the channel the search has never had. Without it the model proposes
    as if every round were the first, which is measurably what happens today.
    """
    if not population:
        return "  (no prior attempts recorded)"
    rows = []
    for t in population[:_POPULATION_K]:
        m = dict(getattr(t, "backtest_metrics", None) or {})
        hyp = (getattr(t, "hypothesis", "") or "")[:150].replace("\n", " ")
        exprs = [f.get("expression", "") for f in (getattr(t, "factors", None) or [])][:2]
        rows.append(
            f"  - hypothesis: {hyp}\n"
            f"    expression: {'; '.join(e[:120] for e in exprs) if exprs else 'n/a'}\n"
            f"    RankIC {_fmt(_get(m, 'rank_ic'))} | t_nw {_fmt(_get(m, 't_nw'))} | "
            f"turnover {_fmt(m.get('turnover_book'))} | "
            f"contribution {_fmt(m.get('delta_mean'))} | admitted {m.get('admitted')}"
        )
    return "\n".join(rows)


_SYSTEM = (
    "You are diagnosing a quantitative equity alpha factor that has just been "
    "evaluated. You are given the factor, every measurement taken on it, and the "
    "attempts that came before it.\n\n"
    "Your job is to DIAGNOSE, not to treat. Say what is actually wrong with this "
    "factor and WHERE in it the measurements locate the problem, reasoning from the "
    "measurements given and the structure of the expression. If the measurements do "
    "not support any specific weakness, say so -- 'nothing is resolvably wrong, this "
    "is within noise' is a valid and useful diagnosis. Do not manufacture a weakness "
    "to have something to say.\n\n"
    "You state the measured problem, the part of the factor the measurements locate "
    "it in (which sub-expression, which scored dimension, which horizon), and the "
    "observed outcomes. You do NOT state a remedy. Never say what to change, how to "
    "fix it, or what the next factor should do -- how to address what you find is "
    "not your call. Reason from the numbers you are shown and from the structure of "
    "the expression. Do not assume facts about this market that is not in the data "
    "given to you -- you are not told which market this is, and any assumption about "
    "which effects should work here would be a guess.\n\n"
    "A factor reaches you for one of two reasons, and the gate's decision (in the "
    "measurements) tells you which. A REJECTED factor has a measured shortfall -- "
    "locate it: which sub-expression, which scored dimension, which horizon the "
    "measurements show is weak. An ADMITTED factor is not broken -- it cleared the "
    "gate; locate where it ranks WEAKEST relative to the repository (its lowest "
    "scored e_* dimensions, its shortest-lived horizon, its highest turnover) as "
    "the axes where a real edge still has room to be pushed further. In neither "
    "case do you state a remedy or what to change; locating the shortfall, or the "
    "room to push, is the diagnosis.\n\n"
    "Use the prior attempts. If several attempts already failed the same way, name "
    "the shared failure and note that the measurements do not support another "
    "variation of the same line. If an attempt succeeded, name what the measurements "
    "show distinguished it. Do not propose a replacement -- naming the pattern is the "
    "diagnosis; what to test instead is not your call.\n\n"
    "Return JSON with exactly these keys:\n"
    '  "diagnosis": what is wrong and how the measurements show it (2-4 sentences)\n'
    '  "weakness": one short label for the primary weakness, or "none"\n'
    '  "layer": "hypothesis" if the premise itself is wrong and should be replaced, '
    '"expression" if the premise is sound and only the construction needs to change\n'
    '  "directive": the measured problem stated for whoever writes the next factor '
    "(3-6 sentences). Name the problem, where in the factor the measurements locate "
    "it, and the observed outcomes. Do NOT state a remedy or what to change; close "
    "with 'how to address that is yours to determine.'\n"
    '  "exhausted": true if the prior attempts show this direction has been tried '
    "enough times without working that another variation is not supported by the "
    "measurements"
)


def _build_prompt(parent: Any, metrics: dict[str, Any], population: list | None,
                  ancestors: list | None, ablation_summary: str = "") -> str:
    hypothesis = getattr(parent, "hypothesis", "") or "(none recorded)"
    factors = getattr(parent, "factors", None) or []
    expr_lines = [
        f"  - {f.get('name', 'unnamed')}: {f.get('expression', '')}" for f in factors
    ] or ["  (no expressions recorded)"]

    lineage = ""
    if ancestors:
        steps = []
        for a in ancestors[-4:]:
            am = dict(getattr(a, "backtest_metrics", None) or {})
            steps.append(
                f"  - {getattr(getattr(a, 'phase', None), 'value', '?')}: "
                f"RankIC {_fmt(_get(am, 'rank_ic'))}, t_nw {_fmt(_get(am, 't_nw'))}, "
                f"contribution {_fmt(am.get('delta_mean'))}"
            )
        lineage = "\n\n## How this factor's own lineage has scored\n" + "\n".join(steps)

    reason = metrics.get("reason") or ""
    verdict_line = f"\n\n## The evaluator's summary\n  {reason}" if reason else ""

    return (
        "## The factor being diagnosed\n"
        f"Hypothesis: {hypothesis}\n"
        "Expressions:\n" + "\n".join(expr_lines) +
        "\n\n## Everything measured on it\n" + _measurement_block(metrics) +
        (f"\n\n## Per-part solo measurement\n{ablation_summary}"
         if ablation_summary else "") +
        verdict_line + lineage +
        "\n\n## Attempts that came before this one\n" + _population_block(population) +
        "\n\nDiagnose this factor: state the measured problem and where in it the "
        "measurements locate that problem. Do not state a remedy; how to address it "
        "is yours to determine."
    )


def _to_directive(payload: dict, parent: Any, metrics: dict[str, Any],
                  verdict: Verdict, ablation_summary: str = "") -> RefinementDirective:
    layer = str(payload.get("layer", "expression")).strip().lower()
    target = RefineTarget.HYPOTHESIS if layer == "hypothesis" else RefineTarget.EXPRESSION
    weakness = str(payload.get("weakness", "") or "").strip() or None
    if weakness and weakness.lower() in ("none", "n/a", "nothing"):
        weakness = None

    diagnosis = str(payload.get("diagnosis", "") or "").strip()
    directive = str(payload.get("directive", "") or "").strip()
    text = f"{diagnosis}\n\n{directive}".strip() if diagnosis else directive

    return RefinementDirective(
        verdict=verdict,
        weakness_dimension=weakness,
        refine_target=target,
        # An expression-refine keeps the premise; a hypothesis-refine replaces it.
        frozen_layers=["hypothesis"] if target is RefineTarget.EXPRESSION else [],
        mechanism_hint=diagnosis,
        directive_text=text,
        parent_hypothesis=getattr(parent, "hypothesis", "") or "",
        parent_expected_ic_sign=str(
            getattr(parent, "expected_ic_sign", "")
            or (getattr(parent, "backtest_metrics", None) or {}).get("sign_predicted", "")
            or "").strip().lower(),
        parent_factors=list(getattr(parent, "factors", None) or []),
        raw_metrics=dict(metrics),
        exhausted_lever=bool(payload.get("exhausted", False)),
        ablation_summary=ablation_summary,
    )


def llm_diagnose(parent: Any, ancestors: list | None = None,
                 population: list | None = None, *,
                 ablation: Any = None, ablation_summary: str = "") -> RefinementDirective | None:
    """Diagnose a parent with the LLM; fall back to the table on any failure.

    Returns ``None`` only when there is nothing to diagnose (no objective
    vector), matching ``diagnosis.diagnose``'s contract exactly so this is a
    drop-in at the ``refine.py`` seam.
    """
    metrics: dict[str, Any] = dict(getattr(parent, "backtest_metrics", None) or {})
    if "U" not in metrics:
        return None

    if not _enabled():
        return _table_diagnose(parent, ancestors, ablation=ablation,
                               ablation_summary=ablation_summary)

    verdict = classify_verdict(metrics)
    try:
        from quantaalpha.llm.client import APIBackend

        raw = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=_build_prompt(parent, metrics, population, ancestors,
                                      ablation_summary),
            system_prompt=_SYSTEM,
            json_mode=True,
        )
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(payload, dict) or not payload.get("directive"):
            raise ValueError(f"diagnosis returned no directive: {str(payload)[:200]}")
        directive = _to_directive(payload, parent, metrics, verdict, ablation_summary)
        logger.info(
            "llm_diagnosis: verdict=%s layer=%s weakness=%s exhausted=%s",
            verdict.value, directive.refine_target.value,
            directive.weakness_dimension, directive.exhausted_lever,
        )
        return directive
    except Exception as exc:
        logger.warning("llm_diagnosis failed (%s); falling back to the table", exc)
        return _table_diagnose(parent, ancestors, ablation=ablation,
                               ablation_summary=ablation_summary)


__all__ = ["llm_diagnose"]
