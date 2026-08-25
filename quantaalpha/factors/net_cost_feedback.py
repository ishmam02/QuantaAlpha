"""Net-cost summarizer: shows the generator net-of-cost, per-dimension feedback.

This is the highest-leverage change in the whole design. Under the frictionless
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
# What the model is TOLD, which must be what the factor is JUDGED on.
#
# These used to be book aggregates (net_ir, net_arr, mdd, cost_bps) plus a RAW
# RankIC. Factors are now admitted per-factor on their NEUTRALIZED significance,
# so the old surface described a different criterion from the one deciding their
# fate -- and its RankIC still carried the size exposure the gate had just
# stripped. On the Amihud-style factors that is a -0.0146 shown where the honest
# figure is -0.0015.
#
# Each row is (key, label, format spec). The spec is BRACED because `_fmt` calls
# `spec.format(value)`; a bare ".4f" would render as the literal ".4f".
_RAW_METRICS: tuple[tuple[str, str, str], ...] = (
    ("rank_ic", "Raw RankIC (before neutralization)", "{:+.4f}"),
    ("rank_ic_neutral", "Neutralized RankIC (size/industry/beta removed)", "{:+.4f}"),
    ("t_nw", "t-statistic (Newey-West, overlap-corrected)", "{:+.2f}"),
    ("rank_icir",
     "Rank IC information ratio (stability; industry weak 0.20, good 0.30-0.50)",
     "{:+.3f}"),
    ("best_horizon", "Horizon where the edge is strongest (days)", "{:.0f}"),
    ("ic_pos_frac", "Days the IC keeps its sign", "{:.1%}"),
    ("monotonicity", "Decile monotonicity (1 = clean gradient, 0 = tails only)", "{:+.2f}"),
    ("q_spread", "Top-decile minus bottom-decile return", "{:+.4f}"),
    ("ls_sharpe", "Long/short Sharpe (dollar-neutral, before cost)", "{:+.2f}"),
    ("ls_mdd", "Long/short max drawdown (dollar-neutral, before cost)", "{:+.2f}"),
    ("exposure_size", "Correlation with SIZE (log circulating market cap)", "{:+.3f}"),
    ("rho_max", "Max correlation with a factor already held", "{:.3f}"),
    ("marginal_er",
     "Independent directions this factor adds to the book (marginal effective rank)",
     "{:.2f}"),
    ("dsr", "[book] Deflated Sharpe (survives the size of the search)", "{:.3f}"),
    ("dsr_n_trials", "[book] Factors searched to find this book", "{:.0f}"),
    ("sign_predicted", "Direction your hypothesis committed to", "{}"),
    ("sign_realized", "Direction the measurement actually produced", "{}"),
    ("mechanism_validated", "Did the measurement confirm your stated mechanism", "{}"),
    ("closest_held", "Closest factor already in the repository", "{}"),
    ("closest_held_t", "That factor's |t| (what you must beat to replace it)", "{:.2f}"),
    ("fdr_t_required", "|t| required by multiple-testing control", "{:.2f}"),
    ("fdr_n_tests", "Factors tested so far this run", "{:.0f}"),
    # THE ECONOMIC BAR. Significance and profitability are different questions,
    # and only the first was ever reported. Measured 2026-08-24: the mined
    # factors clear |t| 4.77 against a required 2.18 -- succeeding at the bar
    # they were shown -- while the book they form loses 3.57pp/yr to a no-alpha
    # baseline. Stating the second number is not a hint about what to build; it
    # is the other half of the measurement the factor was judged against.
    ("ic_breakeven_solo",
     "|IC| at which this factor's own turnover pays for itself", "{:.4f}"),
    ("ic_breakeven_book",
     "|IC| at which a book of these turns net-profitable (oracle-measured)",
     "{:.4f}"),
    ("capacity_cny", "Capacity at 5% of median daily volume (CNY)", "{:.3e}"),
    ("turnover_solo", "Signal turnover (daily, one-way)", "{:.4f}"),
    ("cx", "Complexity (symbol count)", "{:.0f}"),
    # BREADTH as a distribution, not an extremum. rho_max names the closest
    # single neighbour; it cannot say that 37 factors carry 12 independent
    # bets (measured 2026-08-24) or that 15 carry 10.6. Those are the numbers
    # that describe how much the book actually spans.
    ("effective_rank",
     "[book] Independent bets the book carries (exp-entropy of the "
     "correlation spectrum)", "{:.1f}"),
    ("book_n_factors", "[book] Factors in the book", "{:.0f}"),
    # Book aggregates are kept, clearly marked as downstream of selection.
    ("net_ir", "[book] Net IR after cost", "{:+.4f}"),
    ("net_arr", "[book] Net annualized return", "{:+.2%}"),
)

_DIMENSION_HELP = {
    "effectiveness": "net risk-adjusted return of the book containing this factor",
    "arr": "net annualized return",
    "stability": "consistency of the rank correlation (RankICIR)",
    "turnover": "trading volume implied by the signal (lower is better)",
    "diversity": "how distinct this factor is from the ones already held",
    "decay": "whether the edge fades across the evaluation window",
    "overfit": "how much of the in-sample edge survived out of sample",
}

# What each research metric MEANS, and which direction is good. Stated so the
# model can reason from the number instead of guessing a sign convention. These
# describe the measurement only -- never what to do about it.
_METRIC_HELP: dict[str, str] = {
    "rank_ic": "rank correlation with the forward return BEFORE removing risk "
               "exposures. Compare with the neutralized figure: a large gap means "
               "the raw edge was mostly risk exposure, not alpha.",
    "rank_ic_neutral": "rank correlation with the forward return AFTER removing "
                       "size, industry and beta. Higher magnitude is stronger. "
                       "This is what admission scores; the raw RankIC is not.",
    "t_nw": "how many standard errors that correlation sits from zero, with "
            "autocorrelation priced in. |t| >= 3 is the bar for a newly claimed "
            "factor. Sign follows the correlation.",
    "rank_icir": "the information ratio of the rank correlation -- mean RankIC "
                 "divided by its own standard deviation across days. The stability "
                 "metric: |t| can be high off a few big days while ICIR stays low. "
                 "Industry bars are roughly 0.20 weak and 0.30-0.50 good; below the "
                 "weak bar means the edge is not consistent. Measured on the "
                 "neutralized signal, so it is the stability of the alpha, not of a "
                 "repackaged risk exposure. What to do about a low number is yours "
                 "to determine.",
    "best_horizon": "the forecast horizon at which the edge was strongest, "
                    "chosen by measurement across 1/5/20 days rather than assumed.",
    "ic_pos_frac": "share of days the correlation kept its sign. Near 50% means "
                   "the edge came from a few days rather than persistently.",
    "monotonicity": "does return rise steadily from the bottom decile to the top, "
                    "or only at the extremes. A tails-only signal cannot be held "
                    "by a book that owns a few dozen of a few hundred names.",
    "q_spread": "average return of the top decile minus the bottom decile.",
    "ls_sharpe": "risk-adjusted return of a dollar-neutral top-minus-bottom book, "
                 "before costs.",
    "ls_mdd": "largest peak-to-trough drop of that same dollar-neutral top-minus-"
              "bottom book, before costs. A deep drawdown alongside a thin Sharpe "
              "is the same instability ic_pos_frac and rank_icir describe, seen in "
              "the return path instead of the daily correlation.",
    "exposure_size": "correlation between the factor and company size. A large "
                     "magnitude means most of the raw signal WAS size, and the "
                     "neutralized RankIC above is what remains after removing it.",
    "rho_max": "highest correlation with any factor already in the repository.",
    "marginal_er": "how many independent directions the book gains by adding this "
                   "factor (the change in the book's effective rank). A value below "
                   "the admission bar means the factor is redundant at the margin: "
                   "no single held factor duplicates it, so rho_max passes, but it is "
                   "correlated with the collective book and widens it by less than one "
                   "independent bet. What to do about a low number is yours to "
                   "determine.",
    "dsr": "the probability the book's Sharpe is real once it is discounted "
           "for how many factors were searched to find it, and for the skew "
           "and fat tails of its own return path. 0.95 is the conventional "
           "bar. A Sharpe that looks good and a DSR that does not means the "
           "search, not the strategy, produced the number.",
    "dsr_n_trials": "how many factors the deflation was charged for. It grows "
                    "through the run, so the same Sharpe is worth less later.",
    "sign_predicted": "the IC sign your hypothesis committed to BEFORE the "
                      "factor was measured. This is a pre-registered "
                      "prediction, not a description.",
    "sign_realized": "the IC sign the measurement actually produced.",
    "mechanism_validated": "whether those two agree. When they disagree the "
                           "factor may still carry signal, but the mechanism "
                           "you stated does not explain it -- what is left is "
                           "an unexplained fit, which is what a false discovery "
                           "looks like. A mechanism that predicts no direction "
                           "cannot be checked at all.",
    "closest_held": "the factor already in the repository that this one most "
                    "resembles. A near-duplicate is not automatically rejected: "
                    "if it is STRONGER than the factor it duplicates it replaces "
                    "it, so the way past a redundancy rejection is either a "
                    "genuinely different signal or a better version of this one.",
    "closest_held_t": "that incumbent's |t|. Beat it and you take its place.",
    "fdr_t_required": "the |t| this factor had to clear given how many factors "
                      "have been tested THIS RUN. A fixed bar is not enough: "
                      "testing many ideas gives noise many chances to clear any "
                      "threshold, so the requirement rises as the search proceeds.",
    "fdr_n_tests": "how many factors have been scored so far. The bar above is "
                   "derived from this count.",
    "capacity_cny": "the NAV this factor could carry before its own trading "
                    "moved the market, at 5% of each name's daily volume. "
                    "Higher is better; it falls as signal turnover rises.",
    "turnover_solo": "how much of the signal's ranking changes per day; higher "
                     "means more trading and more cost to capture the same edge.",
}

def economic_gap(sheet: dict) -> list[str]:
    """State the distance between the achieved |IC| and the bar that pays.

    Clearing |t| answers "is this signal REAL?". It does not answer "is it
    BIG ENOUGH?", and the two bars are far apart: measured on this protocol,
    |t|=3 needs |IC| 0.0173 over the 3-year valid window, while the book
    does not turn net-profitable until |IC| is an order of magnitude higher.
    The search has been clearing the first bar comfortably (median |t| 4.77
    against a required 2.18) and losing money at the second.

    This states the two numbers and the ratio between them. It does not say
    what to change -- widening a window, smoothing, or any other edit named
    here would be a construction the search was handed rather than one it
    reasoned to, and the named remedy then gets selected for that reason
    alone. What to do about the gap is the model's call.
    """
    ic = sheet.get("rank_ic_neutral")
    try:
        ic = abs(float(ic))
    except (TypeError, ValueError):
        return []
    if ic != ic or ic <= 0:
        return []

    lines: list[str] = []
    solo = sheet.get("ic_breakeven_solo")
    book = sheet.get("ic_breakeven_book")
    try:
        solo = float(solo) if solo is not None else None
    except (TypeError, ValueError):
        solo = None
    try:
        book = float(book) if book is not None else None
    except (TypeError, ValueError):
        book = None

    if solo is not None and solo == solo and solo > 0:
        lines.append(
            f"      Economic bar (own turnover): |IC| {ic:.4f} measured vs "
            f"{solo:.4f} needed to pay for its own trading "
            f"-- {ic / solo:.1f}x the bar" if ic >= solo else
            f"      Economic bar (own turnover): |IC| {ic:.4f} measured vs "
            f"{solo:.4f} needed to pay for its own trading "
            f"-- {solo / ic:.1f}x SHORT")
    # THE BOOK BAR IS A BOOK-LEVEL NUMBER. Comparing a SINGLE factor to it
    # asks "could this factor, alone, be the entire book?" -- which it never
    # had to be, and which reported every factor as "3.9x SHORT" when the book
    # they form is 1.7x short. Under Grinold's law the composite is
    # IC * sqrt(independent bets), so a 0.02 factor in a book of ten
    # independent bets is doing its job; the shortfall belongs to the book.
    #
    # So this reports the BOOK against the bar, and the factor's place in it.
    if book is not None and book == book and book > 0:
        comp = None
        for k in ("book_composite_ic", "rank_ic"):
            v = sheet.get(k)
            try:
                v = abs(float(v))
            except (TypeError, ValueError):
                v = None
            if v is not None and v == v and v > 0:
                comp = v
                break
        if comp is not None:
            gap = (f"{book / comp:.1f}x SHORT" if comp < book else "clears it")
            lines.append(
                f"      Economic bar (BOOK, not this factor): the combined book "
                f"scores |IC| {comp:.4f} against {book:.4f} needed for it to be "
                f"net-profitable -- {gap}. This factor contributes "
                f"|IC| {ic:.4f} to that composite.")
            if comp < book:
                lines.append(
                    "      A book's IC is its factors' IC multiplied by the "
                    "square root of the number of INDEPENDENT bets they carry, "
                    "so the shortfall is a property of the book, not a target "
                    "for any single factor. How to address that is yours to "
                    "determine.")
        else:
            # No composite measured this batch (the book was not priced). State
            # the bar without implying the factor must clear it alone.
            lines.append(
                f"      Book break-even: a book of these factors turns "
                f"net-profitable at |IC| {book:.4f}; this factor contributes "
                f"|IC| {ic:.4f}. The composite was not priced this batch.")
    return lines


def breadth_note(raw: dict) -> list[str]:
    """State how many INDEPENDENT bets the book carries.

    ``rho_max`` names the single closest neighbour, which is an extremum: a
    37-factor book whose worst pair correlated 0.997 still carried 12.0
    independent directions, and a 15-factor book carried 10.6 (measured
    2026-08-24). Neither number is visible from a max, so the feedback
    could not distinguish "many factors, few bets" from "many factors, many
    bets" -- and the search was rewarded for factor COUNT either way.

    Measurement only: the count and the ratio. Whether fewer-but-broader or
    more-but-narrower is the right response is the model's call, and naming
    a remedy here would hand it a construction it did not reason to.
    """
    m = raw.get("metrics") if isinstance(raw, dict) else None
    m = m if isinstance(m, dict) else raw
    if not isinstance(m, dict):
        return []
    er, n = m.get("effective_rank"), m.get("book_n_factors")
    try:
        er, n = float(er), int(n)
    except (TypeError, ValueError):
        return []
    if er != er or n < 2:
        return []
    return ["",
            f"Book breadth: {n} factors carrying {er:.1f} independent bets "
            f"({er / n:.0%} of the slots add a distinct direction). "
            "Correlated factors occupy a slot without widening the book."]


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
            # An infinite multiple-testing bar is a real state, not an error:
            # it means no factor scored so far is strong enough to be a
            # discovery at this trial count. Spell it out -- "inf" in a column
            # of decimals reads as a bug rather than as a verdict.
            if isinstance(value, float) and value in (float("inf"), float("-inf")):
                return "not reachable yet (nothing scored so far qualifies)"
            # Not every measurement is a number. The mechanism verdict is a
            # word, and forcing it through a float spec would drop it.
            if isinstance(value, bool):
                return "yes" if value else "NO"
            if isinstance(value, str):
                return value or "n/a"
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
        lines.append("=== NET-OF-COST EVALUATION (E_theta) ===")
        lines.append(
            f"Utility U = {u}   (0 = worst in the repository so far, 1 = best)"
        )
        n = m.get("n_factors")
        if n:
            lines.append(f"Factors evaluated together this round: {n}")
        lines.append(
            "There are no pass/fail thresholds. Every factor is kept; what varies "
            "is how this round SCORES against the factors already in the "
            "repository, and that bar rises as the repository improves."
        )

        d_ir, d_arr = m.get("delta_net_ir"), m.get("delta_net_arr")
        if d_ir is not None and d_ir == d_ir:
            verdict = "IMPROVED" if d_ir > 0 else "WEAKENED"
            lines.append("")
            lines.append(
                f"This round {verdict} the book: net IR {self._fmt(d_ir, '{:+.4f}')} "
                f"({self._fmt(d_arr, '{:+.2%}')} annualised) versus the same book "
                f"without these factors."
            )

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
                help_text = _METRIC_HELP.get(key, "")
                lines.append(f"  {label}: {self._fmt(m.get(key), spec)}")
                if help_text:
                    lines.append(f"      ({help_text})")

        lines.extend(self._per_fold_lines(m))
        lines.extend(self._per_factor_lines(raw))
        # Breadth is a property of the BOOK, not of any one factor, so it is
        # emitted here rather than inside the per-factor block. That block
        # returns early when `factor_tearsheets` is absent -- which is every
        # batch whose factors were rejected before the tear sheet was built --
        # and breadth was silently dropped on exactly those batches. Measured on
        # gate2: effective_rank was in the ledger on every batch and reached the
        # model on none.
        lines.extend(breadth_note(m))

        if m.get("theta_hash"):
            lines.append("")
            lines.append(
                f"[protocol {m['theta_hash']} | repository size {m.get('zoo_size', 'n/a')}]"
            )

        # Operator-coverage measurement over the population mined so far. Set on
        # this summarizer instance by ``LoopBase.feedback`` (library-so-far plus
        # the current batch). Stated only -- the block names which operators the
        # population has exhausted and which it has never touched, then closes
        # "yours to determine". The construct LLM reads this channel in the same
        # user prompt that lists every declared operator, so this puts the
        # coverage gap right beside the operator menu.
        cov = getattr(self, "operator_coverage_block", "")
        if cov:
            lines.append("")
            lines.append(cov)

        lines.append("")
        lines.append(
            # States HOW the number was produced, and stops. The closing
            # sentence used to be "Prefer signals that are cheap to trade and
            # orthogonal to existing factors" -- a remedy, and the single most
            # widely read line in the system, since the summarizer runs every
            # round. It also pre-empts the search: whether cheap-to-trade or
            # orthogonal signals score better here is what the objective is for
            # measuring, not something to assert up front.
            "NOTE: these figures are NET of transaction cost (commission, "
            "volatility-scaled slippage, and super-linear market impact) and are "
            "measured on the book built from the COMBINED model prediction over "
            "all factors together -- not on any factor in isolation. Turnover "
            "and correlation with the existing repository both enter these "
            "numbers; what follows from that is yours to determine."
        )
        return "\n".join(lines)

    def _per_fold_lines(self, m: dict) -> list[str]:
        """Name WHICH regime failed, not just the average across regimes.

        The per-fold vector is computed and gated on, then averaged away before
        anything reaches the model. Folds of [+0.5,+0.5,+0.5] and [+1.5,0,0] have
        the same mean and are completely different propositions -- one is an edge
        that holds across regimes, the other worked in a single window. Without
        the vector the model cannot tell those apart, and every regime lesson in
        the run is invisible to it.
        """
        series = m.get("fold_net_ir")
        windows = m.get("fold_windows") or []
        if not isinstance(series, (list, tuple)) or len(series) < 2:
            return []
        out = ["", "Per-fold result (each fold is a different market period):"]
        for i, v in enumerate(series):
            label = windows[i] if i < len(windows) else f"fold {i + 1}"
            out.append(f"  {label}: net IR {self._fmt(v, '{:+.4f}')}")
        pos = sum(1 for v in series if isinstance(v, (int, float)) and v > 0)
        out.append(f"  positive in {pos} of {len(series)} periods.")
        if pos and pos < len(series):
            out.append("  The sign is not stable across periods; the average above "
                       "hides that.")
        return out

    def _per_factor_lines(self, raw: dict) -> list[str]:
        """Each factor's OWN measurements, admitted or rejected.

        Rejections carry the lesson. A batch verdict says the round failed; a
        per-factor sheet says WHICH factor failed and on which measurement, and
        those are different fixes -- no predictive content, size in disguise,
        wrong horizon, tails-only, duplicate, or untradeable all arrive here as
        distinguishable numbers rather than as one scalar.
        """
        sheets = raw.get("factor_tearsheets") if isinstance(raw, dict) else None
        if not isinstance(sheets, dict) or not sheets:
            return []
        admitted = set(raw.get("admitted_exprs") or [])

        # Glossary ONCE, before the sheets. The meanings belong next to the
        # numbers, but repeating them for every factor would bury the numbers
        # under boilerplate -- so they are stated once and the sheets stay dense.
        present: list[str] = []
        for key, label, _spec in _RAW_METRICS:
            if key in _METRIC_HELP and any(
                    isinstance(sh, dict) and key in sh for sh in sheets.values()):
                present.append(f"  {label}: {_METRIC_HELP[key]}")
        out: list[str] = []
        if present:
            out += ["", "What each measurement means:"] + present
        out += ["", "Per-factor measurements (this is what admission judged):"]
        for expr, sheet in list(sheets.items())[:12]:
            if not isinstance(sheet, dict):
                continue
            mark = "KEPT" if expr in admitted else "not kept"
            short = expr if len(expr) <= 70 else expr[:67] + "..."
            out.append(f"  [{mark}] {short}")
            for key, label, spec in _RAW_METRICS:
                if key in sheet:
                    out.append(f"      {label}: {self._fmt(sheet.get(key), spec)}")
            out += economic_gap(sheet)
        return out

    # Aliases: the helpers are module-level pure functions (a test Stub that
    # copies selected methods off this class cannot see instance helpers), but
    # reaching them through the class stays supported.
    _economic_gap = staticmethod(economic_gap)
    _breadth_note = staticmethod(breadth_note)

    def _gate_lines(self, m: dict, names: list[str]) -> list[str]:
        """"value vs threshold" for each failed gate."""
        if self.theta is None:
            return names
        from quantaalpha.eval.gates import describe_failures

        metrics = {k: v for k, v in m.items()}
        described = describe_failures(metrics, self.theta, names)
        return described or names


__all__ = ["NetCostFactorFeedback"]
