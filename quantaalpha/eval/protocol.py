"""The frozen evaluation protocol Θ (problem_formulation.tex, Eq. 13).

Θ pins every degree of freedom of the evaluation engine: the splits, the fill
rule and latency, the combiner's fitting procedure ``A`` (algorithm,
hyperparameters *and* random seed), the portfolio map ``g``, the cost
coefficients, the feasibility gates and the utility weights.

Two properties are mechanically enforced here:

* **Property 1 (Immutability)** — every dataclass is ``frozen=True``, so a
  mutation raises ``FrozenInstanceError``. ``Θ.hash`` is the SHA-256 of the
  canonical JSON of the whole structure, logged with every evaluation; a
  changed protocol therefore produces a visibly different hash rather than a
  silent behaviour change.
* **Property 2 (Determinism)** — ``combiner.seed`` is part of Θ, so the
  LightGBM refit that produces the composite prediction is reproducible.

This is a plain dataclass, deliberately **not** an ``ExtendedBaseSettings``:
Θ is loaded once from a file and must not be env-overridable, or Property 1
is unenforceable. ``QA_PROTOCOL`` selects *which file* to load; it never
overrides an individual field.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

# The YAML shipped alongside this module. Resolved against the package rather
# than the cwd, because the runner executes inside per-factor workspaces.
DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol_csi300.yaml")

DateRange = tuple[str, str]


def default_protocol_path() -> Path:
    """Path of the protocol to load, honouring ``QA_PROTOCOL``."""
    override = os.environ.get("QA_PROTOCOL")
    return Path(override) if override else DEFAULT_PROTOCOL_PATH


@dataclass(frozen=True)
class Splits:
    """Evaluation windows. The engine reads these; it never takes a window
    from a caller, so the test split cannot be reached by accident."""

    train: DateRange
    valid: DateRange
    final_test: DateRange
    # Optional extra windows (e.g. an in-loop test split). Unused by default:
    # splits now mirror configs/backtest.yaml so E_Theta trains on the same data
    # as the backtest that compares the arms.
    inloop_test: DateRange | None = None
    oos_proxy: DateRange | None = None
    search_is: str = "train"
    search_oos: str = "valid"

    def window(self, name: str) -> DateRange:
        """Resolve a split name (including the ``search_*`` aliases)."""
        if name in ("search_is", "search_oos"):
            name = getattr(self, name)
        try:
            value = getattr(self, name)
        except AttributeError as exc:  # pragma: no cover - guarded at load
            raise KeyError(f"unknown split {name!r}") from exc
        if not isinstance(value, tuple):
            raise KeyError(f"{name!r} is an alias, not a window")
        return value

    @property
    def is_window(self) -> DateRange:
        return self.window("search_is")

    @property
    def oos_window(self) -> DateRange:
        return self.window("search_oos")

    def walk_forward_folds(self, n: int) -> list[tuple[DateRange, DateRange]]:
        """``n`` expanding-window (train, validate) folds ending at the current split.

        Markets regime-shift, and a single train/validate block can sit entirely
        inside one regime -- the model is then selected on a market that no
        longer exists by the time the test window starts. Walk-forward retrains
        progressively so selection is averaged over several regimes.

        The window EXPANDS rather than slides: every fold trains from the same
        start date, because dropping early history to keep a fixed-length train
        window would confound "later regime" with "less data".

        Fold ``n`` is exactly the configured ``(train, valid)`` split, so ``n=1``
        reproduces today's behaviour and this is a strict generalisation.
        Earlier folds step the validation window back by its own length.

        ``final_test`` is never involved: folds are carved out of the
        train+validate span alone, which is what keeps Property 3 intact.
        """
        import pandas as pd

        if n <= 1:
            return [(self.train, self.valid)]

        v_start, v_end = pd.Timestamp(self.valid[0]), pd.Timestamp(self.valid[1])
        t_start = pd.Timestamp(self.train[0])
        span = v_end - v_start
        day = pd.Timedelta(days=1)

        folds: list[tuple[DateRange, DateRange]] = []
        for k in range(n):
            shift = (span + day) * k
            fv_start, fv_end = v_start - shift, v_end - shift
            ft_end = fv_start - day
            if ft_end <= t_start:
                # Not enough history left to train this fold; stop rather than
                # emit a degenerate window.
                break
            folds.append((
                (t_start.strftime("%Y-%m-%d"), ft_end.strftime("%Y-%m-%d")),
                (fv_start.strftime("%Y-%m-%d"), fv_end.strftime("%Y-%m-%d")),
            ))
        return list(reversed(folds))


@dataclass(frozen=True)
class Execution:
    """Eq. 4-5: the fill rule Φ and the latency δ."""

    delta: int = 1
    fill_rule: str = "open_next"
    label_expr: str = "Ref($close, -2) / Ref($close, -1) - 1"


@dataclass(frozen=True)
class Combiner:
    """Procedure ``A`` (Eq. 2) — an element of Θ, seed included."""

    model: str = "lightgbm"
    # Ensemble of seeds, averaged. A single seed left delta_net_ir with sd
    # comparable to its own magnitude; averaging N cuts that by ~1/sqrt(N).
    seeds: tuple[int, ...] = (42,)
    params: dict[str, Any] = field(default_factory=dict)
    base_features: tuple[str, ...] = ()
    fit_split: str = "train"
    # Which preprocessing pipeline produces the prediction.
    #
    # "native" is the original, written to mirror the baseline handler. It does
    # not: it omits ProcessInf, and it drops label-NaN rows BEFORE rank
    # normalising, so a factor's feature value depended on how many of its
    # cross-sectional neighbours happened to be droppable that day.
    #
    # "backtest_v2" ports the published path exactly (backtest/runner.py,
    # _create_dataset_with_computed_factors). The two engines produced
    # UNCORRELATED daily returns from the same library -- Rank IC +0.0336 here
    # against +0.1179 there on the test split -- so this is not a reporting
    # difference to reconcile but a different model to replace.
    engine: str = "native"


@dataclass(frozen=True)
class Portfolio:
    """The portfolio map ``g`` (Eq. 6). Stateful: w_t = g(ŷ_t, w_drift_t).

    ``cost_aware_dropout`` changes what ``n_drop`` means. With it off, exactly
    ``n_drop`` names are swapped whenever that many have fallen out of the
    top-k, so book turnover is pinned at ``n_drop/topk`` -- measured at
    0.1050-0.1060 across every configuration tested, which leaves the turnover
    dimension of ``U`` carrying no information and the cost model applying to a
    constant trade volume. A capacity-aware objective cannot express itself
    against a construction that trades the same amount whatever it predicts.

    With it on, ``n_drop`` becomes a cap rather than a quota, and a swap happens
    only when the predicted-return gain clears the modelled round-trip cost.
    Turnover then becomes endogenous: persistent signals trade little, noisy
    ones trade more and pay for it.
    """

    construction: str = "topk_dropout"
    topk: int = 50
    n_drop: int = 5
    signed: bool = False
    gross_leverage: float = 1.0
    cost_aware_dropout: bool = False
    # --- mean-variance member (construction: "mean_variance") ---
    # lambda in  max w'mu - (lambda/2) w'Sigma w  - the price of variance.
    risk_aversion: float = 10.0
    # Explicit one-way turnover budget per rebalance. This is the knob top-k
    # dropout does not have: there, turnover is whatever n_drop/topk dictates.
    turnover_cap: float = 0.10
    # Cap on any single position, so the optimiser cannot answer "concentrate
    # everything in the highest-scoring name", which is optimal under a
    # diagonal risk model and untradeable in practice.
    max_weight: float = 0.05
    # Gain must exceed cost by this multiple before a swap is worth making. 1.0
    # is break-even on the modelled cost; above 1.0 demands a margin of safety
    # against the cost model itself being optimistic.
    swap_hurdle: float = 1.0

    # --- risk model -------------------------------------------------------
    # "diagonal" prices each name's own volatility and no correlation, so the
    # optimiser cannot see that two names are the same bet and will happily
    # hold both at full weight. It is the honest limit of the first
    # implementation and it is not what anyone runs a book on.
    #
    # "factor" fits a k-factor statistical model to trailing returns --
    # Sigma = B diag(lambda) B' + diag(d) -- which is the structure every
    # commercial risk model (Barra, Axioma) has, estimated statistically
    # instead of from vendor fundamentals. Low rank plus diagonal keeps
    # Sigma @ w at O(nk) rather than O(n^2), so the optimiser stays fast.
    # Turnover as a price instead of a quota. With this on, the objective
    # carries a per-name transaction-cost term and ``turnover_cap`` stops
    # binding -- how much the book trades becomes an outcome of whether trading
    # pays rather than a number set in advance. A hard cap scales every trade
    # down by the same factor once the book moves too far, spending the budget
    # without choosing what to spend it on, and it also fixes turnover at the
    # same level for every arm, which is precisely the condition under which a
    # net-of-cost objective cannot demonstrate anything.
    cost_in_objective: bool = False
    # Margin of safety on the modelled cost, mirroring swap_hurdle. The
    # optimiser trades whenever the modelled edge exceeds the modelled cost, and
    # both are estimates -- but only one of them is systematically optimistic.
    # Measured: the edge implied by beta is 2.7x larger when calibrated on the
    # split the model was fitted on than on a split it was not.
    cost_hurdle: float = 1.0
    # kappa: inertia charged as (kappa/2) * lambda * dw' Sigma dw.
    #
    # This is what the hard turnover cap was doing by accident. mu is a point
    # estimate whose day-to-day variation is mostly estimation error, and an
    # optimiser that treats it as known chases that error -- Markowitz as an
    # error-maximiser. A cash cost does not restrain it, being roughly constant
    # per unit traded while the apparent edge keeps moving.
    #
    # DIMENSIONLESS on purpose. An absolute nu||dw||^2 was tried first and did
    # not survive the window it was tuned on: nu = 0.5 gave turnover 0.0596 on
    # validation and 0.1092 on test, because mu's dispersion differs between
    # windows. Measured in the same units as the risk term it sits beside,
    # kappa sets the FRACTION of the way the book moves -- 1/(1+kappa)
    # unconstrained -- and that fraction is free of the scale of mu and Sigma.
    # Same measurement after the change: turnover ratios of 1.06, 1.03, 1.25,
    # 0.89 and 0.74 across kappa in {0,1,3,9,30}, against 1.83 before.
    trade_penalty: float = 0.0
    # Which split calibrates beta, the score -> expected-return conversion the
    # cost-aware rules trade against.
    #
    # "train" was the original choice and it is the window the combiner was
    # FITTED on, so beta measures the model's in-sample edge. The measured
    # IS->OOS RankIC gap is +0.28 for both arms against +0.03 for the un-mined
    # baseline, which means the optimiser is told to expect roughly the edge the
    # model shows on data it has already seen. It then trades for that edge and
    # pays real costs for it. "valid" calibrates on a window the model did not
    # fit, so beta reflects what the edge is actually worth out of sample; it is
    # still strictly before final_test, so Property 3 is untouched.
    scale_split: str = "train"
    covariance: str = "diagonal"
    # Trailing window used to estimate the model, and how often it is refit.
    # Both are point-in-time: the window ends strictly before the date being
    # optimised. Monthly refit is the standard institutional cadence and keeps
    # the cost of a 2427-date backtest bounded.
    risk_window: int = 252
    risk_refresh: int = 21
    n_factors: int = 5
    # Shrinkage of the idiosyncratic variances toward their cross-sectional
    # mean. A statistical model estimated on 252 days of 300 names leaves the
    # smallest d_i badly under-estimated, and the optimiser will chase exactly
    # those names because they look riskless.
    idio_shrink: float = 0.20


@dataclass(frozen=True)
class Constraints:
    """Hard A-share feasibility limits (see ``eval/tradability.py``).

    These are market mechanics, not costs, and they apply to both arms: a
    comparison in which one objective is allowed to trade the untradeable is
    not a comparison. Defaults are the main-board rules; ``enabled: false``
    restores the unconstrained behaviour for backwards comparison.
    """

    enabled: bool = True
    enforce_price_limits: bool = True
    enforce_suspension: bool = True
    # T+1 is STRUCTURALLY NON-BINDING for a once-daily book and is off by
    # default for that reason. A name entering at step s has its buy filled at
    # s+delta, and the earliest sale decision is step s+1, filling at
    # s+1+delta -- always at least one session apart, whatever delta is. The
    # machinery is kept because it would bind the moment the rebalance becomes
    # intraday, but enabling it at daily frequency changes no book and claiming
    # it as a modelled constraint would overstate the realism of the result.
    enforce_t1: bool = False
    # Main board ±10%. ST names are ±5% and STAR/ChiNext ±20%; using the
    # main-board band on a mixed universe under-detects locks on ST names
    # rather than inventing ones that did not happen.
    price_limit_pct: float = 0.10


@dataclass(frozen=True)
class Costs:
    """Eq. 7 coefficients. ``nav`` normalizes dollar ADV into weight space."""

    kappa0: float = 0.0020
    kappa1: float = 0.10
    kappa2: float = 0.01
    impact_exponent: float = 1.5
    beta_per_day: float = 0.0004
    beta_offlist: float = float("inf")
    vol_window: int = 20
    adv_window: int = 20
    nav: float = 1e8
    # Converts Qlib cn_data's scaled `$amount` into CNY: see costs.dollar_volume.
    adv_scale: float = 1000.0


@dataclass(frozen=True)
class Gates:
    """F_Θ (Eq. 9): absolute floors, evaluated on train/valid only.

    These are a property of the market and the existing book, not a bar that
    drifts with search progress — which is why they are absolute here and the
    *scoring* in ``scoring.py`` is repository-relative.
    """

    # Every gate is nullable and every one is null by default: hard
    # admissibility was removed after measurement showed the criteria were
    # either anti-correlated with contribution (IC) or not reproducible across
    # combiner seeds (delta_net_ir). Set a float to re-enable an individual gate.
    gamma_delta: float | None = None
    gamma_ic: float | None = None
    gamma_ir: float | None = None
    tau_max: float | None = None
    rho_bar: float | None = None
    gamma_cx: float | None = None
    turnover_basis: str = "book"


@dataclass(frozen=True)
class Decay:
    """§3.4 item 7: OOS sub-period IC trend."""

    K: int = 4
    lambda_d: float = 0.5


@dataclass(frozen=True)
class Overfit:
    """Reference IS→OOS RankIC gap used to normalize ``e_overfit``."""

    delta_ref: float = 0.02


@dataclass(frozen=True)
class Utility:
    """Eq. 12. ``weighted_rank`` is the repository-relative scalarization."""

    mode: str = "weighted_rank"
    omega: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForward:
    """Progressive retraining for the in-loop evaluation (§ walk-forward).

    ``folds=1`` is the single fixed split. Above that, the in-loop score is the
    mean over folds, so a factor selected because it happened to suit one
    regime is penalised relative to one that holds across several.

    Costs one combiner refit per fold, and touches only train/validate --
    ``final_test`` is scored exactly as before.
    """

    enabled: bool = False
    folds: int = 3


@dataclass(frozen=True)
class Admission:
    """When ``U`` admits a candidate to the repository, and when it evicts one.

    ``U`` is a weighted mean of percentile complements, so ``0.5`` is literally
    "the median incumbent". Admitting at ``0.5`` is therefore the adaptive bar
    §3.4 describes: the threshold is constant while the standard it encodes
    rises with the repository.

    ``tau_evict`` must sit **below** ``tau_admit``. Because every member's score
    is a percentile within the repository, roughly half of them fall below the
    median by construction -- evicting at the admission bar would drop half the
    repository every round and cascade toward empty. The gap between the two
    thresholds is hysteresis: a factor has to fall clearly behind, not merely
    below average, before it is dropped.

    ``min_size`` bootstraps the process. Ranking against nothing is
    uninformative (see ``scoring.rank``), so the first few batches are admitted
    unconditionally, and eviction never shrinks the repository below this.
    """

    enabled: bool = True
    tau_admit: float = 0.5
    tau_evict: float = 0.25
    min_size: int = 3

    # --- marginal-contribution mode (see eval/admission.py) ---------------
    # "utility_rank" is the original percentile bar, kept so old protocols
    # reproduce exactly. "marginal_contribution" admits on whether the book
    # actually improves, tested against the noise on that measurement.
    mode: str = "utility_rank"

    # How many standard errors the improvement must clear. This is the bar, and
    # it is dynamic by construction: it scales with how precisely the delta can
    # be measured rather than with a percentile of a winners-only sample. 1.0 is
    # "more likely than not, on this evidence"; raise it to demand more.
    k_sigma: float = 1.0

    # Combiner seeds used to measure the delta. delta_net_ir flipped 3 of 4
    # verdicts across seeds as a point estimate, which is why it needs a spread
    # rather than a single number. Fewer than two seeds leaves the standard
    # error undefined and the test cannot pass.
    test_seeds: tuple[int, ...] = (42, 1, 7)

    # Repository capacity. Once full, admission becomes replacement: a
    # candidate enters only by displacing the incumbent it beats, so the
    # criterion stops decaying as marginal contribution shrinks and the library
    # improves rather than merely growing. None means unbounded.
    capacity: int | None = None

    # Re-measure incumbents against the CURRENT repository every N rounds and
    # drop those whose contribution has gone below evict_below. A factor that
    # was additive at |zoo|=12 can be redundant at 150, and nothing in its own
    # metrics changes to say so. 0 disables re-scoring.
    evict_every: int = 0
    evict_below: float = 0.0

    # Pathology floor. Rejects signals that cannot be priced, never signals
    # that are merely weak -- that distinction is what separates this from the
    # absolute quality gates that were removed for starving the repository.
    min_coverage: float = 0.30


@dataclass(frozen=True)
class Protocol:
    """Θ. Frozen, hashable, and the sole source of every evaluation knob."""

    protocol_id: str
    market: str
    benchmark: str
    periods_per_year: int
    splits: Splits
    execution: Execution
    combiner: Combiner
    portfolio: Portfolio
    costs: Costs
    constraints: Constraints
    gates: Gates
    decay: Decay
    overfit: Overfit
    utility: Utility
    admission: Admission = field(default_factory=Admission)
    walk_forward: WalkForward = field(default_factory=WalkForward)

    def __post_init__(self) -> None:
        # Copy mutable containers so a caller holding a reference to the parsed
        # YAML cannot mutate Θ out from under the (cached) hash.
        object.__setattr__(self.combiner, "params", dict(self.combiner.params))
        object.__setattr__(self.utility, "omega", dict(self.utility.omega))

    @cached_property
    def hash(self) -> str:
        """SHA-256 of the canonical JSON form, truncated to 16 hex chars.

        Logged on every evaluation. Two runs sharing a hash provably shared a
        protocol; a mutated YAML yields a different hash.
        """
        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_range(value: Any, name: str) -> DateRange:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"split {name!r} must be a [start, end] pair, got {value!r}")
    start, end = (str(v) for v in value)
    if start >= end:
        raise ValueError(f"split {name!r} is not well-ordered: {start} >= {end}")
    return (start, end)


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Construct a dataclass from a YAML mapping, rejecting unknown keys.

    Rejecting rather than ignoring is deliberate: a typo in Θ that silently
    fell back to a default would be a Property 1 violation that never surfaced.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown protocol keys {sorted(unknown)}")
    return cls(**payload)


def load_protocol(path: str | os.PathLike[str] | None = None) -> Protocol:
    """Load and validate Θ from YAML.

    Validation is strict on load, because every downstream number depends on
    these values and a malformed protocol is far cheaper to catch here than in
    a five-hour mining run.
    """
    resolved = Path(path) if path is not None else default_protocol_path()
    if not resolved.exists():
        raise FileNotFoundError(f"protocol not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"protocol {resolved} did not parse to a mapping")

    splits_raw = dict(raw["splits"])
    aliases = {
        "search_is": splits_raw.pop("search_is", "train"),
        "search_oos": splits_raw.pop("search_oos", "valid"),
    }
    windows = {name: _as_range(value, name) for name, value in splits_raw.items()}
    for alias, target in aliases.items():
        if target not in windows:
            raise ValueError(f"{alias}: {target!r} is not one of {sorted(windows)}")
    splits = _build(Splits, {**windows, **aliases})

    # Property 3: nothing the search reads may reach into the test window.
    for name in (splits.search_is, splits.search_oos):
        if splits.window(name)[1] >= splits.final_test[0]:
            raise ValueError(
                f"split {name!r} ends at {splits.window(name)[1]}, which is not before "
                f"final_test start {splits.final_test[0]}; the search would see test data"
            )

    combiner_raw = dict(raw["combiner"])
    combiner_raw["base_features"] = tuple(combiner_raw.get("base_features", ()))
    combiner_raw["params"] = dict(combiner_raw.get("params", {}))
    if "seed" in combiner_raw:      # accept the old scalar form
        combiner_raw.setdefault("seeds", [combiner_raw.pop("seed")])
    combiner_raw["seeds"] = tuple(int(s) for s in combiner_raw.get("seeds", (42,)))
    if not combiner_raw["seeds"]:
        raise ValueError("combiner.seeds must contain at least one seed")
    combiner = _build(Combiner, combiner_raw)
    if combiner.fit_split not in ("train",):
        # Fitting anywhere but train would leak the evaluation window into the
        # model that generates the predictions being evaluated.
        raise ValueError(f"combiner.fit_split must be 'train', got {combiner.fit_split!r}")

    utility_raw = dict(raw["utility"])
    utility_raw["omega"] = dict(utility_raw.get("omega", {}))
    utility = _build(Utility, utility_raw)
    if utility.mode not in ("net_ir", "net_arr", "weighted_rank"):
        raise ValueError(f"unknown utility.mode {utility.mode!r}")
    omega_sum = sum(utility.omega.values())
    if abs(omega_sum - 1.0) > 1e-9:
        raise ValueError(f"utility.omega must sum to 1, got {omega_sum!r}")

    admission = _build(Admission, dict(raw.get("admission", {})))
    if not 0.0 <= admission.tau_evict <= admission.tau_admit <= 1.0:
        # Without hysteresis the repository oscillates: U is a percentile, so
        # about half the members sit below the admission bar at any moment.
        raise ValueError(
            f"admission requires 0 <= tau_evict <= tau_admit <= 1, got "
            f"tau_evict={admission.tau_evict}, tau_admit={admission.tau_admit}"
        )
    if admission.min_size < 1:
        raise ValueError("admission.min_size must be at least 1")

    portfolio = _build(Portfolio, dict(raw["portfolio"]))
    if portfolio.n_drop > portfolio.topk:
        raise ValueError("portfolio.n_drop cannot exceed portfolio.topk")
    if portfolio.swap_hurdle < 0:
        raise ValueError("portfolio.swap_hurdle must be non-negative")

    gates = _build(Gates, dict(raw["gates"]))
    if gates.turnover_basis not in ("book", "solo"):
        raise ValueError(f"unknown gates.turnover_basis {gates.turnover_basis!r}")

    return Protocol(
        protocol_id=str(raw["protocol_id"]),
        market=str(raw["market"]),
        benchmark=str(raw["benchmark"]),
        periods_per_year=int(raw["periods_per_year"]),
        splits=splits,
        execution=_build(Execution, dict(raw["execution"])),
        combiner=combiner,
        portfolio=portfolio,
        costs=_build(Costs, dict(raw["costs"])),
        constraints=_build(Constraints, dict(raw.get("constraints", {}))),
        gates=gates,
        decay=_build(Decay, dict(raw["decay"])),
        overfit=_build(Overfit, dict(raw["overfit"])),
        utility=utility,
        admission=admission,
        walk_forward=_build(WalkForward, dict(raw.get("walk_forward", {}))),
    )


__all__ = [
    "Combiner",
    "Costs",
    "DEFAULT_PROTOCOL_PATH",
    "Decay",
    "Execution",
    "Gates",
    "Overfit",
    "Portfolio",
    "Protocol",
    "Splits",
    "Utility",
    "default_protocol_path",
    "load_protocol",
]
