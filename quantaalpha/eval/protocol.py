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


@dataclass(frozen=True)
class Portfolio:
    """The portfolio map ``g`` (Eq. 6). Stateful: w_t = g(ŷ_t, w_drift_t)."""

    construction: str = "topk_dropout"
    topk: int = 50
    n_drop: int = 5
    signed: bool = False
    gross_leverage: float = 1.0


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
    gates: Gates
    decay: Decay
    overfit: Overfit
    utility: Utility
    admission: Admission = field(default_factory=Admission)

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
        gates=gates,
        decay=_build(Decay, dict(raw["decay"])),
        overfit=_build(Overfit, dict(raw["overfit"])),
        utility=utility,
        admission=admission,
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
