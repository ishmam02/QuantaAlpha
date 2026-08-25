"""Procedure ``A`` (Eq. 2) — the combiner refit that produces ŷ.

This module exists because of a load-bearing detail in the formulation: the
portfolio is built from the **combiner's composite prediction**, not from the
candidate factor's own signal. ``M`` (LightGBM) is refit at every
strategy-level evaluation on the repository as it currently stands,
``θ = A(F_zoo ∪ {f}, D_tr)``.

The consequence is that the strategy-level metrics of a candidate are a
property of *the book containing it* — a marginal contribution — rather than
of the factor in isolation. A factor that is individually mediocre but
orthogonal to the zoo can therefore score well, which is exactly the intent.

``A`` is an element of Θ: algorithm, hyperparameters **and** random seed. The
seed is what Task 1 added to the in-loop Qlib templates, where it was missing
entirely; without it Property 2 (determinism) cannot hold.

Preprocessing mirrors ``conf_combined_factors.yaml:38-58`` exactly —
``Fillna(feature)`` → ``DropnaLabel`` → ``CSRankNorm(feature)`` →
``CSRankNorm(label)`` — because the whole point of nesting the baseline is
that the composite prediction is built the same way.
"""

from __future__ import annotations

import os

import hashlib
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from quantaalpha.eval.data import PanelBundle, _init_qlib
from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)

# Qlib's CSRankNorm scales the centred percentile rank towards unit variance.
# Uniform[-0.5, 0.5] has std 1/sqrt(12) = 0.2887, hence ~3.46.
_CS_RANK_SCALE = 3.46


def _cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization, per date (Qlib ``CSRankNorm``)."""
    ranked = df.groupby(level="datetime", group_keys=False).rank(pct=True)
    return (ranked - 0.5) * _CS_RANK_SCALE


def _preprocess_v2(features: pd.DataFrame, label: pd.Series):
    """Backtest V2's preprocessing, step for step.

    Ported rather than approximated, because approximating it is what produced
    the discrepancy this exists to close: on the same 156-factor library our
    prediction scored Rank IC +0.0336 on the test split against Backtest V2's
    +0.1179, and the two engines' daily excess returns were *uncorrelated*
    (-0.029 at every lag from -5 to +5) -- two different models, not two
    reports of one.

    Three differences from the native path, all here:

    1. **ProcessInf.** Infinities are replaced with 0. The native path filled
       NaNs and left ``inf`` alone. They are demonstrably present -- the
       Alpha158(20) seed ``SHADOW_RATIO`` has ``1e-12`` in a denominator and a
       standard deviation of 1.8e11 on real data.
    2. **Rank BEFORE dropping label-NaN rows.** The native path dropped first,
       so every cross-sectional rank was computed over the subset of names that
       happened to have a label that day, and the same factor value received a
       different feature value depending on how many neighbours were droppable.
       Ranking first fixes the cross-section.
    3. **No 3.46 scaling.** Qlib's ``CSRankNorm`` divides by the standard
       deviation of a uniform to give unit variance; Backtest V2's inline
       version does not. Monotone, so it cannot change a tree's splits or the
       IC, but it is kept faithful so the two paths are byte-comparable rather
       than merely similar.

    Grouping is by the datetime level, matching ``groupby(level=dt_level)``
    there.
    """
    feat = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    ranked = feat.groupby(level="datetime", group_keys=False).rank(pct=True) - 0.5
    keep = label.notna()
    ranked, label = ranked[keep], label[keep]
    label = label.groupby(level="datetime", group_keys=False).rank(pct=True) - 0.5
    return ranked.fillna(0.0), label


def _wide_to_long(frame: pd.DataFrame, name: str) -> pd.Series:
    try:  # pandas >= 2.1 spells "keep the NaNs" this way
        out = frame.stack(future_stack=True)
    except TypeError:  # pragma: no cover - older pandas
        out = frame.stack(dropna=False)
    out.index.names = ["datetime", "instrument"]
    out.name = name
    return out


def _base_features(panel: PanelBundle, theta: Protocol) -> pd.DataFrame:
    """Evaluate Θ's four base features through Qlib, on the panel's window."""
    _init_qlib()
    from qlib.data import D

    exprs = list(theta.combiner.base_features)
    if not exprs:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]))

    raw = D.features(
        D.instruments(theta.market),
        exprs,
        start_time=str(panel.dates.min().date()),
        end_time=str(panel.dates.max().date()),
    )
    raw.index = raw.index.set_names(["instrument", "datetime"])
    raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    raw.columns = [f"BASE{i}" for i in range(len(exprs))]
    return raw


def _label(panel: PanelBundle, theta: Protocol) -> pd.Series:
    """Θ's training label, **whatever ``execution.label_expr`` says**.

    This docstring used to claim a close-to-close label ``Ref($close,-2)/
    Ref($close,-1)-1``. That is no longer what Θ trains on: the soft-linear
    protocol sets ``label_expr: Ref($open,-2)/Ref($open,-1)-1``, matching
    ``fill_rule: open_next`` so the target is the return the book actually
    earns between its own fills. There is no label/fill mismatch left to
    resolve here -- the two were deliberately aligned on open.

    The distinction is not cosmetic. Measured on 2022-01-01..2025-12-26 the
    open-to-open and close-to-close one-day labels agree at a cross-sectional
    rank IC of only **+0.069** -- they overlap solely in the overnight gap,
    the intraday legs being disjoint -- so they are close to *different*
    prediction problems. ``backtest_v2`` (``configs/backtest.yaml``) still
    trains and scores on the close label while filling at the open, which is
    why its IC is not comparable with Θ's and reads far higher.
    """
    _init_qlib()
    from qlib.data import D

    raw = D.features(
        D.instruments(theta.market),
        [theta.execution.label_expr],
        start_time=str(panel.dates.min().date()),
        end_time=str(panel.dates.max().date()),
    )
    raw.index = raw.index.set_names(["instrument", "datetime"])
    raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    label = raw.iloc[:, 0]
    label.name = "LABEL0"
    return label


def signal_hash(expression: str) -> str:
    return hashlib.md5(expression.encode()).hexdigest()


def zoo_hash(zoo_signals: dict[str, pd.DataFrame]) -> str:
    """Content hash of the repository state (change #4 in the formulation).

    ``E_Θ(f; zoo)`` is deterministic with respect to the **pair**, so a trial
    is only reproducible if the zoo it was scored against is recorded too.
    """
    if not zoo_signals:
        return "empty"
    joined = "|".join(sorted(signal_hash(expr) for expr in zoo_signals))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


_PREDICTION_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
# T4: per-factor combiner credit, parallel to _PREDICTION_CACHE and keyed
# identically. None for the LightGBM path (no per-factor weights); a list of
# per-factor dicts for the ICIR path. Kept in a parallel cache so the
# prediction cache (and its byte-identical re-reindex on hit) is untouched.
_ATTRIBUTION_CACHE: dict[tuple[str, str, str], list | None] = {}


def _per_date_ic(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-date cross-sectional rank IC of each feature with the label.

    Inputs are already cross-sectionally rank-normalised (``_preprocess_v2`` or
    ``_cs_rank_norm``), so Pearson correlation here *is* the Spearman rank IC.
    Returns a ``(n_dates, n_features)`` frame -- one IC row per training date --
    which the bootstrap in :func:`_fit_predict_icir` re-averages per seed.

    Computed as one vectorised pass rather than ``groupby.apply(corrwith)``.
    The apply form builds a DataFrame per training date (~1200 of them) and
    pays pandas' per-group overhead on each; the sums below are the same
    arithmetic done once over the whole column block, which measured 5-17x
    faster on the real window shape and agrees with the previous result to
    floating-point exactness (<=4e-16 on clean, NaN-holed, ragged and
    ragged+NaN inputs alike).

    The subtlety worth preserving: ``corrwith`` is **pairwise-complete** -- each
    feature is correlated with the label over the rows where *that* feature and
    the label are both present, not over rows where every feature is present.
    A dense ``reshape`` + shared-mask formulation is therefore wrong (it
    disagreed by ~0.13 where NaNs were present) and also cannot represent a
    ragged panel at all. The per-column ``valid`` mask below reproduces the
    pairwise semantics exactly, and ``reduceat`` over group boundaries handles
    unequal instrument counts per date.
    """
    feats = list(x.columns)
    if not feats or len(x.index) == 0:
        return pd.DataFrame(index=pd.Index([], name="datetime"), columns=feats, dtype=float)

    dt = x.index.get_level_values("datetime")
    # reduceat needs contiguous groups. The caller's frames are already sorted
    # by datetime; sort defensively (stable, so ties keep their order) rather
    # than assume it, since an unsorted input would silently mis-group.
    dt_vals = dt.to_numpy()
    if not (dt_vals[1:] >= dt_vals[:-1]).all():
        order = np.argsort(dt_vals, kind="stable")
        xv, yv, dt_vals = (x.to_numpy(dtype=float)[order],
                           y.to_numpy(dtype=float)[order], dt_vals[order])
    else:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)

    starts = np.flatnonzero(np.r_[True, dt_vals[1:] != dt_vals[:-1]])
    uniq = dt_vals[starts]

    yb = np.broadcast_to(yv[:, None], xv.shape)
    valid = np.isfinite(xv) & np.isfinite(yb)      # pairwise mask, per feature
    xz = np.where(valid, xv, 0.0)
    yz = np.where(valid, yb, 0.0)

    def _sum(a):                                    # per-date column sums
        return np.add.reduceat(a, starts, axis=0)

    n = _sum(valid.astype(np.float64))
    sx, sy = _sum(xz), _sum(yz)
    sxx, syy, sxy = _sum(xz * xz), _sum(yz * yz), _sum(xz * yz)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        var_x = sxx - sx * sx / n
        var_y = syy - sy * sy / n
        denom = np.sqrt(var_x * var_y)
        # A date on which a column has no cross-sectional variance is UNDEFINED,
        # not infinite. Dividing straight through produced +/-inf, and inf is
        # poison downstream: `notna()` counts it as present and `np.nanmean`
        # skips NaN but NOT inf, so ONE such date turned a factor's whole mean IC
        # into NaN. Measured on the live design matrix: every one of 19 mined
        # factors had 131 inf dates (all before its signal history began, where
        # fillna(0.0) leaves a constant column) and therefore a NaN mean IC whose
        # true value was -0.00917. The factors were silently invisible to the
        # combiner's weighting.
        ic = np.divide(cov, denom, out=np.full_like(cov, np.nan),
                       where=np.isfinite(denom) & (denom > 0))
        ic[~np.isfinite(ic)] = np.nan
    ic[n < 2] = np.nan                              # corr is undefined on <2 points
    return pd.DataFrame(ic, index=pd.Index(uniq, name="datetime"), columns=feats)


def _icir_weights(ic_arr_sample: np.ndarray, shrink) -> tuple[np.ndarray, float]:
    """ICIR (IC/σ_IC) weights for one bootstrap sample, shrunk toward 1/N.

    ``ic_arr_sample`` is the ``(n_dates, n_feats)`` per-date IC matrix for one
    bootstrap resample. Returns ``(weights, delta)`` where ``delta`` is the
    Ledoit-Wolf shrinkage intensity actually applied (useful for diagnostics).

    Pure function -- no Θ, no panel -- so the Ding-Martin Redux math is unit
    testable without standing up Qlib. See :func:`_fit_predict_icir` for the
    rationale.
    """
    n_feats = ic_arr_sample.shape[1]
    # Defence in depth: scrub non-finite values before any statistic. `nanmean`
    # skips NaN but not +/-inf, so a single infinite date silently NaNs a
    # factor's mean IC -- and a NaN weight then makes `feat_mat @ weights` NaN
    # for EVERY row, taking out the whole composite rather than one factor.
    ic_arr_sample = np.where(np.isfinite(ic_arr_sample), ic_arr_sample, np.nan)
    with np.errstate(invalid="ignore"):
        ic_mean = np.nanmean(ic_arr_sample, axis=0)        # mean IC per factor
        ic_std = np.nanstd(ic_arr_sample, axis=0, ddof=1)  # σ_IC per factor
    # A column with no usable dates at all contributes nothing rather than NaN.
    ic_mean = np.nan_to_num(ic_mean, nan=0.0, posinf=0.0, neginf=0.0)
    ic_std = np.nan_to_num(ic_std, nan=0.0, posinf=0.0, neginf=0.0)
    icir = ic_mean / (ic_std + 1e-8)                   # IC/σ_IC (Ding-Martin IR)
    if shrink == "auto":
        # Ledoit-Wolf: ratio of ESTIMATION-ERROR variance to weight dispersion.
        # →1 (equal weight) when the ICIRs are noisy; →0 (raw ICIR) when they are
        # precisely estimated and well spread.
        #
        # The numerator was ``sum(ic_std**2)`` -- a sum of DAILY IC variances,
        # not the estimation error of the ICIR estimate. Those differ by a
        # factor of T (the number of dates):
        #
        #     ICIR_hat = mean_IC / sd_IC,  Var(mean_IC) = sd_IC**2 / T
        #     => Var(ICIR_hat) ~ 1/T   to leading order
        #
        # At T=727 the old numerator was 727x too large, so delta CLIPPED TO 1.0
        # for any realistic factor set and the ICIR fit never happened: the
        # combiner computed ICIRs spanning [-0.36, -0.10] and then returned
        # identical weights of -1/N to every factor. "ICIR combiner" was an
        # equal-weight combiner in every run to date.
        #
        # Measured 2026-08-24 on two independent factor sets, valid window:
        #     15-factor book: delta 1.000 -> 0.240, |IC| 0.0554 -> 0.0612 (+10.5%)
        #     37-factor book: delta 1.000 -> 0.234, |IC| 0.0593 -> 0.0629 (+6.1%)
        # and the composite crosses from BELOW its own best single factor
        # (0.98x) to above it (1.08x) -- from destroying signal to adding it.
        #
        # n_dates is taken from the sample actually used, not assumed, so a
        # short bootstrap resample shrinks harder (correctly: less data, noisier
        # ICIR). Fewer than 2 dates carries no estimate at all -> full shrinkage.
        disp = float(np.nansum((icir - np.nanmean(icir)) ** 2))
        n_dates = int(np.max(np.sum(np.isfinite(ic_arr_sample), axis=0))) \
            if ic_arr_sample.size else 0
        if n_dates < 2:
            delta = 1.0
        else:
            est_err = float(n_feats) / float(n_dates)
            delta = float(np.clip(est_err / (disp + 1e-12), 0.0, 1.0))
    else:
        delta = float(np.clip(shrink, 0.0, 1.0))
    # SIGN-PRESERVING shrinkage target.
    #
    # This was `np.full(n_feats, 1.0/n_feats)` -- a POSITIVE constant. Ledoit-Wolf
    # shrinks a noisy estimator toward a low-variance structured target, and the
    # equal-weight target is the standard choice *for factors already oriented to
    # positive IC*, which is how the literature defines them. These are raw mined
    # expressions with arbitrary sign, and the consequence is severe:
    #
    #   weight = (1-d)*icir + d*(1/N)
    #
    # flips the sign of any factor with |icir| < d/(1-d) * 1/N. At d=0.5, N=9
    # that threshold is 0.111, and mined factors run icir ~0.01-0.05 -- so EVERY
    # negative-IC factor received a POSITIVE weight. On a market whose factors
    # are predominantly negative-IC (short-horizon reversal), the combiner was
    # inverting nearly every factor's contribution.
    #
    # Measured on the live 9-factor zoo: production combiner rank IC **-0.0063**
    # against a best single factor of **0.0431**, while a sign-aligned equal
    # weight scored **+0.0319** and signed IC-weighting **+0.0396**. The book was
    # losing ~6x the signal a one-line baseline captures.
    #
    # sign(icir)/N is the orientation-respecting analogue: same shrinkage
    # magnitude, target pointed the way the factor actually predicts. Equivalent
    # to orienting each feature by sign(IC) and then shrinking toward 1/N.
    # A factor with icir exactly 0 gets target 0, which is correct -- it carries
    # no directional information to shrink toward.
    equal = np.sign(icir) / n_feats
    weights = (1.0 - delta) * icir + delta * equal
    weights = np.nan_to_num(weights, nan=0.0)
    return weights, delta


def _icir_attribution(
    features: pd.DataFrame,
    ic_arr: np.ndarray,
    seed_weights: list[np.ndarray],
    col_to_expr: dict[str, str] | None,
) -> list[dict]:
    """Per-factor combiner credit from the ICIR weights (T4, full fix #3).

    Each factor column gets: ``weight`` (its share of the composite's absolute
    weight -- positive, sums to 1 across factors; the "strongest sub-pattern"
    signal for diagnosis/crossover), ``weight_raw`` (the signed mean ICIR
    weight, which the operator uses for the turnover attribution),
    ``weight_std`` / ``weight_stability`` (cross-seed estimation noise and
    ``1 - std/|mean|``), and ``ic_mean``/``ic_std``/``rank_ic`` from the
    full-sample per-date IC matrix (CSRankNorm'd features -> the per-date
    Pearson corr IS the rank IC, so the mean over dates is the mean rank IC).
    Columns not in ``col_to_expr`` (the base features) are excluded; ``None``
    -> "every column is a factor" (used by the unit tests).
    """
    if col_to_expr is None:
        col_to_expr = {c: c for c in features.columns}
    w_mat = np.array(seed_weights)                 # (n_seeds, n_feats)
    w_mean = w_mat.mean(axis=0)
    w_std = w_mat.std(axis=0, ddof=1) if w_mat.shape[0] > 1 else np.zeros_like(w_mean)
    ic_mean_full = np.nanmean(ic_arr, axis=0)
    ic_std_full = np.nanstd(ic_arr, axis=0, ddof=1)
    total_abs = float(np.nansum(np.abs(w_mean))) or 1e-12
    out: list[dict] = []
    for col, expr in col_to_expr.items():
        if expr is None or col not in features.columns:
            continue
        i = features.columns.get_loc(col)
        wm = float(w_mean[i]); ws = float(w_std[i])
        out.append({
            "expr": expr,
            "weight": float(abs(wm) / total_abs),
            "weight_raw": wm,
            "weight_std": ws,
            "weight_stability": float(np.clip(1.0 - ws / (abs(wm) + 1e-8), 0.0, 1.0)),
            "ic_mean": float(ic_mean_full[i]),
            "ic_std": float(ic_std_full[i]),
            "rank_ic": float(ic_mean_full[i]),
        })
    return out


def _fit_predict_icir(
    features: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    theta: Protocol,
    panel: PanelBundle,
    col_to_expr: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """ICIR + shrinkage linear combiner (Ding-Martin Redux).

    The composite prediction is ``μ = Σᵢ wᵢ · xᵢ`` where the weights are the
    **information ratio** of each factor's cross-sectional rank IC --
    ``wᵢ = mean(ICᵢ) / std(ICᵢ)`` -- not the mean IC alone. This is the
    Ding-Martin (2017) Redux fix: IR = IC/σ_IC asymptotically, and σ_IC
    ("strategy risk") is non-diversifiable and bounds IR, so a factor whose IC
    is positive but volatile must weight *less* than one whose IC is the same
    on average but stable. Mean-IC weighting ignores σ_IC and lets noisy
    factors dominate the composite; the replay bore this out -- plain
    IC-weighting admitted 9/125 where LightGBM's ``se≈0`` auto-admit gave 26,
    three of which were resolvably *negative* under honest variance.

    The weights are then **shrunk toward equal weight** by a Ledoit-Wolf
    intensity δ (James-Stein / DeMiguel; equal-weight is the high-σ_IC limit
    and the null model that needs no estimation). ``shrinkage`` is read from
    ``theta.combiner.params``:

      * ``"auto"`` (default): δ = clip( Σσ_ICᵢ² / Σ(wᵢ - w̄)², 0, 1 ) per
        bootstrap sample -- →1 when the ICs are noisy (estimation error
        dominates dispersion), →0 when they are precise and spread.
      * a float f: δ = f (fixed intensity, 0 = raw ICIR, 1 = equal weight).

    The five ``test_seeds`` BOOTSTRAP the train dates (resample with
    replacement) before computing each sample's ICIR, so five seeds give five
    different weight vectors, ``μ``s, books, and marginal contributions -- the
    honest finite-sample σ_IC the admission gate's t-test measures. A
    deterministic combiner would return five identical deltas (``se = 0``) and
    collapse the gate; LightGBM's ``se≈0`` is exactly that failure, and is what
    inflated its admission count.

    Returns the **raw composite score** (wide T×N). The operator maps it to
    expected-return units via Grinold α (``α = IC_c · σ_i · s_i``) rather than
    the empirical ``prediction_scale`` β, so the raw scale of ``μ`` is
    irrelevant -- only the *relative* weighting the ICIR+shrinkage induces.

    T4 also returns the **per-factor attribution** (a list of dicts, one per
    factor column) built from the weights the seeds already compute -- the
    prediction itself is unchanged (byte-identical to the pre-T4 path); only
    the weights and full-sample IC stats are summarized into credit. ``None``
    for ``col_to_expr`` attributes every column (used by the unit tests).
    """
    ic_by_date = _per_date_ic(x_train, y_train)
    ic_arr = ic_by_date.to_numpy(dtype=float)          # (n_dates, n_feats)
    n_dates = ic_arr.shape[0]
    feat_mat = features.to_numpy(dtype=float)          # (n_rows, n_feats), full window
    shrink = theta.combiner.params.get("shrinkage", "auto")
    preds = []
    seed_weights: list[np.ndarray] = []  # T4: per-seed weight vectors -> credit
    for seed in theta.combiner.seeds:
        rng = np.random.default_rng(int(seed))
        samp = rng.integers(0, n_dates, size=n_dates)  # bootstrap date indices
        weights, _delta = _icir_weights(ic_arr[samp], shrink)
        seed_weights.append(weights)
        preds.append(feat_mat @ weights)               # (n_rows,)
    raw_pred = pd.Series(np.mean(preds, axis=0), index=features.index, name="score")
    wide = raw_pred.unstack(level="instrument")
    wide.index = pd.to_datetime(wide.index)
    prediction = wide.sort_index().reindex(index=panel.dates, columns=panel.instruments)
    prediction = prediction.where(panel.universe)
    return prediction, _icir_attribution(features, ic_arr, seed_weights, col_to_expr)


def fit_predict(
    zoo_signals: dict[str, pd.DataFrame],
    candidate_signal: pd.DataFrame | None,
    panel: PanelBundle,
    theta: Protocol,
    candidate_expr: str = "CANDIDATE",
) -> tuple[pd.DataFrame, list[dict] | None]:
    """Refit ``A`` on ``zoo ∪ {f}`` and return the composite prediction ŷ.

    Fits on ``Θ.combiner.fit_split`` (train) only and predicts across the whole
    panel window. Returns a wide ``(T × N)`` frame aligned to ``panel``.

    ``candidate_signal=None`` fits the **baseline** book on ``zoo`` alone, which
    is what marginal contribution is measured against. With an empty zoo that
    baseline is the null model: the four base features and nothing else.
    """
    key = (
        zoo_hash(zoo_signals),
        "baseline" if candidate_signal is None else signal_hash(candidate_expr),
        theta.hash,
    )
    cached = _PREDICTION_CACHE.get(key)
    if cached is not None:
        return cached.reindex(index=panel.dates, columns=panel.instruments), \
            _ATTRIBUTION_CACHE.get(key)

    # ---- design matrix: base features + every zoo signal + the candidate ----
    columns: dict[str, pd.Series] = {}
    base = _base_features(panel, theta)
    for col in base.columns:
        columns[col] = base[col]
    # T4: ZOO{i}/CAND -> expression mapping, so the ICIR attribution can report
    # per-factor credit (the base features are intentionally not in the map and
    # are excluded from the per-factor credit).
    col_to_expr: dict[str, str] = {}
    for i, (expr, sig) in enumerate(sorted(zoo_signals.items())):
        columns[f"ZOO{i}"] = _wide_to_long(sig.reindex(index=panel.dates, columns=panel.instruments), f"ZOO{i}")
        col_to_expr[f"ZOO{i}"] = expr
    if candidate_signal is not None:
        columns["CAND"] = _wide_to_long(
            candidate_signal.reindex(index=panel.dates, columns=panel.instruments), "CAND"
        )
        col_to_expr["CAND"] = candidate_expr

    features = pd.DataFrame(columns)
    label = _label(panel, theta).reindex(features.index)

    if getattr(theta.combiner, "engine", "native") == "backtest_v2":
        features, label = _preprocess_v2(features, label)
    else:
        # ---- preprocessing, in the baseline handler's exact order ----
        features = features.fillna(0.0)                      # Fillna(feature)
        keep = label.notna()                                 # DropnaLabel
        features, label = features[keep], label[keep]
        features = _cs_rank_norm(features)                   # CSRankNorm(feature)
        label = _cs_rank_norm(label.to_frame()).iloc[:, 0]   # CSRankNorm(label)
        features = features.fillna(0.0)

    # ---- fit on train only ----
    train_start, train_end = theta.splits.window(theta.combiner.fit_split)
    # PURGE the tail of the fit window. A label at date t is realised at t+1+h,
    # so rows at the very end of `train` are scored on returns that fall after
    # the window closes -- inside the data this fit is about to be judged on.
    # Only the `train` split is purged; a caller fitting on some other window has
    # already chosen it deliberately.
    if theta.combiner.fit_split == "train":
        try:
            _h = int(getattr(theta.execution, "label_horizon", 1) or 1)
            train_start, train_end = theta.splits.train_window_purged(_h)
        except Exception:
            pass
    dates = features.index.get_level_values("datetime")
    in_train = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
    x_train, y_train = features[in_train], label[in_train]
    if x_train.empty:
        raise ValueError(
            f"combiner: no rows in fit split {theta.combiner.fit_split} "
            f"({train_start}..{train_end}) within the loaded panel"
        )

    # ---- model dispatch: ICIR+shrinkage linear combiner (Ding-Martin Redux) ----
    # The frozen LightGBM procedure (model="lightgbm") is the default and is
    # what the frozen / soft protocols select. model="icir" routes here
    # instead: same design matrix and preprocessing, but μ is an ICIR-weighted
    # (IC/σ_IC) linear sum of the factor signals, shrunk toward equal weight,
    # rather than a 500-round boosted fit. No threads, no over-fitting,
    # milliseconds not minutes. See _fit_predict_icir for why each combiner
    # seed bootstraps the train dates.
    model = getattr(theta.combiner, "model", "lightgbm")
    if model == "icir":
        prediction, attribution = _fit_predict_icir(features, x_train, y_train, theta, panel, col_to_expr)
        _PREDICTION_CACHE[key] = prediction
        _ATTRIBUTION_CACHE[key] = attribution
        logger.debug(
            "combiner refit (icir): zoo=%s cand=%s theta=%s rows=%d feats=%d seeds=%d",
            key[0], key[1][:8], key[2], len(x_train), features.shape[1], len(theta.combiner.seeds),
        )
        return prediction, attribution

    import lightgbm as lgb  # just-in-time: the icir path above never imports it

    params = dict(theta.combiner.params)
    # QA_THREADS caps LightGBM to one instance's fair share of the machine.
    # Theta asks for 20 threads, which oversubscribes an 8-core box on its own
    # and thrashes once several runs share it. Capping is NOT a protocol change:
    # thread count does not affect the fitted model, only how fast it is fitted,
    # so Theta's hash and Property 2 are untouched.
    _threads = os.environ.get("QA_THREADS")
    if _threads:
        try:
            params["num_threads"] = max(1, int(_threads))
        except ValueError:
            pass
    num_boost_round = int(params.pop("num_boost_round", 500))
    # Early stopping needs a held-out set, and Θ forbids fitting or tuning on
    # valid -- so it is deliberately inert here and the model always runs the
    # full boosting schedule. Dropping the key keeps LightGBM from warning
    # about an unused parameter on every single evaluation.
    params.pop("early_stopping_round", None)
    params.update(
        objective=params.pop("loss", "mse"),
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )

    # ---- fit one model per seed and AVERAGE the predictions ----
    # LightGBM's row (`subsample`) and feature (`colsample_bytree`) sampling are
    # both seeded, so a single seed carries real variance. Measured on this data:
    # delta_net_ir had sd 0.016-0.071 against signal of 0.02-0.04, flipping the
    # verdict for 3 of 4 sampled factors. Averaging N cuts that by ~1/sqrt(N),
    # and the fixed seed list keeps the result exactly reproducible.
    preds = []
    for seed in theta.combiner.seeds:
        p_seed = dict(params, seed=int(seed), bagging_seed=int(seed),
                      feature_fraction_seed=int(seed), data_random_seed=int(seed))
        # A fresh Dataset per seed: `data_random_seed` is a *Dataset* parameter
        # baked in at construction, so reusing one handle across seeds raises
        # "Cannot change data_random_seed after constructed Dataset handle".
        # Re-binning costs far less than the 500 boosting rounds that follow.
        model = lgb.train(p_seed, lgb.Dataset(x_train, label=y_train),
                          num_boost_round=num_boost_round)
        preds.append(model.predict(features))

    # ---- predict across the whole window ----
    raw_pred = pd.Series(np.mean(preds, axis=0), index=features.index, name="score")
    wide = raw_pred.unstack(level="instrument")
    wide.index = pd.to_datetime(wide.index)
    prediction = wide.sort_index().reindex(index=panel.dates, columns=panel.instruments)
    prediction = prediction.where(panel.universe)

    _PREDICTION_CACHE[key] = prediction
    _ATTRIBUTION_CACHE[key] = None
    logger.debug(
        "combiner refit: zoo=%s cand=%s theta=%s rows=%d feats=%d seeds=%d",
        key[0], key[1][:8], key[2], len(x_train), features.shape[1], len(theta.combiner.seeds),
    )
    return prediction, None


def clear_cache() -> None:
    _PREDICTION_CACHE.clear()
    _ATTRIBUTION_CACHE.clear()


__all__ = ["clear_cache", "fit_predict", "signal_hash", "zoo_hash"]
