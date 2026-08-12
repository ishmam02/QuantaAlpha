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
    """The close-to-close training label from Θ.

    Kept identical to the baseline's ``Ref($close,-2)/Ref($close,-1)-1``. Note
    this is the *training* target only: realized P&L is computed from the fill
    series in ``execution.py``, which is where the label/fill mismatch is
    resolved.
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


def _per_date_ic(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-date cross-sectional rank IC of each feature with the label.

    Inputs are already cross-sectionally rank-normalised (``_preprocess_v2`` or
    ``_cs_rank_norm``), so Pearson correlation here *is* the Spearman rank IC.
    Returns a ``(n_dates, n_features)`` frame -- one IC row per training date --
    which the bootstrap in :func:`_fit_predict_icir` re-averages per seed.
    """
    label_name = "__ic_label__"
    df = x.assign(**{label_name: y})
    feats = list(x.columns)
    ic = df.groupby(level="datetime", group_keys=False).apply(
        lambda g: g[feats].corrwith(g[label_name])
    )
    return ic


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
    ic_mean = np.nanmean(ic_arr_sample, axis=0)        # mean IC per factor
    ic_std = np.nanstd(ic_arr_sample, axis=0, ddof=1)  # σ_IC per factor
    icir = ic_mean / (ic_std + 1e-8)                   # IC/σ_IC (Ding-Martin IR)
    if shrink == "auto":
        # Ledoit-Wolf: ratio of estimation-error variance to weight dispersion.
        # →1 (equal weight) when ICs are noisy; →0 (raw ICIR) when precise & spread.
        disp = np.nansum((icir - np.nanmean(icir)) ** 2)
        delta = float(np.clip(np.nansum(ic_std ** 2) / (disp + 1e-12), 0.0, 1.0))
    else:
        delta = float(np.clip(shrink, 0.0, 1.0))
    equal = np.full(n_feats, 1.0 / n_feats)
    weights = (1.0 - delta) * icir + delta * equal     # shrink toward 1/N
    weights = np.nan_to_num(weights, nan=0.0)
    return weights, delta


def _fit_predict_icir(
    features: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    theta: Protocol,
    panel: PanelBundle,
) -> pd.DataFrame:
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
    """
    ic_by_date = _per_date_ic(x_train, y_train)
    ic_arr = ic_by_date.to_numpy(dtype=float)          # (n_dates, n_feats)
    n_dates = ic_arr.shape[0]
    feat_mat = features.to_numpy(dtype=float)          # (n_rows, n_feats), full window
    shrink = theta.combiner.params.get("shrinkage", "auto")
    preds = []
    for seed in theta.combiner.seeds:
        rng = np.random.default_rng(int(seed))
        samp = rng.integers(0, n_dates, size=n_dates)  # bootstrap date indices
        weights, _delta = _icir_weights(ic_arr[samp], shrink)
        preds.append(feat_mat @ weights)               # (n_rows,)
    raw_pred = pd.Series(np.mean(preds, axis=0), index=features.index, name="score")
    wide = raw_pred.unstack(level="instrument")
    wide.index = pd.to_datetime(wide.index)
    prediction = wide.sort_index().reindex(index=panel.dates, columns=panel.instruments)
    prediction = prediction.where(panel.universe)
    return prediction


def fit_predict(
    zoo_signals: dict[str, pd.DataFrame],
    candidate_signal: pd.DataFrame | None,
    panel: PanelBundle,
    theta: Protocol,
    candidate_expr: str = "CANDIDATE",
) -> pd.DataFrame:
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
        return cached.reindex(index=panel.dates, columns=panel.instruments)

    # ---- design matrix: base features + every zoo signal + the candidate ----
    columns: dict[str, pd.Series] = {}
    base = _base_features(panel, theta)
    for col in base.columns:
        columns[col] = base[col]
    for i, (expr, sig) in enumerate(sorted(zoo_signals.items())):
        columns[f"ZOO{i}"] = _wide_to_long(sig.reindex(index=panel.dates, columns=panel.instruments), f"ZOO{i}")
    if candidate_signal is not None:
        columns["CAND"] = _wide_to_long(
            candidate_signal.reindex(index=panel.dates, columns=panel.instruments), "CAND"
        )

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
        prediction = _fit_predict_icir(features, x_train, y_train, theta, panel)
        _PREDICTION_CACHE[key] = prediction
        logger.debug(
            "combiner refit (icir): zoo=%s cand=%s theta=%s rows=%d feats=%d seeds=%d",
            key[0], key[1][:8], key[2], len(x_train), features.shape[1], len(theta.combiner.seeds),
        )
        return prediction

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
    logger.debug(
        "combiner refit: zoo=%s cand=%s theta=%s rows=%d feats=%d seeds=%d",
        key[0], key[1][:8], key[2], len(x_train), features.shape[1], len(theta.combiner.seeds),
    )
    return prediction


def clear_cache() -> None:
    _PREDICTION_CACHE.clear()


__all__ = ["clear_cache", "fit_predict", "signal_hash", "zoo_hash"]
