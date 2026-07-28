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

    import lightgbm as lgb

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

    params = dict(theta.combiner.params)
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
