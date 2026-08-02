"""Panel loading for ``E_Θ``.

The mining panel (``daily_pv.h5``) has no ``$amount``, no ``$vwap`` and no
benchmark series, so the evaluation engine sources prices, ADV and the
benchmark from **Qlib** instead. That also keeps ``E_Θ`` point-in-time
auditable independently of the mining data.

Everything downstream consumes *aligned* wide ``(T × N)`` frames produced by
:func:`_align`: the same date index, the same instrument columns, and
non-members masked to ``NaN``. Nothing in ``execution``/``costs``/``metrics``
should ever have to think about index alignment again.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from quantaalpha.eval.protocol import Protocol

logger = logging.getLogger(__name__)

# Mirrors factors/library.py:19-21 so we read the same cache the miner writes.
DEFAULT_FACTOR_CACHE_DIR = os.environ.get("FACTOR_CACHE_DIR", "data/results/factor_cache")

_RAW_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap", "$factor"]


@dataclass(frozen=True)
class PanelBundle:
    """Aligned market data for one window.

    All frames share ``(dates × instruments)``. ``universe`` is the
    point-in-time membership mask; every price frame is already masked by it.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    vwap: pd.DataFrame
    factor: pd.DataFrame
    universe: pd.DataFrame

    @property
    def dates(self) -> pd.Index:
        return self.close.index

    @property
    def instruments(self) -> pd.Index:
        return self.close.columns


def _init_qlib() -> None:
    """Initialize Qlib once per process, idempotently.

    Mirrors ``backtest/runner.py:39-51``. ``QLIB_PROVIDER_URI`` wins so a
    caller can point the engine at a different data root without editing Θ —
    the data *location* is not a protocol field, only the data *semantics* are.
    """
    import platform
    import sys

    import qlib
    from qlib.config import C

    if getattr(C, "_registered", False):
        return
    provider_uri = os.path.expanduser(
        os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data")
    )
    region = os.environ.get("QLIB_REGION", "cn")

    # Qlib fans feature loading out over `multiprocessing`. On macOS the start
    # method is `spawn`, which re-imports __main__ in each child -- that blows
    # up when the interpreter was fed from stdin (no importable __main__), and
    # nests badly under the mining loop's own parallel executor. Single-kernel
    # loading sidesteps both; override with QLIB_KERNELS if you want the pool.
    default_kernels = 1 if (platform.system() == "Darwin" or not getattr(sys.modules.get("__main__"), "__file__", None)) else 0
    kernels = int(os.environ.get("QLIB_KERNELS", default_kernels))

    init_kwargs = {"provider_uri": provider_uri, "region": region}
    if kernels > 0:
        init_kwargs["kernels"] = kernels
    qlib.init(**init_kwargs)
    logger.info(
        "Qlib initialized for E_theta: %s (region=%s, kernels=%s)", provider_uri, region, kernels or "default"
    )


def _long_to_wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """``(datetime, instrument)`` MultiIndex column → wide ``(T × N)``."""
    wide = df[column].unstack(level="instrument")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _align(df: pd.DataFrame, panel: PanelBundle, *, mask: bool = True) -> pd.DataFrame:
    """Reindex an arbitrary frame onto the panel's grid.

    ``mask=True`` additionally NaNs out non-members, which is what keeps a
    factor from being scored on names the strategy could never have held.
    """
    out = df.reindex(index=panel.dates, columns=panel.instruments)
    if mask:
        out = out.where(panel.universe)
    return out


def _membership_mask(
    instruments_config: object, dates: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """Point-in-time CSI 300 membership as a boolean ``(T × N)`` mask.

    Qlib's ``D.list_instruments`` returns, per instrument, the list of
    ``(start, end)`` spans during which it was a constituent. Using those spans
    rather than a static end-of-sample list is what prevents survivorship bias
    from entering every downstream metric.
    """
    from qlib.data import D

    spans = D.list_instruments(
        instruments_config,
        start_time=dates.min(),
        end_time=dates.max(),
        as_list=False,
    )
    mask = pd.DataFrame(False, index=dates, columns=columns)
    for inst, ranges in spans.items():
        if inst not in mask.columns:
            continue
        col = mask[inst].copy()
        for start, end in ranges:
            col.loc[pd.Timestamp(start) : pd.Timestamp(end)] = True
        mask[inst] = col
    return mask


def load_panel(theta: Protocol, start: str, end: str) -> PanelBundle:
    """Load the aligned market panel for ``[start, end]``.

    Prices are returned **as reported** (not back-adjusted); the ``factor``
    frame carries Qlib's cumulative adjustment factor so that
    ``execution.fill_prices`` can build a split-consistent fill series.
    """
    _init_qlib()
    from qlib.data import D

    instruments = D.instruments(theta.market)
    raw = D.features(instruments, _RAW_FIELDS, start_time=start, end_time=end)
    if raw.empty:
        raise ValueError(f"Qlib returned no data for {theta.market} over {start}..{end}")

    frames = {f.lstrip("$"): _long_to_wide(raw, f) for f in _RAW_FIELDS}
    dates = frames["close"].index
    columns = frames["close"].columns

    universe = _membership_mask(instruments, dates, columns)
    # A name with no price on a date cannot be traded on it, whatever the
    # index membership file says.
    universe = universe & frames["close"].notna()

    bundle = PanelBundle(
        open=frames["open"],
        high=frames["high"],
        low=frames["low"],
        close=frames["close"],
        volume=frames["volume"],
        amount=frames["amount"],
        vwap=frames["vwap"],
        factor=frames["factor"],
        universe=universe,
    )
    # Mask prices to the tradeable universe up front so no caller can forget.
    masked = {
        name: getattr(bundle, name).where(universe)
        for name in ("open", "high", "low", "close", "volume", "amount", "vwap", "factor")
    }
    return PanelBundle(universe=universe, **masked)


def load_benchmark(theta: Protocol, start: str, end: str) -> pd.Series:
    """Daily simple return of the protocol's benchmark (default SH000300)."""
    _init_qlib()
    from qlib.data import D

    raw = D.features([theta.benchmark], ["$close"], start_time=start, end_time=end)
    if raw.empty:
        raise ValueError(f"no benchmark data for {theta.benchmark} over {start}..{end}")
    close = raw["$close"].droplevel("instrument")
    close.index = pd.to_datetime(close.index)
    ret = close.sort_index().pct_change()
    ret.name = "benchmark"
    return ret


def factor_cache_path(expression: str, cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Path of a mined factor's cached signal (``library.py:153-176``)."""
    root = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)
    return root / f"{hashlib.md5(expression.encode()).hexdigest()}.pkl"


@lru_cache(maxsize=64)
def _read_cached_signal(path: str):
    """Load a cached signal, in whichever shape it was stored.

    Two shapes are legitimate. A signal cached straight from computation is a
    ``(datetime, instrument)`` MultiIndex Series, sometimes wrapped in a
    one-column frame. A signal cached after alignment is a WIDE
    ``(dates x instruments)`` frame.

    The unwrap must therefore be conditional. Taking ``.iloc[:, 0]``
    unconditionally reduces a wide frame to a single instrument's column --
    a DatetimeIndex Series, which ``align_signal`` then rejects outright with
    "expected a (datetime, instrument) MultiIndex Series". That is exactly what
    happened when the cache was compacted to the aligned form: every consumer
    broke, while a test calling ``align_signal`` on the pickle directly passed,
    because it never went through this function.
    """
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex) or obj.shape[1] == 1:
            obj = obj.iloc[:, 0]        # a Series in a one-column wrapper
        # else: a wide, already-aligned frame -- pass it through untouched
    return obj


def load_factor_signal(
    expression: str,
    cache_dir: str | os.PathLike[str] | None = None,
    h5_path: str | os.PathLike[str] | None = None,
) -> pd.Series:
    """Load a mined factor's signal as a ``(datetime, instrument)`` Series.

    Prefers the MD5 pickle cache the library maintains; falls back to a
    workspace ``result.h5``. Returns the *unaligned* signal — call
    :func:`align_signal` to put it on a panel grid.
    """
    path = factor_cache_path(expression, cache_dir)
    if path.exists():
        return _read_cached_signal(str(path))
    if h5_path is not None and Path(h5_path).exists():
        obj = pd.read_hdf(str(h5_path))
        return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    raise FileNotFoundError(
        f"no cached signal for expression (md5={path.stem}); looked in {path}"
        + (f" and {h5_path}" if h5_path else "")
    )


def align_signal(signal: pd.Series | pd.DataFrame, panel: PanelBundle) -> pd.DataFrame:
    """Put a factor signal onto the panel grid as a wide ``(T × N)`` frame."""
    if isinstance(signal, pd.DataFrame):
        if isinstance(signal.index, pd.MultiIndex):
            signal = signal.iloc[:, 0]
        else:
            return _align(signal, panel)
    if isinstance(signal.index, pd.MultiIndex):
        wide = signal.unstack(level=-1)
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
    else:
        raise TypeError("expected a (datetime, instrument) MultiIndex Series")
    return _align(wide, panel)


__all__ = [
    "DEFAULT_FACTOR_CACHE_DIR",
    "PanelBundle",
    "align_signal",
    "factor_cache_path",
    "load_benchmark",
    "load_factor_signal",
    "load_panel",
]
