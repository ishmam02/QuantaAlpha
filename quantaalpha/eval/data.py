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


def estimated_dividend_return(start: str, end: str,
                              max_daily: float = 0.25) -> pd.Series:
    """Daily dividend return of the universe.

    The income component is the gap between a TOTAL-return series and a
    PRICE-return series: qlib's ``$close`` is already adjusted, and dividing by
    ``$factor`` recovers the raw quoted price, so::

        dividend_return = mean(adjusted.pct_change()) - mean(raw.pct_change())

    Take the cross-sectional MEAN, not the median. An earlier version used the
    median to stop a handful of ex-dates dominating a day -- but that is exactly
    backwards: dividends are paid on scattered ex-dates by a MINORITY of names,
    so the median is zero on almost every day and the whole correction silently
    evaluated to +0.00%. An index's dividend return IS the weighted average
    across its constituents, so the mean is the estimator that matches it.

    ``max_daily`` rejects only implausible single-day income (bad prints,
    unhandled splits). It must NOT be a quantile trim: dividends come from a
    minority of names on any day, so trimming the tail deletes the ex-dividend
    names themselves.
    """
    _init_qlib()
    from qlib.data import D

    raw = D.features(D.instruments("csi300"), ["$close", "$factor"],
                     start_time=start, end_time=end)
    if raw.empty:
        return pd.Series(dtype=float)
    adj = raw["$close"].unstack(level="instrument").sort_index()
    fac = raw["$factor"].unstack(level="instrument").sort_index()
    adj.index = pd.to_datetime(adj.index)
    fac.index = pd.to_datetime(fac.index)
    px = adj / fac                       # raw quoted price -> price return

    diff = adj.pct_change() - px.pct_change()
    # Guard against DATA ERRORS only, never against the signal. Quantile
    # trimming was tried and removed: dividends are paid by a minority of names
    # on any given day, so trimming the top of the cross-section deletes exactly
    # the ex-dividend names and the estimate collapses (+0.21%/yr against a
    # measured +4.25%/yr). A single-day income component above `max_daily` is a
    # bad print or an unhandled split, not a payout.
    diff = diff.where(diff.abs() <= max_daily)
    div = diff.mean(axis=1).fillna(0.0).clip(lower=0.0)
    div.name = "dividend"
    return div


def equal_weight_benchmark(theta: Protocol, start: str, end: str) -> pd.Series:
    """Equal-weight portfolio of the POINT-IN-TIME index members.

    The matched comparison for a book whose position cap forces near-equal
    weights. Membership is resolved per date from qlib's spells, so a name
    contributes only while it was actually in the index -- survivorship-free,
    the same basis the book trades on.

    Returns are computed on the same price basis as the book (adjusted closes),
    so ``benchmark_basis`` does not apply here: both sides already include
    dividends and there is nothing to correct.
    """
    _init_qlib()
    from qlib.data import D

    market = getattr(theta, "market", None) or "csi300"
    raw = D.features(D.instruments(market), ["$close"], start_time=start, end_time=end)
    if raw.empty:
        raise ValueError(f"no member data for {market} over {start}..{end}")
    close = raw["$close"].unstack(level="instrument").sort_index()
    close.index = pd.to_datetime(close.index)

    ret = close.pct_change()
    ret = ret.where(_membership_mask(market, ret.index, ret.columns))
    out = ret.mean(axis=1)          # equal weight across the members trading that day
    out.name = "benchmark"
    return out


def _membership_mask(market: str, dates: pd.Index, instruments: pd.Index) -> pd.DataFrame:
    """Point-in-time membership from qlib's spell file.

    ``D.features`` returns the UNION of everything ever in the index -- measured
    at a median 447 names for a 300-name index -- so weighting or averaging its
    output without this mask silently uses a larger, different universe.
    """
    path = Path.home() / ".qlib/qlib_data/cn_data/instruments" / f"{market}.txt"
    mask = pd.DataFrame(False, index=dates, columns=instruments)
    if not path.exists():
        return ~mask.astype(bool) | True        # no spell file -> keep everything
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        inst, lo, hi = parts[0], pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
        if inst in mask.columns:
            mask.loc[(mask.index >= lo) & (mask.index <= hi), inst] = True
    return mask


def load_benchmark(theta: Protocol, start: str, end: str) -> pd.Series:
    """Daily simple return of the protocol's benchmark (default SH000300).

    Honours ``theta.benchmark_basis``. The book prices with ADJUSTED closes --
    dividends reinvested -- so a price-return benchmark hands the strategy the
    market's whole dividend yield as if it were alpha (+4.25 pp/yr on CSI300
    2019-2021). ``estimated_total`` puts both sides on the same basis.
    """
    if str(getattr(theta, "benchmark_construction", "index")).lower() == "equal":
        return equal_weight_benchmark(theta, start, end)

    _init_qlib()
    from qlib.data import D

    raw = D.features([theta.benchmark], ["$close"], start_time=start, end_time=end)
    if raw.empty:
        raise ValueError(f"no benchmark data for {theta.benchmark} over {start}..{end}")
    close = raw["$close"].droplevel("instrument")
    close.index = pd.to_datetime(close.index)
    ret = close.sort_index().pct_change()

    basis = str(getattr(theta, "benchmark_basis", "price")).strip().lower()
    if basis in ("estimated_total", "total"):
        div = estimated_dividend_return(start, end)
        if not div.empty:
            ret = ret.add(div.reindex(ret.index).fillna(0.0), fill_value=0.0)
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


def aligned_cache_path(expression: str, panel: PanelBundle,
                       cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Path of a factor's signal ALREADY aligned to ``panel``'s grid.

    Keyed by the expression AND the grid it was aligned to (first date, last
    date, instrument count) -- an aligned frame is only reusable on the panel it
    was cut for, so a protocol re-split must miss rather than silently return a
    frame with the wrong dates.
    """
    root = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR) / "aligned"
    dates = panel.dates
    grid = (f"{pd.Timestamp(dates[0]).date()}_{pd.Timestamp(dates[-1]).date()}"
            f"_{len(dates)}x{len(panel.instruments)}")
    key = hashlib.md5(f"{expression}||{grid}".encode()).hexdigest()
    return root / f"{key}.pkl"


def load_aligned_signal(expression: str, panel: PanelBundle,
                        cache_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """``align_signal(load_factor_signal(expr), panel)`` with the RESULT cached.

    Why this exists. A cached signal is a ``(datetime, instrument)`` Series over
    the whole 4138 x 5982 factor cache -- 14.2M rows. Putting it on the panel
    grid costs a full ``unstack`` of that MultiIndex, measured at 3.43s, and
    ``_align`` then discards 95% of the result to reach 2610 x 668. The
    repository rehydrates EVERY incumbent on EVERY batch, so that 3.4s is paid
    per factor per batch: measured 5.0s/factor across 59 live batches, making
    backtest time ``100s + 5.0s * |zoo|`` -- the search gets slower the better
    it does.

    The aligned frame is ~14 MB and depends only on (expression, grid), both of
    which are immutable for a run. Caching it turns the per-batch cost into a
    ~0.05s read and makes batch time flat in ``|zoo|``.

    Falls back to computing (and, on any write failure, to simply returning the
    computed frame) so a cache problem can never block an evaluation.
    """
    path = aligned_cache_path(expression, panel, cache_dir)
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:      # corrupt/partial file -- recompute over it
            logger.warning("aligned-signal cache unreadable, recomputing: %s", path.name)

    frame = align_signal(load_factor_signal(expression, cache_dir), panel)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")   # atomic: a reader never sees a partial file
        frame.to_pickle(tmp)
        tmp.replace(path)
    except Exception as exc:
        logger.warning("could not cache aligned signal (%s); continuing uncached", exc)
    return frame


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
    "aligned_cache_path",
    "load_aligned_signal",
    "factor_cache_path",
    "load_benchmark",
    "load_factor_signal",
    "load_panel",
]
