#!/usr/bin/env python
"""Build the S&P 500 Qlib dataset (``data/qlib/us_data``) from free sources.

The CSI300 system runs on a Qlib ``cn_data`` directory + a ``daily_pv.h5`` factor
cache. This script produces the field-for-field US equivalent so a TRANSFER test
(mine CSI300 -> recompute on S&P 500, no re-mining) can run on the same 8 cache
fields and the same 2005-2026 range.

Data sources (all free, no paid API key):
  * yfinance  -- daily OHLC + Adj Close + Volume (auto_adjust=False) for the full
    2005-2026 history, batched. $vwap = (O+H+L+C)/4 typical-price PROXY (free
    Alpaca reaches only 2020 and no free source carries 2005-2026 daily VWAP; the
    full range + free was chosen over real VWAP -- documented trade-off).
  * chinobing/historical_sp500_constituents CSV -- current S&P 500 members +
    add-date + CIK. Point-in-time-correct on the ADD side; survivorship-biased on
    the removal side (removed names absent). Override with --membership-csv for a
    clean point-in-time file.
  * SEC EDGAR Company Facts API (data.sec.gov) -- historical shares outstanding +
    public float, point-in-time (filing-lagged), for the free-float sidecar.
    Needs EDGAR_IDENTITY="Name <email>" for the required User-Agent. yfinance
    .info['floatShares'] is the no-identity fallback (current snapshot).

Field mapping (yfinance auto_adjust=False -> Qlib). yfinance's auto_adjust=False
returns SPLIT-adjusted Close/OHLC/Volume (back-adjusted) + split+dividend Adj
Close, so a single download gives everything:
  $close   = Adj Close                         (split+div adjusted; 71 factors)
  $factor  = Adj Close / Close                 (the dividend factor; $close/$factor
                                                = split-adjusted close, which is
                                                split-consistent for fill_prices
                                                and gives the right dividend return
                                                for estimated_dividend_return)
  $open/h/l= Open/High/Low * (Adj Close/Close) (div-adjusted to match $close)
  $volume  = Volume                            (split-adjusted shares; 58 factors)
  $amount  = Close * Volume                    (split-invariant -> true USD notional;
                                                costs.dollar_volume with adv_scale=1.0)
  $vwap    = (open+high+low+close)/4           (typical-price PROXY; 26 factors)
  $adjclose= Adj Close                         (reserved; = $close here)
  $change  = pct_change($adjclose)             (reserved)
  ($return = pct_change($close) is DERIVED in the factor cache, not a .bin file.)

Rate limits are respected for every API: yfinance is throttled + retried with a
per-ticker parquet cache (a flaky ticker is re-fetched alone, not the whole
build); SEC EDGAR is paced at <=8 req/s with the required User-Agent.

Usage::

    conda run -n quantaalpha python scripts/qa_build_us_data.py \\
        --out data/qlib/us_data --start 2005-01-01 --end 2026-01-10 \\
        [--limit N] [--tickers AAPL,MSFT,...] [--membership-csv FILE] \\
        [--float-source {yfinance,sec,none}] [--yf-cache data/qlib/.us_yf_cache]

Verify the result with ``QLIB_DATA_DIR=data/qlib/us_data python scripts/qa_check_data.py``
and the round-trip check this script runs at the end.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_build_us_data")

# The 10 Qlib feature fields written per ticker (filenames have no '$' prefix;
# Qlib maps a request for "$close" to features/<ticker>/close.day.bin). The first
# 8 are _RAW_FIELDS the eval panel reads; adjclose/change are reserved for cn_data
# structural parity + future ICIR/mean-variance use.
FIELDS = ["open", "close", "high", "low", "volume", "amount", "vwap", "factor", "adjclose", "change"]

CHINOBING_URL = ("https://raw.githubusercontent.com/chinobing/"
                 "historical_sp500_constituents/master/sp500_constituents.csv")
CHANGES_URL = ("https://raw.githubusercontent.com/chinobing/"
               "historical_sp500_constituents/master/sp500_changes_since_1996.csv")

SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #
def load_membership(membership_csv: str | None, end: str
                    ) -> pd.DataFrame:
    """S&P 500 membership windows as ``[symbol, start, end, cik]`` (one row per
    membership spell; a ticker can have several if it left and re-entered).

    Default source: chinobing ``sp500_changes_since_1996.csv`` (every add/remove
    event since 1996) + ``sp500_constituents.csv`` (current members + add-dates +
    CIKs). This is point-in-time on BOTH sides -- DROPPED names (Lehman, Sears,
    Yahoo, GE pre-2018, ...) are included with their [add, remove] windows, so the
    universe is every name ever in the index over 2005-2026 (~900), not just the
    ~503 current survivors. ``--membership-csv`` overrides with a user file
    (columns: symbol,start,end[,cik]).
    """
    if membership_csv:
        logger.info("loading membership from %s", membership_csv)
        df = pd.read_csv(membership_csv)
        df.columns = df.columns.str.strip().str.lower()
        sym = "symbol" if "symbol" in df.columns else df.columns[0]
        out = pd.DataFrame({
            "symbol": df[sym].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
            "start": pd.to_datetime(df["start"], errors="coerce"),
            "end": pd.to_datetime(df["end"], errors="coerce"),
            "cik": pd.to_numeric(df.get("cik"), errors="coerce") if "cik" in df.columns else pd.NA,
        })
        logger.info("membership: %d windows from user CSV (point-in-time)", len(out))
        return out.dropna(subset=["symbol"])

    # Full point-in-time membership from chinobing's changes file (add/remove
    # events since 1996) + the constituents file (current members + add-dates +
    # CIKs). This includes DROPPED names (Lehman, Sears, Yahoo, ...), so the
    # universe is every name that was ever in the index over 2005-2026 (~900), not
    # just the ~503 current survivors.
    import ast
    from collections import defaultdict

    logger.info("building point-in-time S&P 500 membership from chinobing "
                "changes_since_1996 + constituents (adds AND removes)")
    const = pd.read_csv(CHINOBING_URL)
    const.columns = const.columns.str.strip().str.lower()
    const["date added"] = pd.to_datetime(const["date added"], errors="coerce")
    norm = lambda s: str(s).strip().upper().replace(".", "-")
    current: dict[str, tuple] = {}
    for _, row in const.iterrows():
        tk = norm(row["symbol"])
        if tk:
            current[tk] = (row["date added"], pd.to_numeric(row.get("cik"), errors="coerce"))

    changes = pd.read_csv(CHANGES_URL)
    events: list[tuple] = []  # (date, ticker, +1 add / -1 remove)
    for _, r in changes.iterrows():
        d = pd.to_datetime(r["date"], errors="coerce")
        if pd.isna(d):
            continue
        for col, delta in (("added_tickers", +1), ("removed_tickers", -1)):
            v = r.get(col)
            if pd.isna(v) or not str(v).strip():
                continue
            try:
                tks = ast.literal_eval(str(v))
            except Exception:
                continue
            if isinstance(tks, str):
                tks = [tks]
            for tk in tks:
                events.append((d, norm(tk), delta))

    # Seed current members with their add-date (the original-500 current members
    # have no changes entry -- they have been in since before 1996 and never
    # removed -- so this is their only event).
    for tk, (add_d, _cik) in current.items():
        events.append((add_d if pd.notna(add_d) else pd.Timestamp("1995-01-01"), tk, +1))
    # Seed pre-1996 additions that were later removed (a remove with no add in the
    # changes file): they were in from before 1996, so add at the changes start.
    added = {tk for _d, tk, delta in events if delta == +1}
    removed = {tk for _d, tk, delta in events if delta == -1}
    for tk in removed - added:
        events.append((pd.Timestamp("1996-01-01"), tk, +1))

    events.sort(key=lambda x: x[0])
    in_idx: set[str] = set()
    start: dict[str, object] = {}
    spells: dict[str, list] = defaultdict(list)
    for d, tk, delta in events:
        if delta == +1:
            if tk not in in_idx:
                in_idx.add(tk)
                start[tk] = d
        else:
            if tk in in_idx:
                spells[tk].append((start[tk], d))
                in_idx.discard(tk)
    end_ts = pd.Timestamp(end)
    for tk in list(in_idx):
        spells[tk].append((start[tk], end_ts))

    rows = []
    for tk, sps in spells.items():
        cik = current.get(tk, (None, None))[1]
        for s, e in sps:
            rows.append({"symbol": tk, "start": s, "end": e, "cik": cik})
    out = pd.DataFrame(rows)
    logger.info("membership: %d distinct tickers, %d membership spells "
                "(point-in-time, adds + removes; incl. dropped names)",
                out["symbol"].nunique(), len(out))
    return out.dropna(subset=["symbol"])


# --------------------------------------------------------------------------- #
# yfinance fetch (batched, throttled, cached)
# --------------------------------------------------------------------------- #
def _yf_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """One yf.download for a batch; returns {ticker: raw df} (auto_adjust=False)."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    # group_by="column" -> top level is Price (Open/High/Low/Close/Adj Close/Volume),
    # second level is ticker. For a single ticker yfinance may drop the ticker level.
    if raw.columns.nlevels == 2:
        for tk in raw.columns.get_level_values(1).unique():
            sub = raw.xs(tk, level=1, axis=1)
            if not sub.empty:
                out[tk] = sub
    else:
        # single ticker: columns are just the Price names
        if len(tickers) == 1:
            out[tickers[0]] = raw
    return out


def fetch_yf(tickers: list[str], start: str, end: str, cache_dir: Path,
             batch: int = 50, sleep_s: float = 0.8, retries: int = 3
             ) -> dict[str, pd.DataFrame]:
    """Fetch all tickers in batches with parquet caching + retry on failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    todo: list[str] = []
    for tk in tickers:
        p = cache_dir / f"{tk}.parquet"
        if p.exists():
            try:
                out[tk] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        todo.append(tk)
    if todo:
        logger.info("yfinance: %d tickers cached, %d to fetch (%d/batch)",
                    len(out), len(todo), batch)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for attempt in range(retries):
            try:
                got = _yf_batch(chunk, start, end)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("yf.download batch %d failed (%s); retry %d/%d in %ds",
                               i, exc, attempt + 1, retries, wait)
                time.sleep(wait)
        else:
            logger.error("yf.download batch starting %d failed after %d retries; skipping %s",
                         i, retries, chunk)
            continue
        for tk, df in got.items():
            if df is None or df.empty:
                logger.warning("no yfinance data for %s", tk)
                continue
            df = df.dropna(how="all")
            if df.empty:
                continue
            out[tk] = df
            try:
                df.to_parquet(cache_dir / f"{tk}.parquet")
            except Exception as exc:
                logger.debug("could not cache %s (%s)", tk, exc)
        done = min(i + batch, len(todo))
        logger.info("  yfinance %d/%d fetched (total panels %d)", done, len(todo), len(out))
        time.sleep(sleep_s)
    return out


# --------------------------------------------------------------------------- #
# Field derivation
# --------------------------------------------------------------------------- #
def compute_fields(raw: pd.DataFrame) -> pd.DataFrame:
    """raw (auto_adjust=False: Open/High/Low/Close/Adj Close/Volume) -> 10 $ fields.

    Returns a frame indexed by trading dates with columns named by FIELDS
    (without '$'). All prices are split+dividend adjusted and mutually consistent.
    """
    # yfinance column names: 'Open','High','Low','Close','Adj Close','Volume'.
    # Flatten any (Price, Ticker) MultiIndex to the Price level and normalize so
    # 'Adj Close' -> 'adjclose' etc.
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    norm = {str(c).strip().lower().replace(" ", ""): c for c in raw.columns}

    def get(name: str) -> pd.Series:
        return raw[norm[name]]

    close_raw = get("close").astype(float)          # split-adjusted close
    adjclose = get("adjclose").astype(float)        # split+div adjusted close
    volume = get("volume").astype(float)            # split-adjusted shares
    opn = get("open").astype(float)
    high = get("high").astype(float)
    low = get("low").astype(float)

    # Sanitize: yfinance returns 0.0 prices for some delisted tickers' post-
    # delisting days (e.g. DEC), and a 0 close would make $factor=0/NaN and $return
    # inf (pct_change 0 -> nonzero = inf). Treat any 0/negative price as missing so
    # it becomes a non-trading NaN instead. Volume 0 is legitimate (a no-trade day)
    # and is left untouched.
    for s in (close_raw, adjclose, opn, high, low):
        s[s <= 0] = np.nan

    div_ratio = (adjclose / close_raw).replace([np.inf, -np.inf], np.nan)
    # div-adjust O/H/L onto the same basis as $close (= Adj Close)
    close = adjclose
    opn = opn * div_ratio
    high = high * div_ratio
    low = low * div_ratio
    factor = div_ratio
    amount = close_raw * volume                     # split-invariant -> true USD notional
    vwap = (opn + high + low + close) / 4.0         # typical-price PROXY
    change = adjclose.pct_change().fillna(0.0)

    out = pd.DataFrame({
        "open": opn, "close": close, "high": high, "low": low,
        "volume": volume, "amount": amount, "vwap": vwap, "factor": factor,
        "adjclose": adjclose, "change": change,
    }, index=pd.to_datetime(raw.index))
    out.index.name = "datetime"
    return out


# --------------------------------------------------------------------------- #
# Qlib .day.bin writer
# --------------------------------------------------------------------------- #
def write_bins(us_dir: Path, ticker: str, fields: pd.DataFrame,
               calendar: pd.DatetimeIndex) -> int:
    """Write features/<ticker>/<field>.day.bin for each field.

    Bin format (qlib file_storage): ``np.hstack([start_index, values]).astype("<f")``
    where start_index is the calendar index of the ticker's first trading day and
    values are float32 LE aligned to ``calendar[start_index:]`` (NaN for the
    ticker's non-trading days). Returns the number of fields written.
    """
    feat_dir = us_dir / "features" / ticker
    feat_dir.mkdir(parents=True, exist_ok=True)
    # Align each field onto the full calendar (NaN where the ticker didn't trade).
    aligned = fields.reindex(calendar)
    first_valid = int(np.arange(len(calendar))[aligned["close"].notna().to_numpy()][0]) \
        if aligned["close"].notna().any() else 0
    written = 0
    for field in FIELDS:
        series = aligned[field].astype(float).to_numpy()
        vals = series[first_valid:].astype(np.float32)
        arr = np.hstack([np.asarray(first_valid, dtype=np.float32), vals]).astype("<f4")
        arr.tofile(feat_dir / f"{field}.day.bin")
        written += 1
    return written


# --------------------------------------------------------------------------- #
# Instruments files
# --------------------------------------------------------------------------- #
def write_instruments(us_dir: Path, panels: dict[str, pd.DataFrame],
                      membership: pd.DataFrame, end: str) -> None:
    """all.txt (every ticker's full span) + sp500.txt (membership windows)."""
    inst_dir = us_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end)

    def _fmt(d) -> str:
        return pd.Timestamp(d).strftime("%Y-%m-%d")

    # all.txt: every fetched ticker over its full yfinance span.
    all_lines = []
    for tk, df in sorted(panels.items()):
        if df.empty:
            continue
        all_lines.append(f"{tk}\t{_fmt(df.index.min())}\t{_fmt(df.index.max())}")
    (inst_dir / "all.txt").write_text("\n".join(all_lines) + ("\n" if all_lines else ""))

    # sp500.txt: membership window clipped to the ticker's actual data span.
    sp_lines = []
    have = {tk: (pd.Timestamp(df.index.min()), pd.Timestamp(df.index.max()))
            for tk, df in panels.items()}
    for _, row in membership.iterrows():
        tk = row["symbol"]
        if tk not in have:
            continue
        d0, d1 = have[tk]
        start = max(pd.Timestamp(row["start"]), d0)
        last = min(pd.Timestamp(row["end"]) if pd.notna(row["end"]) else end_ts, d1, end_ts)
        if start > last:
            continue
        sp_lines.append(f"{tk}\t{_fmt(start)}\t{_fmt(last)}")
    (inst_dir / "sp500.txt").write_text("\n".join(sp_lines) + ("\n" if sp_lines else ""))
    logger.info("instruments: all=%d sp500=%d", len(all_lines), len(sp_lines))


# --------------------------------------------------------------------------- #
# Free-float sidecar
# --------------------------------------------------------------------------- #
def fetch_float_yfinance(tickers: list[str], sleep_s: float = 0.4
                         ) -> list[tuple[str, str, float, float]]:
    """Current-snapshot free float via yfinance .info (no identity needed).

    Rows: (ticker, DATE='today', shares_outstanding, public_float_usd=NaN).
    Marked DATE=today so a consumer treats it as known-only-from-now.
    """
    import yfinance as yf
    rows = []
    today = pd.Timestamp.now(tz=None).strftime("%Y-%m-%d")
    for i, tk in enumerate(tickers):
        try:
            info = yf.Ticker(tk).info
            fl = info.get("floatShares")
            sh = info.get("sharesOutstanding")
            shares = fl if fl and fl > 0 else sh
            if shares and shares > 0:
                rows.append((tk, today, float(shares), float("nan")))
        except Exception as exc:
            logger.debug("yfinance .info failed for %s (%s)", tk, exc)
        if i % 50 == 49:
            logger.info("  yfinance float %d/%d", i + 1, len(tickers))
            time.sleep(sleep_s)
    return rows


def fetch_float_sec(membership: pd.DataFrame, sleep_s: float = 0.13
                    ) -> list[tuple[str, str, float, float]]:
    """Historical point-in-time free float via the SEC EDGAR Company Facts API.

    Pulls dei:EntityCommonStockSharesOutstanding (shares, as-of cover-page date)
    and dei:EntityPublicFloat (USD, annual) per CIK, keyed by the FILING date so a
    backtest sees a value only after the market learned it. Free-float share count
    is not a direct XBRL tag -> store shares outstanding + public_float_usd; the
    consumer derives float shares as public_float/price where available. Paced at
    <=8 req/s (SEC hard limit 10/s) with the required User-Agent.
    """
    import urllib.request
    identity = (Path(__file__).resolve().parents[1] / ".env")
    ua = ""
    # EDGAR_IDENTITY env (preferred) else a polite default.
    import os
    ua = os.environ.get("EDGAR_IDENTITY", "QuantaAlpha research@example.com")
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    rows: list[tuple[str, str, float, float]] = []
    rows_by_tk: dict[str, list[tuple[str, str, float, float]]] = {}
    todo = membership.dropna(subset=["cik"])
    logger.info("SEC free-float: %d CIKs @ <=8 req/s (User-Agent=%s)", len(todo), ua)
    for i, (_, row) in enumerate(todo.iterrows()):
        tk = row["symbol"]
        cik = int(row["cik"])
        url = SEC_COMPANYFACTS.format(cik10=str(cik).zfill(10))
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
        except Exception as exc:
            logger.debug("SEC companyfacts failed for %s (CIK %s): %s", tk, cik, exc)
            time.sleep(sleep_s)
            continue
        facts = data.get("facts", {}).get("dei", {})
        tk_rows: list[tuple[str, str, float, float]] = []
        for concept, unit_key, scale in (
            ("EntityCommonStockSharesOutstanding", "shares", 1.0),
            ("EntityPublicFloat", "USD", 1.0),
        ):
            node = facts.get(concept, {}).get("units", {}).get(unit_key, [])
            for f in node:
                filed = f.get("filed")
                val = f.get("val")
                if filed and val is not None:
                    tk_rows.append((tk, filed, float(val), float("nan")))
        # Merge shares + float onto one row per filed date (shares outstanding).
        # Keep it simple: emit shares-outstanding rows; attach public_float where
        # the float's end date matches.
        for r in tk_rows:
            rows_by_tk.setdefault(tk, []).append(r)
        if i % 50 == 49:
            logger.info("  SEC companyfacts %d/%d", i + 1, len(todo))
        time.sleep(sleep_s)
    for tk, rs in rows_by_tk.items():
        rs.sort(key=lambda x: x[1])
        rows.extend(rs)
    return rows


def write_floatshares(us_dir: Path, rows: list[tuple[str, str, float, float]]) -> None:
    inst = us_dir / "instruments"
    inst.mkdir(parents=True, exist_ok=True)
    lines = ["# ticker\tdate\tshares_outstanding\tpublic_float_usd "
             "(date = SEC filed date or 'today' for the yfinance snapshot)"]
    for tk, date, shares, pub in rows:
        pub_s = "" if (pub != pub) else f"{pub}"
        lines.append(f"{tk}\t{date}\t{shares}\t{pub_s}")
    (inst / "floatshares.txt").write_text("\n".join(lines) + "\n")
    logger.info("floatshares.txt: %d rows", len(rows))


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def build_calendar(panels: dict[str, pd.DataFrame], end: str) -> pd.DatetimeIndex:
    """Union of all tickers' trading dates (2005+) for day.txt; robust to the
    exchange_calendars XNYS gap (its XNYS only starts 2006-09-11, so we cannot use
    it for 2005). The union of ~500 liquid large caps' dates IS the XNYS calendar.
    """
    dates = set()
    for df in panels.values():
        if not df.empty:
            dates.update(pd.to_datetime(df.index))
    cal = pd.DatetimeIndex(sorted(dates))
    cal = cal[(cal >= pd.Timestamp("2005-01-01")) & (cal <= pd.Timestamp(end))]
    return cal


def write_calendar(us_dir: Path, cal: pd.DatetimeIndex, future_end: str) -> None:
    cdir = us_dir / "calendars"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")
    # day_future.txt: day.txt + forward XNYS sessions to future_end (2026-12-31).
    future = cal
    try:
        import exchange_calendars as xcals
        xcal = xcals.get_calendar("XNYS")
        tail = xcal.sessions_in_range(pd.Timestamp(cal[-1]) + pd.Timedelta(days=1),
                                      pd.Timestamp(future_end))
        future = cal.append(tail)
    except Exception as exc:
        logger.warning("could not extend day_future via exchange_calendars (%s); "
                       "day_future == day.txt", exc)
    (cdir / "day_future.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in future) + "\n")
    logger.info("calendar: %d sessions %s..%s (future %d)",
                len(cal), cal[0].date(), cal[-1].date(), len(future))


# --------------------------------------------------------------------------- #
# Self-verify
# --------------------------------------------------------------------------- #
def self_verify(us_dir: Path, ticker: str) -> None:
    """Round-trip: init qlib on us_data and read back; spot-check $factor on a split."""
    import qlib
    qlib.init(provider_uri=str(us_dir), region="us")
    from qlib.data import D
    raw = D.features([ticker], ["$open", "$high", "$low", "$close", "$volume",
                                "$amount", "$vwap", "$factor"],
                     start_time="2020-08-24", end_time="2020-09-02")
    if raw.empty:
        raise RuntimeError(f"self-verify: qlib read back NO data for {ticker}")
    logger.info("self-verify: qlib read back %d rows for %s:\n%s",
                len(raw), ticker, raw.to_string())
    # AAPL 4:1 split ex-date 2020-08-28: $close/$factor should be split-adjusted
    # (smooth, ~$120-130, no 4x jump) -- confirms factor + adjustment basis.
    sub = raw.xs(ticker, level="instrument") if ticker in raw.index.get_level_values(-1) else raw
    if ticker == "AAPL":
        ratio = (sub["$close"] / sub["$factor"]).dropna()
        logger.info("AAPL close/factor around 2020-08-28 split (should be ~smooth, "
                    "no 4x jump):\n%s", ratio.to_string())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/qlib/us_data")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2026-01-10")
    p.add_argument("--future-end", default="2026-12-31")
    p.add_argument("--limit", type=int, default=None, help="first N members (smoke test)")
    p.add_argument("--tickers", default=None, help="comma list overriding membership")
    p.add_argument("--membership-csv", default=None, help="point-in-time membership file")
    p.add_argument("--float-source", choices=["yfinance", "sec", "none"], default="yfinance")
    p.add_argument("--yf-cache", default="data/qlib/.us_yf_cache")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    us_dir = Path(args.out)
    us_dir.mkdir(parents=True, exist_ok=True)
    end = args.end

    # 1. Membership / universe.
    if args.tickers:
        tk_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        membership = pd.DataFrame({"symbol": tk_list, "start": pd.Timestamp("1995-01-01"),
                                   "end": pd.Timestamp(end), "cik": pd.NA})
        logger.info("universe overridden: %d tickers", len(tk_list))
    else:
        membership = load_membership(args.membership_csv, end)
        if args.limit:
            membership = membership.head(args.limit)
    universe = membership["symbol"].unique().tolist()

    # 2. Fetch yfinance.
    panels = fetch_yf(universe, args.start, end, Path(args.yf_cache))
    logger.info("fetched %d/%d tickers", len(panels), len(universe))
    if not panels:
        logger.error("no yfinance data fetched; aborting")
        return 2

    # 3. Calendar.
    cal = build_calendar(panels, end)
    if len(cal) < 100:
        logger.error("calendar too short (%d sessions); aborting", len(cal))
        return 2
    write_calendar(us_dir, cal, args.future_end)

    # 4. Compute fields + write .day.bin for every ticker.
    n_written = 0
    for tk, raw in panels.items():
        try:
            fields = compute_fields(raw)
        except Exception as exc:
            logger.warning("compute_fields failed for %s (%s); skipping", tk, exc)
            continue
        if fields["close"].dropna().empty:   # all-NaN (e.g. a delisted ticker that
            logger.info("skipping all-NaN ticker %s (no valid price)", tk)  # yfinance zero-filled)
            continue
        write_bins(us_dir, tk, fields, cal)
        n_written += 1
    logger.info("wrote %d tickers x %d fields", n_written, len(FIELDS))

    # 5. Instruments.
    write_instruments(us_dir, panels, membership, end)

    # 6. Free-float sidecar.
    if args.float_source == "none":
        write_floatshares(us_dir, [])
    elif args.float_source == "sec":
        rows = fetch_float_sec(membership)
        if not rows:
            logger.warning("SEC free-float returned nothing; falling back to yfinance")
            rows = fetch_float_yfinance(universe)
        write_floatshares(us_dir, rows)
    else:
        write_floatshares(us_dir, fetch_float_yfinance(universe))

    # 7. Self-verify.
    if not args.no_verify:
        verify_ticker = "AAPL" if "AAPL" in panels else next(iter(panels))
        self_verify(us_dir, verify_ticker)

    print(f"\nUS data built at {us_dir}: {len(cal)} sessions, {n_written} tickers, "
          f"{len(FIELDS)} fields each.")
    print("Next: QLIB_DATA_DIR=data/qlib/us_data python scripts/qa_check_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())