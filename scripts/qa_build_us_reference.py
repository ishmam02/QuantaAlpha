#!/usr/bin/env python
"""Build US neutralization reference data: market cap + GICS industry.

The CSI300 gate neutralizes each factor's signal for **size** (log circ_mv) and
**industry** (CSRC top-level letter) before scoring IC (``eval/neutralize.py``).
The reference data for that -- ``data/reference/market_cap.parquet`` (891 SH/SZ)
and ``industry.parquet`` (CSRC, 5542 SH/SZ) -- is China-only, so a US eval
reindexes US tickers onto it, gets zero overlap, and ``residualize`` silently
degrades to cross-sectional demeaning only (size + industry neutralization
SKIPPED). This script builds the US analogs so the US gate is
methodology-correct:

* ``data/reference/market_cap_us.parquet`` -- a DAILY market-cap series,
  schema-compatible with the CSI300 file (columns ``date, instrument, close,
  float_shares, circ_mv, source``). yfinance exposes only a CURRENT ``marketCap``
  snapshot (no free historical daily market cap), so the series is price-scaled
  from the US factor cache: for each ticker, ``shares = current_market_cap /
  close_at_cache_end`` and ``circ_mv[t] = shares * close[t]``. Because ``close``
  is split+dividend adjusted and the share count is held at its current level,
  this is a point-in-time *size ranking* proxy (what neutralization needs), not a
  point-in-time share-count series -- it captures relative size changes driven by
  price, the dominant signal in a large-cap cross-section. A true point-in-time
  free-float series needs SEC EDGAR historical shares (the ``floatshares.txt``
  sidecar came out empty in the T1 build); this is a disclosed approximation
  whose cross-sectional size RANK -- all ``size_frame`` uses -- is sound.

* ``data/reference/industry_us.parquet`` -- GICS sector per ticker, schema-
  compatible (columns ``code, industry``). Mapped to a single DISTINCT letter
  (T/H/F/D/S/C/I/E/U/M/R) so ``_load_industry``'s ``industry.str.slice(0,1)``
  yields one bucket per sector (11 dummies, 10 after the intercept drop). Like
  the CSI300 industry file (a baostock CURRENT snapshot), this is a current
  classification applied to the whole history; the bias is small and is the same
  disclosure CSI300 already makes.

Sources (no API key; all free):
  * **yfinance ``.info``** (PRIMARY) -- one call returns BOTH ``marketCap`` and
    ``sector``. It needs Yahoo's crumb; a too-fast pace (0.35s) 429'd the crumb
    fetch and every call then 401'd, so this paces at ~1.1s with backoff. At a
    gentle pace the crumb is stable.
  * **yfinance ``.fast_info['market_cap']``** (FALLBACK for market cap) -- uses
    the quote endpoint, NOT crumb-gated, so it works when ``.info`` is auth-
    blocked. Has no sector, hence only a market-cap fallback.
  * **Wikipedia "List of S&P 500 companies"** (FALLBACK for sector) -- one fetch
    (User-Agent required -- the default urllib UA gets 403), ~500 current
    constituents with a GICS Sector column. Covers the current S&P 500; used
    where ``.info`` failed and the ticker is still a current constituent.

Coverage: live tickers get market cap + sector from ``.info``; a ``.info`` 401
falls back to ``fast_info`` (market cap) + Wikipedia (sector, current S&P 500
only). Delisted/renamed symbols (FB->META) return nothing from any source ->
``?`` industry and no size value, so they are NOT neutralized (the same failure
mode as today, but for a small minority rather than 100%). The eval window
(2013-2015) active universe is mostly live survivors, so coverage there is high.
Coverage is logged.

Resumable: each ticker's result is cached as JSON in
``data/reference/_us_ref_cache/``; a cache entry with BOTH null marketCap and
null sector is treated as a failure and re-fetched (so a flaky run fixes itself
on re-run, no manual cache clearing).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("us_reference")

REF = Path("data/reference")
CACHE_DIR = REF / "_us_ref_cache"
MARKET_CAP_US = REF / "market_cap_us.parquet"
INDUSTRY_US = REF / "industry_us.parquet"

# GICS sector -> one DISTINCT letter (slice(0,1) in _load_industry returns it).
# Handles BOTH yfinance and Wikipedia spellings (they differ: "Technology" vs
# "Information Technology", "Healthcare" vs "Health Care", "Consumer Cyclical"
# vs "Consumer Discretionary", "Consumer Defensive" vs "Consumer Staples").
SECTOR_LETTER = {
    "Technology": "T", "Information Technology": "T",
    "Healthcare": "H", "Health Care": "H",
    "Financials": "F", "Financial Services": "F",
    "Consumer Cyclical": "D", "Consumer Discretionary": "D",
    "Consumer Defensive": "S", "Consumer Staples": "S",
    "Communication Services": "C",
    "Industrials": "I",
    "Energy": "E",
    "Utilities": "U",
    "Basic Materials": "M", "Materials": "M",
    "Real Estate": "R",
}

_PACE = 1.1            # seconds between yfinance calls (0.35 429'd the crumb)
_MAX_RETRIES = 3
_CALL_TIMEOUT = 30     # yfinance can hang on a dead symbol; ceiling it


def _with_timeout(fn, *args, timeout: int = _CALL_TIMEOUT):
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn, *args).result(timeout=timeout)
    except _TO:
        return None
    finally:
        ex.shutdown(wait=False)


def _fetch_wikipedia_sectors() -> dict[str, str]:
    """ticker -> GICS sector name, from the S&P 500 Wikipedia table."""
    import requests
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "QuantaAlpha/1.0 (research build)"},
            timeout=30)
        r.raise_for_status()
        tbl = pd.read_html(r.text)
        df = tbl[0]
        sym = [c for c in df.columns if "Symbol" in c][0]
        sec = [c for c in df.columns if "GICS Sector" in c][0]
        out = {str(s).upper().replace(".", "-"): v
               for s, v in zip(df[sym], df[sec]) if pd.notna(s) and pd.notna(v)}
        log.info("Wikipedia S&P 500 sectors: %d constituents", len(out))
        return out
    except Exception as exc:
        log.warning("Wikipedia sector fetch failed: %s", exc)
        return {}


def _fetch_one(ticker: str, session) -> dict:
    """PRIMARY: .info (marketCap + sector). FALLBACK: fast_info market_cap."""
    import yfinance as yf
    mc, sec = None, None
    # Primary: .info (crumb-gated, has both fields).
    try:
        info = _with_timeout(lambda: yf.Ticker(ticker, session=session).info)
        if isinstance(info, dict):
            mc = info.get("marketCap")
            sec = info.get("sector")
    except Exception:
        info = None
    # Fallback: fast_info market_cap (quote endpoint, NOT crumb-gated).
    if not mc:
        try:
            fi = _with_timeout(lambda: yf.Ticker(ticker, session=session).fast_info)
            if fi is not None:
                mc = float(fi.market_cap) if fi.market_cap else None
        except Exception:
            pass
    mc = float(mc) if mc else None
    ok = bool(mc or sec)
    return {"ticker": ticker, "marketCap": mc, "sector": sec,
            "error": None if ok else "no marketCap and no sector"}


def fetch_all(tickers: list[str]) -> dict[str, dict]:
    """Fetch marketCap + sector for every ticker, resumable via per-ticker cache.

    A cache entry with BOTH null marketCap and null sector is a failure and is
    re-fetched (so a flaky run self-heals on re-run). Only real successes are
    skipped on resume.
    """
    import requests
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (QuantaAlpha/us-reference-build)"})
    out: dict[str, dict] = {}
    todo = []
    for t in tickers:
        cf = CACHE_DIR / f"{t}.json"
        if cf.exists():
            try:
                rec = json.loads(cf.read_text())
                if rec.get("marketCap") or rec.get("sector"):  # a real success
                    out[t] = rec
                    continue
            except Exception:
                pass
        todo.append(t)
    log.info("yfinance fetch: %d tickers (%d cached-success, %d todo)",
             len(tickers), len(out), len(todo))
    failed = 0
    for i, t in enumerate(todo, 1):
        rec = None
        for attempt in range(_MAX_RETRIES):
            rec = _fetch_one(t, session)
            if rec["marketCap"] or rec["sector"]:
                break
            # backoff: a 429/401 crumb failure often recovers after a cooldown.
            if attempt < _MAX_RETRIES - 1:
                time.sleep(8.0 * (2 ** attempt))
        out[t] = rec
        (CACHE_DIR / f"{t}.json").write_text(json.dumps(rec))
        if not (rec["marketCap"] or rec["sector"]):
            failed += 1
        time.sleep(_PACE)
        if i % 25 == 0:
            log.info("  %d/%d (%d failed so far)", i, len(todo), failed)
    log.info("fetch done: %d success, %d failed", len(tickers) - failed, failed)
    return out


def build_market_cap(info: dict[str, dict], h5_path: Path) -> pd.DataFrame:
    """Price-scaled daily market cap from the US factor cache close + yfinance mc."""
    log.info("loading close from %s", h5_path)
    df = pd.read_hdf(h5_path)
    close = df["$close"].unstack("instrument")           # date x instrument
    close_ff = close.ffill()
    last_close = close_ff.iloc[-1]                        # per-ticker last close

    rows = []
    got = 0
    for t, rec in info.items():
        mc = rec.get("marketCap")
        if not mc or t not in close.columns:
            continue
        lc = last_close.get(t)
        if not lc or not np.isfinite(lc) or lc <= 0:
            continue
        shares = mc / lc                                  # current shares implied
        circ = close[t] * shares                          # daily mkt cap (NaNs preserved)
        sub = circ.dropna()
        if not len(sub):
            continue
        cl = close[t].reindex(sub.index)
        rows.append(pd.DataFrame({
            "date": sub.index,
            "instrument": t,
            "close": cl.to_numpy(),
            "float_shares": shares,
            "circ_mv": sub.to_numpy(),
            "source": "yfinance",
        }))
        got += 1
    if not rows:
        raise RuntimeError("no tickers produced a market-cap series")
    out = pd.concat(rows, ignore_index=True).sort_values(["instrument", "date"])
    out.to_parquet(MARKET_CAP_US, index=False)
    log.info("wrote %s: %d rows, %d tickers", MARKET_CAP_US, len(out), got)
    ev = out[(out["date"] >= "2013-01-01") & (out["date"] <= "2015-12-31")]
    log.info("  eval-window 2013-2015: %d ticker-days, %d distinct tickers",
             len(ev), ev["instrument"].nunique())
    return out


def build_industry(info: dict[str, dict], wiki: dict[str, str]) -> pd.DataFrame:
    rows = []
    dist = {}
    src_count = {"yfinance": 0, "wikipedia": 0, "unknown": 0}
    for t, rec in info.items():
        sec = rec.get("sector")
        src = "yfinance"
        if not sec and t in wiki:                         # .info failed -> Wikipedia
            sec = wiki[t]
            src = "wikipedia"
        letter = SECTOR_LETTER.get(sec, "?") if sec else "?"
        rows.append({"code": t, "industry": letter})
        dist[letter] = dist.get(letter, 0) + 1
        src_count[src if letter != "?" else "unknown"] = src_count.get(
            src if letter != "?" else "unknown", 0) + 1
    out = pd.DataFrame(rows)
    out.to_parquet(INDUSTRY_US, index=False)
    log.info("wrote %s: %d tickers, sector distribution %s",
             INDUSTRY_US, len(out), dict(sorted(dist.items(), key=lambda kv: -kv[1])))
    n_known = sum(1 for r in rows if r["industry"] != "?")
    log.info("  industry coverage: %d/%d mapped to a GICS bucket (%.0f%%); "
             "sources=%s", n_known, len(rows), 100 * n_known / max(1, len(rows)),
             src_count)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    ap.add_argument("--instruments", default="data/qlib/us_data/instruments/all.txt")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse cached yfinance results; only (re)build the parquets")
    args = ap.parse_args()

    tickers = [l.split()[0] for l in Path(args.instruments).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    log.info("%d tickers in %s", len(tickers), args.instruments)

    wiki = _fetch_wikipedia_sectors()

    if args.skip_fetch:
        info = {t: json.loads((CACHE_DIR / f"{t}.json").read_text())
                for t in tickers if (CACHE_DIR / f"{t}.json").exists()}
    else:
        info = fetch_all(tickers)
    if not info:
        raise SystemExit("no yfinance results available")

    build_market_cap(info, Path(args.h5))
    build_industry(info, wiki)
    log.info("US reference data built: %s, %s", MARKET_CAP_US, INDUSTRY_US)


if __name__ == "__main__":
    main()