#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Fetch the OFFICIAL CSI300 total-return index (沪深300全收益, H00300).

Why
---
Every excess return in this project was measured against the wrong benchmark:

  * btv2 (``quantaalpha/backtest/runner.py``) subtracts the **SH000300 PRICE index**
    while the book itself prices with qlib's ADJUSTED closes (dividends reinvested),
    so the strategy is handed the market's whole dividend yield as if it were alpha
    (~2.4 pp/yr on 2022-2025; +5.26/+4.46/+3.03 pp for 2019/2020/2021 per
    ``quantaalpha/eval/protocol.py:625-629``).
  * the eval operator subtracts an EQUAL-WEIGHTED constituent basket
    (``benchmark_construction: equal``), which is neither the index nor cap-weighted.

The correct benchmark is the actual cap-weighted CSI300 with dividends reinvested.
There is **no total-return series in cn_data**: ``SH000300.$adjclose`` is a constant
unit rescale of ``$close`` (ratio 982.79, std 4.4e-5; return difference 2.4e-7), and
the same holds for the ``SZ399300`` cross-listing. So it must be fetched.

Why not reconstruct it from market caps
---------------------------------------
Already tried, already failed. ``scripts/qa_synth_benchmark.py`` built
``w = circ_mv / sum(circ_mv)`` and validated it against the one published weight
snapshot: ``data/results/synth_benchmark_validation.json`` records
``validated: false``, ``levels_ok: false``, total_variation 0.252.
``quantaalpha/eval/protocol.py:649-656`` records why -- circulating market cap
overweights state holdings, off by +13-15%/yr, tracking error 3.58%/yr. The official
index is the only correct source.

Gates
-----
GATE A -- the fetched series must reproduce the known official annual total returns
to within ``--tol`` pp (default 0.6). Verified during planning for 2017-2025 with
diffs of +0.17..+0.36 pp; the offset is systematically POSITIVE (mean ~+0.25 pp/yr),
consistent with a year-boundary/rounding convention difference in the reference
figures rather than a wrong series -- every sign and magnitude matches.

GATE B -- coverage must span the protocol's train (2005-01-01..2012-12-31) and valid
(2013-01-01..2015-12-31) windows, else the combiner has no benchmark there. If the
official series does not reach back that far the script FAILS rather than silently
producing a short series; the fallback is a manual CSV, never a cap reconstruction.

Fetching
--------
``ak.stock_zh_index_hist_csindex`` is slow (a multi-year pull exceeded 110 s with no
output during planning), so this fetches in ``--chunk-years`` blocks, caches each
chunk as its own parquet under ``data/reference/_csi300_tr_chunks/``, and retries with
backoff. A re-run reuses cached chunks, so an interrupted fetch resumes cheaply.

Usage
-----
    /opt/anaconda3/envs/quantaalpha/bin/python -u scripts/qa_fetch_csi300_total_return.py
    ... --start 2004-12-31 --end 2026-01-09
    ... --refresh            # ignore cached chunks
    ... --no-write           # validate only, write nothing
"""
from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


class _Timeout(Exception):
    pass


@contextmanager
def hard_timeout(seconds: int):
    """Abort a blocking call after ``seconds``.

    Necessary because ``akshare`` exposes no request timeout: the CSIndex endpoint
    throttles by ACCEPTING the connection and then never responding, so a plain
    retry loop (which only catches exceptions) waits forever. Observed live --
    chunks 1 and 2 returned in 1-2 s, chunk 3 then stalled >7 min on an ESTABLISHED
    socket with zero bytes back. SIGALRM turns that stall into a retryable
    exception. ``socket.setdefaulttimeout`` alone is not enough: it bounds
    individual socket ops, not a server that dribbles keepalives.
    """
    def _fire(signum, frame):
        raise _Timeout(f"no response within {seconds}s")
    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "data" / "reference"
CHUNK_DIR = REF / "_csi300_tr_chunks"
OUT = REF / "csi300_total_return.parquet"

SYMBOL = "H00300"          # 沪深300全收益 -- CSI300 Total Return (dividends reinvested)
PRICE_SYMBOL = "SH000300"  # the price index, for the cross-check

# Official CSI300 total-return annual returns (%), dividends reinvested.
# GATE A compares against these.
KNOWN_ANNUAL = {
    2017: 23.99, 2018: -23.81, 2019: 38.87, 2020: 29.62, 2021: -3.69,
    2022: -20.02, 2023: -9.36, 2024: 17.88, 2025: 20.64,
}

# GATE B -- windows that must be covered (protocol train / valid).
REQUIRED_WINDOWS = [("2005-01-01", "2012-12-31"), ("2013-01-01", "2015-12-31")]


def _chunk_path(start: str, end: str) -> Path:
    return CHUNK_DIR / f"{SYMBOL}_{start.replace('-', '')}_{end.replace('-', '')}.parquet"


def fetch_chunk(start: str, end: str, refresh: bool, retries: int = 4,
                pause: float = 5.0, timeout: int = 60) -> pd.DataFrame:
    """One cached, timed-out, retried chunk. Returns the raw akshare frame."""
    cp = _chunk_path(start, end)
    if cp.exists() and not refresh:
        df = pd.read_parquet(cp)
        print(f"  [cache] {start}..{end}: {len(df)} rows", flush=True)
        return df
    import akshare as ak
    socket.setdefaulttimeout(timeout)
    last = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            with hard_timeout(timeout):
                df = ak.stock_zh_index_hist_csindex(
                    symbol=SYMBOL,
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                )
            dt = time.time() - t0
            df = pd.DataFrame() if df is None else df
            CHUNK_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cp)
            print(f"  [fetch] {start}..{end}: {len(df)} rows in {dt:.0f}s", flush=True)
            return df
        except (_Timeout, Exception) as exc:  # stall, network, or schema flakiness
            last = exc
            back = pause * attempt
            print(f"  [retry {attempt}/{retries}] {start}..{end}: "
                  f"{type(exc).__name__}: {exc} -- backing off {back:.0f}s", flush=True)
            time.sleep(back)
    raise RuntimeError(f"chunk {start}..{end} failed after {retries} tries: {last}")


def to_series(df: pd.DataFrame) -> pd.Series:
    """Raw akshare frame -> a clean (datetime -> close) float Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    dcol = next(c for c in df.columns if "日期" in str(c))
    ccol = next(c for c in df.columns if "收盘" in str(c))
    s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol]))
    return s.dropna().sort_index()


def annual_returns(s: pd.Series) -> dict[int, float]:
    """Calendar-year % return, each year based off the prior year's last close."""
    out: dict[int, float] = {}
    for y in sorted({d.year for d in s.index}):
        cur = s[s.index.year == y]
        prev = s[s.index < pd.Timestamp(f"{y}-01-01")]
        if len(cur) == 0 or len(prev) == 0:
            continue
        out[y] = float(cur.iloc[-1] / prev.iloc[-1] - 1.0) * 100.0
    return out


def gate_a(s: pd.Series, tol: float) -> bool:
    """Fetched annual returns must match the known official figures within tol pp."""
    got = annual_returns(s)
    print(f"\nGATE A -- official annual total returns (tol {tol} pp)")
    print(f"  {'yr':<6}{'fetched':>10}{'expected':>10}{'diff':>8}")
    ok, diffs = True, []
    for y, exp in sorted(KNOWN_ANNUAL.items()):
        if y not in got:
            print(f"  {y:<6}{'MISSING':>10}{exp:>10.2f}{'':>8}")
            ok = False
            continue
        d = got[y] - exp
        diffs.append(d)
        flag = "" if abs(d) < tol else "   <-- EXCEEDS TOL"
        ok &= abs(d) < tol
        print(f"  {y:<6}{got[y]:>10.2f}{exp:>10.2f}{d:>8.2f}{flag}")
    if diffs:
        print(f"  mean offset {sum(diffs)/len(diffs):+.2f} pp "
              f"(a small systematic offset is expected: year-boundary/rounding convention)")
    print(f"  GATE A: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_b(s: pd.Series) -> bool:
    """Coverage must span the protocol's train and valid windows."""
    print("\nGATE B -- coverage of the protocol train/valid windows")
    if s.empty:
        print("  series is EMPTY -- GATE B: FAIL")
        return False
    print(f"  span {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")
    ok = True
    for a, b in REQUIRED_WINDOWS:
        n = int(((s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))).sum())
        good = n > 0
        ok &= good
        print(f"  {a}..{b}: {n} rows  {'OK' if good else 'MISSING'}")
    print(f"  GATE B: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  -> The official series does not reach the required windows.\n"
              "     Fallback is a MANUAL CSV of the official index, NOT a circ_mv\n"
              "     reconstruction (already proven to fail: see the module docstring).")
    return ok


def cross_check_price(s: pd.Series) -> None:
    """TR minus the SH000300 price index should be a plausible dividend yield.

    Not a gate -- a sanity print. A large disagreement with the ~2.4%/yr measured by
    ``estimated_dividend_return`` means the wrong series was fetched.
    """
    print("\nCROSS-CHECK -- official TR vs the SH000300 price index")
    try:
        import os
        os.environ.setdefault("QLIB_PROVIDER_URI", str(REPO / "data/qlib/cn_data"))
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=str(REPO / "data/qlib/cn_data"), region="cn")
        raw = D.features([PRICE_SYMBOL], ["$close"],
                         start_time="2016-12-01", end_time="2025-12-31")
        px = raw["$close"].droplevel("instrument")
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()
    except Exception as exc:
        print(f"  (skipped: {type(exc).__name__}: {exc})")
        return
    tr_a, px_a = annual_returns(s), annual_returns(px)
    print(f"  {'yr':<6}{'TR%':>9}{'price%':>9}{'div pp':>9}")
    gaps = []
    for y in sorted(set(tr_a) & set(px_a)):
        g = tr_a[y] - px_a[y]
        gaps.append(g)
        print(f"  {y:<6}{tr_a[y]:>9.2f}{px_a[y]:>9.2f}{g:>9.2f}")
    if gaps:
        print(f"  mean implied dividend yield {sum(gaps)/len(gaps):+.2f} pp/yr "
              f"(estimated_dividend_return measured 2.40%/yr -- these should be close)")


QLIB_ROOT = REPO / "data" / "qlib" / "cn_data"
TR_TICKER = "SH000300TR"


def write_qlib_instrument(s: pd.Series) -> None:
    """Write the TR series into cn_data as the instrument ``SH000300TR``.

    Why an instrument rather than a new code path: btv2
    (``quantaalpha/backtest/runner.py:614``) has no ``Protocol`` awareness at all --
    it passes a bare ticker string to qlib's own backtest, which derives the
    benchmark as ``$close/Ref($close,1)-1``. Making the total-return series a
    READABLE INSTRUMENT therefore fixes btv2 and the eval operator at once, with
    zero code change to either.

    Conventions mirrored from the sibling index ``sh000300`` (verified on disk):
      * ``$close``    normalized to 1.0 at the first calendar date (its base is
                      982.79, the same 2005-01-04 base as the price index, so the
                      two are directly comparable)
      * ``$adjclose`` the raw index level in points
      * ``$factor``   ``close/adjclose`` = 1/base, constant (this is a UNIT rescale,
                      not an economic adjustment -- same as the price index)
      * ``$change``   ``pct_change($close)``, first value NaN
      * ``$open/$high/$low`` mirror ``$close``: CSIndex publishes NO intraday for the
                      total-return variant (开盘/最高/最低 come back NaN), and nothing
                      reads them for a benchmark. Mirroring keeps range-style
                      expressions at 0 rather than propagating NaN.
      * ``$volume/$amount/$vwap`` NaN -- not persisted, and identical to the price
                      index's by construction (same constituents).

    Bin format (from ``qlib/data/storage/file_storage.py``):
    ``np.hstack([start_index, values]).astype("<f").tofile(path)``.
    """
    cal = pd.to_datetime([l.strip() for l in
                          (QLIB_ROOT / "calendars/day.txt").read_text().splitlines()
                          if l.strip()])
    aligned = s.reindex(cal)
    gaps = int(aligned.isna().sum())
    if gaps:
        print(f"  calendar days missing from the TR series: {gaps} "
              f"({[str(d.date()) for d in cal[aligned.isna()]][:5]}) -- forward-filling")
        aligned = aligned.ffill()
    if aligned.isna().any():
        raise RuntimeError("leading NaN in the aligned TR series; cannot normalize")

    base = float(aligned.iloc[0])
    close = aligned / base
    fields = {
        "close": close,
        "adjclose": aligned,
        "factor": pd.Series(1.0 / base, index=cal),
        "change": close.pct_change(),
        "open": close, "high": close, "low": close,
        "volume": pd.Series(float("nan"), index=cal),
        "amount": pd.Series(float("nan"), index=cal),
        "vwap": pd.Series(float("nan"), index=cal),
    }

    fdir = QLIB_ROOT / "features" / TR_TICKER.lower()
    fdir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    for name, series in fields.items():
        arr = np.hstack([0.0, series.to_numpy(dtype="float64")]).astype("<f")
        arr.tofile(fdir / f"{name}.day.bin")
    print(f"  wrote {len(fields)} fields to {fdir.relative_to(REPO)} "
          f"(base {base:.2f}, {len(cal)} days)")

    inst = QLIB_ROOT / "instruments" / "all.txt"
    lines = inst.read_text().splitlines()
    spell = f"{TR_TICKER}\t{cal[0].strftime('%Y-%m-%d')}\t{cal[-1].strftime('%Y-%m-%d')}"
    if any(l.split("\t")[0] == TR_TICKER for l in lines if l.strip()):
        lines = [spell if l.split("\t")[0] == TR_TICKER else l for l in lines if l.strip()]
        print(f"  updated existing {TR_TICKER} spell in instruments/all.txt")
    else:
        lines.append(spell)
        print(f"  appended {TR_TICKER} to instruments/all.txt")
    inst.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2004-12-31")
    ap.add_argument("--end", default="2026-01-09")
    ap.add_argument("--chunk-years", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.6, help="GATE A tolerance in pp")
    ap.add_argument("--refresh", action="store_true", help="ignore cached chunks")
    ap.add_argument("--no-write", action="store_true", help="validate only")
    ap.add_argument("--write-qlib", action="store_true",
                    help=f"also write {TR_TICKER} into data/qlib/cn_data")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"Fetching {SYMBOL} (CSI300 total return) {start.date()} -> {end.date()} "
          f"in {args.chunk_years}y chunks")

    frames, bounds, cur = [], [], start
    while cur <= end:
        nxt = min(pd.Timestamp(f"{cur.year + args.chunk_years}-01-01") - pd.Timedelta(days=1), end)
        bounds.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)

    failed = []
    for a, b in bounds:
        try:
            frames.append(fetch_chunk(a, b, args.refresh))
        except Exception as exc:
            print(f"  [FAILED] {a}..{b}: {exc}", flush=True)
            failed.append((a, b))

    raw = pd.concat([f for f in frames if f is not None and not f.empty],
                    ignore_index=True) if frames else pd.DataFrame()
    s = to_series(raw)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    print(f"\nCombined: {len(s)} unique trading days"
          + (f"  ({len(failed)} chunk(s) failed: {failed})" if failed else ""))

    a_ok = gate_a(s, args.tol)
    b_ok = gate_b(s)
    cross_check_price(s)

    if not (a_ok and b_ok):
        print("\nGATES FAILED -- refusing to write the parquet. Nothing downstream was "
              "changed.\nRe-run with --refresh, or supply a manual CSV of the official "
              "series.")
        return 1

    if args.no_write:
        print("\n--no-write: validated only, nothing written.")
        return 0

    REF.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"date": s.index, "close": s.values})
    out.to_parquet(OUT, index=False)
    print(f"\nBOTH GATES PASS -> {OUT.relative_to(REPO)}  ({len(out)} rows, "
          f"{out['date'].min().date()} -> {out['date'].max().date()})")

    if args.write_qlib:
        print(f"\nWriting {TR_TICKER} into {QLIB_ROOT.relative_to(REPO)}")
        write_qlib_instrument(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
