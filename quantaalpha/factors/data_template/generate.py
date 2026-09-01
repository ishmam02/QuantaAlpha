"""Build the price/volume cache the factor generator computes against.

Writes two HDF5 files next to this script:
  daily_pv_all.h5    every instrument -- what production mining uses
  daily_pv_debug.h5  a 100-instrument subset over a short window, for smoke tests

The columns written here are exactly the fields a mined formula may reference. A field
absent from this cache cannot be used, and any formula referencing it is silently
dropped -- so adding one here is what makes it available to the search.

The defaults reproduce the cache the reference libraries were mined against:

    daily_pv_all.h5    14,215,449 rows x 8 cols, 5,982 instruments, 2008-12-29..2026-01-09
    daily_pv_debug.h5      48,700 rows x 8 cols,   100 instruments, 2018-01-02..2019-12-31

Verify a rebuild with ``python scripts/qa_check_data.py``.
"""
import os

import pandas as pd
import qlib

_provider = os.environ.get("QLIB_DATA_DIR", os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"))
qlib.init(provider_uri=_provider)
from qlib.data import D  # noqa: E402  (must follow qlib.init)

# Fields exposed to the factor generator, in the order the reference cache stores them.
# $vwap and $factor are fetched from Qlib; $return is derived below. The reference
# libraries use $vwap (26 of 150 formulas in the main library reference it), so a cache
# built without it cannot compute them. Add a field only if the Qlib data serves it --
# check with scripts/qa_check_data.py, or the inventory in RUNNING.md section 3.2.
FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor", "$vwap"]

# Start of history for the full cache. The evaluation protocol fits its combiner on
# early data and the walk-forward folds reach back before the reporting window, so this
# must cover the earliest date any protocol asks for -- truncating it silently shortens
# every lookback that depends on it. 2008-12-29 matches the reference cache.
START = os.environ.get("QA_DATA_START", "2008-12-29")

# The debug subset is a small, fast panel for smoke tests, not a sample of the full one.
DEBUG_START = os.environ.get("QA_DEBUG_START", "2018-01-02")
DEBUG_END = os.environ.get("QA_DEBUG_END", "2019-12-31")
DEBUG_N = int(os.environ.get("QA_DEBUG_N", "100"))

# Instruments are loaded in chunks. Asking Qlib for all ~6,000 at once materialises the
# whole panel plus intermediate copies during swaplevel/sort; that peaks at several GB
# and is killed by the OOM reaper on a 16 GB machine, silently, with no output. Chunking
# bounds the peak at one chunk and produces an identical result. Raise on a larger box.
CHUNK = int(os.environ.get("QA_DATA_CHUNK", "400"))


def load(names, start, end=None):
    """Panel for these instruments as (datetime, instrument) with a derived $return."""
    parts = []
    for i in range(0, len(names), CHUNK):
        block = names[i : i + CHUNK]
        df = D.features(block, FIELDS, freq="day", start_time=start, end_time=end)
        if df is not None and not df.empty:
            parts.append(df)
        print(f"  {min(i + CHUNK, len(names))}/{len(names)} instruments", flush=True)
    if not parts:
        raise SystemExit("no data returned -- check QLIB_DATA_DIR and the field list")
    # Qlib returns (instrument, datetime); the cache is keyed (datetime, instrument).
    out = pd.concat(parts).swaplevel().sort_index()
    # Return is computed per instrument, so group by the instrument level of the index.
    # fill_method is pinned to pandas' current default ('ffill'): the default is slated
    # to change, and letting it drift would silently alter $return -- and with it every
    # factor computed against this cache -- on a pandas upgrade.
    out["$return"] = (
        out.groupby(level="instrument")["$close"]
        .pct_change(fill_method="ffill")
        .fillna(0)
    )
    # Column order matches the reference cache.
    return out[["$open", "$close", "$high", "$low", "$volume", "$factor", "$return", "$vwap"]]


def main():
    names = D.list_instruments(D.instruments(), as_list=True)
    print(f"{len(names)} instruments | fields {' '.join(FIELDS)} | from {START} | chunk {CHUNK}")

    data = load(names, START)
    print(data)
    data.to_hdf("./daily_pv_all.h5", key="data")
    print(f"wrote daily_pv_all.h5   {data.shape[0]:,} rows x {data.shape[1]} cols, "
          f"{data.index.get_level_values('instrument').nunique()} instruments")

    # Debug subset: the first N instruments of the full panel that actually trade in the
    # debug window. Two details matter. Candidates come from the FULL panel rather than
    # list_instruments -- the raw listing starts with Beijing tickers that carry no data
    # in this window, so slicing it yields an empty file. And candidates are filtered to
    # the window before taking N, so the result is a complete N x (trading days) panel;
    # taking the first N unfiltered leaves holes wherever a name has since stopped
    # trading, which is how this file drifts to fewer than N instruments over time.
    panel_names = list(dict.fromkeys(data.index.get_level_values("instrument")))
    window = data.loc[str(DEBUG_START):str(DEBUG_END)]
    live = set(window.index.get_level_values("instrument"))
    debug_names = [n for n in panel_names if n in live][:DEBUG_N]
    debug = load(debug_names, DEBUG_START, DEBUG_END)
    debug.to_hdf("./daily_pv_debug.h5", key="data")
    print(f"wrote daily_pv_debug.h5 {debug.shape[0]:,} rows x {debug.shape[1]} cols, "
          f"{debug.index.get_level_values('instrument').nunique()} instruments")


# Qlib fans work out to a process pool. Without this guard each spawned child re-runs
# the module top level, tries to start its own pool, and dies with "An attempt has been
# made to start a new process before the current process has finished its bootstrapping
# phase" -- so the build fails before writing anything.
if __name__ == "__main__":
    main()
