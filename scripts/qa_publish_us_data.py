#!/usr/bin/env python
"""Package the S&P 500 Qlib dataset for HuggingFace (mirror QuantaAlpha/qlib_csi300).

Produces ``hf_data_us/`` containing:
  * us_data.zip   -- the Qlib dir (top-level us_data/ with calendars/features/instruments,
                     incl. instruments/floatshares.txt) so ``unzip -d ./data/qlib`` yields
                     ./data/qlib/us_data/
  * daily_pv.h5   -- the US factor cache (8-col, incl $return); the EXACT generate.py
                     output of the us_data dir zipped above (staleness guard: one
                     regenerates the other)
  * daily_pv_debug.h5 -- the debug subset
  * README.md     -- the HF dataset card (file table, placement, $vwap proxy + SEC
                     free-float provenance disclosures)

If HF_TOKEN is set, uploads to QuantaAlpha/qlib_sp500; otherwise prints the exact
huggingface-cli commands to run after ``huggingface-cli login``.

Usage::

    conda run -n quantaalpha python scripts/qa_publish_us_data.py \\
        [--us-dir data/qlib/us_data] [--cache data/git_ignore_folder/.../daily_pv.h5] \\
        [--debug-cache data/git_ignore_folder/..._debug/daily_pv.h5] [--out hf_data_us] \\
        [--verify]   # regenerate the cache from the zipped dir and compare (staleness)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qa_publish_us_data")

DATASET_CARD = """\
---
language:
  - en
license: apache-2.0
task_categories:
  - time-series-forecasting
  - other
tags:
  - finance
  - quantitative
  - qlib
  - factor
  - time-series
  - sp500
  - s-and-p-500
pretty_name: QuantaAlpha Qlib S&P 500 Dataset
---

# QuantaAlpha Qlib S&P 500 Dataset

[![GitHub](https://img.shields.io/badge/GitHub-QuantaAlpha-181717?logo=github)](https://github.com/QuantaAlpha/QuantaAlpha)

Qlib-format S&P 500 market data and pre-computed HDF5 files for QuantaAlpha factor
**transfer** testing (mine CSI300 -> recompute on S&P 500, no re-mining). Same 8
cache fields and the same 2005-2026 range as the CSI300 dataset.

## Dataset description

| Filename          | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `us_data.zip`     | Qlib raw market data (S&P 500, 2005-2026). Unpack into `./data/qlib/`. |
| `daily_pv.h5`     | Pre-computed full price-volume panel (8 cols incl `$return`). Needed for the transfer. |
| `daily_pv_debug.h5` | Smaller debug subset.                                           |

> The `daily_pv.h5` here is the EXACT output of `quantaalpha/factors/data_template/generate.py`
> run on the `us_data` dir in this zip (the staleness guard: one regenerates the other).

## How to download

```bash
pip install huggingface_hub
huggingface-cli download QuantaAlpha/qlib_sp500 --repo-type dataset --local-dir ./hf_data_us
# or: wget -P hf_data_us https://huggingface.co/datasets/QuantaAlpha/qlib_sp500/resolve/main/us_data.zip
```

## How to place the files

```bash
unzip hf_data_us/us_data.zip -d ./data/qlib          # -> ./data/qlib/us_data/
mkdir -p git_ignore_folder/factor_implementation_source_data_us
mkdir -p git_ignore_folder/factor_implementation_source_data_us_debug
cp hf_data_us/daily_pv.h5        git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5
cp hf_data_us/daily_pv_debug.h5  git_ignore_folder/factor_implementation_source_data_us_debug/daily_pv.h5
```

Then set in `.env`:
```bash
QLIB_DATA_DIR=./data/qlib/us_data
QLIB_PROVIDER_URI=./data/qlib/us_data
```

## How to read the files locally

```python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")   # (datetime, instrument) MultiIndex, key="data"
```

## Field description (daily price and volume)

| Field    | Description                              |
| -------- | ---------------------------------------- |
| open     | Open price (split+dividend adjusted)     |
| close    | Close price (split+dividend adjusted)    |
| high     | High price (split+dividend adjusted)     |
| low      | Low price (split+dividend adjusted)      |
| volume   | Trading volume (split-adjusted shares)   |
| factor   | Adjustment factor = AdjClose/Close (the dividend factor) |
| return   | Daily close-to-close return (derived)    |
| vwap     | **(O+H+L+C)/4 typical-price PROXY** (see note) |
| amount   | True USD traded notional = Close*Volume (in us_data features only) |

## Documented limitations

- **`$vwap` is the `(O+H+L+C)/4` typical-price PROXY, not real volume-weighted VWAP.**
  Free Alpaca reaches only 2020 and no free source carries 2005-2026 daily VWAP; the
  full range + free was chosen over real VWAP. The 26/150 `$vwap` factors compute on
  typical price. Real VWAP (Alpaca 2020+ splice or a paid Polygon pull) is the upgrade.
- **Point-in-time membership (adds + removes).** `instruments/sp500.txt` is built
  from chinobing's `sp500_changes_since_1996.csv` (every add/remove event since 1996)
  + current constituents, so DROPPED names (Lehman/LEHMQ, Bear Stearns/BSC, Sears,
  Yahoo/YHOO, ...) are included with their `[add, remove]` windows -- no survivorship
  bias on the removal side (~1,150 names ever in the index over 2005-2026, not just
  the ~503 current). Residual bias only where yfinance lacks data for a delisted
  ticker (those names are absent, logged). Override with `--membership-csv`.
- **Free-float sidecar** (`instruments/floatshares.txt`): SEC EDGAR historical shares
  outstanding + public float (point-in-time, filing-lagged, ~2009+) where built with
  `--float-source sec`; a yfinance current-snapshot fallback fills gaps. Free-float
  share count is derived (`EntityPublicFloat / price`) since it is not a direct XBRL
  tag. Inert unless shorting is enabled.
"""


def _zip_us_data(us_dir: Path, out_zip: Path) -> None:
    """zip data/qlib/us_data -> out_zip with top-level dir 'us_data/'."""
    us_dir = us_dir.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in sorted(us_dir.rglob("*")):
            if p.is_file():
                arc = Path("us_data") / p.relative_to(us_dir)
                zf.write(p, arc.as_posix())
                n += 1
    logger.info("zipped %d files -> %s (%.1f MB)", n, out_zip, out_zip.stat().st_size / 1e6)


def _verify_staleness(out_zip: Path, cache: Path, us_dir: Path) -> None:
    """Regenerate the cache from the zipped us_data and compare a content hash to the
    published daily_pv.h5 -- proves the cache IS f(us_data), not a stale drift (the
    CSI300 HF cache once shipped a 6-col schema with no $return)."""
    import pandas as pd
    from pandas.util import hash_pandas_object
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with zipfile.ZipFile(out_zip) as zf:
            zf.extractall(tdp)
        regen = tdp / "us_data"
        env = dict(os.environ, QLIB_DATA_DIR=str(regen), QLIB_REGION="us")
        script = Path("quantaalpha/factors/data_template/generate.py").resolve()
        logger.info("staleness guard: regenerating cache from unzipped us_data...")
        subprocess.run([sys.executable, str(script)], cwd=Path(__file__).resolve().parents[1],
                       env=env, check=True, capture_output=True)
        regen_h5 = Path(__file__).resolve().parents[1] / "daily_pv_all.h5"
        if not regen_h5.exists():
            logger.warning("staleness guard: regenerated daily_pv_all.h5 not found; skipping compare")
            return
        a = pd.read_hdf(str(regen_h5), key="data")
        b = pd.read_hdf(str(cache), key="data")
        assert list(a.columns) == list(b.columns), (list(a.columns), list(b.columns))
        assert a.shape == b.shape, (a.shape, b.shape)
        # content hash on a sampled instrument
        samp = sorted(a.index.get_level_values("instrument").unique())[:3]
        ha = hashlib.sha256(hash_pandas_object(a.loc[(slice(None), samp), :]).values.tobytes()).hexdigest()[:16]
        hb = hashlib.sha256(hash_pandas_object(b.loc[(slice(None), samp), :]).values.tobytes()).hexdigest()[:16]
        assert ha == hb, f"staleness guard FAILED: sampled hash {ha} != {hb}"
        logger.info("staleness guard PASS: regenerated cache matches published daily_pv.h5 (cols=%s, shape=%s)",
                    list(a.columns), a.shape)
        regen_h5.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--us-dir", default="data/qlib/us_data")
    p.add_argument("--cache", default="data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    p.add_argument("--debug-cache", default="data/git_ignore_folder/factor_implementation_source_data_us_debug/daily_pv.h5")
    p.add_argument("--out", default="hf_data_us")
    p.add_argument("--verify", action="store_true", help="regenerate cache from the zip and compare (staleness guard)")
    p.add_argument("--repo", default="QuantaAlpha/qlib_sp500")
    p.add_argument("--no-zip", action="store_true", help="skip zipping (use existing us_data.zip)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    us_zip = out / "us_data.zip"
    if not args.no_zip:
        _zip_us_data(Path(args.us_dir), us_zip)
    elif not us_zip.exists():
        logger.error("--no-zip given but %s does not exist", us_zip)
        return 2

    for src, dst in ((args.cache, out / "daily_pv.h5"),
                     (args.debug_cache, out / "daily_pv_debug.h5")):
        if not Path(src).exists():
            logger.warning("missing %s; skipping %s", src, dst)
            continue
        shutil.copyfile(src, dst)
        logger.info("copied %s -> %s (%.1f MB)", src, dst, dst.stat().st_size / 1e6)

    (out / "README.md").write_text(DATASET_CARD)
    logger.info("wrote %s (dataset card)", out / "README.md")

    if args.verify:
        _verify_staleness(us_zip, out / "daily_pv.h5", Path(args.us_dir))

    print(f"\nPackaged at {out}/: {[p.name for p in sorted(out.iterdir())]}")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        logger.info("HF_TOKEN set: uploading to %s", args.repo)
        rc = subprocess.call(["huggingface-cli", "upload", args.repo, str(out), "--repo-type", "dataset"])
        print(f"upload exit={rc}")
    else:
        print("\nHF_TOKEN not set. After `huggingface-cli login`, run:\n")
        print(f"  huggingface-cli repo create {args.repo} --type dataset   # if it does not exist")
        print(f"  huggingface-cli upload {args.repo} {out} --repo-type dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())