# Running QuantaAlpha: Setup, Mining, and Backtesting

A step-by-step guide to reproduce a factor-mining run and evaluate the resulting
factor library. Written to be followed by a person or executed by an AI agent: every
step states what to run, what you should see, and how to tell whether it worked.

> **The short version.** Install into a conda environment, put the Qlib China A-share
> data where `.env` points, build the factor cache with `generate.py` (the downloaded
> one has no `$return` or `$vwap`), add an LLM API key, then run `scripts/qa_mine.sh` to mine
> and `python -m quantaalpha.backtest.run_backtest` to evaluate. Expect **10–30 hours**
> for a full 150-factor mine and about **5 minutes** for a backtest.

---

## 0. What this system does

It uses a large language model to invent stock-picking formulas ("factors"), tests each
one on historical market data, keeps the ones that improve a portfolio, and repeats.
The output is a **factor library**: a JSON file of formulas plus their measured scores.
That library is then evaluated by a backtest that builds a portfolio from the formulas
and reports what it would have earned.

Two things are worth knowing before you start:

- **Mining calls a paid LLM API thousands of times.** A 150-factor run makes roughly
  1,000–2,000 completion calls. Budget accordingly, and prefer a cheap fast model.
- **Mining is long-running.** Launch it detached (§4.2) so a closed laptop or a dropped
  SSH session does not kill it.

---

## 1. Prerequisites

| Requirement | Version / note |
|---|---|
| OS | macOS or Linux. Windows via WSL2. |
| Python | **3.10+** (`requires-python = ">=3.10"`) |
| Conda | Miniconda or Anaconda. A venv works, but conda is what the scripts assume. |
| Disk | **~15 GB**: ~5 GB market data, the rest run artifacts and logs. |
| RAM | 16 GB minimum. The evaluator holds a full price/volume panel in memory. |
| LLM API | An OpenAI-compatible endpoint and key (see §3.3). |

Check your starting point:

```bash
python --version        # need 3.10+
conda --version         # any recent version
df -h .                 # need ~15 GB free
```

---

## 2. Install

```bash
git clone <repository-url> QuantaAlpha
cd QuantaAlpha

conda create -n quantaalpha python=3.10 -y
conda activate quantaalpha

pip install -e .        # installs the package plus everything in requirements.txt
```

**Verify** — every line must succeed:

```bash
python -c "import quantaalpha; print('package OK', quantaalpha.__file__)"
python -c "import qlib, lightgbm; print('qlib', qlib.__version__, '| lightgbm', lightgbm.__version__)"
quantaalpha --help      # the CLI entry point
```

> **If `import quantaalpha` picks up the wrong copy** (for example when you have several
> checkouts), the editable install resolves by path order. Force the right one with
> `PYTHONPATH=$PWD python ...`, or reinstall from inside the checkout you want.

---

## 3. Configure

### 3.1 Create your `.env`

```bash
cp configs/.env.example .env
```

Then edit `.env`. The keys that must be correct before anything runs:

| Key | What it is | Example |
|---|---|---|
| `QLIB_DATA_DIR` | Market data directory. **Use an absolute path.** | `/home/you/QuantaAlpha/data/qlib/cn_data` |
| `QLIB_PROVIDER_URI` | Same directory again; Qlib reads this one. | same as above |
| `DATA_RESULTS_DIR` | Where runs write their output. | `/home/you/QuantaAlpha/data/results` |
| `OPENAI_API_KEY` | Key for your LLM endpoint. | `sk-...` |
| `OPENAI_BASE_URL` | Endpoint URL (any OpenAI-compatible server). | `https://api.openai.com/v1` |
| `CHAT_MODEL` | Model that writes the formulas. | `gpt-4o-mini` |
| `REASONING_MODEL` | Model for planning steps; can be the same. | `gpt-4o-mini` |
| `CONDA_ENV_NAME` | Must match the env you created. | `quantaalpha` |

> **Use absolute paths.** A relative `QLIB_DATA_DIR` breaks: one internal step runs with
> its working directory changed, so `./data/...` resolves somewhere empty and the run
> fails with `ValueError: ... does not contain data for day`.

### 3.2 Get the market data

All the price data comes from the Hugging Face dataset
[`QuantaAlpha/qlib_csi300`](https://huggingface.co/datasets/QuantaAlpha/qlib_csi300):

| File | Size | Contents | Needed for |
|---|---|---|---|
| `cn_data.zip` | 493 MB | Qlib-format China A-share daily data, 2005-01-04 to 2026-01-09, ten fields per stock | Qlib init, the evaluation protocol, backtests |
| `daily_pv.h5` | 398 MB | The **factor cache**: a pre-computed price/volume panel | Factor mining |
| `daily_pv_debug.h5` | 1.4 MB | A 100-stock, 2018–2019 slice of the cache | Mining in debug mode |

Mining and evaluation read prices from different places. Mined formulas are computed
against the factor cache. The evaluation protocol that scores them reads
`$open $high $low $close $volume $amount $vwap $factor` directly from the Qlib directory
([`quantaalpha/eval/data.py:32`](quantaalpha/eval/data.py#L32)). Both must be complete, and
the cache you download is not: Step 3 fixes it.

#### Step 1: Download

```bash
# Option A: huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download QuantaAlpha/qlib_csi300 --repo-type dataset --local-dir ./hf_data

# Option B: wget
mkdir -p hf_data
wget -P hf_data https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/cn_data.zip
wget -P hf_data https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/daily_pv.h5
wget -P hf_data https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/daily_pv_debug.h5
```

#### Step 2: Extract and place

```bash
# 1. Qlib data -> data/qlib/cn_data/
mkdir -p data/qlib
unzip hf_data/cn_data.zip -d ./data/qlib

# 2. Factor cache -> both cache folders (see "Why two folders" below)
for root in git_ignore_folder data/git_ignore_folder; do
  mkdir -p "$root/factor_implementation_source_data" "$root/factor_implementation_source_data_debug"
  cp hf_data/daily_pv.h5       "$root/factor_implementation_source_data/daily_pv.h5"
  cp hf_data/daily_pv_debug.h5 "$root/factor_implementation_source_data_debug/daily_pv.h5"
done
```

> **Note**: `daily_pv_debug.h5` must be renamed to `daily_pv.h5` when placed in the debug
> folder, as above.

**Why two folders.** The cache location is a relative path
([`quantaalpha/factors/coder/config.py:10-13`](quantaalpha/factors/coder/config.py#L10-L13)),
and a mine resolves it in two different places:

- **`data/git_ignore_folder/`**: factor code runs against this copy, which is linked into
  every workspace as `./daily_pv.h5`. `factor.py` resolves the path against the parent of
  `DATA_RESULTS_DIR`, which is `data/` with the §3.1 settings.
- **`git_ignore_folder/` at the repository root**: the list of columns shown to the
  generator comes from this copy's debug file. If this folder is missing, the first mine
  rebuilds the cache here, and only here.

Keep the two copies identical; `qa_check_data.py` checks both. If you launch with
`QA_QLIB_DATA_DIR` set (see the US subsection below), `run.sh` exports an absolute path
instead, and only `data/git_ignore_folder/` is read.

#### Step 3: Add `$return` and `$vwap` to the cache (required)

The `daily_pv.h5` on Hugging Face is out of date. It has six columns
(`$open $close $high $low $volume $factor`), but mining uses eight. The two it lacks are
the base features the generator's prompt offers besides OHLCV
([`prompts.yaml:379`](quantaalpha/factors/prompts/prompts.yaml#L379)):

| Missing column | What it is | Reference-library formulas that use it |
|---|---|---|
| `$return` | Daily close-to-close return, derived per stock | 67 of 150 |
| `$vwap` | Volume-weighted average price, read from Qlib | 26 of 150 |

A formula that uses a missing column fails to compute and is dropped, but the run carries
on. So a mine on the downloaded cache finishes normally while quietly searching a smaller
space.

The script that writes all eight columns is
[`quantaalpha/factors/data_template/generate.py`](quantaalpha/factors/data_template/generate.py).
It reads `$vwap` from the Qlib data you unpacked in Step 2 and computes `$return` from
`$close`:

```bash
export QLIB_DATA_DIR="$PWD/data/qlib/cn_data"      # absolute path; generate.py reads it
cd quantaalpha/factors/data_template
python generate.py                                 # writes daily_pv_all.h5 + daily_pv_debug.h5 here
cd ../../..
for root in git_ignore_folder data/git_ignore_folder; do
  cp quantaalpha/factors/data_template/daily_pv_all.h5   "$root/factor_implementation_source_data/daily_pv.h5"
  cp quantaalpha/factors/data_template/daily_pv_debug.h5 "$root/factor_implementation_source_data_debug/daily_pv.h5"
done
```

The build took 1.5 minutes on a 16 GB Mac (2026-09-16) and ends with
`wrote daily_pv_all.h5   14,215,449 rows x 8 cols, 5982 instruments`. Its first six
columns are identical to the downloaded file, so the build only adds the two missing
columns. It also overwrites both cache files, so if you plan to run it you can skip
downloading `daily_pv.h5` and `daily_pv_debug.h5` in Step 1; it needs only
`cn_data.zip`.

**Everyone gets the same cache.** These commands, run on the Hugging Face `cn_data.zip`,
produce a cache identical in content to the reference cache. It has the same eight
float32 columns in the same order (`$open $close $high $low $volume $factor $return
$vwap`), the same rows, and the same values. This was verified on 2026-09-16 from a fresh
unzip, with pyqlib 0.9.7 and pandas 2.3.3. Leave `QA_DATA_START` and the `QA_DEBUG_*`
variables unset, since they change what gets built. `QA_DATA_CHUNK` changes only memory
use: a build with `QA_DATA_CHUNK=200` produced the same fingerprints.

HDF5 files differ byte for byte between builds even when their content is identical, so
don't compare them with `md5`. Instead, `qa_check_data.py` fingerprints each cache's
content and prints `matches the reference cache` when it agrees with:

| Cache file (in both folders) | Rows | Fingerprint |
|---|---|---|
| `factor_implementation_source_data/daily_pv.h5` | 14,215,449 | `dd5541b38b2d2a69` |
| `factor_implementation_source_data_debug/daily_pv.h5` | 48,700 | `b98eb53ae1fd7769` |

To change which columns the cache carries, see §4.1.1. The check skips the fingerprint
comparison once `FIELDS` differs from the reference list.

#### Step 4: Reference data for the evaluation protocol

Before scoring a candidate, the evaluation protocol strips out its size and industry
exposure ([`quantaalpha/eval/neutralize.py`](quantaalpha/eval/neutralize.py)). Neither
download includes size or industry data.
[`quantaalpha/data/reference_sync.py`](quantaalpha/data/reference_sync.py) builds both
from free sources into `data/reference/`:

| File | Contents | Built by |
|---|---|---|
| `industry.parquet` | CSRC industry code for every A-share (5,553 rows on 2026-09-16) | `fetch_industry()`: one baostock query, a few seconds |
| `market_cap.parquet` | Daily circulating market cap for the 934 stocks ever in CSI300 | `fetch_market_cap()`: one paced akshare request per stock. About 6 s per listed stock (2026-09-16), longer for delisted ones, so allow two hours or more. Resumable. |

```bash
pip install akshare baostock      # not installed by `pip install -e .`
python - <<'PY'
from quantaalpha.data.reference_sync import fetch_industry, fetch_market_cap
fetch_industry()
# Called with no arguments, fetch_market_cap() reads CSI300 membership from
# ~/.qlib/qlib_data/cn_data, which exists only after run.sh has run once.
# Pass the membership from Step 2 instead.
codes = sorted({line.split()[0] for line in open("data/qlib/cn_data/instruments/csi300.txt") if line.strip()})
fetch_market_cap(codes=codes)     # re-run to resume if interrupted
PY
```

If either file is missing, a mine still runs. Each candidate logs
`neutralization failed for <formula> (FileNotFoundError); scoring raw` and is judged on
its raw signal, which cannot tell a real factor from a disguised size bet. If that line
appears in a mine's log, these files are missing.

#### Verify

You must have all three Qlib subdirectories:

```bash
ls data/qlib/cn_data          # expect: calendars  features  instruments
```

Then check the Qlib fields and the factor cache:

```bash
python scripts/qa_check_data.py
```

A full Qlib copy has ten fields (`open close high low volume amount vwap adjclose factor
change`); the bare minimum is `open close high low volume`. The script then checks all
four cache files, in `data/git_ignore_folder/` and at the repository root. After Step 3,
each should show eight columns and `matches the reference cache`, followed by `READY`.

- `MISSING $vwap $return` means that copy is still the downloaded one: run Step 3.
- `DIFFERS from the reference cache` means it was built from other data or settings (or
  is left over from an older `generate.py`): rebuild it with Step 3.

```bash
python - <<'PY'
import qlib
from qlib.data import D
qlib.init(provider_uri="data/qlib/cn_data", region="cn")
cal = D.calendar(start_time="2005-01-01", end_time="2026-12-31")
print(f"{len(cal)} trading days, {cal[0].date()} to {cal[-1].date()}")
names = D.list_instruments(D.instruments("csi300"), as_list=True)
print(f"{len(names)} stocks have appeared in CSI300")
PY
```

Expect **~5,100 trading days ending in 2026** and several hundred stocks. If the
calendar is short or the dates are wrong, the download is incomplete — fix it now
rather than debugging a failed mine later.

#### S&P 500 (US transfer) — optional

The US dataset has the same two parts (a Qlib directory and `daily_pv.h5`), on
[`QuantaAlpha/qlib_sp500`](https://huggingface.co/datasets/QuantaAlpha/qlib_sp500).
Its `daily_pv.h5` already has all eight columns, including `$return` and `$vwap`, so it
needs no Step 3. `run.sh` reads the US cache only from `data/git_ignore_folder/`; it
switches there when `QA_QLIB_DATA_DIR` points at `us_data`:

```bash
huggingface-cli download QuantaAlpha/qlib_sp500 --repo-type dataset --local-dir ./hf_data_us
mkdir -p data/qlib
unzip hf_data_us/us_data.zip -d ./data/qlib          # -> ./data/qlib/us_data/
mkdir -p data/git_ignore_folder/factor_implementation_source_data_us
mkdir -p data/git_ignore_folder/factor_implementation_source_data_us_debug
cp hf_data_us/daily_pv.h5        data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5
cp hf_data_us/daily_pv_debug.h5  data/git_ignore_folder/factor_implementation_source_data_us_debug/daily_pv.h5
```

Point `.env` at `./data/qlib/us_data` (`QLIB_DATA_DIR` + `QLIB_PROVIDER_URI`) and
verify the same way:

```bash
python - <<'PY'
import qlib
from qlib.data import D
qlib.init(provider_uri="data/qlib/us_data", region="us")
cal = D.calendar(start_time="2005-01-01", end_time="2026-12-31")
print(f"{len(cal)} trading days, {cal[0].date()} to {cal[-1].date()}")
names = D.list_instruments(D.instruments("sp500"), as_list=True)
print(f"{len(names)} tickers in sp500 membership")
PY
```

Expect **~5,300 NYSE sessions 2005→2026** and ~490 S&P 500 tickers.

The US version of Step 4 is `data/reference/market_cap_us.parquet` and
`industry_us.parquet`. [`scripts/qa_build_us_reference.py`](scripts/qa_build_us_reference.py)
builds both from yfinance, falling back to Wikipedia for sectors. It is resumable and
reads tickers from the US cache above:

```bash
pip install yfinance                         # the script's only dependency beyond pip install -e .
python scripts/qa_build_us_reference.py
```

Then run the transfer, the decisive method-vs-market test. For the matched baseline,
compare against `qa_eval_oneshot.py` on the CSI300 protocol:

```bash
conda run -n quantaalpha python scripts/qa_transfer_us.py \
  --library data/factorlib/all_factors_library_meanvar_20260828_194432.json \
  --protocol quantaalpha/eval/protocol_sp500_meanvar_soft_linear.yaml \
  --cache data/git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \
  --qlib-dir data/qlib/us_data --report
```

To rebuild the US data yourself (yfinance + chinobing membership + SEC EDGAR
free-float), see `scripts/qa_build_us_data.py`; to repackage/publish it, see
`scripts/qa_publish_us_data.py`. **Limitations**: `$vwap` is the `(O+H+L+C)/4`
typical-price proxy (no free 2005-2026 daily VWAP); `instruments/sp500.txt` is
point-in-time (adds + removes since 1996; includes dropped names like Lehman/Bear
Stearns/Sears).

### 3.3 Choose an LLM

Any OpenAI-compatible endpoint works: OpenAI, a local Ollama server, or a hosted
gateway. Two properties matter far more than raw quality:

- **Speed.** 75–85% of a mine's wall-clock time is waiting on the LLM. A model twice as
  slow makes the run twice as long. In our testing a model that was ~5.8× slower per
  call turned a 12-hour mine into a multi-day one without improving factor quality.
- **Cost.** Thousands of calls per run.

A small fast model such as `gpt-4o-mini` is a sensible default. Test it before mining:

```bash
python - <<'PY'
import os, openai
c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ.get("OPENAI_BASE_URL"))
r = c.chat.completions.create(model=os.environ["CHAT_MODEL"],
                              messages=[{"role":"user","content":"Reply with OK"}])
print("LLM OK:", r.choices[0].message.content)
PY
```

---

## 4. Run the mine

### 4.1 Smoke test first (~15 minutes)

Never start a 20-hour run without proving the pipeline end to end. The default config
is deliberately tiny (2 directions, 3 rounds):

```bash
conda activate quantaalpha
CONFIG_PATH=configs/experiment.yaml ./run.sh "cross-sectional equity factors from daily price and volume"
```

**What success looks like:** log lines showing hypotheses being proposed, factors being
computed, and a backtest running; then a library file appearing under
`data/factorlib/`. Confirm:

```bash
ls -la data/factorlib/all_factors_library_*.json | tail -1
python -c "import json,glob; f=sorted(glob.glob('data/factorlib/all_factors_library_*.json'))[-1]; d=json.load(open(f)); print(f, len(d['factors']), 'factors')"
```

> **Counting factors:** the library is a dict `{"metadata": ..., "factors": {...}}`.
> `len(json.load(...))` returns **2** (the two top-level keys), not the factor count.
> Always use `len(d["factors"])`.

**First-run note.** Build the factor cache (§3.2, Step 3) before the first mine. A mine
that finds no cache folder at the repository root builds the cache itself, but only
there: `data/git_ignore_folder/`, the copy factor code actually runs against, stays empty.
And if you launch many parallel tasks on a cold cache, they all try to build it at once
and can exceed the internal timeout.

### 4.1.1 Which fields the generator can use — check this before a long run

The cache, not the raw Qlib data, is what the generator sees, so a field missing from
the cache means every formula referencing it is silently dropped. One command checks
both the data and the cache:

```bash
python scripts/qa_check_data.py
```

It lists the fields your Qlib copy serves, the columns your cache exposes, and prints
`READY` or the specific thing to fix. Run it before any long mine.

**The cache carries eight columns:** `$open $close $high $low $volume $factor $vwap`
fetched from Qlib, plus a `$return` computed from close. The `daily_pv.h5` on Hugging
Face has only the first six; §3.2 Step 3 adds the other two. Your Qlib copy also serves
`$amount`, `$adjclose` and `$change`, but the cache deliberately does not expose them —
the reference libraries were mined without them, so adding one changes what the search
can reach. That the cache is a subset matters because mined formulas use the richer
fields it *does* carry. In the reference library, **26 of 150 formulas reference
`$vwap`** (for example `RANK(TS_MEAN($vwap * $volume, 20) / (TS_MEAN($vwap * $volume,
120) + 1e-8))`) and **67 reference `$return`**. A cache missing either column cannot
compute those formulas.

To change which fields are exposed, edit the `FIELDS` list at the top of
`quantaalpha/factors/data_template/generate.py`. `qa_check_data.py` reads that same
list, so the check follows your edit automatically. Then rebuild with the §3.2 Step 3
commands, which overwrite both cache folders, so you don't need to delete anything first.
Don't delete the cache files and wait for a mine to rebuild them: a mine rebuilds only
when the repo-root cache *folder* is missing, and then only that copy.

Only add a field your Qlib data actually serves. `qa_check_data.py` lists them, and
requesting a missing one fails the rebuild. The cache starts at 2008 by default;
override with `QA_DATA_START=2005-01-01` if a protocol needs more history.

Building the cache with `generate.py` (§3.2 Step 3) is the only way to get a complete
cache without spending LLM credits, and it is deterministic: rebuilding against the same
Qlib snapshot reproduces every value of the reference cache **bit for bit** (the HDF5
file bytes still differ, which is why `qa_check_data.py` compares content fingerprints).
This was verified on 2026-09-01 and again on 2026-09-16: 14,215,449 rows × 8 columns,
5,982 instruments, maximum absolute difference 0.000. If your machine has 16 GB of RAM
or less and the build dies with no error message, it was killed for running out of
memory. Lower the chunk size with `QA_DATA_CHUNK=200 python generate.py`; the result is
the same.

### 4.2 Full production mine (10–30 hours)

```bash
screen -dmS qa_mine bash -lc './scripts/qa_mine.sh'
```

This launches the production configuration: 10 directions, 15 rounds, target 150
factors, seed 42. `screen -dmS` detaches the run so it survives your terminal closing;
the script also holds a wake-lock so an idle laptop does not sleep mid-run.

Watch progress:

```bash
screen -r qa_mine                                     # attach (Ctrl-A then D to detach)
tail -f data/results/ledger_*.jsonl                   # one line per decision
watch -n 60 'python -c "import json,glob;f=sorted(glob.glob(\"data/factorlib/all_factors_library_*.json\"))[-1];print(len(json.load(open(f))[\"factors\"]),\"factors\")"'
```

**Deliverables** when it finishes:

| File | Contents |
|---|---|
| `data/factorlib/all_factors_library_<id>.json` | Every formula mined (~150) |
| `data/factorlib/all_factors_library_<id>_zoo.json` | Only the formulas the quality gate kept |
| `data/results/ledger_<id>.jsonl` | One record per keep/reject decision |
| `data/results/trajectory_pool_<id>.json` | Full breeding lineage |

### 4.3 Resuming

A mine that stops (crash, reboot, manual kill) resumes from its ledger — it does not
start over, and nothing already mined is deleted. Pass the experiment ID, which is the
middle part of your library's filename:

```bash
# for data/factorlib/all_factors_library_meanvar_20260828_194432.json:
./scripts/qa_resume.sh meanvar_20260828_194432
```

It prints how many trajectories and factors it found, then relaunches under `screen`
itself — do not wrap it in `screen` again. If it reports `no pool at ...`, the ID is
wrong; list your runs with `ls data/results/trajectory_pool_*.json`.

> The script activates conda from `/opt/anaconda3`. If your conda lives elsewhere, edit
> that path or launch `EXPERIMENT_ID=<id> ./scripts/qa_mine.sh` yourself.

To stop a mine, kill the Python process; `screen -X quit` alone can leave it orphaned.

---

## 5. Run the backtest

Evaluates a factor library: builds a portfolio of the 50 highest-scoring stocks each
day and reports what it earned.

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library_<id>.json
```

Check the library loads before committing to a full run:

```bash
python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml \
  --factor-source custom --factor-json <your-library>.json --dry-run
# expect: "Factor load result: Qlib 0, custom (LLM) <N>"
```

**Expected output** (takes ~5 minutes):

```
[IC Metrics]
  IC: 0.130   ICIR: 0.899   Rank IC: 0.129   Rank ICIR: 0.876
[Strategy Metrics]
  Ann. Return: 0.3002   Max DD: -0.0794
Results saved: data/results/backtest_v2_results/<name>_backtest_metrics.json
```

Results land in `data/results/backtest_v2_results/`:
`<name>_backtest_metrics.json` (headline numbers) and `<name>_cumulative_excess.csv`
(daily returns, for plotting or per-year breakdowns).

### 5.1 Changing the test period

`configs/backtest.yaml` sets the date ranges. Train on earlier years, test on later
ones — never overlap them:

```yaml
dataset:
  segments:
    train: ["2016-01-01", "2020-12-31"]
    valid: ["2021-01-01", "2021-12-31"]
    test:  ["2022-01-01", "2025-12-26"]
backtest:
  backtest:
    start_time: "2022-01-01"     # must match the test segment
    end_time:   "2025-12-26"
```

Copy the file and pass `-c your_config.yaml` rather than editing the original, so runs
stay reproducible.

### 5.2 Two backtests, and which to believe

| | Trading cost | Use it for |
|---|---|---|
| **Simple** (`configs/backtest.yaml`) | Flat fee, independent of size | Comparing libraries against each other |
| **Realistic** (`quantaalpha/eval/protocol_csi300.yaml`) | Impact grows with fund size | Whether a strategy actually makes money |

The simple backtest also measures the market using a price index that **excludes
dividends**, which flatters returns by roughly the dividend yield (~4 pp/yr on CSI300).
Use it to rank libraries, not to judge profitability.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Error: .env file not found` | Run `cp configs/.env.example .env` and fill it in. |
| `ValueError: ... does not contain data for day` | `QLIB_DATA_DIR` is relative or wrong. Use an absolute path; confirm `calendars/ features/ instruments/` exist. |
| `No module named 'qlib'` | Wrong interpreter. `conda activate quantaalpha`, or call the env's Python by full path. |
| Library shows "2 factors" | You measured `len(json.load(...))`. Use `len(d["factors"])`. |
| All LLM calls fail | Check `OPENAI_API_KEY` / `OPENAI_BASE_URL` with the §3.3 snippet; check quota. |
| Mine dies when the terminal closes | Launch under `screen` as in §4.2. |
| First run times out building data | Many parallel tasks are building a cold HDF5 cache at once. Build it first (§3.2, Step 3), then relaunch. |
| Formulas using `$vwap` or `$return` fail to compute | Your cache is the Hugging Face download, which has neither column. Run `python scripts/qa_check_data.py`, then rebuild (§3.2, Step 3). |
| `qa_check_data.py` says `DIFFERS from the reference cache` | That copy was built from different Qlib data or settings, or by an older `generate.py`. Rebuild with the §3.2 Step 3 commands, which write both folders. |
| Factor feedback shows `FileNotFoundError` for `./daily_pv.h5` | `data/git_ignore_folder/` has no cache. With default settings, a mine's automatic build fills only the repo-root folder. Place and build the cache in both (§3.2, Steps 2–3). |
| Log shows `neutralization failed ... scoring raw` | `data/reference/market_cap.parquet` or `industry.parquet` is missing, so candidates are scored without size/industry neutralization. Build them (§3.2, Step 4). |
| `pytest tests/` aborts with `INTERNALERROR ... SystemExit` | Expected: most files here are standalone scripts, not pytest tests. Run them individually — see §6.1. |
| Fewer factors than expected survive | Same cause: formulas referencing a missing field are dropped. Check the cache columns first (§4.1.1). |
| Mine "stuck" but CPU is high | It is working. Factor evaluation is compute-bound and slows as the library grows. |
| Rate-limit (HTTP 429) errors in the log | Tolerated — the run retries and continues. |
| Disk filling up | `log/` grows ~1.5 GB per round and is never cleaned. Delete old run directories. |

**Diagnostic snapshot** — run this and read the output before asking for help:

```bash
echo "python : $(python --version 2>&1) @ $(which python)"
python -c "import quantaalpha,qlib,lightgbm;print('imports OK')" 2>&1 | tail -1
echo "env    : $(grep -c . .env 2>/dev/null) keys in .env"
echo "data   : $(ls data/qlib/cn_data 2>/dev/null | tr '\n' ' ')"
echo "libs   : $(ls data/factorlib/*.json 2>/dev/null | wc -l) library files"
echo "disk   : $(df -h . | tail -1 | awk '{print $4}') free"
```

### 6.1 Running the checks

`tests/` holds 53 files, but only 6 define `pytest` test functions. The other 47 are
standalone scripts that assert at module level and print their own `PASS` lines — several
also call `raise SystemExit(0)` to skip themselves when a run artefact they need is
absent. `pytest tests/` therefore aborts the entire session on the first such skip. That
is a property of the suite, not a broken install.

Run one check directly:

```bash
python tests/eval/test_remine_split.py       # prints R1..R6 PASS, exits non-zero on failure
```

Or run them all, letting each report independently:

```bash
for t in tests/*/test_*.py; do
  python "$t" >/dev/null 2>&1 && echo "PASS $t" || echo "FAIL $t"
done
```

A file that needs a mined library or a warm signal cache will print `SKIP` on a clean
checkout. That is correct behaviour, not a failure.

---

## 7. Reproducing the report's comparison

The mid-research report compares two systems. This repository is the **treatment**
(`main`); the baseline lives on the `original` branch. To reproduce both:

```bash
# treatment
./scripts/qa_mine.sh

# baseline, in a separate working copy so the two never share state
git worktree add ../qa_original original
cd ../qa_original
cp ../QuantaAlpha/.env .env          # then edit every path to point HERE
PYTHONPATH=$PWD ./scripts/qa_mine.sh
```

Three things to keep straight when comparing runs:

1. **Set `PYTHONPATH=$PWD`** in the second copy. Without it the editable install
   resolves imports back to the first checkout and you will silently mine with the
   wrong code.
2. **Give each copy its own absolute paths** in `.env`. Sharing a results directory
   makes the two runs contaminate each other.
3. **A "round" is not comparable between the two.** The baseline emits three formulas
   per idea and the treatment one, so equal rounds mean very different amounts of work.
   Compare per formula, or at equal library size.

For the analysis scripts that produced the report's tables, see `scripts/qa_report_*.py`.

---

## 8. What git does not carry

Cloning the repository gives you the code but **not** the data, the credentials, or the
results. Everything below is excluded by `.gitignore`. The first group is small enough
to hand over directly; the second must be regenerated.

### 8.1 Copy these (~9 MB, about 2 MB compressed)

| Path | Size | What it is |
|---|---|---|
| `data/factorlib/*.json` | 7.6 MB | The four mined libraries the report analyses |
| `reports/` | 884 KB | The report PDF, its source, and its figures |
| `data/results/report_*.json` | 400 KB | Every number in the report's tables |

Without the libraries you cannot reproduce the report's tables without re-mining, which
costs 10–30 hours and money. Copy them:

```bash
tar czf quantaalpha_artifacts.tgz \
    data/factorlib/*.json reports data/results/report_*.json
```

Unpack into the same relative paths in the new checkout. `scripts/` is no longer listed
here because it is tracked in git as of 2026-09-01 — see §8.1.1.

### 8.1.1 `scripts/` and `tests/` are tracked

Every command in this guide comes from a tracked file, so a fresh clone can run all of
them. `scripts/` holds the launcher (`qa_mine.sh`), the resume helper (`qa_resume.sh`),
the data checker (`qa_check_data.py`) and the analysis scripts behind the report's
tables (`qa_report_*.py`); `tests/` holds the suite referenced in §6. Both were excluded
by `.gitignore` until 2026-09-01 — if you are working from an older clone, pull before
following §4.2 onward.

`tools/` is still ignored: it is scratch, and nothing here depends on it.

The report scripts default to finding the baseline mine at `../qa_orig_mine`, a sibling
of this repository. If yours is elsewhere, point `QA_ORIG_DIR` at it:

```bash
QA_ORIG_DIR=/path/to/qa_orig_mine python scripts/qa_report_learning.py
```

### 8.2 Regenerate these

| Path | Size | How to get it |
|---|---|---|
| `.env` | — | `cp configs/.env.example .env`, then fill in (§3.1). Never copy a filled-in one — it holds an API key. |
| `data/qlib/` | 706 MB | Download the Qlib dataset (§3.2). |
| Cache folders in `data/git_ignore_folder/` and `git_ignore_folder/` | 490 MB per CSI300 copy | CSI300: download, then add `$return` and `$vwap` with `generate.py` (§3.2, Steps 2–3). The content matches the reference fingerprints. US: copy from the US download (§3.2). |
| `data/reference/` | 64 MB | CSI300: `reference_sync` (§3.2, Step 4). US: `scripts/qa_build_us_reference.py`. |
| `data/results/workspace_*` | 190 GB | Per-run scratch. **Do not copy.** Regenerated by mining. |
| `log/`, `mlruns/` | 175 GB | Run logs. **Do not copy.** |

### 8.3 Confirming a handover worked

In the new checkout:

```bash
python scripts/qa_check_data.py     # data + cache usable
ls data/factorlib/*.json            # four libraries present
python -m quantaalpha.backtest.run_backtest -c configs/backtest_1725.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library_meanvar_20260828_194432.json
```

The last command takes about 28 minutes and should reproduce the report's headline
figures for the full `main` library over 2017–2025: **Rank IC 0.129, IC 0.134, annual
return 30.0%, max drawdown −7.9%, information ratio 4.00**. Note the config: it is
`backtest_1725.yaml`, whose test window is 2017–2025. Plain `backtest.yaml` tests
2022–2025 and will print different, equally correct numbers.

Matching numbers mean data, code, and libraries all transferred correctly. Differing
numbers point at the data (§3.2) or the cache (§4.1.1) before anything else.

## 9. Configuration reference

| File | Controls |
|---|---|
| `.env` | Paths, API keys, model names |
| `configs/experiment.yaml` | Small config for smoke tests (2 directions, 3 rounds) |
| `configs/experiment_paper.yaml` | Production mine (10 directions, 15 rounds) |
| `configs/backtest.yaml` | Simple backtest: dates, portfolio size, costs |
| `quantaalpha/eval/protocol_csi300.yaml` | Realistic-cost evaluation |

Useful environment overrides (set on the command line, no file edits needed):

```bash
QA_SEED=7            ./scripts/qa_mine.sh   # different random seed
QA_CHAT_MODEL=gpt-4o ./scripts/qa_mine.sh   # different model for one run
```

**To shorten a run, lower `max_rounds` in the config — not `QA_TARGET_MINED`.** The
round budget is what actually stops a mine: the target is only consulted *after*
`max_rounds` is reached, and then only to keep mining further. Setting
`QA_TARGET_MINED=50` will not stop the run at 50 factors. For a half-length mine, copy
`configs/experiment_paper.yaml`, set `max_rounds: 8`, and launch with
`CONFIG_PATH=your_config.yaml ./run.sh "<direction>"`.

Each round yields roughly ten factors, so 15 rounds produces about 150.

---

## 10. Checklist

Before a long run, confirm all seven:

- [ ] `python -c "import quantaalpha, qlib, lightgbm"` succeeds
- [ ] `ls data/qlib/cn_data` shows `calendars features instruments`
- [ ] The calendar check in §3.2 prints ~5,100 days ending 2026
- [ ] `python scripts/qa_check_data.py` prints `READY`, and the cache has all eight
      columns, including `$return` and `$vwap` (§3.2, Step 3)
- [ ] `data/reference/market_cap.parquet` and `industry.parquet` exist (§3.2, Step 4)
- [ ] The LLM test in §3.3 prints `LLM OK`
- [ ] The §4.1 smoke test produced a library file with a non-zero factor count

All seven passing means a full mine will run.
