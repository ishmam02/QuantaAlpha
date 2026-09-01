# Running QuantaAlpha: Setup, Mining, and Backtesting

A step-by-step guide to reproduce a factor-mining run and evaluate the resulting
factor library. Written to be followed by a person or executed by an AI agent: every
step states what to run, what you should see, and how to tell whether it worked.

> **The short version.** Install into a conda environment, put the Qlib China A-share
> data where `.env` points, add an LLM API key, then run `scripts/qa_mine.sh` to mine
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

The system needs Qlib-format China A-share daily data (2005–2026), roughly 5 GB.

```bash
mkdir -p data/qlib
# Option A — Hugging Face (recommended)
huggingface-cli download <dataset-id> --local-dir data/qlib --repo-type dataset
# Option B — direct download, then unpack
# wget <url> -O cn_data.zip && unzip cn_data.zip -d data/qlib
```

**Verify** — you must have all three subdirectories:

```bash
ls data/qlib/cn_data          # expect: calendars  features  instruments
```

Then check which price/volume fields your copy carries — this determines what the
generator can build formulas from:

```bash
python scripts/qa_check_data.py
```

A full copy has ten fields: `open close high low volume amount vwap adjclose factor
change`; the bare minimum is `open close high low volume`. The same script also
inspects the factor cache (§4.1.1) and prints `READY` when both are usable.

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

**First-run note.** The first mine builds an HDF5 cache of price/volume data before it
can compute anything. This takes several minutes. If you launch many parallel tasks on a
cold cache they will all try to build it at once and can exceed the internal timeout —
so run the smoke test first and let it populate
`data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5`. Every later run
reuses it.

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
fetched from Qlib, plus a `$return` computed from close. Your Qlib copy also serves
`$amount`, `$adjclose` and `$change`, but the cache deliberately does not expose them —
the reference libraries were mined without them, so adding one changes what the search
can reach. That the cache is a subset matters because mined formulas use the richer
fields it *does* carry: in the reference library **26 of 150 formulas reference
`$vwap`**, for example `RANK(TS_MEAN($vwap * $volume, 20) / (TS_MEAN($vwap * $volume,
120) + 1e-8))`. A cache lacking that column cannot compute any of them.

To change which fields are exposed, edit the `FIELDS` list at the top of
`quantaalpha/factors/data_template/generate.py` — `qa_check_data.py` reads that same
list, so the check follows your edit automatically. Then delete the stale cache so the
next run rebuilds it:

```bash
rm -f data/git_ignore_folder/factor_implementation_source_data*/daily_pv*.h5
```

Only add a field your Qlib data actually serves — `qa_check_data.py` lists them, and
requesting a missing one fails the rebuild. The cache starts at 2008 by default;
override with `QA_DATA_START=2005-01-01` if a protocol needs more history.

To rebuild the cache directly rather than waiting for a mine to do it (about 25 minutes,
and the only way to get a cache without spending LLM credits):

```bash
cd quantaalpha/factors/data_template
python generate.py                     # writes daily_pv_all.h5 + daily_pv_debug.h5 here
cp daily_pv_all.h5   ../../../data/git_ignore_folder/factor_implementation_source_data/daily_pv.h5
cp daily_pv_debug.h5 ../../../data/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5
cd ../../..
python scripts/qa_check_data.py        # expect READY
```

This is deterministic: rebuilding against the same Qlib snapshot reproduces the
reference cache **bit for bit** (verified 2026-09-01 — 14,215,449 rows × 8 columns,
5,982 instruments, maximum absolute difference 0.000). If your machine has 16 GB of RAM
or less and the build dies with no error message, it was killed for memory — lower the
chunk size with `QA_DATA_CHUNK=200 python generate.py`.

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
| First run times out building data | Cold HDF5 cache built by many parallel tasks. Run the §4.1 smoke test first, then relaunch. |
| Formulas using `$vwap` fail to compute | Your cache lacks the column. Run `python scripts/qa_check_data.py`; rebuild per §4.1.1. |
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
| `data/git_ignore_folder/` | 490 MB | Built automatically on the first mine, or directly via `generate.py` (§4.1.1). Reproduces bit for bit. |
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

Before a long run, confirm all five:

- [ ] `python -c "import quantaalpha, qlib, lightgbm"` succeeds
- [ ] `ls data/qlib/cn_data` shows `calendars features instruments`
- [ ] The calendar check in §3.2 prints ~5,100 days ending 2026
- [ ] The LLM test in §3.3 prints `LLM OK`
- [ ] The §4.1 smoke test produced a library file with a non-zero factor count
- [ ] The cache (§4.1.1) contains every field you intend the generator to use ---
      in particular `$vwap`, if you are reproducing the reference run

All six passing means a full mine will run.
