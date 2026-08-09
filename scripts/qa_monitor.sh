#!/bin/bash
# One screen of "is the run healthy", refreshed on an interval.
#
#   ./scripts/qa_monitor.sh            # every 5 min, until interrupted
#   ./scripts/qa_monitor.sh 60 1       # every 60s, once
#
# Looks at the four places a run goes wrong, in the order it usually does:
# the process is gone, the LLM is failing, the search is stalling, the disk is
# filling. Each line is a fact and a threshold, not a colour.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-300}"
ONCE="${2:-}"

while true; do
    echo "════════════════════════════════════════════════════════════ $(date '+%F %H:%M:%S')"

    # --- alive? ---------------------------------------------------------
    if pgrep -f "qa_rerun_all.sh" >/dev/null 2>&1; then
        echo "RUN      driver alive (pid $(pgrep -f qa_rerun_all.sh | head -1))"
    else
        echo "RUN      driver NOT running"
    fi
    # The binary, not the string: the repository path contains "quantaalpha",
    # so a bare match picks up this monitor's own shell.
    arm=$(pgrep -fl "bin/quantaalpha|qa_run_arms.sh" 2>/dev/null | head -1 | cut -c1-70)
    [ -n "${arm}" ] && echo "         miner: ${arm}"

    # --- disk -----------------------------------------------------------
    free_g=$(df -g . | awk 'NR==2{print $4}')
    echo "DISK     ${free_g} GB free$( [ "${free_g}" -lt 25 ] && echo "   <-- LOW" )"
    for d in data/results/workspace_* data/results/pickle_cache_*; do
        [ -d "$d" ] || continue
        printf "         %-52s %s\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    done
    sweep=$(tail -1 data/results/logs/compact_auto.log 2>/dev/null | cut -c1-70)
    [ -n "${sweep}" ] && echo "         last sweep: ${sweep}"

    # Will we make it? The sweeper already stamps free space once per cycle, so
    # the drain rate is measurable rather than guessed -- and it is the NET rate,
    # after reclamation, which is the only one that predicts anything. Gross
    # write rate is ~2.5x higher and projecting from it cries wolf.
    PYTHONPATH=. python - <<'PY' 2>/dev/null
import re, pathlib, datetime as dt
p = pathlib.Path("data/results/logs/compact_auto.log")
if not p.exists():
    raise SystemExit
pts = []
for line in p.read_text(errors="ignore").splitlines():
    m = re.match(r"\[([\d-]+ [\d:]+)\] free: (\d+)Gi", line.strip())
    if m:
        pts.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M"), int(m.group(2))))
if len(pts) < 2:
    print(f"         drain: need 2 sweep cycles to project ({len(pts)} so far)")
    raise SystemExit
hours = (pts[-1][0] - pts[0][0]).total_seconds() / 3600.0
drop = pts[0][1] - pts[-1][1]
if hours <= 0:
    raise SystemExit
rate = drop / hours
free = pts[-1][1]
if rate <= 0:
    print(f"         drain: {rate:+.1f} GB/h net over {hours:.1f}h -- not shrinking")
else:
    hrs = free / rate
    flag = "   <-- TIGHT, prune" if hrs < 24 else ""
    print(f"         drain: {rate:.1f} GB/h net over {hours:.1f}h "
          f"-> {hrs:.0f}h of headroom at {free} GB{flag}")
PY

    # --- the LLM, where a run dies quietly ------------------------------
    newest=$(ls -t data/results/logs/rerun_seed_*.log 2>/dev/null | head -1)
    if [ -n "${newest}" ]; then
        age=$(( ($(date +%s) - $(stat -f %m "${newest}")) / 60 ))
        echo "LOG      $(basename "${newest}")  (${age} min since last write)$( [ "${age}" -gt 30 ] && echo "   <-- STALLED?" )"
        printf "         %-26s %s\n" "empty/truncated responses" "$(grep -ci 'empty response' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "tracebacks"                "$(grep -c 'Traceback' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "HTTP 4xx/5xx"              "$(grep -cE 'HTTP (4|5)[0-9][0-9]' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "expression rejected"       "$(grep -c 'Expression has no nodes' "${newest}" 2>/dev/null)"
        # Anchored on log LEVELS, not the word: factor feedback routinely says
        # "executed successfully, without errors", which a bare match reports as
        # the most recent error.
        last_err=$(grep -E "\| (ERROR|CRITICAL)|Traceback|^[A-Za-z]*Error:" "${newest}" 2>/dev/null | tail -1 | cut -c1-88)
        [ -n "${last_err}" ] && echo "         last error: ${last_err}"
    else
        echo "LOG      no run log yet"
    fi

    # --- is the search actually finding anything? -----------------------
    led=$(ls -t data/results/ledger_treatment_*.jsonl 2>/dev/null | head -1)
    if [ -n "${led}" ]; then
        PYTHONPATH=. python - "${led}" <<'PY' 2>/dev/null
import json, sys, statistics as st
rows=[]
for l in open(sys.argv[1]):
    l=l.strip()
    if l:
        try: rows.append(json.loads(l))
        except: pass
ev=[r for r in rows if "admitted" in r]
if not ev:
    print("SEARCH   ledger exists, no decisions yet"); raise SystemExit
adm=[r for r in ev if r["admitted"]]
n=sum(int(r.get("n_factors",0) or 0) for r in adm)
print(f"SEARCH   {len(ev)} batches, {len(adm)} admitted, {n} factors (target 150)")
tail=ev[-12:]
run=0
for r in reversed(ev):
    if r["admitted"]: break
    run+=1
print("         last 12: " + "".join("A" if r["admitted"] else "." for r in tail)
      + f"   {run} consecutive rejections" + ("   <-- STALLED" if run>=15 else ""))
# NaN is a float, so an isinstance check alone lets bootstrap batches -- which
# record delta_mean = NaN by design -- through, and one of them turns the mean
# and the max into nan. The v==v test is what actually excludes them.
num=lambda k:[float(r[k]) for r in ev
              if isinstance(r.get(k),(int,float)) and r[k]==r[k]]
d=num("delta_mean")
if len(d)>=6:
    h=len(d)//2
    print(f"         delta_mean: first half {st.mean(d[:h]):+.5f}  "
          f"second half {st.mean(d[h:]):+.5f}   <- the learning signal")
elif d:
    print(f"         delta_mean: {len(d)} measured ({', '.join(f'{v:+.4f}' for v in d[-6:])})"
          f"; need 6 for a trend")
t=num("delta_t")
if t:
    print(f"         t-stat: median {st.median(t):+.2f}, max {max(t):+.2f} (bar 1.0)")
ev_rows=[r for r in rows if r.get("evicted_exprs")]
n_out=sum(len(r.get("evicted_exprs") or []) for r in ev_rows)
print(f"         eviction: {len(ev_rows)} event(s), {n_out} factor(s) removed"
      + ("" if ev_rows else "  (fires on every 20th ADMITTED batch)"))
PY
    else
        echo "SEARCH   no ledger yet"
    fi

    # --- what has actually been produced --------------------------------
    libs=$(ls -1 data/factorlib/all_factors_library_*.json 2>/dev/null | grep -v _zoo | wc -l | tr -d ' ')
    if [ "${libs}" -gt 0 ]; then
        echo "LIBRARY"
        for f in data/factorlib/all_factors_library_*.json; do
            case "$f" in *_zoo.json) continue ;; esac
            PYTHONPATH=. python - "$f" <<'PY' 2>/dev/null
import json, sys, hashlib, os
from pathlib import Path
d=json.load(open(sys.argv[1])); fs=d.get("factors",{})
# The cache is PER SEED (factor_cache_s42), so the plain default finds an empty
# directory and reports "0 cached" for a run whose signals are all present --
# a monitor that reports total data loss on a healthy run.
_env=os.environ.get("FACTOR_CACHE_DIR")
if _env:
    cache=Path(_env)
else:
    _per=sorted(Path("data/results").glob("factor_cache_s*"),
                key=lambda q:q.stat().st_mtime, reverse=True)
    cache=_per[0] if _per else Path("data/results/factor_cache")
cached=sum(1 for v in fs.values() if v.get("factor_expression")
           and (cache/f"{hashlib.md5(v['factor_expression'].encode()).hexdigest()}.pkl").exists())
nocode=sum(1 for v in fs.values() if not v.get("factor_implementation_code"))
seeded=sum(1 for v in fs.values() if (v.get("metadata") or {}).get("source")=="alpha158_20_seed_pool")
print(f"         {Path(sys.argv[1]).name[:46]:<46} {len(fs):>4} factors, "
      f"{cached:>4} cached, {nocode:>3} never computed, {seeded:>3} seed")
PY
        done
    fi
    echo ""

    [ -n "${ONCE}" ] && break
    sleep "${INTERVAL}"
done
