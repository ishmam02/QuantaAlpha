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

    # --- the LLM, where a run dies quietly ------------------------------
    newest=$(ls -t data/results/logs/rerun_seed_*.log 2>/dev/null | head -1)
    if [ -n "${newest}" ]; then
        age=$(( ($(date +%s) - $(stat -f %m "${newest}")) / 60 ))
        echo "LOG      $(basename "${newest}")  (${age} min since last write)$( [ "${age}" -gt 30 ] && echo "   <-- STALLED?" )"
        printf "         %-26s %s\n" "empty/truncated responses" "$(grep -ci 'empty response' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "tracebacks"                "$(grep -c 'Traceback' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "HTTP 4xx/5xx"              "$(grep -cE 'HTTP (4|5)[0-9][0-9]' "${newest}" 2>/dev/null)"
        printf "         %-26s %s\n" "expression rejected"       "$(grep -c 'Expression has no nodes' "${newest}" 2>/dev/null)"
        last_err=$(grep -iE "error|exception" "${newest}" 2>/dev/null | tail -1 | cut -c1-88)
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
d=[r.get("delta_mean") for r in ev if isinstance(r.get("delta_mean"),(int,float))]
if len(d)>=6:
    h=len(d)//2
    print(f"         delta_mean: first half {st.mean(d[:h]):+.5f}  "
          f"second half {st.mean(d[h:]):+.5f}   <- the learning signal")
t=[r.get("delta_t") for r in ev if isinstance(r.get("delta_t"),(int,float))]
if t:
    print(f"         t-stat: median {st.median(t):+.2f}, max {max(t):+.2f} (bar 1.0)")
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
cache=Path(os.environ.get("FACTOR_CACHE_DIR","data/results/factor_cache"))
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
