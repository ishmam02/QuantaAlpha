#!/bin/bash
# Remove everything a previous run produced, and nothing else.
#
#   ./scripts/qa_clean.sh          # show what would go
#   ./scripts/qa_clean.sh --yes    # actually remove it
#
# The protect list is explicit rather than implied by a pattern, because the
# expensive-to-replace directories sit right next to the disposable ones:
# data/qlib is the market data the whole engine reads, hf_data is a download,
# and git_ignore_folder is load-bearing for the mining loop. Deleting any of
# them costs hours to recover and nothing here needs it gone.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROTECT=(data/qlib data/git_ignore_folder git_ignore_folder hf_data)
REMOVE=(data/results data/factorlib mlruns)

echo "PROTECTED (never touched):"
for d in "${PROTECT[@]}"; do
    [ -e "$d" ] && printf "  %-28s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

echo ""
echo "TO REMOVE:"
TOTAL=0
for d in "${REMOVE[@]}"; do
    if [ -e "$d" ]; then
        printf "  %-28s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    fi
done
N_LOGS=$(ls -1 /tmp/qa_*.log 2>/dev/null | wc -l | tr -d ' ')
printf "  %-28s %s file(s)\n" "/tmp/qa_*.log" "${N_LOGS}"

# Refuse to run if a protected path would be caught by a removal path.
for p in "${PROTECT[@]}"; do
    for r in "${REMOVE[@]}"; do
        case "$p" in "$r"/*|"$r") echo ""; echo "ABORT: $p sits inside $r"; exit 1;; esac
    done
done

if [ "${1:-}" != "--yes" ]; then
    echo ""
    echo "Dry run. Re-run with --yes to remove."
    exit 0
fi

echo ""
for d in "${REMOVE[@]}"; do
    [ -e "$d" ] && { rm -rf "$d"; echo "removed $d"; }
done
rm -f /tmp/qa_*.log 2>/dev/null
mkdir -p data/results data/factorlib
echo "removed /tmp/qa_*.log"
echo ""
echo "Done. Free space: $(df -h . | awk 'NR==2{print $4}')"
