#!/usr/bin/env bash
# GPT and Claude pilot on this machine (OpenRouter): vignette check, stance, trials.
# Each model runs in its own process with its own cache; rerunning resumes from the cache.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/pilot.yaml}
LOG=${LOG:-runs/pilot-296551ab/logs}
MODELS=${MODELS:-gpt claude}
LIMIT=${LIMIT:-}   # e.g. LIMIT=60 for a small paid test of the trials
mkdir -p "$LOG"
one() {
  local m=$1
  local lim=""; [ -n "$LIMIT" ] && lim="--limit $LIMIT"
  for cmd in "cues --model $m" "stance --model $m" "elicit --model $m $lim"; do
    echo "[$(date)] idea $cmd" >> "$LOG/api_$m.log"
    .venv/bin/idea $cmd --config "$CFG" >> "$LOG/api_$m.log" 2>&1 || { echo "[$(date)] FAILED: $cmd" >> "$LOG/api_$m.log"; return 1; }
  done
  echo "[$(date)] DONE $m" >> "$LOG/api_$m.log"
}
pids=()
for m in $MODELS; do one "$m" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "models: $MODELS exit=$rc"
