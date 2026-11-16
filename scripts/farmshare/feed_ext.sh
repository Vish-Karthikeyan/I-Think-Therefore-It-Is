#!/usr/bin/env bash
# Submit the remaining exploratory jobs as GPU-queue slots free up. FarmShare's gpu QOS allows
# 32 queued GPU jobs per user; this waits for room for each model's prep + pieces, then submits
# them through submit.sh ext-split (one model at a time). Run on the login node with nohup:
#   nohup bash scripts/farmshare/feed_ext.sh "olmo13_sft:3 ..." > $IDEA_SCRATCH/logs/feed_ext.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
LIMIT="${LIMIT:-32}"
for spec in $1; do
  n=$(echo "$spec" | cut -d: -f2); need=$((n + 1))
  while :; do
    used=$(squeue -u "$USER" -h -o %P | grep -c gpu)
    [ $((used + need)) -le "$LIMIT" ] && break
    sleep 300
  done
  echo "[$(date)] submitting $spec (gpu jobs queued: $used)"
  EXT_DIAG_ONLY="" EXT_SPLIT="$spec" bash scripts/farmshare/submit.sh ext-split 2>&1 | grep -v "^Watch"
done
echo "[$(date)] feeder done"
