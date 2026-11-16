#!/usr/bin/env bash
# Submit queued job groups in priority order as GPU-queue slots free up (gpu QOS: 32 queued jobs
# per user). Queue file: one group per line, "<jobs in group>|<command>", e.g.
#   6|AUDIT_MODELS=olmo_instruct AUDIT_CONDS=steered bash scripts/farmshare/submit.sh rq3-audit
# Lines run in order; a group is submitted only when all its jobs fit. DRY=1: print, never submit
# (USED=<n> fakes the queue count). Run on the login node with nohup.
set -uo pipefail
cd "$(dirname "$0")/../.."
Q="${1:?queue file}"; LIMIT="${LIMIT:-32}"
while IFS='|' read -r need cmd; do
  [ -z "$need" ] && continue
  case "$need" in \#*) continue ;; esac
  while :; do
    if [ "${DRY:-0}" = 1 ]; then used="${USED:-0}"; else used=$(squeue -u "$USER" -h -o %P < /dev/null | grep -c gpu); fi
    [ $((used + need)) -le "$LIMIT" ] && break
    sleep 300
  done
  echo "[$(date)] submitting ($need jobs; gpu jobs queued: $used): $cmd"
  if [ "${DRY:-0}" = 1 ] && [ "${DRY_EXEC:-0}" != 1 ]; then echo "  DRY: would run: $cmd"; else bash -c "$cmd" < /dev/null 2>&1 | grep -v "^Watch"; fi
done < "$Q"
echo "[$(date)] queue done"
