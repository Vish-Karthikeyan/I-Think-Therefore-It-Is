#!/usr/bin/env bash
# Submit IDEA work as 1-GPU jobs, each running a short list of tasks through idea_bundle.sbatch
# (one lane; deadline timeout + requeue; .done markers so a rerun skips finished tasks).
#
#   DRY=1 bash scripts/farmshare/submit_singles.sh <plan file> <work dir>
#
# Plan file, one task per line (jobs keep the order of first appearance; tasks run in file order):
#   <job>|<qos gpu|normal>|<time>|<mem>|<dep job or empty>|<config>|<model>|<command>|<extra args>
# dep: afterok on another job of the same plan (submitted earlier in this run, or recorded in
# <work dir>/<dep>.jobid by an earlier run). DRY=1 prints the sbatch commands and submits nothing.
set -uo pipefail
cd "$(dirname "$0")/../.."
PLAN="${1:?plan file}"; DIR="${2:?work dir}"; DRY="${DRY:-0}"
mkdir -p "$DIR"
jobs_in_order=$(grep -v '^\s*#' "$PLAN" | grep -v '^\s*$' | cut -d'|' -f1 | awk '!seen[$0]++')
for j in $jobs_in_order; do
  rows=$(grep -v '^\s*#' "$PLAN" | awk -F'|' -v j="$j" '$1 == j')
  IFS='|' read -r _ qos tlim mem dep _ <<< "$(head -1 <<< "$rows")"
  # every row of a job must agree on qos/time/mem/dep
  if [ "$(cut -d'|' -f2-5 <<< "$rows" | sort -u | wc -l)" -ne 1 ]; then echo "ERROR: $j rows disagree on qos/time/mem/dep" >&2; exit 2; fi
  lanef="$DIR/$j.txt"
  cut -d'|' -f6- <<< "$rows" | sed 's/^/0|/' > "$lanef"
  part=(-p gpu --qos gpu); [ "$qos" = normal ] && part=(-p gpu,normal)
  depa=()
  if [ -n "$dep" ]; then
    [ -f "$DIR/$dep.jobid" ] || { echo "ERROR: $j depends on $dep, which has no job id yet" >&2; exit 2; }
    did=$(cat "$DIR/$dep.jobid")
    dst=""; [ "$DRY" = 1 ] || dst=$(sacct -n -X -P -j "$did" -o State < /dev/null | head -1 | cut -d' ' -f1)
    case "$dst" in
      COMPLETED) depa=() ;;   # already done (and maybe gone from squeue): no dependency needed
      ""|PENDING|RUNNING|REQUEUED|CONFIGURING|SUSPENDED) depa=(--dependency "afterok:$did" --kill-on-invalid-dep=yes) ;;
      *) echo "ERROR: $j depends on $dep (job $did), which ended $dst; not submitting" >&2; exit 2 ;;
    esac
  fi
  cmd=(sbatch --parsable -J "idea-s-$j" "${part[@]}" -G 1 -c 4 --mem "$mem" -t "$tlim" ${depa[@]+"${depa[@]}"}
       --export="ALL,IDEA_BUNDLE=$lanef,IDEA_LANES=0" scripts/farmshare/idea_bundle.sbatch)
  ntask=$(wc -l < "$lanef" | tr -d ' ')
  if [ "$DRY" = 1 ]; then
    echo "DRY $j ($ntask tasks): ${cmd[*]}"; echo "DRY-$j" > "$DIR/$j.jobid"
  else
    id=$("${cmd[@]}") || { echo "ERROR: sbatch failed for $j; stopping" >&2; exit 1; }
    echo "$id" > "$DIR/$j.jobid"; echo "$j ($ntask tasks): $id"
  fi
done
