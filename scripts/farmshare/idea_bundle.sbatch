#!/usr/bin/env bash
#SBATCH --requeue
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
# Whole-node bundle: one job holding all 4 GPUs of an oat node (and its memory) runs four lanes
# of IDEA tasks side by side, one lane per GPU, so no GPU or memory sits idle.
#
#   IDEA_BUNDLE   lane file: one task per line, "<lane 0-3>|<config>|<model>|<command>|<extra args>",
#                 e.g. "2|configs/main.yaml|olmo_instruct|audit|--part steered/mmlu_pro:1of4"
#                 Lanes run in parallel; tasks in a lane run in order.
#   IDEA_BIN      the idea executable (default .venv/bin/idea; tests point it at a stub)
#   IDEA_LANES    lanes to run (default "0 1 2 3"). With a single lane (a 1-GPU job, e.g. "0") the
#                 task keeps the GPU Slurm assigned (CUDA_VISIBLE_DEVICES is left alone).
#
# A finished task leaves a .done marker beside the lane file, so a requeued bundle skips it. A task
# that stops at the deadline (exit 3) stops its lane; once every lane has stopped, the bundle
# requeues itself and resumes (the elicitor and lm-eval caches keep partial work). Any other
# failure skips that model's later tasks in its lane (other models go on); the bundle then
# exits non-zero after all lanes finish.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
[ -f scripts/farmshare/env.sh ] && source scripts/farmshare/env.sh
BUNDLE="${IDEA_BUNDLE:?set IDEA_BUNDLE to a lane file}"
I="${IDEA_BIN:-.venv/bin/idea}"
STATE="${BUNDLE%.txt}.state"; mkdir -p "$STATE"
LANES="${IDEA_LANES:-0 1 2 3}"; read -r -a LANE_A <<< "$LANES"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
  if [ -n "$end" ] && [ "$end" != "N/A" ]; then
    export IDEA_DEADLINE=$(( $(date -d "$end" +%s) - ${IDEA_DEADLINE_MARGIN:-1200} ))
  fi
fi
echo "[$(date)] bundle $BUNDLE job ${SLURM_JOB_ID:-local} restart=${SLURM_RESTART_COUNT:-0}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv

lane() {  # lane <n>: run this lane's tasks in order
  local n=$1 rc=0 bad=" "
  while IFS='|' read -r ln cfg model cmd extra; do
    [ "$ln" = "$n" ] || continue
    case "$bad" in *" $model "*) echo "[lane $n] skip (earlier task of $model failed): $cmd $extra"; continue ;; esac
    local key; key=$(printf '%s|%s|%s|%s' "$cfg" "$model" "$cmd" "$extra" | md5sum | cut -c1-16)
    if [ -f "$STATE/$key.done" ]; then echo "[lane $n] skip (done): $model $cmd $extra"; continue; fi
    echo "[$(date)] [lane $n] start: $model $cmd $extra"
    read -r -a xa <<< "$extra"
    # lm-eval does not watch IDEA_DEADLINE: stop any task at the deadline ourselves (its caches keep
    # finished requests), so the bundle can requeue instead of being killed at the time limit
    local tmo=()
    if [ -n "${IDEA_DEADLINE:-}" ]; then
      local left=$(( IDEA_DEADLINE - $(date +%s) ))
      if [ "$left" -le 60 ]; then echo "[lane $n] deadline before start: $model $cmd $extra"; touch "$STATE/lane$n.deadline"; return 0; fi
      tmo=(timeout --signal=TERM --kill-after=120 "$left")
    fi
    local gpu=(); [ "${#LANE_A[@]}" -gt 1 ] && gpu=(env "CUDA_VISIBLE_DEVICES=$n")   # 1 lane: keep Slurm's GPU
    ${gpu[@]+"${gpu[@]}"} ${tmo[@]+"${tmo[@]}"} "$I" "$cmd" --config "$cfg" --model "$model" ${xa[@]+"${xa[@]}"} < /dev/null
    rc=$?
    [ $rc -eq 124 ] && rc=3   # stopped by the deadline timeout
    if [ $rc -eq 0 ]; then touch "$STATE/$key.done"; echo "[$(date)] [lane $n] done: $model $cmd $extra"
    elif [ $rc -eq 3 ]; then echo "[$(date)] [lane $n] deadline: $model $cmd $extra"; touch "$STATE/lane$n.deadline"; return 0
    else echo "[$(date)] [lane $n] FAILED (exit $rc): $model $cmd $extra" >&2; touch "$STATE/lane$n.failed"
         bad="$bad$model "   # skip this model's later tasks; the lane goes on with the next model
    fi
  done < "$BUNDLE"
}

rm -f "$STATE"/lane*.deadline "$STATE"/lane*.failed
pids=()
for n in "${LANE_A[@]}"; do lane $n > >(sed "s/^/[L$n] /") 2>&1 & pids+=($!); done
wait "${pids[@]}"

if ls "$STATE"/lane*.deadline >/dev/null 2>&1; then   # requeue first: unfinished work resumes, failed tasks retry
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -ge "${IDEA_MAX_REQUEUES:-20}" ]; then echo "requeue limit hit" >&2; exit 3; fi
    echo "[$(date)] deadline reached; requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0
  fi
  echo "[$(date)] deadline reached (local run)"; exit 3
fi
if ls "$STATE"/lane*.failed >/dev/null 2>&1; then
  echo "[$(date)] bundle finished with failures: $(ls "$STATE"/lane*.failed | xargs -n1 basename)" >&2; exit 1
fi
echo "[$(date)] bundle complete"
