#!/usr/bin/env bash
#SBATCH --job-name=idea
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# Partition, GPUs, time and memory are set by submit.sh on the sbatch command line.
#
# Runs IDEA stages for ONE model, from env vars:
#   IDEA_CONFIG   configs/pilot.yaml | configs/main.yaml
#   IDEA_MODEL    olmo_instruct, llama, ... (empty for model-free stages)
#   IDEA_FORMATS  "chat completion" (default: the model's formats)
#   IDEA_STAGES   space-separated: cues stance elicit screen pilot power prereg fit steer audit
#   IDEA_EXTRA    extra CLI args for every command (e.g. "--rq2-folds 4")
# If a stage stops at the deadline (exit 3), the job requeues itself with the same id and
# resumes from the cache, so dependent jobs keep waiting on it.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
source scripts/farmshare/env.sh

# Stop starting new chunks IDEA_DEADLINE_MARGIN seconds (default 20 min) before this allocation ends.
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
if [ -n "$end" ] && [ "$end" != "N/A" ]; then
  export IDEA_DEADLINE=$(( $(date -d "$end" +%s) - ${IDEA_DEADLINE_MARGIN:-1200} ))
fi
echo "[$(date)] job $SLURM_JOB_ID on $(hostname): model=${IDEA_MODEL:-none} stages=${IDEA_STAGES} restart=${SLURM_RESTART_COUNT:-0}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

I=.venv/bin/idea
C=(--config "$IDEA_CONFIG")
read -r -a EXTRA <<< "${IDEA_EXTRA:-}"
if [ -z "${IDEA_FORMATS:-}" ] && [ -n "${IDEA_MODEL:-}" ]; then
  IDEA_FORMATS=$(.venv/bin/python -c "from idea.models import get_model; print(' '.join(get_model('$IDEA_MODEL').formats))")
fi

run() {  # run one command; requeue on deadline, stop on error
  echo "[$(date)] idea $*"
  "$I" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  if [ $rc -eq 3 ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -ge "${IDEA_MAX_REQUEUES:-20}" ]; then
      echo "deadline reached and requeue limit hit; resubmit by hand" >&2; exit 3
    fi
    echo "[$(date)] deadline reached; requeueing job $SLURM_JOB_ID to resume"
    scontrol requeue "$SLURM_JOB_ID"; exit 0
  elif [ $rc -ne 0 ]; then
    echo "[$(date)] FAILED (exit $rc): idea $*" >&2; exit $rc
  fi
}

for stage in $IDEA_STAGES; do
  case "$stage" in
    cues)    case " $IDEA_FORMATS " in *" chat "*) run cues "${C[@]}" --model "$IDEA_MODEL" ;;
             *) echo "skip cues for $IDEA_MODEL (no chat format)" ;; esac ;;
    stance)  for f in $IDEA_FORMATS; do run stance "${C[@]}" --model "$IDEA_MODEL" --fmt "$f"; done ;;
    elicit)  for f in $IDEA_FORMATS; do run elicit "${C[@]}" --model "$IDEA_MODEL" --fmt "$f"; done ;;
    diag)    for f in $IDEA_FORMATS; do run diag "${C[@]}" --model "$IDEA_MODEL" --fmt "$f"; done ;;
    steer)   run steer "${C[@]}" --model "$IDEA_MODEL" ;;
    audit)   run audit "${C[@]}" --model "$IDEA_MODEL" ;;
    power)   if [ -n "${IDEA_MODEL:-}" ]; then run power "${C[@]}" --model "$IDEA_MODEL"; else run power "${C[@]}"; fi ;;
    generate) if [ -n "${IDEA_PILOT_RUN:-}" ]; then run generate "${C[@]}" --pilot-run "$IDEA_PILOT_RUN"; else run generate "${C[@]}"; fi ;;
    screen|pilot|prereg|fit) run "$stage" "${C[@]}" ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
echo "[$(date)] done: ${IDEA_STAGES}"
