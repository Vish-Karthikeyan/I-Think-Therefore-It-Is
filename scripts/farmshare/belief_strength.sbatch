#!/usr/bin/env bash
#SBATCH --job-name=idea-bs
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# Belief-strength experiment (exploratory, 2026-11-16). Partition, GPU, time and memory on the sbatch line.
#   BS_CONFIG  configs/bs_probe.yaml | configs/bs.yaml
#   BS_MODELS  space-separated aliases to elicit, in order (e.g. "olmo_instruct_bs0 olmo_instruct_bs1 olmo_instruct_bs2")
#   BS_BCHECK  space-separated aliases to run the manipulation check for (optional)
# Each command resumes from its cache; at the deadline (exit 3) the job requeues itself.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
source scripts/farmshare/env.sh
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
if [ -n "$end" ] && [ "$end" != "N/A" ]; then
  export IDEA_DEADLINE=$(( $(date -d "$end" +%s) - ${IDEA_DEADLINE_MARGIN:-600} ))
fi
echo "[$(date)] job $SLURM_JOB_ID on $(hostname): config=$BS_CONFIG models=$BS_MODELS bcheck=${BS_BCHECK:-} restart=${SLURM_RESTART_COUNT:-0}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
I=.venv/bin/idea
run() {
  echo "[$(date)] idea $*"
  "$I" "$@"; rc=$?
  if [ $rc -eq 3 ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -ge 5 ]; then echo "requeue limit hit" >&2; exit 3; fi
    echo "[$(date)] deadline reached; requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0
  elif [ $rc -ne 0 ]; then echo "[$(date)] FAILED (exit $rc): idea $*" >&2; exit $rc; fi
}
read -r -a MODELS <<< "$BS_MODELS"
read -r -a CHECKS <<< "${BS_BCHECK:-}"
for m in ${MODELS[@]+"${MODELS[@]}"}; do run elicit --config "$BS_CONFIG" --model "$m" --fmt chat; done
for m in ${CHECKS[@]+"${CHECKS[@]}"}; do run bcheck --config "$BS_CONFIG" --model "$m"; done
echo "[$(date)] done"
