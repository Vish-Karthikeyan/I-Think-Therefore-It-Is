#!/usr/bin/env bash
#SBATCH --job-name=idea-readout
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --open-mode=append
# Readout calibration (reviewer request, 2026-11-16; docs/READOUT_CALIBRATION_PLAN.md). RC_MODEL = olmo_instruct | llama.
# The script checkpoints per prompt, so a rerun after a time-out resumes where it stopped.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
source scripts/farmshare/env.sh
echo "[$(date)] job $SLURM_JOB_ID on $(hostname): model=$RC_MODEL"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/readout_calibration.py --model "$RC_MODEL" --subset 0/3 --batch-size 32
echo "[$(date)] done (exit $?)"
