#!/usr/bin/env bash
#SBATCH --job-name=idea-doses
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
[ -f scripts/farmshare/env.sh ] && source scripts/farmshare/env.sh
P=.venv/bin/python
echo "[doses] start $(date) on $(hostname)"
$P scripts/capture_doses.py --config configs/main.yaml --model olmo_instruct || echo "WARN olmo failed"
$P scripts/capture_doses.py --config configs/main.yaml --model llama        || echo "WARN llama failed"
$P scripts/capture_doses.py --config configs/ext.yaml  --model tulu_final    || echo "WARN tulu failed"
echo "[doses] done $(date)"
