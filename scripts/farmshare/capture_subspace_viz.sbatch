#!/usr/bin/env bash
#SBATCH --job-name=idea-subviz
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --requeue
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
[ -f scripts/farmshare/env.sh ] && source scripts/farmshare/env.sh
P=.venv/bin/python
echo "[subviz] start $(date) on $(hostname)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$P scripts/capture_subspace_viz.py --config configs/main.yaml --model olmo_instruct || echo "WARN olmo failed"
$P scripts/capture_subspace_viz.py --config configs/main.yaml --model llama        || echo "WARN llama failed"
$P scripts/capture_subspace_viz.py --config configs/ext.yaml  --model tulu_final    || echo "WARN tulu failed"
echo "[subviz] done $(date)"
