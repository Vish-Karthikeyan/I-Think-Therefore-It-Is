#!/usr/bin/env bash
#SBATCH --job-name=idea-varstab
#SBATCH --output=/scratch/users/%u/idea/logs/%x-%j.out
#SBATCH --requeue
# Subspace stability under prompt rewording, for a subset of models (reviewer request, 2026-11-16).
#   VS_CONFIG  configs/main.yaml | configs/ext.yaml
#   VS_MODELS  space-separated model keys run in order with that config
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/idea}"
[ -f scripts/farmshare/env.sh ] && source scripts/farmshare/env.sh
echo "[varstab] start $(date) on $(hostname): $VS_CONFIG $VS_MODELS"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
read -r -a MS <<< "$VS_MODELS"
for m in "${MS[@]}"; do .venv/bin/python scripts/capture_variant_stability.py --config "$VS_CONFIG" --model "$m" || echo "WARN $m failed"; done
echo "[varstab] done $(date)"
