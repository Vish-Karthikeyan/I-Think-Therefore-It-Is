# Shared environment for IDEA on Stanford FarmShare. Sourced by setup.sh and idea_job.sbatch.
# Override any of these by exporting them before you run setup.sh / submit.sh.

# Code lives in home (50 GB quota, backed up); weights, cache, activations and runs live in
# scratch (purged of files untouched for 90 days, so keep a copy of results/ elsewhere).
export IDEA_HOME="${IDEA_HOME:-$HOME/idea}"
export IDEA_SCRATCH="${IDEA_SCRATCH:-/scratch/users/$USER/idea}"
export IDEA_RUNS_DIR="${IDEA_RUNS_DIR:-$IDEA_SCRATCH/runs}"
export HF_HOME="${HF_HOME:-$IDEA_SCRATCH/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$IDEA_SCRATCH/uv-cache}"
export TOKENIZERS_PARALLELISM=false
export IDEA_SQLITE_JOURNAL=DELETE   # scratch is NFS: no SQLite WAL

# Set to 1 if GPU nodes cannot reach the internet: jobs then read weights and datasets only
# from what setup.sh downloaded into $HF_HOME on the login node.
export IDEA_OFFLINE="${IDEA_OFFLINE:-0}"
if [ "$IDEA_OFFLINE" = "1" ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

# Hugging Face token for the gated Llama weights: keep it in a file only you can read.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.idea_hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.idea_hf_token")"
fi

# OpenRouter key for GPT/Claude jobs, if you run the API models here instead of on your Mac.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/.idea_openrouter_key" ]; then
  export OPENROUTER_API_KEY="$(cat "$HOME/.idea_openrouter_key")"
fi

# uv lives in ~/.local/bin (setup.sh installs it there if missing)
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$IDEA_SCRATCH" "$IDEA_RUNS_DIR" "$HF_HOME" "$IDEA_SCRATCH/logs"
module load python/3.12.3 >/dev/null 2>&1 || module load python >/dev/null 2>&1 || true
