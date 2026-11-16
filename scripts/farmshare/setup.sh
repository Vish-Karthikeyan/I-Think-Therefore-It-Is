#!/usr/bin/env bash
# One-time setup on a FarmShare LOGIN node (has internet): virtualenv, package, model
# weights and benchmark datasets, all into scratch. Safe to rerun; downloads resume.
#
#   bash scripts/farmshare/setup.sh            # everything
#   SKIP_WEIGHTS=1 bash scripts/farmshare/setup.sh   # code only
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/farmshare/env.sh

if ! command -v uv >/dev/null 2>&1; then
  # FarmShare's Python refuses `pip install --user` (PEP 668), so give uv its own venv.
  python3 -m venv "$HOME/.local/uv-venv"
  "$HOME/.local/uv-venv/bin/pip" install -q uv
  mkdir -p "$HOME/.local/bin" && ln -sf "$HOME/.local/uv-venv/bin/uv" "$HOME/.local/bin/uv"
fi
uv --version
uv venv --allow-existing .venv --python 3.11
uv pip install -p .venv -e ".[hf,audit,dev]"
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "NOTE: no HF_TOKEN (put it in ~/.idea_hf_token, chmod 600). Llama-3.1-8B is gated and will be skipped." >&2
  fi
  .venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
from idea.models import MODELS
token = os.environ.get("HF_TOKEN")
for spec in MODELS.values():
    if spec.backend != "hf" or not spec.hf_id:
        continue
    if spec.hf_id.startswith("meta-llama/") and not token:
        print("skip (no token):", spec.hf_id); continue
    path = snapshot_download(spec.hf_id, revision=spec.revision, token=token,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"])
    print("ok", spec.hf_id, "->", path)
# Datasets used by RQ3 contrast sets and the audit (cached for offline GPU nodes)
from datasets import load_dataset
for args in (("cais/mmlu", "all"),):
    for split in ("test", "validation"):
        load_dataset(*args, split=split); print("ok dataset", args, split)
try:
    load_dataset("TIGER-Lab/MMLU-Pro", split="test"); print("ok dataset MMLU-Pro")
except Exception as e:
    print("MMLU-Pro prefetch failed (audit will need internet):", e)
PY
fi
du -sh "$HF_HOME" 2>/dev/null || true
echo "Setup done. Next: bash scripts/farmshare/check_gpu.sh"
