#!/usr/bin/env bash
# Gemini 2.5 Flash (full design) and Claude Sonnet 5 (6,000-trial labels-only subset) through OpenRouter
# with OPENROUTER_API_KEY_CLAUDE (project .env). Budgets (user, 2026-11-15): Gemini $20, Claude $30 in total.
# Each step's cost = change in the key's usage counter; appended to results/compute/api_spend_2026-11-15.csv.
set -uo pipefail
cd "$(dirname "$0")/.."
LEDGER=results/compute/api_spend_2026-11-15.csv
LOG=runs/main-8b15a8c4/logs; mkdir -p "$LOG"
C=(--config configs/main.yaml)

usage() { .venv/bin/python - <<'EOF'
import json, urllib.request
from pathlib import Path
env = dict(l.split("=", 1) for l in Path("../.env").read_text().splitlines() if "=" in l and not l.startswith("#"))
k = env["OPENROUTER_API_KEY_CLAUDE"].strip().strip('"').strip("'")
req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {k}"})
print(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
EOF
}
spent() { .venv/bin/python -c "import pandas as pd; d=pd.read_csv('$LEDGER'); print(round(d[d.model=='$1'].cost_usd.sum(), 6))"; }

step() {  # step <model> <label> <cap> <idea args...>
  local model=$1 label=$2 cap=$3; shift 3
  local u0 u1 rc
  u0=$(usage)
  echo "[$(date +%T)] $model: $label (cap \$$cap)"
  IDEA_MAX_COST=$cap .venv/bin/idea "$@" "${C[@]}" > "$LOG/api2_${model}_${label// /_}.log" 2>&1; rc=$?
  u1=$(usage)
  local cost; cost=$(.venv/bin/python -c "print(round($u1 - $u0, 6))")
  echo "2026-11-15,$model,$label,,$cost,OPENROUTER_API_KEY_CLAUDE,exit $rc" >> "$LEDGER"
  echo "[$(date +%T)] $model: $label exit=$rc cost=\$$cost (model total \$$(spent $model))"
  return $rc
}

# ---- Gemini: vignette check, stance, trials (total budget $20)
step gemini "vignette check" 2 cues --model gemini || exit 1
step gemini "stance" 5 stance --model gemini --fmt chat || exit 1
left=$(.venv/bin/python -c "print(round(min(15, 20 - $(spent gemini)), 4))")
step gemini "trials" "$left" elicit --model gemini --fmt chat || exit 1

# ---- Claude: trials in pieces under the $15 per-run cap (total budget $30)
for piece in 1 2; do
  left=$(.venv/bin/python -c "print(round(min(15, 30 - $(spent claude)), 4))")
  step claude "trials piece $piece" "$left" elicit --model claude --fmt chat; rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -ne 4 ] && exit 1      # 4 = stopped at the spending cap: continue in the next piece (cache resumes)
done
echo "[$(date +%T)] done. Totals: gemini \$$(spent gemini), claude \$$(spent claude)"
