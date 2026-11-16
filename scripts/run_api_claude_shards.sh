#!/usr/bin/env bash
# Claude Sonnet 5 trials (6,000-trial labels-only subset) in 3 shards, each under the $15 per-run cap and
# capped at what is left of Claude's $30 budget. Cost per shard = the run's own reported API usage.
set -uo pipefail
cd "$(dirname "$0")/.."
LEDGER=results/compute/api_spend_2026-11-15.csv
LOG=runs/main-8b15a8c4/logs
spent() { .venv/bin/python -c "import pandas as pd; d=pd.read_csv('$LEDGER'); print(round(d[d.model=='claude'].cost_usd.sum(), 6))"; }
for s in 0 1 2; do
  left=$(.venv/bin/python -c "print(round(min(15, 30 - $(spent)), 4))")
  echo "[$(date +%T)] claude shard $s/3 (cap \$$left)"
  IDEA_MAX_COST=$left .venv/bin/idea elicit --config configs/main.yaml --model claude --fmt chat --shard $s/3 \
      > "$LOG/api2_claude_shard_$s.log" 2>&1; rc=$?
  cost=$(.venv/bin/python - "$LOG/api2_claude_shard_$s.log" <<'EOF'
import ast, re, sys
m = re.findall(r"API usage: (\{.*\})", open(sys.argv[1]).read())
print(ast.literal_eval(m[-1])["cost"] if m else 0.0)
EOF
)
  calls=$(grep -o "'calls': [0-9]*" "$LOG/api2_claude_shard_$s.log" | tail -1 | grep -o "[0-9]*$")
  echo "2026-11-15,claude,trials shard $s/3,${calls:-0},$cost,OPENROUTER_API_KEY_CLAUDE,exit $rc" >> "$LEDGER"
  echo "[$(date +%T)] claude shard $s exit=$rc cost=\$$cost (claude total \$$(spent))"
  [ $rc -ne 0 ] && { echo "stopping: shard $s exit $rc"; exit 1; }
done
echo "[$(date +%T)] done. Claude total \$$(spent)"
