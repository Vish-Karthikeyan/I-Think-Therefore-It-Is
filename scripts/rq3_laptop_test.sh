#!/usr/bin/env bash
# Laptop test of the open-weight path with OLMo 2 1B (configs/rq3_test.yaml). Free; results meaningless.
# Usage: bash scripts/rq3_laptop_test.sh [stage ...]   (default: all stages, stops at the first failure)
set -uo pipefail
cd "$(dirname "$0")/.."
C=(--config configs/rq3_test.yaml)
M=(--model olmo1b_test)
STAGES=${*:-cues stance elicit fit steer audit}
for s in $STAGES; do
  echo "[$(date +%T)] === $s"
  case $s in
    cues|stance|steer|audit) .venv/bin/idea $s "${C[@]}" "${M[@]}" ;;
    elicit) .venv/bin/idea elicit "${C[@]}" "${M[@]}" --capture ;;
    fit) .venv/bin/idea fit "${C[@]}" --rq2-folds 2 --n-boot 20 ;;
  esac
  rc=$?
  echo "[$(date +%T)] === $s exit=$rc"
  [ $rc -ne 0 ] && exit $rc
done
echo "ALL STAGES OK"
