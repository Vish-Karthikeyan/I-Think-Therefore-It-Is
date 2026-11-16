#!/usr/bin/env bash
# End-to-end pipeline test on the simulated model (planted Delta_b=-0.25, theta=-0.15).
set -euo pipefail
cd "$(dirname "$0")/.."
I="${IDEA_BIN:-.venv/bin/idea}"
C=(--config configs/smoke.yaml)
$I generate "${C[@]}"
$I cues "${C[@]}" --model mock
$I stance "${C[@]}" --model mock
$I stance "${C[@]}" --model mock --fmt completion
$I stance "${C[@]}" --model mock_sample
$I screen "${C[@]}"
$I elicit "${C[@]}" --model mock                      # captures activations (capture.models)
$I elicit "${C[@]}" --model mock --fmt completion --shard 0/2
$I elicit "${C[@]}" --model mock --fmt completion --shard 1/2
$I elicit "${C[@]}" --model mock_sample
$I pilot "${C[@]}"
$I power "${C[@]}" --reps 20
$I prereg "${C[@]}" > /dev/null
$I fit "${C[@]}" --rq2-folds 2 --n-boot 20
$I steer "${C[@]}" --model mock
echo "SMOKE OK"
