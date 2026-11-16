#!/usr/bin/env bash
# Package code + data + generated trial designs for upload as a Kaggle Dataset ("idea-code").
# Usage: scripts/make_kaggle_bundle.sh [run-folder ...]   (default: every runs/*/trials.parquet)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/idea-code
rm -rf "$OUT" && mkdir -p "$OUT"
cp -R pyproject.toml README.md src data configs scripts notebooks tests "$OUT"/
runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then runs=(runs/*/); fi
for r in "${runs[@]}"; do
  r=${r%/}
  [ -f "$r/trials.parquet" ] || continue
  mkdir -p "$OUT/$r"
  # design + small results only (no cache, no activations)
  for f in config.yaml trials.parquet items.parquet design_summary.csv screen.json prior_means.json; do
    [ -f "$r/$f" ] && cp "$r/$f" "$OUT/$r/"
  done
  for d in cues stance responses; do
    [ -d "$r/$d" ] && cp -R "$r/$d" "$OUT/$r/"
  done
done
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
rm -f "$OUT/.env"
(cd dist && rm -f idea-code.zip && zip -qr idea-code.zip idea-code)
echo "wrote dist/idea-code.zip ($(du -h dist/idea-code.zip | cut -f1)). Upload it as a Kaggle Dataset named idea-code."
