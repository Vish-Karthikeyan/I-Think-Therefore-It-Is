#!/usr/bin/env bash
# After the corrected-RQ3 audits finish on FarmShare: copy their result files here, rebuild the
# corrected-RQ3 tables (analysis/11) and the paper tables (analysis/09), and rebuild the secondary paper.
#
#   bash scripts/finish_rq3v2.sh          # copy + rebuild
#   DRY=1 bash scripts/finish_rq3v2.sh    # only list what would be copied
#
# Copies only the small result files (*.csv, *.json, *_l2d_answers.parquet), not the lm-eval caches (*.db).
# Needs the FarmShare SSH connection (ControlPath ~/.ssh/cm/farmshare) to be open.
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=(ssh -o ControlPath=~/.ssh/cm/farmshare -o BatchMode=yes -o ConnectTimeout=15)
HOST=vk27@login.farmshare.stanford.edu
REMOTE=/scratch/users/vk27/idea/runs
# model folder -> run folder on FarmShare (Tulu ran in the exploratory run)
declare -a MODELS=("olmo_instruct_chat main-8b15a8c4" "llama_chat main-8b15a8c4" "tulu_final_chat ext-8b15a8c4")
for m in "${MODELS[@]}"; do
  read -r model run <<< "$m"
  src="$REMOTE/$run/results/rq3_v2/$model/audit2"
  dst="results/rq3_v2/$model/audit2"
  mkdir -p "$dst"
  files=$("${SSH[@]}" "$HOST" "cd $src 2>/dev/null && ls *.csv *.json *_l2d_answers.parquet 2>/dev/null" || true)
  echo "== $model: $(wc -w <<< "$files" | tr -d ' ') files"
  [ -z "$files" ] && continue
  if [ "${DRY:-0}" = 1 ]; then echo "$files" | tr '\n' ' '; echo; continue; fi
  for f in $files; do
    "${SSH[@]}" "$HOST" "cat $src/$f" > "$dst/$f.part" && mv "$dst/$f.part" "$dst/$f"
  done
done
[ "${DRY:-0}" = 1 ] && exit 0
.venv/bin/python analysis/11_rq3_corrected.py | grep -A20 "^== audit"
.venv/bin/python analysis/09_paper_tables.py > /dev/null
(cd ../paper/secondary && bash build.sh > /dev/null 2>&1 && grep -E "Output written|undefined" main.log)
echo "done: tables in analysis/output/rq3_v2/, paper in ../paper/secondary/main.pdf"
