#!/usr/bin/env bash
# Run every IDEA analysis in order. Usage (from the repository root):
#   bash analysis/run_all.sh            # analyses only (datasets already built)
#   REBUILD=1 bash analysis/run_all.sh  # also rebuild datasets/ from runs/ first
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
[ -x "$PY" ] || PY=python
[ "${REBUILD:-0}" = 1 ] && "$PY" 00_build_datasets.py
for s in main exploratory pilot; do "$PY" 01_descriptives_controls.py --study "$s"; done
for s in main exploratory pilot; do "$PY" 02_rq1_primary.py --study "$s"; done
for s in main exploratory pilot; do "$PY" 03_rq1_secondary.py --study "$s"; done
"$PY" 04_robustness.py
"$PY" 05_rq2_and_secondary_specs.py --study main
"$PY" 06_rq3_steering_audit.py
"$PY" 11_rq3_corrected.py        # corrected RQ3 (answer position)
"$PY" 07_exploratory.py
"$PY" 08_compute_accounting.py   # refresh results/compute/sacct_raw.txt on FarmShare first (see the script)
"$PY" 09_paper_tables.py        # LaTeX tables for paper/secondary/tables/
echo "all analyses done; tables in analysis/output/"
