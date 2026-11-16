#!/usr/bin/env bash
# Submit an IDEA phase to FarmShare as a chain of Slurm jobs. Run from the repo root on a
# login node, after setup.sh and check_gpu.sh.
#
#   bash scripts/farmshare/submit.sh pilot
#   PILOT_RUN=$IDEA_RUNS_DIR/pilot-0efff380 bash scripts/farmshare/submit.sh main
#   bash scripts/farmshare/submit.sh main-rq3      # only steer + audit (after `fit` ran)
#   PILOT_RUN=... bash scripts/farmshare/submit.sh main-split   # main run in short parallel pieces
#   bash scripts/farmshare/submit.sh ext-split    # exploratory additions (configs/ext.yaml), low priority
#   RQ3_MODELS=llama bash scripts/farmshare/submit.sh rq3-steer        # steering only (no audit)
#   AUDIT_MODELS=olmo_instruct AUDIT_CONDS=steered [AUDIT_DEP=<jobid>] bash scripts/farmshare/submit.sh rq3-audit
#
# pilot:    per open model (GPU, in parallel): cues, stance, elicit
#           [API models as CPU jobs if IDEA_API=1]
#           then (CPU, after all succeed): screen, pilot report, power, prereg
# main:     generate from the pilot (CPU) -> per model cues, stance, elicit with activations (GPU)
#           -> fit: RQ1, secondary, RQ2, RQ2b (CPU) -> steer + audit for olmo_instruct, llama (GPU)
#
# Settings (export to override):
GPU_PART="${GPU_PART:-gpu,normal}"   # either partition reaches the L40S nodes; whichever starts first
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"       # FarmShare oat nodes: NVIDIA L40S 48 GB, so 1 per model
GPU_TIME="${GPU_TIME:-06:00:00}"         # short requests backfill sooner; jobs requeue and resume at the limit
GPU_MEM="${GPU_MEM:-64000M}"   # one GPU's share of an oat node (256000M / 4 GPUs); 16 CPUs at 4000M/CPU
CPU_PART="${CPU_PART:-normal}"
CPU_TIME="${CPU_TIME:-12:00:00}"
CPU_MEM="${CPU_MEM:-32G}"
QOS="${QOS:-}"                           # e.g. long, if your jobs need more than the usual limit
OPEN_MODELS="${OPEN_MODELS:-olmo_base olmo_sft olmo_dpo olmo_instruct llama}"
RQ3_MODELS="${RQ3_MODELS:-olmo_instruct llama}"
IDEA_API="${IDEA_API:-0}"                # 1: also run gpt + claude here as CPU jobs
API_MODELS="${API_MODELS:-gpt claude}"
DRY="${DRY:-0}"                          # 1: print the sbatch commands only
# main-split: pieces per model, piece time limit, and the stop margin before it (seconds).
# Short pieces fit into scheduler gaps; each piece has its own cache file (safe on NFS).
SPLIT="${SPLIT:-olmo_base:2 olmo_sft:3 olmo_dpo:3 olmo_instruct:8 llama:4}"
PIECE_TIME="${PIECE_TIME:-01:00:00}"
PIECE_MARGIN="${PIECE_MARGIN:-600}"
PREP_TIME="${PREP_TIME:-01:00:00}"
# ext-split: model:pieces[:formats] (formats default to the model's own). Jobs get a large nice
# value so the main run's pieces always go first among this user's jobs.
# Sized from the first main-run piece (2026-11-11: 6,325 OLMo-7B trials in 11 min) to ~40 min each.
EXT_SPLIT="${EXT_SPLIT:-olmo13_base:2 olmo13_sft:3 olmo13_dpo:3 olmo13_instruct:4 phi4:3
  tulu_base:1 tulu_sft:2 tulu_dpo:2 tulu_final:2 olmo_instruct_sp:2 llama_sp:2
  olmo_sft:1:chat olmo_dpo:1:chat olmo_instruct_pv1:1 olmo_instruct_pv2:1 llama_pv1:1 llama_pv2:1
  olmo13_instruct_pv1:1 olmo13_instruct_pv2:1 phi4_pv1:1 phi4_pv2:1}"
EXT_DIAG_ONLY="${EXT_DIAG_ONLY-olmo_base olmo_sft olmo_dpo olmo_instruct llama}"   # main-run models
EXT_NICE="${EXT_NICE:-100000000}"
# rq3-audit: the benchmark audit in parts (audit --part), one job per condition x component.
AUDIT_MODELS="${AUDIT_MODELS:-olmo_instruct llama}"
AUDIT_CONDS="${AUDIT_CONDS:-unsteered steered random}"
AUDIT_MMLU_GROUPS="${AUDIT_MMLU_GROUPS:-4}"
AUDIT_SHORT_TIME="${AUDIT_SHORT_TIME:-01:00:00}"   # syceval, bbq
AUDIT_MMLU_TIME="${AUDIT_MMLU_TIME:-04:00:00}"     # 3-4 MMLU-Pro subjects; cut-off parts resume from the lm-eval cache
AUDIT_DEP="${AUDIT_DEP:-}"
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/farmshare/env.sh
PHASE="${1:?usage: submit.sh pilot|main|main-rq3}"
case "$PHASE" in
  pilot) CONFIG=configs/pilot.yaml ;;
  main|main-rq3|main-split|rq3-steer|rq3-audit) CONFIG=configs/main.yaml ;;
  ext-split) CONFIG=configs/ext.yaml ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
[ -f "$CONFIG" ] || { echo "missing $CONFIG" >&2; exit 1; }
[ -x .venv/bin/idea ] || { echo "run scripts/farmshare/setup.sh first" >&2; exit 1; }

qos_arg=(); [ -n "$QOS" ] && qos_arg=(--qos "$QOS")
JOBS=()
submit() {  # submit <name> <gpu|cpu> <model> <stages> [dependency]
  # optional: TIME=hh:mm:ss and EXTRA_EXPORT=",VAR=value,..." set per call
  local name=$1 kind=$2 model=$3 stages=$4 dep=${5:-}
  local res=()
  if [ "$kind" = gpu ]; then
    res=(-p "$GPU_PART" -G "$GPUS_PER_JOB" -t "${TIME:-$GPU_TIME}" --mem "$GPU_MEM" -c 4)
  else
    res=(-p "$CPU_PART" -t "$CPU_TIME" --mem "$CPU_MEM" -c 4)
  fi
  local depa=(); [ -n "$dep" ] && depa=(--dependency "afterok:$dep" --kill-on-invalid-dep=yes)
  local exp="ALL,IDEA_CONFIG=$CONFIG,IDEA_MODEL=$model,IDEA_STAGES=$stages,IDEA_PILOT_RUN=${PILOT_RUN:-}${EXTRA_EXPORT:-}"
  local nicea=(); [ -n "${NICE:-}" ] && nicea=(--nice="$NICE")
  local cmd=(sbatch --parsable -J "idea-$name" "${res[@]}" ${qos_arg[@]+"${qos_arg[@]}"} ${depa[@]+"${depa[@]}"} ${nicea[@]+"${nicea[@]}"} --export="$exp" scripts/farmshare/idea_job.sbatch)
  if [ "$DRY" = 1 ]; then echo "${cmd[*]}" >&2; echo "DRY$RANDOM"; return; fi
  "${cmd[@]}"
}
join() { local IFS=:; echo "$*"; }

if [ "$PHASE" = pilot ]; then
  [ -f "$IDEA_RUNS_DIR"/pilot-*/trials.parquet ] 2>/dev/null || submit_gen=1
  gen=""
  if [ "${submit_gen:-0}" = 1 ]; then gen=$(submit gen-pilot cpu "" "generate"); echo "generate: $gen"; fi
  for m in $OPEN_MODELS; do id=$(submit "$m" gpu "$m" "cues stance elicit" "$gen"); JOBS+=("$id"); echo "$m: $id"; done
  if [ "$IDEA_API" = 1 ]; then
    for m in $API_MODELS; do id=$(submit "$m" cpu "$m" "cues stance elicit" "$gen"); JOBS+=("$id"); echo "$m: $id"; done
  fi
  rep=$(submit pilot-report cpu "" "screen pilot power prereg" "$(join "${JOBS[@]}")")
  echo "pilot report (after all): $rep"
  [ "$IDEA_API" = 1 ] || echo "NOTE: copy the API models' responses/ stance/ cues/ into the pilot run folder before the report job starts, or it will screen and report on the open models only."
fi

if [ "$PHASE" = main ]; then
  : "${PILOT_RUN:?set PILOT_RUN to the pilot run folder (for screen.json and prior_means.json)}"
  gen=$(submit gen-main cpu "" "generate"); echo "generate: $gen"
  for m in $OPEN_MODELS; do id=$(submit "$m" gpu "$m" "cues stance elicit" "$gen"); JOBS+=("$id"); echo "$m: $id"; done
  if [ "$IDEA_API" = 1 ]; then
    for m in $API_MODELS; do id=$(submit "$m" cpu "$m" "cues stance elicit" "$gen"); JOBS+=("$id"); echo "$m: $id"; done
  fi
  fit=$(submit fit cpu "" "fit" "$(join "${JOBS[@]}")"); echo "fit (after all): $fit"
  for m in $RQ3_MODELS; do echo "$m rq3 (after fit): $(submit "rq3-$m" gpu "$m" "steer audit" "$fit")"; done
fi

if [ "$PHASE" = main-split ]; then
  # trials.parquet must already exist in the main run folder (idea generate ... --pilot-run).
  # Per model: one prep job (cues + stance), then its pieces (elicit --shard i/n) in parallel;
  # fit after every piece; RQ3 after fit.
  for spec in $SPLIT; do
    m=${spec%%:*}; n=${spec##*:}
    prep=$(TIME="$PREP_TIME" submit "prep-$m" gpu "$m" "cues stance"); echo "$m prep: $prep"
    for ((i = 0; i < n; i++)); do
      id=$(TIME="$PIECE_TIME" EXTRA_EXPORT=",IDEA_EXTRA=--shard $i/$n,IDEA_DEADLINE_MARGIN=$PIECE_MARGIN" \
           submit "$m-$i-of-$n" gpu "$m" "elicit" "$prep")
      JOBS+=("$id"); echo "$m piece $i/$n: $id"
    done
  done
  fit=$(submit fit cpu "" "fit" "$(join "${JOBS[@]}")"); echo "fit (after all pieces): $fit"
  for m in $RQ3_MODELS; do echo "$m rq3 (after fit): $(submit "rq3-$m" gpu "$m" "steer audit" "$fit")"; done
fi

if [ "$PHASE" = ext-split ]; then
  # trials.parquet must already exist in the ext run folder (idea generate --config configs/ext.yaml ...).
  # Per model: prep (cues, stance, evidence diagnostic), then its elicit pieces in parallel.
  export NICE="$EXT_NICE"
  for m in $EXT_DIAG_ONLY; do echo "$m diag: $(TIME="$PREP_TIME" submit "diag-$m" gpu "$m" "diag")"; done
  for spec in $EXT_SPLIT; do
    IFS=: read -r m n fm <<< "$spec"
    fexp=""; [ -n "${fm:-}" ] && fexp=",IDEA_FORMATS=$fm"
    prep_stages="cues stance diag"
    case "$m" in *_sp|*_pv1|*_pv2) prep_stages="cues stance" ;; esac          # aliases: same weights as a diag'd model
    case "$m" in olmo_sft|olmo_dpo) prep_stages="cues" ;; esac                # activations only; diag runs above
    prep=$(TIME="$PREP_TIME" EXTRA_EXPORT="$fexp" submit "xprep-$m" gpu "$m" "$prep_stages"); echo "$m prep: $prep"
    for ((i = 0; i < n; i++)); do
      id=$(TIME="$PIECE_TIME" EXTRA_EXPORT="$fexp,IDEA_EXTRA=--shard $i/$n,IDEA_DEADLINE_MARGIN=$PIECE_MARGIN" \
           submit "x-$m-$i-of-$n" gpu "$m" "elicit" "$prep")
      echo "$m piece $i/$n: $id"
    done
  done
fi

if [ "$PHASE" = rq3-steer ]; then
  for m in $RQ3_MODELS; do echo "$m steer: $(submit "steer-$m" gpu "$m" "steer")"; done
fi

if [ "$PHASE" = rq3-audit ]; then
  for m in $AUDIT_MODELS; do
    for c in $AUDIT_CONDS; do
      for w in syceval bbq $(for ((i = 0; i < AUDIT_MMLU_GROUPS; i++)); do echo "mmlu_pro:${i}of${AUDIT_MMLU_GROUPS}"; done); do
        t="$AUDIT_SHORT_TIME"; case "$w" in mmlu_pro*) t="$AUDIT_MMLU_TIME" ;; esac
        id=$(TIME="$t" EXTRA_EXPORT=",IDEA_EXTRA=--part $c/$w" submit "aud-$m-$c-${w//:/-}" gpu "$m" "audit" "$AUDIT_DEP")
        echo "$m audit $c/$w: $id"
      done
    done
  done
fi

if [ "$PHASE" = main-rq3 ]; then
  for m in $RQ3_MODELS; do echo "$m rq3: $(submit "rq3-$m" gpu "$m" "steer audit")"; done
fi
echo "Watch: squeue -u $USER    Logs: $IDEA_SCRATCH/logs/"
