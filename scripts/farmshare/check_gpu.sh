#!/usr/bin/env bash
# What GPUs does the gpu partition have, and can its nodes reach Hugging Face?
# Run on a login node; takes a minute or two of queue time.
set -uo pipefail
PART="${IDEA_GPU_PARTITION:-gpu}"
echo "== nodes, GPUs (GRES) and time limit in partition '$PART'"
sinfo -p "$PART" -o "%N %G %l %D %t"
echo "== partition limits"
scontrol show partition "$PART" | tr ' ' '\n' | grep -E "MaxTime|DefaultTime|MaxNodes|QoS|TRES" || true
echo "== your QoS limits"
sacctmgr -n show assoc user="$USER" format=account,partition,qos%40,maxjobs,grptres%40 2>/dev/null || true
echo "== GPU memory and internet from a GPU node"
srun -p "$PART" -G 1 -t 0:10:00 bash -c '
  nvidia-smi --query-gpu=name,memory.total --format=csv
  if curl -sI --max-time 10 https://huggingface.co >/dev/null; then echo "internet: yes"; else echo "internet: NO -> export IDEA_OFFLINE=1"; fi'
echo
echo "Memory per GPU decides GPUS_PER_JOB in submit.sh: >= 24 GB -> 1 GPU per model (fp16/bf16); 16 GB -> 2."
