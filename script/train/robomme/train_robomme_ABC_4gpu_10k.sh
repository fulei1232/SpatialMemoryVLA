#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_GPU="${N_GPU:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
export SPATIAL_PER_DEVICE_BATCH_SIZE="${SPATIAL_PER_DEVICE_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
export SEED="${SEED:-42}"

for group in A B C; do
  echo "Starting RoboMME ablation group ${group}"
  case "${group}" in
    A) run_id="memoryvla_robomme_A_4gpu_10k" ;;
    B) run_id="spatial_forcing_robomme_B_4gpu_10k" ;;
    C) run_id="spatial_memory_robomme_C_4gpu_10k" ;;
  esac
  RUN_ID="${run_id}" "${script_dir}/train_robomme_${group}.sh"
done
