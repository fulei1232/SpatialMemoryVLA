#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_GPU="${N_GPU:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# The original global batch 256 suite was estimated at roughly 113 hours.
# Global batch 64 cuts the per-step compute to about one quarter, giving an
# estimated 28.3-hour ABC run. Keep these values identical across ablations.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export SPATIAL_PER_DEVICE_BATCH_SIZE="${SPATIAL_PER_DEVICE_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export SEED="${SEED:-42}"

for group in A B C; do
  case "${group}" in
    A) default_run_id="memoryvla_robomme_A_4gpu_5k_gb64_30h" ;;
    B) default_run_id="spatial_forcing_robomme_B_4gpu_5k_gb64_30h" ;;
    C) default_run_id="spatial_memory_robomme_C_4gpu_5k_gb64_30h" ;;
  esac
  run_id_var="${group}_RUN_ID"
  run_id="${!run_id_var:-${default_run_id}}"

  echo "Starting RoboMME ablation group ${group}: run_id=${run_id}"
  RUN_ID="${run_id}" "${script_dir}/train_robomme_${group}.sh"
done
