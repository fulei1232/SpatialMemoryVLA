#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fast parameter-efficient ABC ablation: keep the visual encoder and most of
# Llama frozen, while training the last LLM layer plus the MemoryVLA spatial,
# memory, and action modules.
export N_GPU="${N_GPU:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export SPATIAL_PER_DEVICE_BATCH_SIZE="${SPATIAL_PER_DEVICE_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export SEED="${SEED:-42}"
export REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-1}"
export FREEZE_LLM_BACKBONE="${FREEZE_LLM_BACKBONE:-True}"
export FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE:-True}"
export UNFREEZE_LAST_LLM_LAYER="${UNFREEZE_LAST_LLM_LAYER:-True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-False}"

for group in A B C; do
  case "${group}" in
    A) default_run_id="memoryvla_robomme_A_4gpu_5k_fast" ;;
    B) default_run_id="spatial_forcing_robomme_B_4gpu_5k_fast" ;;
    C) default_run_id="spatial_memory_robomme_C_4gpu_5k_fast" ;;
  esac
  run_id_var="${group}_RUN_ID"
  run_id="${!run_id_var:-${default_run_id}}"

  echo "Starting fast RoboMME ablation group ${group}: run_id=${run_id}"
  RUN_ID="${run_id}" "${script_dir}/train_robomme_${group}.sh"
done
