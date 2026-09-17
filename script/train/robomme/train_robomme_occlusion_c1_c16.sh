#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"
manifest="${EPISODE_MANIFEST_PATH:-${storage_root}/datasets/robomme_memory/manifest_occ_200_len16.npz}"

[[ -f "${manifest}" ]] || {
  echo "ERROR: missing episode manifest: ${manifest}" >&2
  echo "Build it with script/data/build_robomme_episode_manifest.py first." >&2
  exit 2
}

export N_GPU="${N_GPU:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export SEED="${SEED:-42}"
export REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-1}"
export FREEZE_LLM_BACKBONE="${FREEZE_LLM_BACKBONE:-True}"
export FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE:-True}"
export UNFREEZE_LAST_LLM_LAYER="${UNFREEZE_LAST_LLM_LAYER:-True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-False}"
export DATALOADER_TYPE=stream
export EPISODE_MANIFEST_PATH="${manifest}"
export MEMORY_CURRICULUM_ENABLED=True
export MEMORY_CURRICULUM_TYPE=occlusion
export OCCLUSION_PROBABILITY="${OCCLUSION_PROBABILITY:-0.5}"
export OCCLUSION_START_RATIO="${OCCLUSION_START_RATIO:-0.4}"
export OCCLUSION_DURATION_RATIO="${OCCLUSION_DURATION_RATIO:-0.2}"
export OCCLUSION_STRENGTH="${OCCLUSION_STRENGTH:-full}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${storage_root}/runs/robomme_memory}"

for memory_length in 1 16; do
  run_id="spatial_memory_robomme_occ_c_mem${memory_length}_${MAX_STEPS}step"
  echo "Starting C-memory${memory_length}: ${run_id}"
  EXPERIMENT_MODE=spatial_memory \
  MEM_LENGTH="${memory_length}" \
  RUN_ID="${run_id}" \
  GATE_DIAGNOSTICS_PATH="${RUN_ROOT_DIR}/${run_id}--image_aug/diagnostics/gate.csv" \
  "${script_dir}/train_robomme_common.sh"
done
