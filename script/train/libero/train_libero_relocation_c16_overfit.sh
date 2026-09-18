#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"
storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"
data_root_dir="${DATA_ROOT_DIR:-${storage_root}/datasets/libero_relocation_c16_50}"
pretrained_ckpt="${PRETRAINED_CHECKPOINT:-${storage_root}/runs/libero_spatial_memory/libero_spatial_occ_C_mem16_500step_det_noaug_rds1/checkpoints/step-000500-epoch-00-loss=0.0860.pt}"
vggt_ckpt="${VGGT_CHECKPOINT:-${storage_root}/pretrained/VGGT-1B/model.pt}"
run_root_dir="${RUN_ROOT_DIR:-${storage_root}/runs/libero_spatial_memory}"
n_gpu="${N_GPU:-4}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-8}"
global_batch_size="${GLOBAL_BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-20}"
save_interval="${SAVE_INTERVAL:-${max_steps}}"
seed="${SEED:-42}"
run_id="${RUN_ID:-libero_relocation_C16_overfit_${max_steps}step}"

[[ -d "${data_root_dir}/episodes" && -f "${data_root_dir}/action_stats.json" ]] || {
  echo "ERROR: relocation dataset missing under ${data_root_dir}" >&2; exit 2;
}
[[ -f "${pretrained_ckpt}" && -f "${vggt_ckpt}" ]] || {
  echo "ERROR: pretrained checkpoint or VGGT checkpoint missing" >&2; exit 3;
}
if (( global_batch_size % (n_gpu * per_device_batch_size) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by N_GPU * PER_DEVICE_BATCH_SIZE" >&2; exit 4
fi
export LIBERO_CONFIG_PATH="${project_root}/.libero"
export HF_HOME="${HF_HOME:-${storage_root}/hf-cache}"
export PRISMATIC_LLAMA2_7B_REPO="${storage_root}/pretrained/NousResearch-Llama-2-7b-hf"

CUDA_VISIBLE_DEVICES="${visible_gpus}" .venv/bin/torchrun --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" --is_resume False \
  --experiment_mode spatial_memory \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix libero_relocation_npz \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${per_device_batch_size}" \
  --vla.global_batch_size "${global_batch_size}" \
  --vla.learning_rate 2e-5 --vla.max_steps "${max_steps}" \
  --vla.freeze_llm_backbone True --vla.freeze_vision_backbone True \
  --vla.unfreeze_last_llm_layer True --vla.enable_gradient_checkpointing False \
  --vla.shuffle_buffer_size 1 \
  --data_root_dir "${data_root_dir}" --run_root_dir "${run_root_dir}" --run_id "${run_id}" \
  --save_interval "${save_interval}" --seed "${seed}" --image_aug False \
  --future_action_window_size 15 --action_dim 7 --action_model_type DiT-L \
  --repeated_diffusion_steps 1 --dataloader_type stream --group_size 16 --mem_length 16 \
  --memory_curriculum_enabled False --memory_curriculum_type normal \
  --use_spatial_forcing True --use_spatial_memory True \
  --spatial_align_layer 24 --spatial_align_coeff 0.5 \
  --spatial_teacher_path "${vggt_ckpt}" --spatial_teacher_feature_layer -1 \
  --spatial_debug_asserts True \
  --gate_diagnostics_path "${run_root_dir}/${run_id}/diagnostics/gate.csv" \
  --trackers jsonl
