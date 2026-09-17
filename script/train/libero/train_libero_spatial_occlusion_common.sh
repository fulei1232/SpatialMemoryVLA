#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"
experiment_mode="${EXPERIMENT_MODE:?Set EXPERIMENT_MODE to spatial_forcing or spatial_memory.}"
case "${experiment_mode}" in
  spatial_forcing)
    default_run_id="spatial_forcing_libero_spatial_occ_B"
    spatial_args=(
      --use_spatial_forcing True --use_spatial_memory False
      --spatial_align_layer 24 --spatial_align_coeff 0.5
      --spatial_teacher_feature_layer -1 --spatial_debug_asserts True
    )
    ;;
  spatial_memory)
    default_run_id="spatial_memory_libero_spatial_occ_C_mem${MEM_LENGTH:-16}"
    spatial_args=(
      --use_spatial_forcing True --use_spatial_memory True
      --spatial_align_layer 24 --spatial_align_coeff 0.5
      --spatial_teacher_feature_layer -1 --spatial_debug_asserts True
    )
    ;;
  *)
    echo "ERROR: EXPERIMENT_MODE must be spatial_forcing or spatial_memory." >&2
    exit 2
    ;;
esac

data_root_dir="${DATA_ROOT_DIR:-${storage_root}/datasets/libero-rlds}"
dataset_dir="${data_root_dir}/libero_spatial_no_noops/1.0.0"
shard_count="$(find "${dataset_dir}" -maxdepth 1 -type f -name '*.tfrecord-*' 2>/dev/null | wc -l)"
[[ "${shard_count}" -eq 16 && -f "${dataset_dir}/dataset_info.json" ]] || {
  echo "ERROR: expected 16 LIBERO-Spatial shards under ${dataset_dir}; found ${shard_count}." >&2
  exit 3
}

pretrained_ckpt="${PRETRAINED_CHECKPOINT:-${storage_root}/pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
vggt_ckpt="${VGGT_CHECKPOINT:-${storage_root}/pretrained/VGGT-1B/model.pt}"
[[ -f "${pretrained_ckpt}" ]] || { echo "ERROR: missing checkpoint: ${pretrained_ckpt}" >&2; exit 4; }
[[ -f "${vggt_ckpt}" ]] || { echo "ERROR: missing VGGT teacher: ${vggt_ckpt}" >&2; exit 5; }
spatial_args+=(--spatial_teacher_path "${vggt_ckpt}")

n_gpu="${N_GPU:-4}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-8}"
global_batch_size="${GLOBAL_BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-20}"
save_interval="${SAVE_INTERVAL:-20}"
seed="${SEED:-42}"
mem_length="${MEM_LENGTH:-16}"
run_id="${RUN_ID:-${default_run_id}_${max_steps}step}"
run_root_dir="${RUN_ROOT_DIR:-${storage_root}/runs/libero_spatial_memory}"
repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS:-1}"
shuffle_buffer_size="${SHUFFLE_BUFFER_SIZE:-1024}"
image_aug="${IMAGE_AUG:-False}"
run_dir_name="${run_id}"
[[ "${image_aug}" == "True" ]] && run_dir_name+="--image_aug"

if (( global_batch_size % (n_gpu * per_device_batch_size) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by N_GPU * PER_DEVICE_BATCH_SIZE." >&2
  exit 6
fi
if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${visible_gpus}" .venv/bin/python -c \
    "import torch; n=torch.cuda.device_count(); assert torch.cuda.is_available() and n == ${n_gpu}, f'expected ${n_gpu} GPUs, found {n}'"
fi

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export HF_HOME="${HF_HOME:-${storage_root}/hf-cache}"
export PRISMATIC_LLAMA2_7B_REPO="${LLAMA2_7B_REPO:-${storage_root}/pretrained/NousResearch-Llama-2-7b-hf}"

cat <<EOF
Matched LIBERO-Spatial temporal-occlusion training
group: ${experiment_mode}
run id: ${run_id}
dataset: ${dataset_dir} (16 shards)
initial checkpoint: ${pretrained_ckpt}
GPUs / per-device / global batch: ${n_gpu} / ${per_device_batch_size} / ${global_batch_size}
max steps / save interval: ${max_steps} / ${save_interval}
seed / memory length: ${seed} / ${mem_length}
dataloader: stream
shuffle buffer: ${shuffle_buffer_size}
generic image augmentation: ${image_aug}
curriculum: probability=0.5, visible<0.30, ramp=0.20, full<0.75, then recovery
run root: ${run_root_dir}
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="${visible_gpus}" \
.venv/bin/torchrun --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" \
  --is_resume False \
  --experiment_mode "${experiment_mode}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix libero_spatial_no_noops \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${per_device_batch_size}" \
  --vla.global_batch_size "${global_batch_size}" \
  --vla.learning_rate 2e-5 \
  --vla.max_steps "${max_steps}" \
  --vla.freeze_llm_backbone True \
  --vla.freeze_vision_backbone True \
  --vla.unfreeze_last_llm_layer True \
  --vla.enable_gradient_checkpointing False \
  --vla.shuffle_buffer_size "${shuffle_buffer_size}" \
  --data_root_dir "${data_root_dir}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --save_interval "${save_interval}" \
  --seed "${seed}" \
  --image_aug "${image_aug}" \
  --future_action_window_size 15 \
  --action_dim 7 \
  --action_model_type DiT-L \
  --repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --dataloader_type stream \
  --group_size 16 \
  --mem_length "${mem_length}" \
  --memory_curriculum_enabled True \
  --memory_curriculum_type occlusion \
  --occlusion_probability 0.5 \
  --occlusion_start_ratio 0.3 \
  --occlusion_duration_ratio 0.2 \
  --occlusion_recovery_ratio 0.75 \
  --occlusion_strength full \
  --gate_diagnostics_path "${run_root_dir}/${run_dir_name}/diagnostics/gate.csv" \
  "${spatial_args[@]}" \
  --trackers jsonl
