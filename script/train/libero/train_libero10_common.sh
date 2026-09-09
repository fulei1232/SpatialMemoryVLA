#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

experiment_mode="${EXPERIMENT_MODE:?Set EXPERIMENT_MODE to memoryvla, spatial_forcing, or spatial_memory.}"
case "${experiment_mode}" in
  memoryvla)
    default_run_id="memoryvla_libero10_A"
    spatial_args=(--use_spatial_forcing False --use_spatial_memory False)
    ;;
  spatial_forcing)
    default_run_id="spatial_forcing_libero10_B"
    spatial_args=(
      --use_spatial_forcing True
      --use_spatial_memory False
      --spatial_align_layer 24
      --spatial_align_coeff 0.5
      --spatial_teacher_feature_layer -1
      --spatial_debug_asserts True
    )
    ;;
  spatial_memory)
    default_run_id="spatial_memory_libero10_C"
    spatial_args=(
      --use_spatial_forcing True
      --use_spatial_memory True
      --spatial_align_layer 24
      --spatial_align_coeff 0.5
      --spatial_teacher_feature_layer -1
      --spatial_debug_asserts True
    )
    ;;
  *)
    echo "Unknown EXPERIMENT_MODE=${experiment_mode}" >&2
    exit 2
    ;;
esac

data_root_dir="${DATA_ROOT_DIR:-${project_root}/data/libero-rlds}"
dataset_mix="libero_10_no_noops"
dataset_dir="${data_root_dir}/${dataset_mix}/${DATASET_VERSION:-1.0.0}"
if [[ ! -d "${dataset_dir}" ]]; then
  echo "ERROR: required LIBERO-10 RLDS dataset is missing: ${dataset_dir}" >&2
  echo "Refusing to substitute libero_spatial_no_noops or libero_100_no_noops." >&2
  exit 3
fi

pretrained_ckpt="${PRETRAINED_CHECKPOINT:-${project_root}/pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
[[ -f "${pretrained_ckpt}" ]] || {
  echo "ERROR: common MemoryVLA pretrained checkpoint is missing: ${pretrained_ckpt}" >&2
  exit 4
}
is_resume="${IS_RESUME:-False}"
if [[ "${is_resume}" != "True" && "$(basename "${pretrained_ckpt}")" =~ ^step-[0-9]+- ]]; then
  echo "ERROR: matched LIBERO-10 runs must not initialize from an experiment step checkpoint: ${pretrained_ckpt}" >&2
  exit 5
fi
if [[ "${is_resume}" != "True" ]]; then
  checkpoint_run_dir="$(dirname "$(dirname "${pretrained_ckpt}")")"
  checkpoint_config="${checkpoint_run_dir}/config.json"
  checkpoint_stats="${checkpoint_run_dir}/dataset_statistics.json"
  [[ -f "${checkpoint_config}" && -f "${checkpoint_stats}" ]] || {
    echo "ERROR: checkpoint requires config.json and dataset_statistics.json in ${checkpoint_run_dir}" >&2
    exit 5
  }
fi

vggt_ckpt="${VGGT_CHECKPOINT:-${project_root}/pretrained/VGGT-1B/model.pt}"
if [[ "${experiment_mode}" != "memoryvla" ]]; then
  [[ -f "${vggt_ckpt}" ]] || { echo "ERROR: VGGT checkpoint is missing: ${vggt_ckpt}" >&2; exit 6; }
  spatial_args+=(--spatial_teacher_path "${vggt_ckpt}")
fi

torchrun_bin="${TORCHRUN_BIN:-${project_root}/.venv/bin/torchrun}"
[[ -x "${torchrun_bin}" ]] || { echo "ERROR: torchrun is missing: ${torchrun_bin}" >&2; exit 7; }

n_gpu="${N_GPU:-8}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-32}"
global_batch_size="${GLOBAL_BATCH_SIZE:-256}"
max_steps="${MAX_STEPS:-10000}"
save_interval="${SAVE_INTERVAL:-2000}"
seed="${SEED:-42}"
run_id="${RUN_ID:-${default_run_id}}"
run_root_dir="${RUN_ROOT_DIR:-${project_root}/log/libero10}"

if (( global_batch_size % (n_gpu * per_device_batch_size) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by N_GPU * PER_DEVICE_BATCH_SIZE." >&2
  exit 8
fi

resume_args=(--is_resume "${is_resume}")
if [[ "${is_resume}" == "True" ]]; then
  : "${RESUME_STEP:?Set RESUME_STEP when IS_RESUME=True.}"
  : "${RESUME_EPOCH:?Set RESUME_EPOCH when IS_RESUME=True.}"
  resume_args+=(--resume_step "${RESUME_STEP}" --resume_epoch "${RESUME_EPOCH}")
fi

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export PRISMATIC_LLAMA2_7B_REPO="${project_root}/pretrained/NousResearch-Llama-2-7b-hf"

cat <<EOF
LIBERO-10 matched training
pretrained checkpoint: ${pretrained_ckpt}
dataset directory: ${dataset_dir}
dataset mix: ${dataset_mix}
experiment mode: ${experiment_mode}
max steps: ${max_steps}
global batch size: ${global_batch_size}
per-device batch size: ${per_device_batch_size}
gradient accumulation: $((global_batch_size / (n_gpu * per_device_batch_size)))
learning rate: 2e-5
random seed: ${seed}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"${torchrun_bin}" --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" \
  "${resume_args[@]}" \
  --experiment_mode "${experiment_mode}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix "${dataset_mix}" \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${per_device_batch_size}" \
  --vla.global_batch_size "${global_batch_size}" \
  --vla.learning_rate 2e-5 \
  --vla.max_steps "${max_steps}" \
  --vla.shuffle_buffer_size 128000 \
  --data_root_dir "${data_root_dir}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --save_interval "${save_interval}" \
  --seed "${seed}" \
  --image_aug True \
  --future_action_window_size 15 \
  --action_model_type DiT-L \
  --repeated_diffusion_steps 4 \
  --dataloader_type group \
  --mem_length 16 \
  "${spatial_args[@]}" \
  --trackers jsonl
