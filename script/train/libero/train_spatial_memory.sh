#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"
torchrun_bin="${TORCHRUN_BIN:-${project_root}/.venv/bin/torchrun}"
[[ -x "${torchrun_bin}" ]] || { echo "Missing venv torchrun: ${torchrun_bin}" >&2; exit 2; }
export LIBERO_CONFIG_PATH="${project_root}/.libero"

# Keep the HTTP download proxy while preventing Codex's SOCKS proxy from
# overriding requests made by timm/Hugging Face during first-time caching.
unset ALL_PROXY all_proxy
export http_proxy="${http_proxy:-http://192.168.32.28:18000}"
export https_proxy="${https_proxy:-http://192.168.32.28:18000}"

pretrained_ckpt="${PRETRAINED_CHECKPOINT:-./pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
vggt_ckpt='./pretrained/VGGT-1B/model.pt'
export PRISMATIC_LLAMA2_7B_REPO='./pretrained/NousResearch-Llama-2-7b-hf'

experiment_mode="${EXPERIMENT_MODE:-spatial_memory}"
spatial_args=()
case "${experiment_mode}" in
  baseline)
    default_run_id='memoryvla_libero_spatial'
    spatial_args=(--use_spatial_forcing False --use_spatial_memory False)
    ;;
  spatial_forcing)
    default_run_id='memoryvla_spatial_forcing_libero_spatial'
    spatial_args=(--use_spatial_forcing True --use_spatial_memory False --spatial_align_layer 24 --spatial_align_coeff 0.5 --spatial_teacher_path "${vggt_ckpt}" --spatial_teacher_feature_layer -1 --spatial_debug_asserts True)
    ;;
  spatial_memory)
    default_run_id='spatial_memory_libero_spatial'
    spatial_args=(--use_spatial_forcing True --use_spatial_memory True --spatial_align_layer 24 --spatial_align_coeff 0.5 --spatial_teacher_path "${vggt_ckpt}" --spatial_teacher_feature_layer -1 --spatial_debug_asserts True)
    ;;
  *)
    echo "Unknown EXPERIMENT_MODE=${experiment_mode}; choose baseline, spatial_forcing, or spatial_memory." >&2
    exit 2
    ;;
esac

data_root_dir='./data/libero-rlds'
data_mix='libero_spatial_no_noops'
run_root_dir="${RUN_ROOT_DIR:-./log/libero}"
run_id="${RUN_ID:-${default_run_id}}"
n_gpu="${N_GPU:-8}"
bs="${PER_DEVICE_BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-20000}"
save_interval="${SAVE_INTERVAL:-1000}"
is_resume="${IS_RESUME:-False}"

resume_args=(--is_resume "${is_resume}")
if [[ "${is_resume}" == "True" ]]; then
  : "${RESUME_STEP:?Set RESUME_STEP when IS_RESUME=True.}"
  : "${RESUME_EPOCH:?Set RESUME_EPOCH when IS_RESUME=True.}"
  resume_args+=(--resume_step "${RESUME_STEP}" --resume_epoch "${RESUME_EPOCH}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"${torchrun_bin}" --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" \
  "${resume_args[@]}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix "${data_mix}" \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${bs}" \
  --vla.global_batch_size "$((n_gpu * bs))" \
  --vla.learning_rate 2e-5 \
  --vla.max_steps "${max_steps}" \
  --vla.shuffle_buffer_size 128000 \
  --data_root_dir "${data_root_dir}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --save_interval "${save_interval}" \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --action_model_type DiT-L \
  --dataloader_type group \
  "${spatial_args[@]}" \
  --trackers jsonl
