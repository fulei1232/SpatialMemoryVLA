#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

# Keep large datasets, pretrained weights, caches, and run artifacts on shared storage.
storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"

experiment_mode="${EXPERIMENT_MODE:?Set EXPERIMENT_MODE to memoryvla, spatial_forcing, or spatial_memory.}"
case "${experiment_mode}" in
  memoryvla)
    default_run_id="memoryvla_robomme_A_4gpu_5k"
    spatial_args=(--use_spatial_forcing False --use_spatial_memory False)
    ;;
  spatial_forcing)
    default_run_id="spatial_forcing_robomme_B_4gpu_5k"
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
    default_run_id="spatial_memory_robomme_C_4gpu_5k"
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
    echo "ERROR: unknown EXPERIMENT_MODE=${experiment_mode}" >&2
    exit 2
    ;;
esac

data_root_dir="${ROBOMME_DATA_ROOT:-${storage_root}/datasets/robomme_preprocessed_data}"
[[ -d "${data_root_dir}/data" && -f "${data_root_dir}/meta/stats.json" ]] || {
  echo "ERROR: RoboMME preprocessed data is missing under ${data_root_dir}." >&2
  echo "Expected data/part_*.zip (or extracted data/*.pkl) and meta/stats.json from Yinpei/robomme_preprocessed_data." >&2
  exit 3
}
if [[ ! -f "${data_root_dir}/meta/norm_stats.json" ]]; then
  cp "${project_root}/assets/robomme/norm_stats.json" "${data_root_dir}/meta/norm_stats.json"
fi

pretrained_ckpt="${PRETRAINED_CHECKPOINT:-${storage_root}/pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
[[ -f "${pretrained_ckpt}" ]] || {
  echo "ERROR: shared MemoryVLA initialization is missing: ${pretrained_ckpt}" >&2
  echo "Download shihao1895/memvla-libero-spatial into ${storage_root}/pretrained/memvla-libero-spatial." >&2
  exit 4
}
pretrained_run_dir="$(dirname "$(dirname "${pretrained_ckpt}")")"
[[ -f "${pretrained_run_dir}/config.json" && -f "${pretrained_run_dir}/dataset_statistics.json" ]] || {
  echo "ERROR: config.json or dataset_statistics.json is missing beside ${pretrained_ckpt}." >&2
  exit 5
}

vggt_ckpt="${VGGT_CHECKPOINT:-${storage_root}/pretrained/VGGT-1B/model.pt}"
if [[ "${experiment_mode}" != "memoryvla" ]]; then
  [[ -f "${vggt_ckpt}" ]] || {
    echo "ERROR: VGGT checkpoint is required for groups B/C: ${vggt_ckpt}" >&2
    exit 6
  }
  spatial_args+=(--spatial_teacher_path "${vggt_ckpt}")
fi

torchrun_bin="${TORCHRUN_BIN:-${project_root}/.venv/bin/torchrun}"
[[ -x "${torchrun_bin}" ]] || { echo "ERROR: torchrun is missing: ${torchrun_bin}" >&2; exit 7; }

n_gpu="${N_GPU:-4}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-32}"
global_batch_size="${GLOBAL_BATCH_SIZE:-256}"
max_steps="${MAX_STEPS:-5000}"
save_interval="${SAVE_INTERVAL:-2000}"
seed="${SEED:-42}"
repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS:-4}"
freeze_llm_backbone="${FREEZE_LLM_BACKBONE:-False}"
freeze_vision_backbone="${FREEZE_VISION_BACKBONE:-False}"
unfreeze_last_llm_layer="${UNFREEZE_LAST_LLM_LAYER:-False}"
enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
run_id="${RUN_ID:-${default_run_id}}"
run_root_dir="${RUN_ROOT_DIR:-${storage_root}/runs/robomme}"
mem_length="${MEM_LENGTH:-16}"
dataloader_type="${DATALOADER_TYPE:-group}"
memory_curriculum_enabled="${MEMORY_CURRICULUM_ENABLED:-False}"
memory_curriculum_type="${MEMORY_CURRICULUM_TYPE:-normal}"
occlusion_probability="${OCCLUSION_PROBABILITY:-0.5}"
occlusion_start_ratio="${OCCLUSION_START_RATIO:-0.4}"
occlusion_duration_ratio="${OCCLUSION_DURATION_RATIO:-0.2}"
occlusion_strength="${OCCLUSION_STRENGTH:-full}"
episode_manifest_path="${EPISODE_MANIFEST_PATH:-}"
gate_diagnostics_path="${GATE_DIAGNOSTICS_PATH:-${run_root_dir}/${run_id}--image_aug/diagnostics/gate.csv}"

manifest_args=()
if [[ -n "${episode_manifest_path}" ]]; then
  [[ -f "${episode_manifest_path}" ]] || { echo "ERROR: episode manifest is missing: ${episode_manifest_path}" >&2; exit 10; }
  manifest_args=(--episode_manifest_path "${episode_manifest_path}")
fi

if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${visible_gpus}" "${project_root}/.venv/bin/python" -c \
    "import torch; count=torch.cuda.device_count(); assert torch.cuda.is_available() and count == ${n_gpu}, f'expected ${n_gpu} CUDA devices, found {count}'" || {
      echo "ERROR: CUDA is unavailable or the visible GPU count does not match N_GPU=${n_gpu}." >&2
      echo "CUDA_VISIBLE_DEVICES=${visible_gpus}" >&2
      exit 9
    }
fi

if (( global_batch_size % (n_gpu * per_device_batch_size) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by N_GPU * PER_DEVICE_BATCH_SIZE." >&2
  exit 8
fi

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export HF_HOME="${HF_HOME:-${storage_root}/hf-cache}"
export PRISMATIC_LLAMA2_7B_REPO="${LLAMA2_7B_REPO:-${storage_root}/pretrained/NousResearch-Llama-2-7b-hf}"

cat <<EOF
RoboMME SpatialMemoryVLA ablation training
group: ${experiment_mode}
run id: ${run_id}
dataset: ${data_root_dir}
pretrained checkpoint: ${pretrained_ckpt}
VGGT checkpoint: ${vggt_ckpt}
GPUs: ${n_gpu}
CUDA_VISIBLE_DEVICES: ${visible_gpus}
per-device batch: ${per_device_batch_size}
global batch: ${global_batch_size}
gradient accumulation: $((global_batch_size / (n_gpu * per_device_batch_size)))
action dimension: 8
action horizon: 16
max steps: ${max_steps}
save interval: ${save_interval}
run root: ${run_root_dir}
learning rate: 2e-5
seed: ${seed}
repeated diffusion steps: ${repeated_diffusion_steps}
freeze LLM backbone: ${freeze_llm_backbone}
freeze vision backbone: ${freeze_vision_backbone}
unfreeze last LLM layer: ${unfreeze_last_llm_layer}
gradient checkpointing: ${enable_gradient_checkpointing}
memory length/group size: ${mem_length} / 16
dataloader type: ${dataloader_type}
memory curriculum: ${memory_curriculum_enabled} (${memory_curriculum_type})
episode manifest: ${episode_manifest_path:-none}
occlusion probability/start/duration/strength: ${occlusion_probability} / ${occlusion_start_ratio} / ${occlusion_duration_ratio} / ${occlusion_strength}
gate diagnostics: ${gate_diagnostics_path}
EOF

CUDA_VISIBLE_DEVICES="${visible_gpus}" \
"${torchrun_bin}" --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" \
  --is_resume False \
  --experiment_mode "${experiment_mode}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix robomme \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${per_device_batch_size}" \
  --vla.global_batch_size "${global_batch_size}" \
  --vla.learning_rate 2e-5 \
  --vla.max_steps "${max_steps}" \
  --vla.freeze_llm_backbone "${freeze_llm_backbone}" \
  --vla.freeze_vision_backbone "${freeze_vision_backbone}" \
  --vla.unfreeze_last_llm_layer "${unfreeze_last_llm_layer}" \
  --vla.enable_gradient_checkpointing "${enable_gradient_checkpointing}" \
  --vla.shuffle_buffer_size 128000 \
  --data_root_dir "${data_root_dir}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --save_interval "${save_interval}" \
  --seed "${seed}" \
  --image_aug True \
  --future_action_window_size 15 \
  --action_dim 8 \
  --action_model_type DiT-L \
  --repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --dataloader_type "${dataloader_type}" \
  --group_size 16 \
  --mem_length "${mem_length}" \
  --memory_curriculum_enabled "${memory_curriculum_enabled}" \
  --memory_curriculum_type "${memory_curriculum_type}" \
  --occlusion_probability "${occlusion_probability}" \
  --occlusion_start_ratio "${occlusion_start_ratio}" \
  --occlusion_duration_ratio "${occlusion_duration_ratio}" \
  --occlusion_strength "${occlusion_strength}" \
  --gate_diagnostics_path "${gate_diagnostics_path}" \
  "${manifest_args[@]}" \
  "${spatial_args[@]}" \
  --trackers jsonl
