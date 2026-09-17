#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

output_root="${OUTPUT_ROOT:-/media/fulei/jlu/SpatialMemoryVLA/evaluations/libero_memory_occlusion_smoke_deterministic}"
checkpoint="${CHECKPOINT:-/media/fulei/jlu/SpatialMemoryVLA/pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-27820}"
trials="${NUM_TRIALS_PER_TASK:-2}"
mkdir -p "${output_root}"

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export PRISMATIC_LLAMA2_7B_REPO="/media/fulei/jlu/SpatialMemoryVLA/pretrained/NousResearch-Llama-2-7b-hf"
export MKL_INTERFACE_LAYER=GNU

deploy_pid=""
cleanup() {
  if [[ -n "${deploy_pid}" ]] && kill -0 "${deploy_pid}" 2>/dev/null; then
    kill -TERM "${deploy_pid}" 2>/dev/null || true
    wait "${deploy_pid}" 2>/dev/null || true
  fi
  deploy_pid=""
}
trap cleanup EXIT

start_deploy() {
  local mode="$1"
  local reset_arg=()
  [[ "${mode}" == "reset" ]] && reset_arg+=(--reset_memory_every_step)
  CUDA_VISIBLE_DEVICES="${gpu_id}" .venv/bin/python deploy.py \
    --saved_model_path "${checkpoint}" \
    --unnorm_key libero_spatial_no_noops \
    --cfg_scale 1.5 --num_ddim_steps 10 --use_ddim --use_bf16 \
    --inference_seed 7 \
    --port "${port}" --action_chunking --action_chunking_window 8 \
    "${reset_arg[@]}" \
    > "${output_root}/deploy-${mode}.log" 2>&1 &
  deploy_pid=$!
  for _ in $(seq 1 240); do
    kill -0 "${deploy_pid}" 2>/dev/null || return 1
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_condition() {
  local memory_mode="$1" condition="$2" task_id="$3"
  local task_dir="${output_root}/${memory_mode}/${condition}/task-${task_id}"
  mkdir -p "${task_dir}"
  env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES="${gpu_id}" \
    .venv/bin/python evaluation/libero/eval_libero.py \
      --model "C16-${memory_mode}" \
      --task_suite_name libero_spatial \
      --num_trials_per_task "${trials}" \
      --spcial_task_id "${task_id}" \
      --seed 7 \
      --hard_case "${condition}" \
      --run_id_note "${memory_mode}-${condition}-task${task_id}" \
      --local_log_dir "${task_dir}" \
      --port "${port}" \
      > "${task_dir}/console.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] starting LIBERO temporal-occlusion smoke"
start_deploy memory
for condition in normal temporal_occlusion; do
  for task_id in 0 8; do
    run_condition memory "${condition}" "${task_id}"
  done
done
cleanup

start_deploy reset
for task_id in 0 8; do
  run_condition reset temporal_occlusion "${task_id}"
done
cleanup
echo "[$(date --iso-8601=seconds)] smoke complete: ${output_root}"
