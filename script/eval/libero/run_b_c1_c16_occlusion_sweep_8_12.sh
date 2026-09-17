#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"
output_root="${OUTPUT_ROOT:-${storage_root}/evaluations/libero_spatial_occ_b_c1_c16_sweep_8_12}"
trials="${NUM_TRIALS_PER_TASK:-50}"
inference_seed="${INFERENCE_SEED:-7}"

labels=(B C1 C16)
gpus=(0 1 2)
ports=(27840 27841 27842)
checkpoints=(
  "${storage_root}/runs/libero_spatial_memory/libero_spatial_occ_B_500step_det_noaug_rds1/checkpoints/step-000500-epoch-00-loss=0.0809.pt"
  "${storage_root}/runs/libero_spatial_memory/libero_spatial_occ_C_mem1_500step_det_noaug_rds1/checkpoints/step-000500-epoch-00-loss=0.0876.pt"
  "${storage_root}/runs/libero_spatial_memory/libero_spatial_occ_C_mem16_500step_det_noaug_rds1/checkpoints/step-000500-epoch-00-loss=0.0860.pt"
)

for checkpoint in "${checkpoints[@]}"; do
  [[ -f "${checkpoint}" ]] || { echo "ERROR: missing checkpoint ${checkpoint}" >&2; exit 2; }
done
if find "${output_root}" -name rollouts.jsonl -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: ${output_root} already contains rollout records; refusing to append duplicates." >&2
  exit 3
fi
mkdir -p "${output_root}"

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export PRISMATIC_LLAMA2_7B_REPO="${storage_root}/pretrained/NousResearch-Llama-2-7b-hf"
export MKL_INTERFACE_LAYER=GNU

worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_model() {
  local label="$1" gpu="$2" port="$3" checkpoint="$4"
  local deploy_pid=""
  local model_log_root="${output_root}/services/${label}"
  mkdir -p "${model_log_root}"

  model_cleanup() {
    if [[ -n "${deploy_pid}" ]] && kill -0 "${deploy_pid}" 2>/dev/null; then
      kill -TERM "${deploy_pid}" 2>/dev/null || true
      wait "${deploy_pid}" 2>/dev/null || true
    fi
  }
  trap model_cleanup EXIT INT TERM

  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python deploy.py \
    --saved_model_path "${checkpoint}" \
    --unnorm_key libero_spatial_no_noops \
    --cfg_scale 1.5 --num_ddim_steps 10 --use_ddim --use_bf16 \
    --inference_seed "${inference_seed}" \
    --port "${port}" --action_chunking --action_chunking_window 8 \
    > "${model_log_root}/deploy.log" 2>&1 &
  deploy_pid=$!

  local ready=0
  for _ in $(seq 1 240); do
    kill -0 "${deploy_pid}" 2>/dev/null || break
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" -eq 1 ]] || { echo "ERROR: ${label} deploy failed to become healthy" >&2; return 4; }

  # The ratios below give exactly 5 visible calls, 3 partial calls, then
  # full_calls fully black calls. Recovery begins at call 8 + full_calls.
  for full_calls in 8 12; do
    local schedule_steps partial_ratio full_ratio
    if [[ "${full_calls}" -eq 8 ]]; then
      schedule_steps=17
      partial_ratio=0.3125  # 5 / 16
      full_ratio=0.5       # 8 / 16
    else
      schedule_steps=21
      partial_ratio=0.25   # 5 / 20
      full_ratio=0.4       # 8 / 20
    fi
    for task_id in 0 8; do
      local task_dir="${output_root}/full${full_calls}/${label}/temporal_occlusion/task-${task_id}"
      mkdir -p "${task_dir}"
      env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
        .venv/bin/python evaluation/libero/eval_libero.py \
          --model "${label}" \
          --task_suite_name libero_spatial \
          --num_trials_per_task "${trials}" \
          --spcial_task_id "${task_id}" \
          --seed "${inference_seed}" \
          --hard_case temporal_occlusion \
          --occlusion_schedule_steps "${schedule_steps}" \
          --occlusion_partial_start_ratio "${partial_ratio}" \
          --occlusion_full_start_ratio "${full_ratio}" \
          --occlusion_recovery_start_ratio 1.0 \
          --save_rollout_videos False \
          --run_id_note "${label}-full${full_calls}-task${task_id}" \
          --local_log_dir "${task_dir}" \
          --port "${port}" \
          > "${task_dir}/console.log" 2>&1
    done
    echo "[$(date --iso-8601=seconds)] ${label} full${full_calls} complete" >> "${output_root}/progress.log"
  done

  model_cleanup
  trap - EXIT INT TERM
  echo "[$(date --iso-8601=seconds)] ${label} all lengths complete" >> "${output_root}/progress.log"
}

echo "[$(date --iso-8601=seconds)] sweep start: visible=5, partial=3, full={8,12}; ${trials} states/task" > "${output_root}/progress.log"
for i in "${!labels[@]}"; do
  run_model "${labels[$i]}" "${gpus[$i]}" "${ports[$i]}" "${checkpoints[$i]}" \
    > "${output_root}/${labels[$i]}.worker.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "${pid}" || status=1
done
worker_pids=()
[[ "${status}" -eq 0 ]] || { echo "ERROR: at least one model evaluation failed" >&2; exit 5; }
echo "[$(date --iso-8601=seconds)] sweep complete" >> "${output_root}/progress.log"
