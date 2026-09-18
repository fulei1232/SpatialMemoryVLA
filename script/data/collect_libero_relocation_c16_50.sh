#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
storage_root="${SPATIALMEMORYVLA_STORAGE_ROOT:-/media/fulei/jlu/SpatialMemoryVLA}"
dataset_root="${DATASET_ROOT:-${storage_root}/datasets/libero_relocation_c16_50}"
staging_root="${dataset_root}/staging"
checkpoint="${storage_root}/runs/libero_spatial_memory/libero_spatial_occ_C_mem16_500step_det_noaug_rds1/checkpoints/step-000500-epoch-00-loss=0.0860.pt"
source_stats="$(find "${storage_root}/datasets/libero-rlds/libero_spatial_no_noops/1.0.0" -name 'dataset_statistics_*.json' -print -quit)"
[[ -f "${checkpoint}" && -f "${source_stats}" ]] || { echo "ERROR: checkpoint or source stats missing" >&2; exit 2; }
if [[ -d "${dataset_root}/episodes" ]] && find "${dataset_root}/episodes" -type f -print -quit | grep -q .; then
  echo "ERROR: final dataset already exists: ${dataset_root}/episodes" >&2; exit 3
fi
mkdir -p "${staging_root}"

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export PRISMATIC_LLAMA2_7B_REPO="${storage_root}/pretrained/NousResearch-Llama-2-7b-hf"
export MKL_INTERFACE_LAYER=GNU
gpus=(0 1 2 3)
ports=(28010 28011 28012 28013)
seeds=(7 17 27 37)
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_worker() {
  local worker="$1" gpu="$2" port="$3" seed="$4"
  local root="${staging_root}/worker-${worker}" deploy_pid=""
  mkdir -p "${root}/episodes" "${root}/eval"
  worker_cleanup() {
    if [[ -n "${deploy_pid}" ]] && kill -0 "${deploy_pid}" 2>/dev/null; then
      kill -TERM "${deploy_pid}" 2>/dev/null || true; wait "${deploy_pid}" 2>/dev/null || true
    fi
  }
  trap worker_cleanup EXIT INT TERM
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python deploy.py \
    --saved_model_path "${checkpoint}" --unnorm_key libero_spatial_no_noops \
    --cfg_scale 1.5 --num_ddim_steps 10 --use_ddim --use_bf16 \
    --inference_seed "${seed}" --port "${port}" --action_chunking --action_chunking_window 8 \
    > "${root}/deploy.log" 2>&1 &
  deploy_pid=$!
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${deploy_pid}" 2>/dev/null || break
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  [[ "${ready}" -eq 1 ]] || { echo "ERROR: worker ${worker} deploy not ready" >&2; return 4; }
  env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
    .venv/bin/python evaluation/libero/eval_libero.py \
      --model "C16-seed${seed}" --task_suite_name libero_spatial \
      --num_trials_per_task 100 --repeat_initial_states True --spcial_task_id 8 \
      --episode_offset "$((worker * 100))" --seed "${seed}" --hard_case relocation \
      --intervention_policy_call 4 --target_object akita_black_bowl_1 \
      --relocation_dx 0.0954 --relocation_dy 0.03 --relocation_dz 0.0 \
      --save_rollout_videos False --max_successful_episodes 13 \
      --successful_episode_dir "${root}/episodes" \
      --run_id_note "collect-worker${worker}" --local_log_dir "${root}/eval" --port "${port}" \
      > "${root}/console.log" 2>&1
  worker_cleanup
  trap - EXIT INT TERM
}

for i in "${!gpus[@]}"; do
  run_worker "${i}" "${gpus[$i]}" "${ports[$i]}" "${seeds[$i]}" \
    > "${staging_root}/worker-${i}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do wait "${pid}" || status=1; done
worker_pids=()
[[ "${status}" -eq 0 ]] || { echo "ERROR: at least one collection worker failed" >&2; exit 5; }
.venv/bin/python script/data/prepare_libero_relocation_npz.py \
  --staging-root "${staging_root}" --output-root "${dataset_root}" \
  --source-stats "${source_stats}" --episodes 50 | tee "${dataset_root}/prepare.log"
