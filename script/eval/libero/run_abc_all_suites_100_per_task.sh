#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

start_at="${START_AT:-2026-09-06 20:00:00 +0800}"
output_root="${OUTPUT_ROOT:-/media/fulei/jlu/libero_abc_spatial_50_per_task_20260906}"
task_timeout_hours="${TASK_TIMEOUT_HOURS:-4}"
worker_count="${WORKER_COUNT:-8}"
base_port="${DEPLOY_BASE_PORT:-26820}"
if (( worker_count < 1 || worker_count > 8 )); then
  echo "WORKER_COUNT must be between 1 and 8" >&2
  exit 2
fi

mkdir -p "${output_root}"
pipeline_log="${output_root}/pipeline.log"
status_file="${output_root}/task_status.tsv"
model_status_file="${output_root}/model_status.tsv"
pid_file="${output_root}/scheduler.pid"
lock_file="${output_root}/scheduler.lock"

exec >> >(tee -a "${pipeline_log}") 2>&1
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "[$(date --iso-8601=seconds)] ERROR: another rollout scheduler holds ${lock_file}"
  exit 1
fi
echo "$$" > "${pid_file}"

deploy_pids=()
deploy_ports=()
eval_pids=()
cleanup_deploy() {
  local pid
  for pid in "${deploy_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${deploy_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  deploy_pids=()
  deploy_ports=()
}
cleanup_all() {
  local pid
  for pid in "${eval_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${eval_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  cleanup_deploy
}
trap 'status=$?; cleanup_all; echo "[$(date --iso-8601=seconds)] scheduler exit status=${status}"' EXIT

[[ -f "${status_file}" ]] || printf 'timestamp\tmodel\tsuite\ttask_id\tstatus\texit_code\tlog_dir\n' > "${status_file}"
[[ -f "${model_status_file}" ]] || printf 'timestamp\tmodel\tstatus\tdetail\n' > "${model_status_file}"

target_epoch="$(date --date="${start_at}" +%s)"
echo "[$(date --iso-8601=seconds)] scheduled A/B/C rollout matrix"
echo "target=$(date --date="@${target_epoch}" --iso-8601=seconds)"
echo "protocol=50 official episodes/task, seed=7, cfg=1.5, action_chunk=8"
echo "parallelism=${worker_count} task workers; models remain sequential A->B->C"
echo "output_root=${output_root}"

while (( $(date +%s) < target_epoch )); do
  remaining="$((target_epoch - $(date +%s)))"
  (( remaining > 60 )) && sleep_for=60 || sleep_for="${remaining}"
  (( sleep_for > 0 )) && sleep "${sleep_for}"
done

echo "[$(date --iso-8601=seconds)] scheduled time reached"
if pgrep -af 'train.py'; then
  echo "ERROR: training is still active; refusing to contend for GPU or evaluate a partial checkpoint"
  exit 2
fi

export PRISMATIC_LLAMA2_7B_REPO='./pretrained/NousResearch-Llama-2-7b-hf'
export LIBERO_CONFIG_PATH="${project_root}/.libero"

declare -A run_dirs=(
  [A]='/media/fulei/jlu/memoryvla_libero_spatial_ab_A--image_aug'
  [B]='/media/fulei/jlu/memoryvla_spatial_forcing_libero_spatial_ab_B--image_aug'
  [C]='/media/fulei/jlu/spatial_memory_libero_spatial_functional--image_aug'
)
models=(A B C)
IFS=',' read -r -a suites <<< "${SUITES_CSV:-libero_spatial}"
if [[ -n "${TASK_IDS_CSV:-}" ]]; then
  IFS=',' read -r -a requested_task_ids <<< "${TASK_IDS_CSV}"
else
  requested_task_ids=()
fi

find_checkpoint() {
  local run_dir="$1"
  local matches=()
  shopt -s nullglob
  matches=("${run_dir}"/checkpoints/step-010000-*.pt)
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] || return 1
  printf '%s\n' "${matches[0]}"
}

validate_checkpoint() {
  local run_dir="$1" checkpoint="$2" bytes
  [[ -f "${run_dir}/config.json" ]] || return 1
  [[ -f "${run_dir}/dataset_statistics.json" ]] || return 1
  bytes="$(stat -c %s "${checkpoint}")"
  [[ "${bytes}" -ge 30000000000 ]] || return 1
  echo "Validated ${checkpoint} (${bytes} bytes)"
}

task_already_recorded() {
  local model="$1" suite="$2" task_id="$3"
  awk -F '\t' -v m="${model}" -v s="${suite}" -v t="${task_id}" \
    'NR > 1 && $2 == m && $3 == s && $4 == t {found=1} END {exit !found}' "${status_file}"
}

record_task() {
  local model="$1" suite="$2" task_id="$3" status="$4" exit_code="$5" log_dir="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${model}" "${suite}" "${task_id}" "${status}" "${exit_code}" "${log_dir}" \
    >> "${status_file}"
  .venv/bin/python evaluation/libero/summarize_abc_rollouts.py --root "${output_root}" || \
    echo "WARNING: live summary update failed"
}

start_deploy_replicas() {
  local model="$1" checkpoint="$2"
  local gpu worker_port deploy_log pid ready deadline all_ready
  mkdir -p "${output_root}/${model}"

  deploy_pids=()
  deploy_ports=()
  for ((gpu=0; gpu<worker_count; gpu++)); do
    worker_port="$((base_port + gpu))"
    if ss -ltn | awk '{print $4}' | grep -qE "(^|:)${worker_port}$"; then
      echo "ERROR: port ${worker_port} is already in use"
      return 1
    fi
    deploy_log="${output_root}/${model}/deploy-gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python deploy.py \
      --saved_model_path "${checkpoint}" \
      --unnorm_key "${UNNORM_KEY:-libero_spatial_no_noops}" \
      --cfg_scale 1.5 \
      --port "${worker_port}" \
      --use_bf16 \
      --action_chunking \
      --action_chunking_window 8 \
      > "${deploy_log}" 2>&1 &
    pid=$!
    deploy_pids+=("${pid}")
    deploy_ports+=("${worker_port}")
    echo "Started ${model} replica gpu=${gpu} port=${worker_port} pid=${pid}"
  done

  deadline="$(( $(date +%s) + 1200 ))"
  while (( $(date +%s) < deadline )); do
    all_ready=1
    for ((gpu=0; gpu<worker_count; gpu++)); do
      pid="${deploy_pids[${gpu}]}"
      worker_port="${deploy_ports[${gpu}]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: ${model} replica gpu=${gpu} exited during startup"
        return 1
      fi
      if ! curl --noproxy '*' -fsS "http://127.0.0.1:${worker_port}/health" >/dev/null 2>&1; then
        all_ready=0
      fi
    done
    if (( all_ready == 1 )); then
      echo "[$(date --iso-8601=seconds)] all ${worker_count} replicas for model ${model} are ready"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: not all deploy replicas for ${model} were ready within 20 minutes"
  return 1
}

for model in "${models[@]}"; do
  run_dir="${run_dirs[${model}]}"
  if ! checkpoint="$(find_checkpoint "${run_dir}")" || ! validate_checkpoint "${run_dir}" "${checkpoint}"; then
    echo "[$(date --iso-8601=seconds)] SKIP model ${model}: missing or incomplete step-10000 checkpoint"
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${model}" skipped checkpoint_invalid \
      >> "${model_status_file}"
    continue
  fi

  if ! start_deploy_replicas "${model}" "${checkpoint}"; then
    cleanup_deploy
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${model}" skipped deploy_failed \
      >> "${model_status_file}"
    continue
  fi
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${model}" running "${checkpoint}" \
    >> "${model_status_file}"

  for suite in "${suites[@]}"; do
    if (( ${#requested_task_ids[@]} > 0 )); then
      task_ids=("${requested_task_ids[@]}")
    elif [[ "${suite}" == "libero_90" ]]; then
      task_ids=($(seq 0 89))
    else
      task_ids=($(seq 0 9))
    fi
    task_count=${#task_ids[@]}

    for ((batch_start=0; batch_start<task_count; batch_start+=worker_count)); do
      eval_pids=()
      eval_task_ids=()
      eval_task_dirs=()
      slots_to_launch="${worker_count}"
      if [[ "${EPISODE_SHARDING:-0}" != "1" ]]; then
        slots_to_launch=$((task_count - batch_start))
        (( slots_to_launch > worker_count )) && slots_to_launch="${worker_count}"
      fi

      for ((slot=0; slot<slots_to_launch; slot++)); do
        if [[ "${EPISODE_SHARDING:-0}" == "1" ]]; then
          task_id="${task_ids[$batch_start]}"
        else
          task_id="${task_ids[$((batch_start + slot))]}"
        fi
        if task_already_recorded "${model}" "${suite}" "${task_id}"; then
          echo "SKIP previously recorded ${model}/${suite}/task-${task_id}"
          continue
        fi

        shard_suffix=""
        if [[ "${EPISODE_SHARDING:-0}" == "1" ]]; then
          shard_suffix="/shard-${slot}"
        fi
        task_dir="${output_root}/${model}/${suite}/task-$(printf '%03d' "${task_id}")${shard_suffix}"
        mkdir -p "${task_dir}"
        worker_port="${deploy_ports[${slot}]}"
        echo "[$(date --iso-8601=seconds)] START ${model}/${suite}/task-${task_id} worker=${slot} port=${worker_port}"

        eval_trials="${NUM_TRIALS_PER_TASK:-50}"
        episode_shard_args=()
        if [[ "${EPISODE_SHARDING:-0}" == "1" ]]; then
          eval_trials=$(( (eval_trials + worker_count - 1) / worker_count ))
          episode_shard_args+=(--episode_offset "${slot}" --episode_stride "${worker_count}")
        fi
        timeout --signal=TERM "${task_timeout_hours}h" \
          env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
          .venv/bin/python evaluation/libero/eval_libero.py \
            --task_suite_name "${suite}" \
            --num_trials_per_task "${eval_trials}" \
            --spcial_task_id "${task_id}" \
            --seed 7 \
            --run_id_note "${model}-50ep-task${task_id}-seed7-ac8" \
            --local_log_dir "${task_dir}" \
            --port "${worker_port}" \
            "${episode_shard_args[@]}" \
            > "${task_dir}/console.log" 2>&1 &
        eval_pids+=("$!")
        eval_task_ids+=("${task_id}")
        eval_task_dirs+=("${task_dir}")
      done

      for ((job=0; job<${#eval_pids[@]}; job++)); do
        task_id="${eval_task_ids[${job}]}"
        task_dir="${eval_task_dirs[${job}]}"
        if wait "${eval_pids[${job}]}"; then
          exit_code=0
          task_status=ok
        else
          exit_code=$?
          task_status=skipped_error
          echo "WARNING: ${model}/${suite}/task-${task_id} exited ${exit_code}; continuing"
        fi
        record_task "${model}" "${suite}" "${task_id}" "${task_status}" "${exit_code}" "${task_dir}"
      done
    done
  done

  cleanup_deploy
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${model}" completed all_suites_attempted \
    >> "${model_status_file}"
done

.venv/bin/python evaluation/libero/summarize_abc_rollouts.py --root "${output_root}"
echo "[$(date --iso-8601=seconds)] all scheduled A/B/C suite tasks attempted"
