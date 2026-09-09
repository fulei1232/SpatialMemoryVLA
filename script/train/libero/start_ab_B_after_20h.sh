#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

wait_hours="${WAIT_HOURS:-20}"
wait_seconds="${WAIT_SECONDS:-$((wait_hours * 3600))}"

run_root="${RUN_ROOT_DIR:-/media/fulei/jlu}"
a_run_id="${A_RUN_ID:-memoryvla_libero_spatial_ab_A}"
b_run_id="${B_RUN_ID:-memoryvla_spatial_forcing_libero_spatial_ab_B}"
a_run_dir="${run_root}/${a_run_id}--image_aug"
b_run_dir="${run_root}/${b_run_id}--image_aug"
automation_dir="${run_root}/ab_automation"
log_file="${automation_dir}/start_B_after_20h.log"
pid_file="${automation_dir}/start_B_after_20h.pid"
lock_file="${automation_dir}/start_B_after_20h.lock"

mkdir -p "${automation_dir}"
exec >> >(tee -a "${log_file}") 2>&1
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "[$(date --iso-8601=seconds)] ERROR: another B scheduler already holds ${lock_file}"
  exit 1
fi

echo "$$" > "${pid_file}"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] scheduler exit status=${status}"; exit "${status}"' EXIT

find_final_checkpoint() {
  local checkpoint_dir="$1/checkpoints"
  local matches=()
  shopt -s nullglob
  matches=("${checkpoint_dir}"/step-010000-*.pt)
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] || return 1
  printf '%s\n' "${matches[0]}"
}

validate_checkpoint_pair() {
  local model_path="$1"
  local optimizer_path="${model_path%.pt}.optimizer"
  local model_bytes optimizer_bytes

  [[ -f "${model_path}" ]] || { echo "ERROR: missing model ${model_path}"; return 1; }
  [[ -f "${optimizer_path}" ]] || { echo "ERROR: missing optimizer ${optimizer_path}"; return 1; }
  model_bytes="$(stat -c %s "${model_path}")"
  optimizer_bytes="$(stat -c %s "${optimizer_path}")"
  [[ "${model_bytes}" -ge 30000000000 ]] || { echo "ERROR: model is too small (${model_bytes} bytes)"; return 1; }
  [[ "${optimizer_bytes}" -ge 60000000000 ]] || { echo "ERROR: optimizer is too small (${optimizer_bytes} bytes)"; return 1; }
  echo "Validated checkpoint pair: ${model_path} (${model_bytes} bytes), ${optimizer_path} (${optimizer_bytes} bytes)"
}

start_epoch="$(date +%s)"
target_epoch="$((start_epoch + wait_seconds))"
echo "[$(date --iso-8601=seconds)] scheduled B launch"
echo "wait_hours=${wait_hours} wait_seconds=${wait_seconds}"
echo "target=$(date --date="@${target_epoch}" --iso-8601=seconds)"
echo "A=${a_run_dir}"
echo "B=${b_run_dir}"

while (( $(date +%s) < target_epoch )); do
  remaining="$((target_epoch - $(date +%s)))"
  (( remaining > 60 )) && sleep_for=60 || sleep_for="${remaining}"
  (( sleep_for > 0 )) && sleep "${sleep_for}"
done

echo "[$(date --iso-8601=seconds)] 20-hour wait complete; running preflight checks"

if pgrep -af 'train.py' | grep -F -- "--run_id ${a_run_id}"; then
  echo "ERROR: A is still running; refusing to overlap A and B"
  exit 2
fi

if pgrep -af 'train.py'; then
  echo "ERROR: another training process is active; refusing to start B"
  exit 3
fi

a_checkpoint="$(find_final_checkpoint "${a_run_dir}")" || {
  echo "ERROR: expected exactly one A step-10000 checkpoint under ${a_run_dir}/checkpoints"
  exit 4
}
validate_checkpoint_pair "${a_checkpoint}"

if b_checkpoint="$(find_final_checkpoint "${b_run_dir}")"; then
  validate_checkpoint_pair "${b_checkpoint}"
  echo "B step-10000 already exists; nothing to do"
  exit 0
fi

if [[ -d "${b_run_dir}" ]]; then
  echo "ERROR: partial B directory already exists (${b_run_dir}); refusing to overwrite it"
  exit 5
fi

echo "[$(date --iso-8601=seconds)] starting matched B: alignment=True, spatial_memory=False"
EXPERIMENT_MODE=spatial_forcing \
MAX_STEPS=10000 \
SAVE_INTERVAL=2000 \
RUN_ROOT_DIR="${run_root}" \
RUN_ID="${b_run_id}" \
bash script/train/libero/train_spatial_memory.sh

b_checkpoint="$(find_final_checkpoint "${b_run_dir}")" || {
  echo "ERROR: B training exited without exactly one step-10000 checkpoint"
  exit 6
}
validate_checkpoint_pair "${b_checkpoint}"
echo "[$(date --iso-8601=seconds)] B training completed successfully"
