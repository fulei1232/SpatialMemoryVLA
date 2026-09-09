#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

# Keep the LIBERO-10 A/B/C runs matched to the completed LIBERO-Spatial
# experiment. The three entrypoints share every training hyperparameter and
# differ only in their ablation mode.
run_root_dir="${RUN_ROOT_DIR:-/media/fulei/jlu}"
max_steps="${MAX_STEPS:-10000}"
save_interval="${SAVE_INTERVAL:-2000}"
start_model="${START_MODEL:-A}"
data_root_dir="${DATA_ROOT_DIR:-${project_root}/data/libero-rlds}"
dataset_dir="${data_root_dir}/libero_10_no_noops/${DATASET_VERSION:-1.0.0}"
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-${project_root}/pretrained/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
vggt_checkpoint="${VGGT_CHECKPOINT:-${project_root}/pretrained/VGGT-1B/model.pt}"

declare -A entrypoints=(
  [A]="script/train/libero/train_libero10_A.sh"
  [B]="script/train/libero/train_libero10_B.sh"
  [C]="script/train/libero/train_libero10_C.sh"
)
declare -A run_ids=(
  [A]="${A_RUN_ID:-memoryvla_libero10_A}"
  [B]="${B_RUN_ID:-spatial_forcing_libero10_B}"
  [C]="${C_RUN_ID:-spatial_memory_libero10_C}"
)

case "${start_model}" in
  A) models=(A B C) ;;
  B) models=(B C) ;;
  C) models=(C) ;;
  *) echo "ERROR: START_MODEL must be A, B, or C; got ${start_model}." >&2; exit 2 ;;
esac

for required in "${dataset_dir}" "${pretrained_checkpoint}" "${vggt_checkpoint}"; do
  [[ -e "${required}" ]] || { echo "ERROR: required input is missing: ${required}" >&2; exit 3; }
done

mkdir -p "${run_root_dir}/libero10_abc_automation"
automation_dir="${run_root_dir}/libero10_abc_automation"
log_file="${automation_dir}/train_abc_$(date +%Y%m%d_%H%M%S).log"
lock_file="${automation_dir}/train_abc.lock"
pid_file="${automation_dir}/train_abc.pid"

exec > >(tee -a "${log_file}") 2>&1
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "ERROR: another LIBERO-10 A/B/C trainer holds ${lock_file}."
  exit 4
fi
echo "$$" > "${pid_file}"
trap 'status=$?; echo "[$(date --iso-8601=seconds)] trainer exit status=${status}"; exit "${status}"' EXIT

printf -v step_tag '%06d' "${max_steps}"

find_final_checkpoint() {
  local run_dir="$1"
  local matches=()
  shopt -s nullglob
  matches=("${run_dir}/checkpoints/step-${step_tag}-"*.pt)
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] || return 1
  printf '%s\n' "${matches[0]}"
}

validate_completed_run() {
  local run_dir="$1"
  local checkpoint
  checkpoint="$(find_final_checkpoint "${run_dir}")" || return 1
  [[ -s "${checkpoint}" ]] || return 1
  [[ -s "${checkpoint%.pt}.optimizer" ]] || return 1
  [[ -s "${run_dir}/config.json" ]] || return 1
  [[ -s "${run_dir}/dataset_statistics.json" ]] || return 1
  printf '%s\n' "${checkpoint}"
}

echo "[$(date --iso-8601=seconds)] LIBERO-10 matched A/B/C training"
echo "models=${models[*]} dataset=libero_10_no_noops max_steps=${max_steps} save_interval=${save_interval}"
echo "run_root_dir=${run_root_dir} log=${log_file}"

for model in "${models[@]}"; do
  run_id="${run_ids[${model}]}"
  run_dir="${run_root_dir}/${run_id}--image_aug"

  if checkpoint="$(validate_completed_run "${run_dir}")"; then
    echo "[$(date --iso-8601=seconds)] SKIP ${model}: completed checkpoint ${checkpoint}"
    continue
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "ERROR: ${model} has a partial or invalid run directory: ${run_dir}" >&2
    echo "Resume it explicitly or move it aside; this script will not overwrite it." >&2
    exit 5
  fi

  echo "[$(date --iso-8601=seconds)] START ${model}: run_id=${run_id}"
  RUN_ROOT_DIR="${run_root_dir}" \
  RUN_ID="${run_id}" \
  MAX_STEPS="${max_steps}" \
  SAVE_INTERVAL="${save_interval}" \
  "${entrypoints[${model}]}"

  checkpoint="$(validate_completed_run "${run_dir}")" || {
    echo "ERROR: ${model} exited without a valid step-${step_tag} checkpoint pair." >&2
    exit 6
  }
  echo "[$(date --iso-8601=seconds)] DONE ${model}: ${checkpoint}"
done

a_config="${run_root_dir}/${run_ids[A]}--image_aug/config.json"
b_config="${run_root_dir}/${run_ids[B]}--image_aug/config.json"
c_config="${run_root_dir}/${run_ids[C]}--image_aug/config.json"
if [[ -f "${a_config}" && -f "${b_config}" && -f "${c_config}" ]]; then
  "${project_root}/.venv/bin/python" script/train/libero/verify_libero10_ABC_fairness.py \
    "${a_config}" "${b_config}" "${c_config}"
else
  echo "Fairness check deferred: one or more A/B/C config files do not exist yet."
fi

echo "[$(date --iso-8601=seconds)] requested LIBERO-10 training sequence completed"
