#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

base_root="${STAGED_ROOT:-/media/fulei/jlu/libero_staged_20260906}"
runner="script/eval/libero/run_abc_all_suites_100_per_task.sh"
stage1="${base_root}/libero10_task4"
master_log="${base_root}/staged_priority.log"
mkdir -p "${base_root}"
exec > >(tee -a "${master_log}") 2>&1

echo "[$(date --iso-8601=seconds)] staged priority scheduler started"

wait_for_stage() {
  local root="$1"
  while [[ ! -f "${root}/pipeline.log" ]] || ! grep -q 'scheduler exit status=' "${root}/pipeline.log"; do
    sleep 30
  done
  echo "[$(date --iso-8601=seconds)] completed ${root}"
}

run_stage() {
  local name="$1" root="$2" suites="$3" tasks="$4" workers="$5"
  echo "[$(date --iso-8601=seconds)] starting ${name} suites=${suites} tasks=${tasks}"
  START_AT=now \
    SUITES_CSV="${suites}" \
    TASK_IDS_CSV="${tasks}" \
    WORKER_COUNT="${workers}" \
    OUTPUT_ROOT="${root}" \
    TASK_TIMEOUT_HOURS=4 \
    UNNORM_KEY=libero_spatial_no_noops \
    bash "${runner}"
  echo "[$(date --iso-8601=seconds)] finished ${name}"
}

wait_for_stage "${stage1}"
run_stage "LIBERO-Goal" "${base_root}/libero_goal" "libero_goal" "" 8
run_stage "LIBERO-Object" "${base_root}/libero_object" "libero_object" "" 8
run_stage "LIBERO-90" "${base_root}/libero_90" "libero_90" "" 8
echo "[$(date --iso-8601=seconds)] all staged priority evaluations finished"
