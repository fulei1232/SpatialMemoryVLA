#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export START_AT="${START_AT:-now}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/fulei/jlu/libero10_abc_50_per_task_20260910}"
export WORKER_COUNT="${WORKER_COUNT:-8}"
export SUITES_CSV="libero_10"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
export UNNORM_KEY="libero_10_no_noops"
export A_RUN_DIR="/media/fulei/jlu/memoryvla_libero10_A--image_aug"
export B_RUN_DIR="/media/fulei/jlu/spatial_forcing_libero10_B--image_aug"
export C_RUN_DIR="/media/fulei/jlu/spatial_memory_libero10_C--image_aug"

exec "${script_dir}/run_abc_all_suites_100_per_task.sh" "$@"
