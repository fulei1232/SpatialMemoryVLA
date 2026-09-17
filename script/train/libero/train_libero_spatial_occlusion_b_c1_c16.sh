#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common="${script_dir}/train_libero_spatial_occlusion_common.sh"
max_steps="${MAX_STEPS:-20}"
run_tag="${RUN_TAG:-det_noaug}"

echo "[1/3] B: spatial forcing without temporal memory"
EXPERIMENT_MODE=spatial_forcing MEM_LENGTH=1 \
RUN_ID="libero_spatial_occ_B_${max_steps}step_${run_tag}" \
"${common}"

echo "[2/3] C1: spatial memory with one-frame capacity"
EXPERIMENT_MODE=spatial_memory MEM_LENGTH=1 \
RUN_ID="libero_spatial_occ_C_mem1_${max_steps}step_${run_tag}" \
"${common}"

echo "[3/3] C16: spatial memory with sixteen-frame capacity"
EXPERIMENT_MODE=spatial_memory MEM_LENGTH=16 \
RUN_ID="libero_spatial_occ_C_mem16_${max_steps}step_${run_tag}" \
"${common}"
