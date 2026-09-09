#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=spatial_memory MAX_STEPS=10000 SAVE_INTERVAL=2000 \
RUN_ROOT_DIR=/media/fulei/jlu \
RUN_ID=spatial_memory_libero_spatial_functional exec "${script_dir}/train_spatial_memory.sh"
