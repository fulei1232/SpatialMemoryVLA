#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=spatial_memory MAX_STEPS=20 SAVE_INTERVAL=10 \
RUN_ID=spatial_memory_libero_spatial_smoke exec "${script_dir}/train_spatial_memory.sh"
