#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STEPS=500 SAVE_INTERVAL=500 RUN_ID="${RUN_ID:-spatial_memory_libero10_C_functional}" \
exec "${script_dir}/train_libero10_C.sh"
