#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=spatial_memory RUN_ID="${RUN_ID:-spatial_memory_libero10_C}" \
exec "${script_dir}/train_libero10_common.sh"
