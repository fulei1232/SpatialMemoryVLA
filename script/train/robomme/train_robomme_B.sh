#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=spatial_forcing \
PER_DEVICE_BATCH_SIZE="${B_PER_DEVICE_BATCH_SIZE:-${SPATIAL_PER_DEVICE_BATCH_SIZE:-8}}" \
RUN_ID="${RUN_ID:-spatial_forcing_robomme_B_4gpu_5k}" \
exec "${script_dir}/train_robomme_common.sh"
