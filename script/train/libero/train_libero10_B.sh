#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=spatial_forcing RUN_ID="${RUN_ID:-spatial_forcing_libero10_B}" \
exec "${script_dir}/train_libero10_common.sh"
