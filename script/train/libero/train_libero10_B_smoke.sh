#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STEPS=20 SAVE_INTERVAL=20 RUN_ID="${RUN_ID:-spatial_forcing_libero10_B_smoke}" \
exec "${script_dir}/train_libero10_B.sh"
