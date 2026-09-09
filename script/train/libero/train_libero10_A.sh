#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=memoryvla RUN_ID="${RUN_ID:-memoryvla_libero10_A}" \
exec "${script_dir}/train_libero10_common.sh"
