#!/bin/bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_MODE=memoryvla RUN_ID="${RUN_ID:-memoryvla_robomme_A_4gpu_5k}" \
exec "${script_dir}/train_robomme_common.sh"
