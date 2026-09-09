#!/bin/bash
set -euo pipefail
checkpoint="${1:?Usage: $0 <smoke-checkpoint.pt>}"
[[ -f "${checkpoint}" ]] || { echo "Checkpoint not found: ${checkpoint}" >&2; exit 2; }
name="$(basename "${checkpoint}")"
[[ "${name}" =~ ^step-([0-9]+)-epoch-([0-9]+)-loss=.*\.pt$ ]] || { echo "Unexpected checkpoint name: ${name}" >&2; exit 2; }
step=$((10#${BASH_REMATCH[1]}))
epoch=$((10#${BASH_REMATCH[2]}))
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_CHECKPOINT="${checkpoint}" IS_RESUME=True RESUME_STEP="${step}" RESUME_EPOCH="${epoch}" \
EXPERIMENT_MODE=spatial_memory MAX_STEPS="$((step + 1))" SAVE_INTERVAL="$((step + 1))" \
RUN_ID=spatial_memory_libero_spatial_smoke_resume exec "${script_dir}/train_spatial_memory.sh"
