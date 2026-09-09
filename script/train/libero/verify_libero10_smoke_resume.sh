#!/bin/bash
set -Eeuo pipefail

mode="${1:?Usage: $0 <memoryvla|spatial_forcing|spatial_memory> <checkpoint.pt>}"
checkpoint="${2:?Usage: $0 <memoryvla|spatial_forcing|spatial_memory> <checkpoint.pt>}"
[[ -f "${checkpoint}" ]] || { echo "Checkpoint not found: ${checkpoint}" >&2; exit 2; }
[[ -f "${checkpoint%.pt}.optimizer" ]] || { echo "Optimizer state not found: ${checkpoint%.pt}.optimizer" >&2; exit 2; }

name="$(basename "${checkpoint}")"
[[ "${name}" =~ ^step-([0-9]+)-epoch-([0-9]+)-loss=.*\.pt$ ]] || {
  echo "Unexpected checkpoint name: ${name}" >&2
  exit 2
}
step=$((10#${BASH_REMATCH[1]}))
epoch=$((10#${BASH_REMATCH[2]}))
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${mode}" in
  memoryvla) entrypoint="${script_dir}/train_libero10_A.sh" ;;
  spatial_forcing) entrypoint="${script_dir}/train_libero10_B.sh" ;;
  spatial_memory) entrypoint="${script_dir}/train_libero10_C.sh" ;;
  *) echo "Unknown mode: ${mode}" >&2; exit 2 ;;
esac

PRETRAINED_CHECKPOINT="${checkpoint}" \
IS_RESUME=True RESUME_STEP="${step}" RESUME_EPOCH="${epoch}" \
MAX_STEPS="$((step + 1))" SAVE_INTERVAL="$((step + 1))" \
RUN_ID="$(basename "$(dirname "$(dirname "${checkpoint}")")")_resume" \
exec "${entrypoint}"
