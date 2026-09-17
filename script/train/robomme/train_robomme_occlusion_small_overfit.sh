#!/bin/bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_STEPS="${MAX_STEPS:-500}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/media/fulei/jlu/SpatialMemoryVLA/runs/robomme_memory}"
exec "${script_dir}/train_robomme_occlusion_c1_c16.sh"
