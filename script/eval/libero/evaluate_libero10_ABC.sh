#!/bin/bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

: "${A_CHECKPOINT:?Set A_CHECKPOINT to the completed A10 checkpoint.}"
: "${B_CHECKPOINT:?Set B_CHECKPOINT to the completed B10 checkpoint.}"
: "${C_CHECKPOINT:?Set C_CHECKPOINT to the completed C10 checkpoint.}"

declare -A checkpoints=(
  [A]="${A_CHECKPOINT}"
  [B]="${B_CHECKPOINT}"
  [C]="${C_CHECKPOINT}"
)
for model in A B C; do
  [[ -f "${checkpoints[${model}]}" ]] || {
    echo "ERROR: ${model} checkpoint not found: ${checkpoints[${model}]}" >&2
    exit 2
  }
done

python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
[[ -x "${python_bin}" ]] || { echo "ERROR: Python environment missing: ${python_bin}" >&2; exit 2; }

output_root="${OUTPUT_ROOT:-${project_root}/log/libero10_eval}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-26820}"
seed="${SEED:-7}"
cfg_scale="${CFG_SCALE:-1.5}"
ddim_steps="${DDIM_STEPS:-10}"
action_chunk="${ACTION_CHUNK:-8}"
num_trials="${NUM_TRIALS_PER_TASK:-50}"
task_suite="libero_10"
unnorm_key="libero_10_no_noops"
mkdir -p "${output_root}"

export LIBERO_CONFIG_PATH="${project_root}/.libero"
export PRISMATIC_LLAMA2_7B_REPO="${project_root}/pretrained/NousResearch-Llama-2-7b-hf"
export MKL_INTERFACE_LAYER=GNU

deploy_pid=""
cleanup() {
  if [[ -n "${deploy_pid}" ]] && kill -0 "${deploy_pid}" 2>/dev/null; then
    kill -TERM "${deploy_pid}" 2>/dev/null || true
    wait "${deploy_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat <<EOF
LIBERO-10 A/B/C rollout protocol
tasks: 10
official initial states per task: ${num_trials}
seed: ${seed}
CFG: ${cfg_scale}
DDIM steps: ${ddim_steps}
action chunk: ${action_chunk}
episode horizon: 520
normalization key: ${unnorm_key}
EOF

for model in A B C; do
  checkpoint="${checkpoints[${model}]}"
  model_dir="${output_root}/${model}"
  mkdir -p "${model_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" deploy.py \
    --saved_model_path "${checkpoint}" \
    --unnorm_key "${unnorm_key}" \
    --cfg_scale "${cfg_scale}" \
    --num_ddim_steps "${ddim_steps}" \
    --use_ddim \
    --use_bf16 \
    --port "${port}" \
    --action_chunking \
    --action_chunking_window "${action_chunk}" \
    > "${model_dir}/deploy.log" 2>&1 &
  deploy_pid=$!

  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${deploy_pid}" 2>/dev/null; then
      echo "ERROR: ${model} deployment exited during startup; see ${model_dir}/deploy.log" >&2
      exit 3
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == 1 ]] || { echo "ERROR: ${model} deployment did not become ready." >&2; exit 3; }

  MUJOCO_GL="${MUJOCO_GL:-egl}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${python_bin}" evaluation/libero/eval_libero.py \
    --model "${model}" \
    --task_suite_name "${task_suite}" \
    --num_trials_per_task "${num_trials}" \
    --seed "${seed}" \
    --run_id_note "${model}-seed${seed}-cfg${cfg_scale}-ddim${ddim_steps}-ac${action_chunk}" \
    --local_log_dir "${model_dir}" \
    --port "${port}"

  cleanup
  deploy_pid=""
  "${python_bin}" evaluation/libero/summarize_libero10_abc.py --root "${output_root}"
done

"${python_bin}" evaluation/libero/summarize_libero10_abc.py --root "${output_root}"
echo "Results: ${output_root}/libero10_rollouts.csv"
echo "Summary: ${output_root}/LIBERO10_ABC_SUMMARY.md"
