#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SCRES_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "SCRES_PYTHON must point to the project Python interpreter" >&2
  exit 2
fi

cd "${repo_root}"
output_root="results/program_q2/qrdqn_dynamic_probe_v1"
mkdir -p "${output_root}"
evaluation_tapes="7490001,7490002,7490003,7490004,7490005,7490006,7490007,7490008,7490009,7490010,7490011,7490012,7490013,7490014,7490015,7490016,7490017,7490018,7490019,7490020,7490021,7490022,7490023,7490024"

optimizer_seeds=(20261101 20261102 20261103)
training_lows=(757100001 757110001 757120001)
training_highs=(757107716 757117716 757127716)
pids=()

for index in 0 1 2; do
  seed="${optimizer_seeds[$index]}"
  output="${output_root}/seed_${seed}"
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite existing output: ${output}" >&2
    exit 3
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${python_bin}" scripts/run_program_q2_minimal_learner.py \
    --algorithm qrdqn \
    --output "${output}" \
    --total-timesteps 60000 \
    --optimizer-seed "${seed}" \
    --training-seed-start "${training_lows[$index]}" \
    --training-seed-end "${training_highs[$index]}" \
    --evaluation-tapes "${evaluation_tapes}" \
    --reward-mode raw_terminal \
    > "${output_root}/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done

printf '%s\n' "${pids[@]}" > "${output_root}/child_pids.txt"
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
