#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root=$1
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run root: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root/models" "$run_root/logs" "$run_root/custody"

repo_root=$(git rev-parse --show-toplevel)
commit=$(git rev-parse HEAD)
sid=$(ps -o sid= -p $$ | tr -d ' ')
pgid=$(ps -o pgid= -p $$ | tr -d ' ')
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "RUNNING" > "$run_root/STATUS"
printf 'commit=%s\nlauncher_pid=%s\nsid=%s\npgid=%s\nstarted_utc=%s\n' \
  "$commit" "$$" "$sid" "$pgid" "$started" > "$run_root/custody/session.env"

pids=()
for learner_index in {0..9}; do
  "$repo_root/.venv/bin/python" "$repo_root/scripts/train_program_o_ret_learner.py" \
    --learner-index "$learner_index" \
    --output "$run_root/models" \
    > "$run_root/logs/train_${learner_index}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf '%s %s\n' "$learner_index" "$pid" >> "$run_root/custody/producer_pids.tsv"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ "$failed" -ne 0 ]]; then
  printf '%s\n' "FAILED_TRAINING" > "$run_root/STATUS"
  printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
  exit 1
fi

find "$run_root/models" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$run_root/custody/model_files.sha256"
printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
printf '%s\n' "COMPLETE_PENDING_EVALUATION" > "$run_root/STATUS"
