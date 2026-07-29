#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root=$1
if [[ -e "$run_root" ]]; then
  unexpected=$(find "$run_root" -mindepth 1 -maxdepth 1 ! -name custody -print -quit)
  if [[ -n "$unexpected" ]]; then
    echo "refusing to overwrite initialized run root: $run_root" >&2
    exit 2
  fi
fi
mkdir -p "$run_root/models" "$run_root/logs" "$run_root/custody" "$run_root/artifacts/training"

repo_root=$(git rev-parse --show-toplevel)
commit=$(git rev-parse HEAD)
sid=$(ps -o sid= -p $$ | tr -d ' ')
pgid=$(ps -o pgid= -p $$ | tr -d ' ')
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "RUNNING" > "$run_root/STATUS"
printf 'commit=%s\nlauncher_pid=%s\nsid=%s\npgid=%s\nstarted_utc=%s\n' \
  "$commit" "$$" "$sid" "$pgid" "$started" > "$run_root/custody/session.env"
printf '{"producer_pid":%s,"producer_pgid":%s,"producer_sid":%s,"stage":"training","started_at_utc":"%s","commit":"%s"}\n' \
  "$$" "$pgid" "$sid" "$started" "$commit" > "$run_root/custody/producer_control.json"
printf '{"status":"RUNNING_TRAINING","completed":0,"total":10,"updated_at_utc":"%s"}\n' \
  "$started" > "$run_root/artifacts/training/progress.json"

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
completed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
  completed=$((completed + 1))
  updated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"status":"RUNNING_TRAINING","completed":%s,"total":10,"updated_at_utc":"%s"}\n' \
    "$completed" "$updated" > "$run_root/artifacts/training/progress.json"
done

finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ "$failed" -ne 0 ]]; then
  printf '%s\n' "FAILED_TRAINING" > "$run_root/STATUS"
  printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
  printf '{"returncode":1,"finished_at_utc":"%s","status":"FAILED_TRAINING"}\n' \
    "$finished" > "$run_root/custody/producer_exit.json"
  exit 1
fi

find "$run_root/models" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$run_root/custody/model_files.sha256"
printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
printf '{"status":"COMPLETE_PENDING_EVALUATION","completed":10,"total":10,"updated_at_utc":"%s"}\n' \
  "$finished" > "$run_root/artifacts/training/progress.json"
printf '{"schema_version":"program_o_ret_only_learner_training_result_v1","status":"COMPLETE_PENDING_EVALUATION","learner_count":10,"finished_at_utc":"%s"}\n' \
  "$finished" > "$run_root/artifacts/training/result.json"
printf '{"returncode":0,"finished_at_utc":"%s","status":"COMPLETE_PENDING_EVALUATION"}\n' \
  "$finished" > "$run_root/custody/producer_exit.json"
printf '%s\n' "COMPLETE_PENDING_EVALUATION" > "$run_root/STATUS"
