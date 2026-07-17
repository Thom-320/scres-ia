#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRAINING_ROOT CALIBRATION_RUN_ROOT" >&2
  exit 2
fi

training_root=$1
run_root=$2
if [[ ! -f "$training_root/custody/model_files.sha256" ]]; then
  echo "missing training model manifest" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  unexpected=$(find "$run_root" -mindepth 1 -maxdepth 1 ! -name custody -print -quit)
  if [[ -n "$unexpected" ]]; then
    echo "refusing to overwrite initialized calibration root: $run_root" >&2
    exit 2
  fi
fi

mkdir -p "$run_root/custody" "$run_root/logs" "$run_root/artifacts/calibration"
repo_root=$(git rev-parse --show-toplevel)
commit=$(git rev-parse HEAD)
sid=$(ps -o sid= -p $$ | tr -d ' ')
pgid=$(ps -o pgid= -p $$ | tr -d ' ')
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "RUNNING_CALIBRATION" > "$run_root/STATUS"
printf 'commit=%s\nlauncher_pid=%s\nsid=%s\npgid=%s\nstarted_utc=%s\ntraining_root=%s\n' \
  "$commit" "$$" "$sid" "$pgid" "$started" "$training_root" \
  > "$run_root/custody/session.env"
printf '{"producer_pid":%s,"producer_pgid":%s,"producer_sid":%s,"stage":"calibration","started_at_utc":"%s","commit":"%s"}\n' \
  "$$" "$pgid" "$sid" "$started" "$commit" \
  > "$run_root/custody/producer_control.json"
printf '{"status":"RUNNING_EVALUATOR","completed":0,"total":3,"updated_at_utc":"%s"}\n' \
  "$started" > "$run_root/artifacts/calibration/progress.json"

on_error() {
  rc=$?
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
  printf '{"returncode":%s,"finished_at_utc":"%s","status":"FAILED_INFRASTRUCTURE"}\n' \
    "$rc" "$finished" > "$run_root/custody/producer_exit.json"
  printf '%s\n' "FAILED_INFRASTRUCTURE" > "$run_root/STATUS"
  exit "$rc"
}
trap on_error ERR

(
  cd "$training_root"
  sha256sum -c custody/model_files.sha256
) > "$run_root/logs/model_hash_verification.log" 2>&1

"$repo_root/.venv/bin/python" "$repo_root/scripts/evaluate_program_o_ret_learner.py" \
  --models "$training_root/models" \
  --output "$run_root/artifacts/calibration/evaluation" \
  --phase calibration \
  > "$run_root/logs/evaluator.log" 2>&1
updated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"status":"RUNNING_DIRECT_AUDIT","completed":1,"total":3,"updated_at_utc":"%s"}\n' \
  "$updated" > "$run_root/artifacts/calibration/progress.json"

"$repo_root/.venv/bin/python" "$repo_root/scripts/audit_program_o_ret_learner_full_des.py" \
  --evaluation "$run_root/artifacts/calibration/evaluation" \
  --output "$run_root/artifacts/calibration/direct_audit" \
  > "$run_root/logs/direct_audit.log" 2>&1
updated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"status":"RUNNING_ADJUDICATION","completed":2,"total":3,"updated_at_utc":"%s"}\n' \
  "$updated" > "$run_root/artifacts/calibration/progress.json"

set +e
"$repo_root/.venv/bin/python" "$repo_root/scripts/adjudicate_program_o_ret_calibration.py" \
  --calibration-result "$run_root/artifacts/calibration/evaluation/result.json" \
  --direct-audit "$run_root/artifacts/calibration/direct_audit/independent_full_des_audit.json" \
  --output "$run_root/artifacts/calibration/result.json" \
  > "$run_root/logs/adjudication.log" 2>&1
adjudication_rc=$?
set -e

finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'finished_utc=%s\n' "$finished" >> "$run_root/custody/session.env"
printf '{"status":"COMPLETE_PENDING_RETRIEVAL","completed":3,"total":3,"updated_at_utc":"%s"}\n' \
  "$finished" > "$run_root/artifacts/calibration/progress.json"
printf '{"returncode":0,"adjudication_returncode":%s,"finished_at_utc":"%s","status":"COMPLETE_PENDING_RETRIEVAL"}\n' \
  "$adjudication_rc" "$finished" > "$run_root/custody/producer_exit.json"
printf '%s\n' "COMPLETE_PENDING_RETRIEVAL" > "$run_root/STATUS"

# A scientific STOP is a completed adjudication, not an infrastructure failure.
exit 0
