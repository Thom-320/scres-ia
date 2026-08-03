#!/usr/bin/env bash
# Passive watcher for the single G3-obs confirmation run.
# It checks once per minute and writes only a terminal status receipt.
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PID RESULT_PATH LOG_PATH STATUS_PATH" >&2
  exit 2
fi

pid="$1"
result="$2"
log_path="$3"
status_path="$4"

while kill -0 "$pid" 2>/dev/null; do
  if [[ -f "$result" ]]; then
    printf '{"status":"COMPLETED","pid":%s,"result":"%s","log":"%s"}\n' \
      "$pid" "$result" "$log_path" > "$status_path"
    exit 0
  fi
  sleep 60
done

if [[ -f "$result" ]]; then
  printf '{"status":"COMPLETED","pid":%s,"result":"%s","log":"%s"}\n' \
    "$pid" "$result" "$log_path" > "$status_path"
else
  printf '{"status":"PROCESS_DIED_WITHOUT_RESULT","pid":%s,"result":"%s","log":"%s"}\n' \
    "$pid" "$result" "$log_path" > "$status_path"
  exit 1
fi
