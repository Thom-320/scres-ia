#!/usr/bin/env python3
"""Record the controlled termination of the obsolete-metric VPS producer."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "outputs/vps_metric_quarantine_20260714"
SOURCE_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_visible_v1_source_semantics_audit_20260714.json"
)
OUTPUT = (
    ROOT
    / "research/paper2_exhaustive_search/mtr_switch4_vps_metric_quarantine_20260714.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    latest = json.loads((RUN / "watcher_latest.json").read_text())
    frozen = json.loads((RUN / "watcher_at_metric_quarantine.json").read_text())
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(RUN.iterdir())
        if path.is_file()
    }
    result = {
        "schema_version": "mtr_switch4_vps_metric_quarantine_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "CONTROLLED_TERMINATION_OBSOLETE_METRIC_NO_RESULT",
        "run_identity": "mtr-switch4-producer-8f31f410-20260714T074723Z",
        "host": latest["hostname"],
        "metric_scored": "ret_excel_visible_v1_OAT_ledger",
        "metric_disposition": "QUARANTINED_NOT_CANONICAL",
        "termination_reason": "The source-semantics audit established that visible-v1 reconstructs Bt/Ut at OATj although the thesis request barrier and raw workbook ordering support request-generation snapshots. Continuing the estimated 20-21 hour producer could not yield admissible canonical evidence.",
        "source_semantics_audit": {
            "path": str(SOURCE_AUDIT.relative_to(ROOT)),
            "sha256": sha256(SOURCE_AUDIT),
        },
        "last_frozen_running_observation": {
            "observed_at_utc": frozen["observed_at_utc"],
            "state": frozen["state"],
            "progress": frozen["progress"],
            "scientific_process_tree_cpu_percent": frozen[
                "scientific_process_tree_cpu_percent"
            ],
            "scientific_process_tree_rss_bytes": frozen[
                "scientific_process_tree_rss_bytes"
            ],
            "stderr_bytes": frozen["stderr_bytes"],
        },
        "final_watcher_observation": {
            "observed_at_utc": latest["observed_at_utc"],
            "state": latest["state"],
            "progress": latest["progress"],
            "scientific_pid_alive": latest["scientific_pid_alive"],
            "result_exists": latest["result_exists"],
            "stderr_bytes": latest["stderr_bytes"],
        },
        "retrieved_artifacts": files,
        "watcher_custody_assessment": {
            "prestart_and_parent_pid_monitoring_verified": True,
            "whole_scientific_session_termination_verified": False,
            "defect": "The v1 watcher keyed terminal state to the parent PID. Six multiprocessing workers were reparented to PID 1 after the parent stopped and continued running outside watcher coverage.",
            "remediation": "The orphan workers were identified from cwd and stdout/stderr descriptors, terminated with SIGTERM, and the replacement watcher now monitors the complete POSIX scientific session.",
        },
        "orphan_worker_remediation": {
            "discovered_at_utc": "2026-07-14T10:26:42Z",
            "terminated_at_utc": "2026-07-14T10:27:24Z",
            "worker_pids": [820794, 820854, 820862, 820872, 820901, 820984],
            "worker_cwd": "/home/ubuntu/paper2-bound-runs/mtr-switch4-preflight-8f31f410-20260714T053810Z/source",
            "identification": "All six workers retained stdout/stderr descriptors to the quarantined run and consumed approximately 80-100 percent CPU each.",
            "signal": "SIGTERM",
            "postcheck_at_utc": "2026-07-14T10:31:19Z",
            "postcheck": "All six PIDs absent; load average had fallen to 0.15 over one minute.",
            "scientific_evidence_effect": "NONE; no result was produced and the obsolete metric was already quarantined.",
        },
        "virgin_tapes_opened": False,
        "scientific_evidence_value": "NONE; five partial calibration evaluations are custody evidence only and may not enter H_PI, H_obs, a comparator ceiling, a null or a positive result.",
        "required_restart": "commit and independently audit ret_excel_request_snapshot_v2, then rerun from the first calibration tape under a new immutable run identity and a watcher that starts before science and monitors the complete POSIX scientific session, including reparented workers",
    }
    result["content_sha256"] = json_sha256(result)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "progress": result["final_watcher_observation"]["progress"],
        "content_sha256": result["content_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
