#!/usr/bin/env python3
"""Persist an agent's output into the repository, so no agent's work dies with its process.

WHY THIS EXISTS. On 2026-08-08 two external review agents produced substantial findings and both
reports were lost: one was blocked by a read-only sandbox when it tried to write its file, the other
by a permission prompt on a path outside its working directory. Their conclusions survived only
because they happened to also appear in a terminal log that was still being tailed. A subagent's
answer that lives in one process's stdout is one `pkill` away from never having existed.

WHAT IT DOES. Reads a hook payload (or an arbitrary blob) on stdin and writes a dated markdown file
under `docs/agent_runs/`. The payload is written whole, in a fenced block, and the fields worth
reading are lifted to the top -- because the shape of hook payloads is not guaranteed across
versions and a saver that only understands today's schema loses tomorrow's content.

IT NEVER FAILS THE CALLER. Every path returns 0. A hook that can break a session by failing to save
a log is a worse problem than the one it solves.

Usage:
    <json on stdin> | tools/save_agent_output.py --event SubagentStop
    <any text>      | tools/save_agent_output.py --event codex --label normaliser-review
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "agent_runs"

#: Fields worth lifting to the top of the file, in the order a reader wants them. Absent ones are
#: skipped rather than rendered empty; the raw payload below always carries everything.
HEADER_FIELDS = ("event", "agent_type", "subagent_type", "description", "label",
                 "session_id", "agent_id", "model", "cwd", "transcript_path",
                 "duration_ms", "total_tokens", "stop_reason")

#: Keys whose value is likely to BE the answer. First hit wins.
ANSWER_KEYS = ("final_response", "response", "result", "output", "text", "content", "message")


def slug(value: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()
    return (s[:limit] or "agent")


def find_answer(node, depth: int = 0):
    """First plausible final-response string, searched breadth-first over the payload."""
    if depth > 4:
        return None
    if isinstance(node, dict):
        for k in ANSWER_KEYS:
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for v in node.values():
            found = find_answer(v, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_answer(v, depth + 1)
            if found:
                return found
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", default="agent")
    ap.add_argument("--label", default=None)
    args, _unknown = ap.parse_known_args()

    try:
        raw = sys.stdin.read()
    except Exception:                                   # noqa: BLE001
        raw = ""
    if not raw.strip():
        return 0                                        # nothing to save is not a failure

    try:
        payload = json.loads(raw)
        pretty = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False)
        parsed = True
    except Exception:                                   # noqa: BLE001
        payload, pretty, parsed = {}, raw, False

    now = datetime.now(timezone.utc)
    meta = {"event": args.event, "label": args.label}
    if isinstance(payload, dict):
        for f in HEADER_FIELDS:
            if payload.get(f) is not None and meta.get(f) is None:
                meta[f] = payload[f]

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                   # noqa: BLE001
        commit = ""

    name = (f"{now.strftime('%Y-%m-%dT%H%M%SZ')}__{slug(args.event, 24)}"
            f"__{slug(args.label or meta.get('description') or meta.get('agent_type') or 'run')}.md")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name

    answer = find_answer(payload) if parsed else raw
    lines = [f"# Agent run — {args.event}", ""]
    lines += [f"- **{k}**: `{v}`" for k, v in meta.items() if v not in (None, "")]
    lines += [f"- **saved_at**: `{now.isoformat()}`"]
    if commit:
        lines += [f"- **commit**: `{commit}`"]
    lines += ["", "## Final response", ""]
    lines += [answer.strip() if answer else "_(no final-response field found in the payload; "
              "the raw payload below is the complete record)_"]
    lines += ["", "## Raw payload", "", "```json" if parsed else "```text", pretty, "```", ""]
    path.write_text("\n".join(lines))
    print(str(path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:                                   # noqa: BLE001
        # A saver that can fail a session is worse than the loss it prevents.
        raise SystemExit(0)
