#!/usr/bin/env python3
"""Recover what subagents and workflows already said, before the retention window eats it.

WHY. Claude Code writes every subagent's conversation to
`~/.claude/projects/<slug>/<session>/subagents/agent-<id>.jsonl` and every workflow's script and
journal beside it, and `cleanupPeriodDays` deletes both after thirty days by default. That is a
large amount of work this project paid for -- audits, reconnaissance sweeps, reviewer simulations --
sitting outside the repository on a timer. The `SubagentStop` hook added on 2026-08-08 saves future
runs; this recovers the ones that predate it.

WHAT IT EXTRACTS. Each agent's task (the first user turn, which is the prompt it was given) and its
final answer (the last assistant text). That pair is what a reader wants and what a hook payload
does not contain -- the payload carries a POINTER to the transcript, which is why the saver follows
it now.

Read-only over the transcript store. Writes markdown into the repository and nothing else.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
STORE = Path.home() / ".claude" / "projects" / "-Users-thom-Projects-research-scres-ia"


def text_of(message) -> str:
    """The assistant's prose. `message` is sometimes a dict and sometimes its repr as a string."""
    if isinstance(message, str):
        try:
            message = json.loads(message.replace("'", '"'))
        except Exception:                                       # noqa: BLE001
            # A repr with apostrophes inside the prose defeats that; fall back to a regex over the
            # raw string rather than losing the content entirely.
            out = re.findall(r"'text':\s*'(.*?)'\}", message, re.S)
            return "\n\n".join(out).replace("\\n", "\n").replace("\\'", "'") if out else ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n\n".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
    return ""


def slug(value: str, limit: int = 56) -> str:
    return (re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()[:limit] or "agent")


def read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:                                       # noqa: BLE001
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7, help="0 recovers everything")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="skip answers shorter than this; a one-line reply is not a report")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "agent_runs" / "recovered")
    args = ap.parse_args()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)) if args.days else None
    args.out.mkdir(parents=True, exist_ok=True)
    kept, skipped_old, skipped_short, index = 0, 0, 0, []

    for path in sorted(STORE.glob("*/subagents/agent-*.jsonl")):
        rows = read(path)
        if not rows:
            continue
        stamps = [r.get("timestamp") for r in rows if r.get("timestamp")]
        when = stamps[-1] if stamps else None
        if cutoff and when:
            try:
                if datetime.fromisoformat(when.replace("Z", "+00:00")) < cutoff:
                    skipped_old += 1
                    continue
            except Exception:                                   # noqa: BLE001
                pass

        answers = [text_of(r.get("message")) for r in rows if r.get("type") == "assistant"]
        answers = [a for a in answers if a and a.strip()]
        if not answers:
            continue
        final = answers[-1]
        if len(final) < args.min_chars:
            skipped_short += 1
            continue

        prompts = [text_of(r.get("message")) for r in rows if r.get("type") == "user"]
        prompts = [p for p in prompts if p and p.strip()]
        task = prompts[0] if prompts else ""
        meta = rows[-1]
        agent = meta.get("attributionAgent") or "subagent"
        title = (task.strip().splitlines() or ["sin tarea"])[0][:90]

        # THE agent_id IS IN THE NAME BECAUSE WITHOUT IT FILES OVERWRITE EACH OTHER. Three
        # Explore agents launched in the same minute with the same prompt produce the same slug,
        # and the first run of this tool wrote 15 files for 23 transcripts before anyone counted.
        # A recovery tool that silently loses a third of what it recovers is the defect it exists
        # to prevent.
        name = (f"{(when or '')[:10]}__{slug(agent, 20)}__{slug(title, 46)}"
                f"__{str(meta.get('agentId'))[:8]}.md")
        body = [
            f"# {agent} — {title}", "",
            f"- **agent_id**: `{meta.get('agentId')}`",
            f"- **session**: `{meta.get('sessionId')}`",
            f"- **branch**: `{meta.get('gitBranch')}`",
            f"- **finished**: `{when}`",
            f"- **effort**: `{meta.get('effort')}` · **version**: `{meta.get('version')}`",
            f"- **transcript**: `{path}`",
            f"- **turns**: {len(rows)} · **assistant messages**: {len(answers)}",
            "", "## Task given", "", "```text", task.strip()[:4000], "```",
            "", "## Final answer", "", final.strip(), "",
        ]
        (args.out / name).write_text("\n".join(body))
        index.append((when or "", agent, title, name, len(final)))
        kept += 1

    # Workflow scripts and journals are the other half: what was orchestrated, not just what one
    # agent said.
    wf = 0
    for script in sorted(STORE.glob("*/workflows/scripts/*.js")):
        dest = args.out / "workflows" / script.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(script.read_text(errors="replace"))
        wf += 1

    if index:
        index.sort(reverse=True)
        lines = ["# Recovered agent runs", "",
                 f"Extracted from the local transcript store on "
                 f"{datetime.now(timezone.utc).date()}; the store is pruned after "
                 f"`cleanupPeriodDays` (30 by default), so these were on a timer.", "",
                 "| finished | agent | task | chars | file |", "|---|---|---|---:|---|"]
        lines += [f"| {w[:16]} | {a} | {t} | {n} | [{f}]({f}) |" for w, a, t, f, n in index]
        (args.out / "INDEX.md").write_text("\n".join(lines) + "\n")

    print(f"recuperados : {kept} transcripts de subagente")
    print(f"workflows   : {wf} scripts")
    print(f"omitidos    : {skipped_old} por antigüedad · {skipped_short} por brevedad")
    print(f"-> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
