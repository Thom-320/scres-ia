# Agent runs — every agent leaves a file, or it never happened

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

## Why

On 2026-08-08 two external review agents produced substantial findings and both reports were lost.
One was blocked by a read-only sandbox when it tried to write its file; the other by a permission
prompt on a path outside its working directory. Their conclusions survived only because they also
happened to appear in a terminal log that was still being tailed. **A subagent's answer that lives
in one process's stdout is one `pkill` away from never having existed** — and this repository's
whole discipline is that a result which cannot be pointed at is not a result.

## What is automatic

`.claude/settings.json` registers `SubagentStart` and `SubagentStop` hooks that pipe the hook
payload into `tools/save_agent_output.py`, which writes a dated markdown file here. Nothing is
required of the agent or of whoever launched it.

Each file carries the fields worth reading at the top, the final response as prose, and the **whole
payload** in a fenced block — because hook payload shapes are not guaranteed across versions and a
saver that only understands today's schema loses tomorrow's content.

The hook cannot fail a session. Every path in the saver returns 0, and the command is wrapped in
`|| true`. A hook that breaks a session by failing to save a log is a worse problem than the one it
solves.

## What is not automatic, and how to make it so

External CLI agents — `codex`, `opencode`, `hermes` — are ordinary shell processes, not subagents.
No hook sees them. Pipe them through the same saver:

```bash
codex exec --sandbox read-only "..." | tee >(tools/save_agent_output.py --event codex --label lo-que-revisa)
```

The saver accepts plain text as readily as JSON, so nothing needs to be structured first. And use
`tee`, not a bare pipe: the output should still reach the terminal.

## Reading them later

Filenames sort chronologically: `<UTC timestamp>__<event>__<label>.md`. They are committed, so
`git log -- docs/agent_runs/` gives the history of what was asked and what came back.
