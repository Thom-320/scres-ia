# M/T/R W12 execution-discipline correction

**Binding status:** the six local key-v4 W12 attempt pairs at commit
`b9916ed5db082d8e408d5f13dec77ea2ce350173` are invalid and provide no
scientific evidence. The authoritative inventory is
`mtr_w12_keyv4_attempt_invalidation_b9916ed.json`.

Attempts 1–5 did not cross the signed ACK gate. Attempt 6 crossed it and launched
both children, but the processes remained attached to a live local execution
session. Automatic continuation aborted that session after about 22 minutes,
leaving two reserved zero-byte outputs, no exact receipts, and no completed
pair. This is an execution-discipline failure, not a numerical result.

The next W12 and W16 pairs may run on the local Mac or the verified
`ovh-agent-lab` VPS. On either host, each pair must use detached supervisors. A
separate detached watcher must begin before ACK and persist its own UTC log,
PID/session manifest, heartbeat-age checks, process-liveness checks,
output-size observations, disk checks, and terminal status after the
interactive Codex call ends. Producer and replay remain separately keyed and
separately supervised. Completion still requires both signed receipts,
launch-host verification, independently hashed transfer manifests, retrieval
when applicable, and relocated-pair verification.

W24 and the full frontier are large executions and must run on the verified
`ovh-agent-lab` VPS under the same detached-controller and independent-watcher
discipline. Local execution is not authorized for those large stages.

The five W12 tapes are classified
`BURNED_DEVELOPMENT_OR_CORRECTIVE`. They are useful only for the corrective
W12 gate and can never be called virgin or confirmatory tapes.

Until the full chain above is complete, the binding verdict is:

`W12_CERTIFIED = false; H_PI_COMPUTED = false; PAPER2_EVIDENCE = false`.
