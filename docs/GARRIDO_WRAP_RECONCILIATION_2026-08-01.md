# Garrido–WRAP reconciliation ledger — 2026-08-01

**Status:** `RECONCILIATION_COMPLETE_FASE4_90_DONE_288_PENDING`

This ledger reconciles the live repository after the driver-leak discovery. It does
not rewrite frozen contracts or promote development artifacts into manuscript claims.

## 1. Evidence disposition

| object | disposition | allowed use |
|---|---|---|
| `results/garrido_wrap_source_audit/result.json` | valid development audit | source/Cf provenance; Cf1–Cf20 author workbooks, Cf21–Cf90 regenerated |
| `results/garrido_wrap_q1/result.json` | valid development result | Q1 neural-vs-linear panel; `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL` |
| `results/garrido_wrap_q2_smoke_2016h/result.json` | valid smoke only | interface test; not H4 confirmation |
| `results/garrido_meta_learner/result.json` | **retired for contrasts** | provenance only; its search used the leaked driver ranking |
| `results/garrido_meta_learner_v2/result.json` | legacy/pending | VPS process opened before the v2 f3/f4 runtime checks; do not use for contrasts |
| `results/garrido_meta_learner_thesis90_v2/result.json` | valid replay only | 90 thesis-native cells; all six runtime falsifiers pass; `SURFACE_REPLAY_ONLY` |
| `results/garrido_meta_learner_v2_corrected/result.json` | pending | 288 DES rerun under v2 seed block and contract |
| H1/H3 | pending | wait for corrected final-configuration selections |
| Fase 1A contention | valid but subcritical | mechanism is live; `H_regime` is below the 0.01 authorization bar |
| Fase 1B expedition | valid complete `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` | 240 held-out episodes; all nine falsifiers pass; ReT diagnostic gain does not survive the service-first gate |

The old meta-learner contrasts (`+6.31`, `+5.18`, `+12.31`) remain withdrawn.
The 90-cell replay is now the relevant thesis-native validation object for the search
interface, but its `ret_excel` endpoint remains provisional and its replay repetitions
are not independent DES replications. The 288 corrected DES artifact is still required
before reporting a physical confirmation effect size.

## 2. Cobb–Douglas boundary

Cobb–Douglas is retained as a **secondary construct-sensitivity metric** because it
includes the repository's cost ledger and can disagree with canonical order-level
ReT. It is not automatically the WRAP metric:

- the 2024 factory paper is an aggregate APP/factory construct, not the WRAP order ledger;
- the repository mapping from MFSC state to factory variables is our adaptation;
- `c = 1`, omitted cost components, and the mapping require their own declaration;
- a Cobb–Douglas result cannot repair a ReT endpoint that rewards abandonment.

The current six-regime CSSU audit gives Cobb–Douglas a stronger status than the
canonical ReT variants: its optimum coincides with the service optimum and it does
not select the abandonment extreme. That is enough to use it as a **predeclared
secondary headroom gate** for the extended DES. It still does not make it a
thesis-native WRAP metric or erase the need to report service, queue and cost
components separately.

The metric decision for the paper therefore remains open: `ret_excel` cannot be
the sole learning objective, while Cobb–Douglas must remain labelled as a
researcher-constructed factory-level construct rather than a replacement for
Garrido's order-level ReT.

## 3. CSSU action interface

The split extension now accepts the allocation decision through the public
`MFSCSimulation.step()` action interface:

```text
{"cssu_allocation_a": alpha, "cssu_service_rule": rule}
```

The action keeps the frozen one-day activation latency and is rejected in aggregate
topology. The targeted physics tests pass, including mass conservation, live
allocation movement, delayed activation, and aggregate-mode fail-closed behavior.

This closes the API gap, not every physical gap. `op11_handling_hours` is still not
connected to the split `op9_linked` dispatch path; no claim about finite Op11
handling should be made until that path has a dedicated liveness test and a new
contract if its timing changes.

## 4. Immediate gates

1. Do not open H1/H3 on the retired search result.
2. Do not duplicate the VPS meta-learner process.
3. Keep the expedition result bounded to its extended-DES contract; it does not
   authorize MLP/PPO because the service-first timing gate is negative.
4. Keep canonical ReT, service/backorder outcomes, and Cobb–Douglas in separate
   columns; do not let one construct silently replace another.
5. Do not authorize MLP/PPO from a headroom signal whose objective still rewards
   abandonment.

## 5. Next artifacts

- corrected 288 VPS result: `results/garrido_meta_learner_v2_corrected/result.json`;
- thesis-native 90-configuration search result: `results/garrido_meta_learner_thesis90_v2/result.json`;
- separate 288-configuration extended search result;
- CSSU physical Op11 timing addendum or explicit no-go;
- corrected expedition result and metric adjudication;
- metric contract with an abandonment falsifier and service/queue estimands.
