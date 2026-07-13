# Paper 2 / Paper 3 provenance reconciliation

**Audit date:** 2026-07-13
**Auditor:** root PI session with full local filesystem, live VPS, and full RL/DES/SA stack.
**Status of this document:** discharges Section 0 (capability, provenance, truth preflight) of the Paper 2/3 charter. It supersedes the connector-only `docs/SESSION_PROVENANCE_GATE_2026-07-13.md` on the provenance axis only (see §5).

---

## 1. Capability preflight (verified this session)

| Capability | Status | Evidence |
|---|---|---|
| Local FS `/Users/thom/Projects/research/scres-ia` | ✅ | working tree read/write |
| Git objects, branches, tags, remotes | ✅ | `git fetch --all` clean; refs enumerated below |
| Python env | ✅ | `.venv` Python **3.11.15** |
| RL / DES / SA stack | ✅ | gymnasium 1.3.0, stable_baselines3 2.9.0, **sb3_contrib 2.9.0 (MaskablePPO)**, simpy 4.1.2, torch 2.12.1, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0, SALib present |
| Local simulation execution | ✅ | K3 confound audit re-ran: `5 passed in 0.10s`, result SHA-256 stable `6d509cdd…` |
| VPS remote execution | ✅ | `ovh-agent-lab` (135.148.42.12) reachable, **6 vCPU / 11 GB / idle**, up 4 days; identity `~/.ssh/id_ed25519_ovh_2026_06_24` |
| Branch/commit/tag/push ability | ✅ (push gated on user approval) | write access to local git; SSH remote `git@github.com:Thom-320/scres-ia.git` |
| Multiagent orchestration | ✅ | Workflow tool active (ultracode); candidate theory-screen launched |

No capability is missing. This session is **not** `EXECUTION_BLOCKED_BY_TOOLING`.

---

## 2. Verified remote state (after `git fetch --all`)

| Remote ref | SHA | Meaning |
|---|---|---|
| `origin/main` | `c6e6d08b` | Stale base (risk modulation + inventory rebuild lead time). **Not** the July source of truth. |
| `origin/codex/garrido-replication-experiments` | `3bcf6e96` | Draft PR #2 head; `STOP_DRA2B_PRE_TREE_GATE`. |
| `origin/program-e/oracle-to-policy` | `59bfd218` | `STOP_PROGRAM_E_VALIDATION`. |
| `origin/program-f/risk-mitigation-portfolio` | `4ab68be9` | Program F noisy-signal audit. |
| `origin/program-g/structured-spatial-headroom` | `9b758d45` | Furthest **pushed** science line (through Program I / Program G terminal). |
| `origin/audit/provenance-gate-2026-07-13` | `ac6bb790` | Prior connector-only session's provenance gate (2 commits on `9b758d45`). |
| `origin/codex/garrido-postfix-reruns` | `e901f860` | Garrido2024 CPU rerun branch. |

## 3. Verified local state

| Local ref | SHA | Relationship |
|---|---|---|
| `codex/paper2-maintenance-headroom` (HEAD) | `ef6b53b7` | **7 commits ahead of every remote ref**, 0 behind. |
| `codex/garrido-replication-experiments` | `a167ddc9` | 1 commit ahead of its remote. |

### 3.1 The 7 unpushed commits (the decisive gap)

```
ef6b53b7  audit K3 adaptive win against full-horizon static schedule   ← K3 RETRACTION
81f00317  close K3 ReT screen against resource-matched classical controls
407b5ea8  K2 strong-comparator screen: EVPI-dominated, no convertible headroom
53350f4d  Concede K critique: reclassify Program K exploratory, freeze corrected K2 prereg
4bc9a6e5  Program K (perishable replenishment): first genuine learned RL win  ← later retracted
9a586cf3  Paper 2 (adaptive maintenance Op5-Op7): reach RL, PPO 0/6 beats static
373aa6ab  screen finite maintenance control for adaptive headroom
```

`git branch -r --contains ef6b53b7` → **NONE**. The Program J (maintenance), K (perishable), K2 (holding-cost replenishment), and K3 (budgeted replenishment) evidence — including the **K3 open-loop-confound retraction** — exists **only on the local disk**. This is a single point of failure and must be pushed to an immutable ref before any Paper 2 claim rests on it (charter §0.2, §16).

## 4. Three-way divergence off `9b758d45`

```
                         9b758d45  (pushed: Program I terminal)
                        /          \
   (remote) ac6bb790  ●            ● ef6b53b7  (local HEAD, UNPUSHED)
   provenance-gate   2 commits     7 commits: J / K / K2 / K3 / K3-audit
   (BLOCKED verdict)               (the actual Paper-2 science)
```

- **Remote-ahead = 2** (`0a453ad7`, `ac6bb790`): a machine-readable provenance gate + verdict, authored by a prior connector-only session. Adds only `docs/SESSION_PROVENANCE_GATE_2026-07-13.md` and `results/session_provenance_gate_2026-07-13.json`.
- **Local-ahead = 7**: the maintenance/replenishment science line.
- These two lines have **not** been merged. Neither contains the other.

## 5. Reconciliation of the prior provenance gate

The prior session returned `BLOCKED_PROVENANCE_NO_TERMINAL_SCIENTIFIC_VERDICT`. Its own "Required provenance package" listed exactly five things it lacked: (1) a ref containing `ef6b53b7` and parents; (2) `results/k3/open_loop_confound_audit.json` + trajectories + full static frontier + resource ledgers; (3) J/K/K2 contracts + verdict JSONs + raw tables; (4) dependency lock + seed registry; (5) an executable compute endpoint.

**This session possesses all five.** Item (2) was re-executed and reproduced bit-for-bit (`fixed_minus_mpc_ordered_D0 = [0,0,0]` → identical resources; fixed period-8 beats MPC by ReT ≈ 0.0177 on learner-test tapes 6900001–6900120). Therefore the prior gate is **superseded on the provenance axis**: the local K3 retraction is now independently verified, not a `PROVISIONAL_LOCAL_ASSERTION`. The gate remains correct that no *positive Paper 2* is authorized.

## 6. K3 retraction — independently confirmed

- Contract: `program_k3_ret_budgeted_replenishment_v1` (weekly order multipliers under exact total + weekly caps).
- PPO emitted **one** unique test sequence on every tape: `(1.5,1.5,1.5,1.5,1.5,1.5,1.0,0.0)·D0`.
- A full-horizon **period-8 fixed** schedule reproduces PPO exactly and beats the "adaptive" MPC under identical resources.
- Effective verdict: **`RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND`**. `paper2_adaptive_confirmed = false`, `paper3_neural_retention_authorized = false`.
- Superseded artifacts (`4bc9a6e5` "first genuine learned RL win", positive K3/PPO JSONs) retain positive booleans in-file; machine readers must honor `superseded_by` / `effective_verdict`.

## 7. Note on an in-session untracked-file mutation

`research/paper2_exhaustive_search/phase0_failure_taxonomy.json` was augmented during this session (300 → 354 lines) by a search subagent, prepending the three Paper-1 lanes (Track A buffer/shift, Track B downstream, Track C campaign) to the family list. The addition is **content-correct** — those lanes are legitimate closed families — so the canonical copy frozen at `results/paper2_search/failure_taxonomy.json` carries all **17** families. Flagged here for full provenance transparency; no tracked file was altered.

## 8. Truth-hierarchy conflicts found

1. **Program H prose** ("no Programs I–K could proceed", "last computational extension") is **superseded** by the executed Programs I, J, K, K2, K3.
2. **Draft PR #2 description** (Track-B / Cobb–Douglas) is stale relative to the terminal same-contract reversal and everything after `3bcf6e96`.
3. **`4bc9a6e5` commit message** ("first genuine learned RL win") is **retracted** by `ef6b53b7`.
   In every case the machine artifact (verdict JSON / corrective audit) governs, per charter §0.3.

## 9. Required provenance actions (recommended; push gated on user approval)

1. **Tag + push** the K3 retraction chain to an immutable ref (e.g. tag `program-k3-retraction-2026-07-13` at `ef6b53b7`, push branch `codex/paper2-maintenance-headroom`). Until done, the decisive evidence is unbacked.
2. Leave `origin/audit/provenance-gate-2026-07-13` as historical; annotate it as superseded by this document.
3. Do **not** fold research branches into `main` yet — orthogonal to Paper 2.
4. Keep raw K3 trajectories / large artifacts out of ordinary Git history; prefer LFS/artifact store with checksums (charter §16). `results/paper2_search/artifact_index.json` records SHA-256 for 58 verdict JSONs.

---

**Binding conclusion:** provenance is reconciled and the local evidence is independently verified. This session is authorized to proceed to a *terminal scientific verdict* (Paper 2 boundary certificate or confirmed positive), not merely another provenance gate. No positive Paper 2 and no Paper 3 is authorized by the evidence audited so far.
