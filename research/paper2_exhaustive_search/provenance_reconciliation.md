# Paper 2 / Paper 3 provenance reconciliation

Date refreshed: 2026-07-14

Status: machine-evidenced Phase 0 record. Git refs were refreshed with
`git fetch --all --prune`; GitHub PR metadata was then read directly with
authenticated `gh`. Narrative descriptions never override the commit, tag,
contract or verdict artifacts.

## 1. Capability preflight

| Capability | Current status | Evidence |
|---|---|---|
| Local repository | OK | `/Users/thom/Projects/research/scres-ia` |
| Git/GitHub | OK | SSH remote `git@github.com:Thom-320/scres-ia.git`; authenticated `gh` |
| Local Python | OK | Python 3.11.15 and the pinned scientific environment |
| Tests | OK for current focused gates | Boundary proof, terminal-return, switch-shell, custody and canonical-metric focused tests pass; the final broad rerun remains pending after current documentation changes |
| VPS | OK and active | `ovh-agent-lab`, 6 CPUs, about 11 GiB RAM; immutable environment preflight completed and the watched `<=4` producer is running |
| Multi-agent audit | OK | Independent proof, runtime and candidate-family reviews executed |
| Push authority | Technically available, not exercised | No push was requested; outward publication remains a user decision |

There is no tooling blocker. A submitted or running local/VPS job is not
scientific evidence; only completed, retrieved, hashed and independently
audited artifacts may change a family state.

## 2. Exact local and remote refs at refresh

```text
local codex/paper2-maintenance-headroom  2490248f9f0accfcb99d1dbb1405c15fecd43de9
origin/codex/paper2-maintenance-headroom a91890bfd3d815fc2bd614076c576487e13e0d06
origin/main                              c6e6d08bfc99db7d842c30edb88e5586227b7729
origin/program-g/structured-spatial-headroom
                                         9b758d45659e7279c55a295f674cf16068e22934
GitHub draft PR #2 head                   3bcf6e96f3dd0c3282d6b051da0a47fa87684b3d
```

At this refresh the current local branch was 19 commits ahead and zero behind
its origin branch, and 268 commits ahead and zero behind `origin/main`.
There were 7 local branches, 10 remote-tracking refs and 27 local/remote tags.

The decisive correction to the earlier provenance note is:

- K3 corrective commit `ef6b53b7` **is now on**
  `origin/codex/paper2-maintenance-headroom` and has the published tag
  `program-k3-retraction-2026-07-13`.
- Concurrent boundary commit `a91890bf` is also published on that origin
  branch, but its terminal boundary claim is scientifically retracted because
  the equal-resource M/T/R exact bound remains unresolved.
- Program-I's own remote branch remains at `9b758d45`.
- Draft PR #2 remains open/draft at `3bcf6e96`, branch
  `codex/garrido-replication-experiments`; its description/head therefore does
  not contain the K3/M/T/R corrective chain and must not be used as current
  scientific status.
- The 19 commits after `a91890bf` through `2490248f` were local-only at refresh.
  They include the invalidation of `179f7c2`, the complete watched/deep-replayed
  `<=3` shell, the frozen `<=4` preflight/producer custody chain and the local
  replay-readiness freeze. Publication would not turn an incomplete shell into
  `H_PI` evidence.

## 3. Branch/tag table

| Ref | Remote state | Scientific interpretation |
|---|---|---|
| `origin/main` | `c6e6d08b` | Repository base; far behind the research branch |
| `origin/codex/paper2-maintenance-headroom` | `a91890bf` | Contains K3 retraction and the now-retracted concurrent boundary commit |
| local `codex/paper2-maintenance-headroom` | `2490248f` at refresh | 19 commits ahead; `<=4` producer active, result absent, no H_PI/H_obs claim |
| `origin/codex/garrido-replication-experiments` / PR #2 | `3bcf6e96` | Draft historical Excel/Cobb-Douglas lane, not current Paper 2 truth |
| `origin/program-g/structured-spatial-headroom` | `9b758d45` | Program-I remote endpoint |
| `program-k3-retraction-2026-07-13` | present remotely | Immutable K3 corrective tag |
| `paper2-bottleneck-negative-2026-07-13` | present remotely | Historical tested-policy null; not an exact family ceiling |

## 4. Supersession and truth rules

Several raw K3 artifacts retain positive booleans. They are superseded by
`results/k3/open_loop_confound_audit.json`, whose effective verdict is
`RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND`. A machine
reader must honor `superseded_by` and `effective_verdict`; the older positive
fields are not evidence.

The same rule now applies to the earlier search boundary:

- `SEARCH_ENVELOPE_BOUNDARY_CERTIFIED` is retracted;
- M/T/R remains `ACTIVE_EXACT_EQUAL_RESOURCE_HPI_BOUND_NOT_YET_CERTIFIED`;
- W12/W16/W24 jobs from `179f7c2` and the older W24 job were intentionally
  stopped after an adversarial HOLD and authorize nothing;
- the later complete `<=3` shell is valid comparator-development evidence but
  boundary-active; the running `<=4` shell is not evidence until retrieval,
  custody validation, independent deep replay and all-exact-ties audit pass;
- neither shell is the complete 11,184,811-calendar W24 comparator or a
  resource-restricted `H_PI` computation;
- Cobb-Douglas remains a secondary construct-sensitivity measure for the
  current canonical-ReT Paper 2 contract.

## 5. Verified scientific-state claims

| Claim | Machine status |
|---|---|
| Programs D-K3 do not currently establish deployable adaptive value | Verified contract-local STOP/RETRACT/null |
| Program E learner did not convert DRA-2b | Verified; CI crosses zero and 0/10 seeds favorable |
| Program I positive region violates worst-CSSU guardrail | Verified |
| Program J PPO beats static in 0/6 seeds | Verified |
| Program K short perishability contradicts the approximately three-year product | Verified |
| K2 gap is perfect-information dominated | Verified, but not a universal replenishment impossibility |
| K3 learner equals one fixed period-8 calendar | Reproduced and remotely tagged |
| Integrated M/T/R has a tested signal-policy null | Verified only as a policy result |
| Integrated M/T/R has a full-horizon H_PI ceiling | **False / unresolved** |
| All 17 current mechanism families have terminal-B proof objects | **False; 0/17 terminal eligible** |
| Paper 2 positive confirmed | False |
| Paper 3 authorized | False |

## 6. Publication and confirmation boundary

No push is performed in this task without explicit authorization. The current
local corrective work must eventually be published or archived at immutable
hashes before any Paper 2 confirmation consumes a genuinely new virgin block.
That provenance requirement is separate from scientific validity: pushing an
invalid proof does not make it evidence, while a valid development screen may
remain local until its pre-confirmation freeze.

Current Phase 0 conclusion:

`EXECUTION_CAPABLE__REMOTE_REFS_RECONCILED__LOCAL_PROOF_WORK_UNPUSHED__NO_TERMINAL_A_OR_B`.
