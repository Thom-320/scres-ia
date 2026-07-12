# Program G — intervention ledger (frozen 2026-07-12)

Status: **FROZEN with V1.1. Build unauthorized until every `face_validation` cell below reads
CONFIRMED by Garrido.** "Realism is not a substance poured over invented parameters." Each Program G
extension is accounted for on six axes: operational interpretation, range, source / face-validation
status, falsifier (what would show the mechanism is absent), sensitivity (later robustness check),
and claim limit (what it may NOT be used to assert).

Source tags: **[T]** thesis / repo-config anchor · **[R]** repository prior result · **[X]**
researcher-imposed extension (needs Garrido face validation) · **[L]** literature.

## Physical extensions

| Parameter | Operational interpretation | Range (v1.1) | Source | Falsifier | Sensitivity | Claim limit |
|---|---|---|---|---|---|---|
| Emergency-reserve overlay | Extraordinary reserve moved on top of normal MFSC flow | 10,000 rations, 2×5,000 loads | [X] (overlay); 5,000 = [T] `RATIONS_PER_BATCH` | G0: normal flow unaffected when reserve idle; A+B demand conserved | Overlay vs replace-downstream (Option A) | Prepositioning value only — NOT total-logistics value |
| Load size 5,000 | One indivisible convoy load | 5,000, no partial | [T] batch anchor `config.py:44` | full-load accounting in G0 | 2,500 / 7,500 load size | "batch anchor", NOT vehicle capacity evidence |
| Convoy count 1 | Single shared lift SB↔A/B | 1 | [X] | dispatch to A denies B in G0 | 2 convoys = explicit later contract (forbidden as a rescue) | scarcity is by construction, disclosed |
| Convoy cycle 24h+24h | Outbound + return travel | 24h/24h | [X] | travel clock in G0 | 12/36h asymmetric routes | not a measured road time |
| Decision epoch 48h | Convoy re-decided when available | 48h | [X] (fixes the 120h-idle contradiction) | availability persists between epochs in G0 | 24h / 72h epoch | persistence claim rests on this |
| No in-episode replenishment | Reserve is spent, not topped up | none, 56d | [X] | reserve monotone-nonincreasing in G0 | route-aware replenishment w/ frozen lead | no "top-up because code wished it" |
| R22 = bidirectional pause | LOC closure halts travel, no destruction | pause only | [T] R22 exists; [X] no-destruction | route-down masks dispatch; clock pauses in G0 | destruction variant = separate contract | v1 makes no vehicle/cargo-loss claim |
| S1 fixed (S2 secondary) | Production shift held constant in spatial screen | S1 only primary | [R] Program L: S1/S2/S3 identical ReT for positive buffers | S2 preflight must change reachable states before entering | S2 as gated extension | spatial value not conflated with production value |

## Demand & tempo extensions

| Parameter | Operational interpretation | Range (v1.1) | Source | Falsifier | Sensitivity | Claim limit |
|---|---|---|---|---|---|---|
| Aggregate conservation | Tempo moves demand between A/B, not total up | `D_A+D_B=D_t` | [X] identification choice | aggregate tape hash invariant across allocations | unequal-total variant | isolates allocation, not demand-magnitude value |
| Semi-Markov tempo | Persistent operational regimes | low/routine/surge, dwell U{4,5,6}wk | [X]; [T] thesis notes non-stationarity limits buffering | dwell + transition ledger in G0 | 6–8wk persistence sensitivity | latent, never observed by policy |
| Surge weight | Demand multiplier in surge | 2.0 / 3.0 | [X] | dose-response across the two levels | wider weights later | not "make it brutal until PPO smiles" |
| Commonality | P(both CSSU surge in lead window) | 0.10 / 0.35 | [X] — **the key omitted axis** | value-of-WHERE collapses at high commonality | full concurrency sweep | headroom must survive realistic concurrency |

## Information extensions

| Parameter | Operational interpretation | Range (v1.1) | Source | Falsifier | Sensitivity | Claim limit |
|---|---|---|---|---|---|---|
| Signal = sensitivity/FPR | Advance operational warning per CSSU | moderate 0.70/0.20, high 0.85/0.10 | [X] — replaces scalar accuracy | must beat shuffled AND delayed AND wrong-CSSU placebo | continuous-score variant | not "accuracy 0.75" (prevalence-dependent) |
| Lead | How far ahead the alert fires | 7 / 14 days | [X] must ≥ action lead | alert after commitment = no advance info (delayed placebo) | intermediate leads | a late signal is not advance information |
| Wrong-CSSU placebo | Swap A/B alerts | required | [X] new in v1.1 | value must drop under swap | — | proves value is knowing WHERE |

## Candidate observations pending Garrido (`advance_information.candidate_signals`)

`planned_troop_level`, `deployment_calendar_pressure`, `route_threat_score` — each needs Garrido to
confirm it exists operationally, its realistic lead, and its sensitivity/FPR envelope. Until then they
are [X] and cannot enter the policy observation.

## V1.2 revisions (transport-binding correction — supersede the rows above where they conflict)

| Parameter | Operational interpretation | Range (v1.2) | Source | Falsifier | Sensitivity | Claim limit |
|---|---|---|---|---|---|---|
| Convoy = downstream transport | The scarce SB→CSSU lift at thesis scale | 2500/day (5000/48h) | [T] matches thesis 2400–2600/day + `RATIONS_PER_SHIFT` | G0: convoy binds when both CSSU need service | overlay (Option B) = Garrido either/or | not a historical single vehicle |
| Action space {A,B,HOLD} | Weekly dispatch priority, all departures | 3 actions | [X] + [R] Program L dead-shift evidence | G0: priority yields multiple departures/wk, no auto-reorient | 6-action w/ dynamic shift = forbidden mid-study | dynamic shift is a DEAD dimension (S2 unmovable) |
| 10k SB stock | Initial finished stock, genealogy-tracked | 10,000 | [X] (thesis Op9 wk = 15,750, [T]) | G0: mass conservation, no inventory creation | 15,750 as single later sensitivity | researcher param, NOT thesis reserve level |
| Surge multiplier | Sustained regime demand | 1.25 / 1.50 | [X] | dose-response across the 2 levels | 2×/3× only for spike models, not regimes | a 2-month regime ≠ a 1-day R24 spike |
| Primary risks | Isolate tempo+signal+convoy | R22 on; R24-native/R23/R11/R21/R3 off | [X] identification choice | G0: only R22 fires in primary tapes | all-risk-current after a pass | off-risks NOT claimed absent in reality |
| Signal = balanced accuracy | sens = spec = q, tempo content only | 0.65/0.75/0.85, lead 1–2wk | [X] | must beat block-shuffle + delayed placebo on rollout | asymmetric sens≠spec tier | route_threat excluded (uninterpretable if mixed) |
| Periodic static bar | All period-1–4 calendars | 120 | [R] Track B statue-vs-sequence lesson | G0: calendars reproducible under CRN | — | learner must beat THIS, not always-A/B/HOLD |
| TRSC cost | Holding as inventory-time | resource/Pareto vector | [X]; [T] thesis omits cost coefficients | — | monetized version needs Garrido coefficients | no invented money; TRSC deferred |
| episode_weeks | Long enough for regimes to matter | 52 | [X] | dwell 4–8wk plays out ~7–13× | — | — |

## Standing

No parameter tagged [X] may enter a generated tape until Garrido confirms it in the sign-off template.
[T]/[R] anchors are already defensible. Every [X] row above has a falsifier that runs in G0 (physics)
or G1 (mechanism) BEFORE any learner — so a mis-specified extension is caught by a preflight, not by a
reviewer.
