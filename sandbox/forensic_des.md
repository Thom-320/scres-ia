# Forensic DES Audit — `supply_chain/config.py` + `supply_chain/supply_chain.py` vs `thesis.txt`

Audit performed 2026-06-19. **Source of truth: `thesis.txt`** (Garrido-Rios 2017, extracted text). Repo-internal docs (`THESIS_FIDELITY_AUDIT.md`, `docs/*`) were read as CLAIMS to verify, never as truth.

Verdict legend: **BUG** (code diverges from thesis in a way that biases output) · **EXTENSION** (declared, non-thesis addition) · **AMBIGUITY** (thesis is internally inconsistent or underspecified; repo picked one reading) · **MATCH** (code ≡ thesis) · **UNVERIFIABLE** (cannot confirm without re-running).

Counts: 7 BUG, 6 EXTENSION, 4 AMBIGUITY, 17 MATCH, 1 UNVERIFIABLE.

---

## Section A — Constants / Config

### Finding A1: Year basis 8,064 h
- **Thesis says:** 1 year = 2 semesters = 12 months = 48 weeks = 336 days = **8,064 h** (`thesis.txt:3999-4002`, Table 6.2 EC1=2,304h/yr at S=1).
- **Repo does:** `HOURS_PER_YEAR_THESIS = 8_064`, `DEFAULT_YEAR_BASIS = "thesis"` (`config.py:34-36`).
- **Verdict:** MATCH
- **Impact:** —
- **Evidence:** `1 shift = 8h, 1 day = 3 shifts, 1 week = 7 days, 1 month = 4 weeks, 1 semester = 6 months, 1 year = 2 semesters` ⇒ 8,064h. Repo identical.

### Finding A2: Simulation horizon 161,280 h (20 yr)
- **Thesis says:** `t = 1…161,280` (`thesis.txt:5469`); R3 expected once per horizon (`thesis.txt:5562, 5601`).
- **Repo does:** `SIMULATION_HORIZON = 161_280` (`config.py:40`).
- **Verdict:** MATCH
- **Impact:** —
- **Evidence:** Thesis: "Up to 20 years or 161,280 hours" (`thesis.txt:5615`).

### Finding A3: MAX_ORDERS = 6,000
- **Thesis says:** `j = 1…6,000` (`thesis.txt:5467`).
- **Repo does:** `MAX_ORDERS = 6_000` (`config.py:41`).
- **Verdict:** MATCH

### Finding A4: BACKORDER_QUEUE_CAP = 60
- **Thesis says:** "list of up to 60 delayed orders … last order … removed and labelled as lost or unattended" (`thesis.txt:5105-5107`).
- **Repo does:** `BACKORDER_QUEUE_CAP = 60` (`config.py:45`).
- **Verdict:** MATCH

### Finding A5: λ = 320.5, RATIONS_PER_SHIFT = 2,564, RATIONS_PER_BATCH = 5,000
- **Thesis says:** λ=320.5 rations/h (`thesis.txt:4053, 5473`); "After a work shift of 8 hours, Op5,j will have transferred a total of 2,564 pre-assemblies" (`thesis.txt:4152`); Q=5,000 (`thesis.txt:4171`).
- **Repo does:** `ASSEMBLY_RATE = 320.5`, `RATIONS_PER_SHIFT = int(320.5*8) = 2564`, `RATIONS_PER_BATCH = 5_000` (`config.py:28, 44, 46`).
- **Verdict:** MATCH

### Finding A6: HOURS_PER_MONTH = 672 (28-day month)
- **Thesis says:** "1 month is equal to 4 weeks" ⇒ 4×168 = 672 h (`thesis.txt:4001`). Note: this gives 13.14 months/year (not 12) — thesis arithmetic is loose, but the convention is "month = 672 h".
- **Repo does:** `HOURS_PER_MONTH = 672` (`config.py:33`).
- **Verdict:** MATCH (follows thesis convention; thesis-internal inconsistency acknowledged but not reproduced).

### Finding A7: 13 operations (Op1–Op13), PT/Q/ROP
- **Thesis says:** Figure 6.2 + Section 6.3.3 operation-by-operation (`thesis.txt:4113-4222`).
- **Repo does:** `OPERATIONS` dict (`config.py:162-313`).
- **Verdict:** MATCH
- **Evidence:** Spot-check all 13:
  | Op | Thesis PT/Q/ROP/Risks | Repo | ✓ |
  |---|---|---|---|
  | 1 | 672 / 12 cntr / 4,032 / R12 | 672 / NUM_SUPPLIERS=12 / 4_032 / ['R12'] | ✓ |
  | 2 | 24 / 190,000 each rm / 672 / R13 | 24 / 190_000 / 672 / ['R13'] | ✓ |
  | 3 | 24 / 15,500 each / 168 / R21, I=0 | 24 / 15_500 / 168 / ['R21'], init 0 | ✓ |
  | 4 | 24 / 15,500 / 168 / R22 | identical | ✓ |
  | 5 | 1/λ / 1 / 1/λ / R11,R21,R3 | 1/ASSEMBLY_RATE / 1 / ditto / ['R11','R21','R3'] | ✓ |
  | 6 | 1/λ / 1 / 1/λ / R11,R21,R3 | identical | ✓ |
  | 7 | 1/λ / 5,000 / 48 / R14,R21,R3 | 1/ASSEMBLY_RATE / 5,000 / 48 / ['R14','R21','R3'] | ✓ |
  | 8 | 24 / 5,000 / 48 / R22 | identical | ✓ |
  | 9 | 24 / 2,400–2,600 / 24 / R21,R3 | identical | ✓ |
  | 10 | 24 / 2,400–2,600 / 24 / R22 | identical | ✓ |
  | 11 | 0 / 2,400–2,600 / 24 / R23, 2 CSSUs | identical, num_units=2 | ✓ |
  | 12 | 24 / 2,400–2,600 / 24 / R22 | identical | ✓ |
  | 13 | — / — / — / R24 | pt=0, q=0, rop=0, risks=['R24'] | ✓ |

### Finding A8: Op5/6/7 PT — Table 6.20 says PT=0
- **Thesis says:** Section 6.3.3 (`thesis.txt:4149-4175`) → PT5=PT6=PT7 = 1/λ = 0.003125 h/ration. Table 6.20 (`thesis.txt:6001-6047`) lists PT=0 for Op5/6/7. **Internal thesis inconsistency.**
- **Repo does:** PT = `1/ASSEMBLY_RATE` = 0.003121 h/ration (Section-6.3.3 reading).
- **Verdict:** AMBIGUITY (thesis is self-contradictory; repo picked the text-Section reading, which is more physically meaningful).
- **Impact:** None — picking PT=0 would make the AL non-causal (instant assembly). Repo choice is correct.

### Finding A9: Op9/10/11/12 Q range — Fig 6.2 vs Table 6.20 mismatch
- **Thesis says:** Figure 6.2 + Section 6.3.3 → Q = U(2,400, 2,600). Table 6.20 (`thesis.txt:6068, 6072, 6076`) → Q = "2,000 to 2,500 rations". **Internal thesis inconsistency.**
- **Repo does:** Default Q=(2,400, 2,600); alternative via `THESIS_DOWNSTREAM_Q_RANGES["table_6_20"] = (2_000, 2_500)` (`config.py:110-123`).
- **Verdict:** AMBIGUITY (defensible choice; both readings available).

### Finding A10: Inventory buffers Table 6.16
- **Thesis says:** I168,1={15360, 15360, 15750}, I336,1={30720, …}, I504,1={46080, …}, I672,1={61440, …, 63000}, I1344,1={122880, …, 126000} (`thesis.txt:5798-5821`).
- **Repo does:** `INVENTORY_BUFFERS` (`config.py:632-638`).
- **Verdict:** MATCH (all 15 numeric values identical).

### Finding A11: Capacity by shifts Table 6.20
- **Thesis says:** S=1: Op3/4 Q=15,500, Op7/8 Q=5,000 ROP=48; S=2: 31,000, 5,000 ROP=24; S=3: 47,000, 7,000 ROP=24 (`thesis.txt:5974-6062`).
- **Repo does:** `CAPACITY_BY_SHIFTS[1..3]` (`config.py:646-677`).
- **Verdict:** MATCH on Op3/4/7/8 Q and ROP. `theoretical_capacity_hrs` field name is unit-ambiguous (per-day value stored under "h" name), but unused by the engine.

### Finding A12: VALIDATION_TABLE_6_10
- **Thesis says:** Pt = [711,808; 901,131; 806,454; 719,344; 731,016; 629,429; 707,203; 728,878]; ECS = [725,021; 773,675; 735,389; 771,434; 888,776; 712,315; 732,883; 801,239]; RMSE = 87,918 (`thesis.txt:5337-5379`).
- **Repo does:** `VALIDATION_TABLE_6_10` (`config.py:686-709`).
- **Verdict:** MATCH (all 17 numeric values identical, including RMSE 87,918).

---

## Section B — Risk sampling

### Finding B1: Default risk occurrence mode is `legacy_renewal` — **biases all uniform risks ~2× too frequent**
- **Thesis says:** Table 6.11 (`thesis.txt:5580-5613`) gives exact expected event counts per 20-year horizon: R11=960, R21=10, R22=40, R23=20, R24=240, R3=1. These are consistent ONLY with **one-event-per-interval (periodic)** sampling, NOT with U(a,b) renewal.
  - Sanity check (periodic): R11 = 8064/168 × 20 = 960 ✓; R21 = 8064/16128 × 20 = 10 ✓; R3 = 8064/161280 × 20 = 1 ✓.
  - Sanity check (renewal, mean IA = (1+b)/2): R11 = 8064/84.5 × 20 = 1909 ✗; R21 = 8064/8064.5 × 20 = 20 ✗; R3 = 8064/80640.5 × 20 = 2 ✗.
- **Repo does:** Default `risk_occurrence_mode = "legacy_renewal"` (`supply_chain.py:142`). In that mode `_uniform_risk_interarrival` returns `rng.integers(a, b+1)` every event (`supply_chain.py:1483-1484`), which is renewal sampling with mean IA = b/2. So uniform risks fire **~2× too often**.
- **Verdict:** BUG
- **Impact:** When `run_static.py` is invoked (the documented entry-point), the stochastic baseline has:
  - R11 workstation breakdowns ≈ 1,909/20 yr instead of 960 → assembly capacity degraded ~2×.
  - R21 natural disasters ≈ 20/20 yr instead of 10.
  - R22 LOC attacks ≈ 80/20 yr instead of 40.
  - R23 forward-unit attacks ≈ 40/20 yr instead of 20.
  - R24 contingent surges ≈ 480/20 yr instead of 240.
  - R3 black swans ≈ 2/20 yr instead of 1.
  All push throughput DOWN relative to thesis. The thesis-faithful alternative exists (`"thesis_periodic"`), but `run_static.py:22-71` does not pass it.
- **Evidence:** `run_static.py:28-35, 64-71` (no `risk_occurrence_mode` kwarg); `supply_chain.py:1483-1487`:
  ```python
  if self.risk_occurrence_mode == "legacy_renewal":
      return float(self.rng.integers(a, b_val + 1))   # mean = b/2
  if first:
      return float(self.rng.integers(a, b_val + 1))
  return float(b_val)                                  # periodic mode: mean = b
  ```

### Finding B2: `thesis_periodic` mode is not literally the thesis either
- **Thesis says:** "U(x: occurrence of a breakdown at the workstation in an interval of 168 hours" (`thesis.txt:4678-4681`) — i.e., time-of-occurrence within each 168-h interval is uniform. Expected: 1 event per interval.
- **Repo does:** `thesis_periodic` mode = first event U(a,b), then every subsequent event at fixed `b` (`supply_chain.py:1485-1487`). This is **fixed-offset periodic**, not **uniform-within-each-interval periodic**. Mean rates match Table 6.11, but variance is wrong (variance should be (b-a)²/12 per event in true periodic-within-interval; here variance is 0 after the first event).
- **Verdict:** EXTENSION (rates match thesis Table 6.11; higher moments differ).
- **Impact:** Negligible for first-order metrics (annual throughput, RMSE); noticeable for tail-event statistics.

### Finding B3: R13 period — `legacy_renewal` gives 4× too FEW events
- **Thesis says:** R13 = 58 events/year (`thesis.txt:5595, 5604`).
- **Repo does:** `_binomial_risk_period("R13")` returns `self.params["op2_rop"] = 672` (monthly) in legacy_renewal mode (`supply_chain.py:1494-1496`). That gives 12 cycles/year × 12 × 0.10 = 14.4/year. In `thesis_periodic` mode it returns `HOURS_PER_WEEK = 168`, giving 48 × 1.2 = 57.6/year ✓.
- **Verdict:** BUG (in default mode).
- **Impact:** R13 (raw-material delivery delays) is **4× too rare** under `legacy_renewal`. The mismatched sign vs B1 (where uniform risks are too frequent) means default-mode risk balance is corrupted in BOTH directions.

### Finding B4: R12 contract delays — MATCH in both modes
- **Thesis says:** R12 = 2⅙ events/year (`thesis.txt:5594`); n=12, p=1/11 (`thesis.txt:5638`); biannual contracting cycle (Op1 ROP=4032h ⇒ 2 cycles/year).
- **Repo does:** `_binomial_risk_period("R12") = self.params["op1_rop"] = 4032` regardless of mode (`supply_chain.py:1491-1492`). 2 cycles/year × 12 × 1/11 = 2.18/year ✓.
- **Verdict:** MATCH

### Finding B5: R14 defects — approximately MATCH
- **Thesis says:** R14 = 22,153 events/year, B(n=2564, p=3/100) (`thesis.txt:5574, 5596, 5640`).
- **Repo does:** Samples binomial daily using `produced` (line 1559, ~2564 at S=1), period = 24h. With no production on Sunday, 288 active days/year × 2564 × 0.03 ≈ 22,153/year.
- **Verdict:** MATCH (gated by actual production, which zeroes on Sundays — matches thesis 6-day week).

### Finding B6: R11 repair time clipped at ≥1h
- **Thesis says:** exp(β=2) hours (`thesis.txt:4682-4686, 4751`); no floor mentioned.
- **Repo does:** `repair = max(1, self.rng.exponential(beta))` (`supply_chain.py:1510`).
- **Verdict:** EXTENSION (defensible to avoid sub-hour events at hourly tick resolution, but biases R11 downtime upward: P(exp(2)<1)≈39% of repairs are clipped to 1h).
- **Impact:** Small — mean repair shifts from 2h to ~1.83h after clipping, partially cancelling the B1 over-frequency bias.

### Finding B7: R22 picks ONE LOC per event
- **Thesis says:** "destruction of **one** line-of-communication" per event (`thesis.txt:4915`); "likelihood that Op4, Op8, Op10 and Op12 are destroyed is identical" (`thesis.txt:4930-4931`). Table 6.11 = 40 events/20 yr.
- **Repo does:** `target = int(self.rng.choice(loc_ops))` (`supply_chain.py:1630`). One LOC per event, chosen uniformly.
- **Verdict:** MATCH (the alternative — all four LOCs simultaneously — would give 4×40=160 op-down events, inconsistent with Table 6.11).

### Finding B8: R3 duration fixed 672h
- **Thesis says:** "Op5, Op6, Op7 and Op9 are simultaneously taken out of operation during 672 consecutive hours" (`thesis.txt:5068-5069`).
- **Repo does:** `recovery: {"dist": "fixed", "duration": 672}` (`config.py:412`); `_risk_R3` waits exactly `duration` (`supply_chain.py:1685`).
- **Verdict:** MATCH

### Finding B9: R3 expected event count in horizon = 1
- **Thesis says:** 1 event per 20 yr (`thesis.txt:5562, 5601`).
- **Repo does (default mode):** legacy_renewal mean IA = (1+161280)/2 = 80,640.5 h ⇒ ~2 events per 161,280-h horizon.
- **Repo does (thesis_periodic mode):** first event at U(1, 161280), then every 161,280 h ⇒ exactly 1 event in horizon (modulo horizon edges).
- **Verdict:** BUG in default mode (R3 fires 2× per simulation); MATCH in `thesis_periodic` mode.
- **Impact:** Doubles expected black-swan impact on RL training and stochastic baseline.
- **Evidence:** `config.py:411` `{"dist": "uniform", "a": 1, "b": 161_280}`; `supply_chain.py:1483-1484` legacy_renewal returns `rng.integers(1, 161281)` ⇒ mean 80,640.5h.

### Finding B10: R24 surge size U(2,400, 2,600)
- **Thesis says:** U2(Dcn ∈ Z+, c=2,400, d=2,600) (`thesis.txt:4971-4990, 5644`).
- **Repo does:** `surge: {"lo": 2400, "hi": 2600}` (`config.py:404`); `_risk_R24` samples `rng.integers(surge_lo, surge_hi+1)` (`supply_chain.py:1661`).
- **Verdict:** MATCH
- **Note on the "24h variant" question:** the surge size (2400–2600) is monthly-sized per thesis ("rations/672 hours"). The repo correctly uses b=672 for inter-arrival and 2400–2600 for size. **There is no 24h-coded variant in `RISKS_CURRENT`.** (The 24h figure in `THESIS_FIDELITY_AUDIT.md` is the daily demand rate, not R24.) Verdict: no bug.

### Finding B11: R24 surge added to a single day's demand
- **Thesis says:** Surge is "rations/672 hours" (`thesis.txt:4972-4973`); also "2,400 to 2,600 rations/month" (`thesis.txt:4417`). Temporal interpretation ambiguous (event-spike vs sustained monthly lift).
- **Repo does:** `_risk_R24` adds full surge to `_contingent_demand_pending`, which is added in one shot to the next daily demand order (`supply_chain.py:1660-1672, 1369-1372`). Annual aggregate matches (12 events × 2500 ≈ 30,000 = 28,800–31,200/yr), but the daily distribution has 12 spikes instead of 12 sustained lifts.
- **Verdict:** AMBIGUITY (annual totals MATCH; intra-month profile differs; thesis wording supports either reading).

### Finding B12: R14 affected ops
- **Thesis says:** R14 at Op7 only (Fig 6.2, `thesis.txt:4379`, Table 6.6b row R14 — quality problems at quality-control step).
- **Repo does:** `RISKS_CURRENT["R14"]["affected_ops"] = [7]` (`config.py:373`). ✓ But `THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"] = "thesis_strict_op6"` and the engine default is `"reprocess"` (`supply_chain.py:136`). The "thesis_strict_op6" name is misleading — R14 is at Op7 and defects return to the *previous* op (Op6) for rework per Table 6.6b (`thesis.txt:4734-4736`). The default `"reprocess"` mode dumps defects into `raw_material_al` (skipping Op5 entirely) — diverges from thesis.
- **Verdict:** BUG (default defect routing is wrong): defective units should re-enter at Op6 rework buffer, not at the AL's raw-material buffer.
- **Impact:** Default mode under-counts AL rework load and over-feeds Op5; bias direction depends on R14 frequency.
- **Evidence:** `supply_chain.py:136` default; `supply_chain.py:1576` `yield self.raw_material_al.put(defects)` vs `supply_chain.py:1570` `yield self.rework_op6.put(defects)`.

---

## Section C — Buffer / Container mechanics

### Finding C1: All 8 buffers `capacity=INF`
- **Thesis says:** "storage capacities of WDC, SBs and CSSBs are assumed to be unlimited" (`thesis.txt:5116-5118`). The AL WIP and the theatre inventory have no stated capacity bound.
- **Repo does:** Lines 252-260, all 8 containers use `capacity=INF` (INF=10,000,000).
- **Verdict:** MATCH (the "unbounded Container = classic optimistic-model bug" hypothesis is **rejected** for this codebase — the thesis explicitly authorizes unlimited storage at all stock points).
- **Evidence:** `supply_chain.py:252-260`.

### Finding C2: Containers, not Stores (bulk flow)
- **Thesis says:** MFSC handles continuous bulk quantities (ration counts, rm-unit counts) — not discrete identities.
- **Repo does:** All material buffers are `simpy.Container` (`supply_chain.py:253-260`). Orders are tracked as separate `OrderRecord` objects in `self.orders`, not as Store items.
- **Verdict:** MATCH (AGENTS.md rule "use Container, not Store" is satisfied).

### Finding C3: `_raw_units_per_ration` defaults to 1.0 (`legacy_validated`)
- **Thesis says:** BOM = 1 unit of each of the 12 raw materials per ration (`thesis.txt:51-89` of config comment; thesis Table 6.1 lists 12 rm/ration). So strict BOM factor = 12.
- **Repo does:** `_raw_units_per_ration = 1.0` unless `raw_material_flow_mode` starts with `"bom_total_units"` (`supply_chain.py:198-202`). Default mode is `"legacy_validated"` ⇒ factor 1.0.
- **Verdict:** EXTENSION (1:1 raw↔ration equivalence is a deliberately-loose legacy calibration that lets Op3's 15,500-rm weekly delivery match AL's ~2,564-ration weekly production without 6× pile-up; alternative BOM modes are available).
- **Impact:** Numerical effect: with 1:1, the `raw_material_al` buffer always has slack; with 12:1, AL can be raw-starved. The thesis's Q figures at Op3/4 are themselves inconsistent with the AL's λ-derived consumption rate (15,500 ≫ 2,564), so a strict BOM mapping has no clean thesis target.

### Finding C4: Strategic-buffer replenishment is opt-in
- **Thesis says:** Scenario II: "every t = 168, 336, 504, 672, or 1,344 hours, and the level of It,S is replenished in the quantities of raw material and rations indicated in Table 6.16" (`thesis.txt:5902-5903`).
- **Repo does:** `_inventory_buffer_replenishment` runs only if `self.inventory_replenishment_period` is truthy (`supply_chain.py:377-387`). The constructor takes this as `Optional[float] = None` (`supply_chain.py:139`). Caller must set it; otherwise buffers only get primed once at t=0 and then drain.
- **Verdict:** EXTENSION (matches thesis WHEN caller passes the period; otherwise Scenario II is not reproduced).

---

## Section D — ROP / Ordering

### Finding D1: ROPs are time-triggered cadences, not inventory-triggered thresholds
- **Thesis says:** ROPs throughout Fig 6.2 / Sec 6.3.3 are expressed as "every X hours" (e.g. Op1 ROP = "biannual or every 4,032 hrs", `thesis.txt:4118`). Also "re-order point (ROP) of the chain is of the type (ROP, Q), i.e., a fixed quantity … is placed whenever the inventory position in each operation falls below ROP" (`thesis.txt:4109-4111`). **The thesis uses "ROP" inconsistently** — text says inventory-triggered, tables/figure say time-triggered.
- **Repo does:** Time-triggered: every Op's ROP param is used as `yield self.env.timeout(rop)` (`supply_chain.py:1138, 1150, 1169, 1285, 1307, 1329`).
- **Verdict:** AMBIGUITY (the thesis textual definition of (ROP,Q) is the classic inventory-triggered policy; the thesis's own ROP values are pure time cadences. Repo follows the table/figure reading. If a true (s,S) inventory policy were intended, the repo is wrong; but the thesis's own parameters don't support that reading.)
- **Impact:** For Cf0 (deterministic, no demand-driven reorders), time-trigger ≡ inventory-trigger because consumption is regular. Diverges only under heavy disruption.

### Finding D2: Scenario II uses order-up-to (matches thesis)
- **Thesis says:** Scenario II buffers topped up to fixed target every t hours (`thesis.txt:5902-5903`).
- **Repo does:** `_top_up_inventory_buffer` fills `shortfall = target - level` (`supply_chain.py:367-369`).
- **Verdict:** MATCH (for Scenario II).

### Finding D3: Q semantics — "ship Q every cycle" (not "ship down to Q")
- **Thesis says:** Each Op's Q is the per-cycle shipment quantity.
- **Repo does:** Op2/3 ship `params[*_q] × NUM_RAW_MATERIALS` each cycle (`supply_chain.py:1159, 1172`). Op7/8 batch into 5,000-packs (`supply_chain.py:1255-1257`). Op9/10/12 dispatch `min(U(q_min,q_max), available)` (`supply_chain.py:1292-1294, 1314-1316, 1336-1338`).
- **Verdict:** MATCH (downstream ops correctly cap at available inventory — this is the `min(target, available)` pattern, not the "Q = ROP quantity" antipattern).

---

## Section E — Warm-up / Horizon

### Finding E1: Default `warmup_trigger="production"` — **wrong**
- **Thesis says:** Warm-up completes when "the first arrival of an order Q = 5,000 rations is verified" at Op9 (`thesis.txt:6379-6392`). 838.8h is the deterministic estimate ONLY.
- **Repo does:** Default `warmup_trigger="production"` (`supply_chain.py:134`), which fires when `total_produced >= batch_size` at the AL output (`supply_chain.py:1259-1263`) — i.e., **before** the 24h Op8 transport to Op9.
- **Verdict:** BUG (in default mode). Triggers warm-up ≈ 24h early; data-collection window includes the AL→SB transit transient that the thesis explicitly excludes.
- **Impact:** Bias on year-1 throughput; small for steady-state metrics but propagates into the "−29% RMSE" comparison.

### Finding E2: `run_static.py` does not pass `warmup_trigger` or `risk_occurrence_mode`
- **Thesis says:** (implicit) the entry point should reproduce thesis protocol.
- **Repo does:** `run_static.py:28-35, 64-71` instantiate `MFSCSimulation` with neither kwarg, so both default to non-thesis values (`warmup_trigger="production"`, `risk_occurrence_mode="legacy_renewal"`). Only `scripts/report_table_6_10_reproduction.py:41` sets `warmup_trigger="op9_arrival"`.
- **Verdict:** BUG (the documented entry point does not run the thesis protocol).
- **Impact:** Anyone running `python run_static.py` (per AGENTS.md and README) gets the divergent configuration.

### Finding E3: `env.py` uses FIXED 838.8h warm-up, ignoring event trigger
- **Thesis says:** Event-trigger at first Op9 arrival (`thesis.txt:6390-6391`).
- **Repo does:** `env.py:81` `self.warmup_hours = float(WARMUP["estimated_deterministic_hrs"])` = 838.8; `env.py:141` `self.sim.env.run(until=self.warmup_hours)`. Underlying `MFSCSimulation` is constructed without `warmup_trigger` ⇒ defaults to `"production"`, but `until=838.8` overrides any trigger by running for fixed wall-clock.
- **Verdict:** EXTENSION (RL env uses a fixed-warmup approximation; the natural event-trigger is bypassed). Acceptable for training stability, but diverges from thesis under stochastic risks where the true warm-up can exceed 838.8h.

### Finding E4: `env_experimental_shifts.py` adds a "priming" phase not in the thesis
- **Thesis says:** No priming concept exists; warm-up is the only initial-transient handling (`thesis.txt:6374-6393`).
- **Repo does:** After warm-up, `_prime_after_warmup` runs S=2 for up to 2016h (`env_experimental_shifts.py:959-974, 2119-2120`); `WARMUP["priming_shifts"] = 2`, `max_priming_hours = 2016` (`config.py:726-728`).
- **Verdict:** EXTENSION (declared RL-side crutch; `THESIS_FAITHFUL_PROTOCOL["priming_enabled"] = False` (`config.py:106`) but the experimental env defaults `priming_enabled=True` (`env_experimental_shifts.py:355`)).
- **Impact:** Adds ~336h of extra pre-collection production that inflates the data-collection window's starting inventory. The audit doc itself admits this introduces a "~336h offset" vs thesis (`THESIS_FIDELITY_AUDIT.md:127`).

### Finding E5: `env.py` `step()` returns 4-tuple, violates Gymnasium ≥0.26 API
- **Thesis says:** n/a (RL wrapper concern).
- **Repo does:** `env.py:172` `return out_obs, float(reward), bool(terminated), bool(truncated), out_info` — wait, re-reading, it does return 5 values. Type annotation at `env.py:146` also says 5-tuple. AGENTS.md rule satisfied.
- **Verdict:** MATCH (on re-inspection, the 5-tuple is present; my initial concern was wrong).

---

## Section F — Demand / Disruptions

### Finding F1: Regular demand U(2,400, 2,600) per day, 6 days/week
- **Thesis says:** U(X ∈ Z+, a=2,400, b=2,600) every 24h, 6 days/week (`thesis.txt:4242-4253, 4408-4409`).
- **Repo does:** `DEMAND` dict (`config.py:322-328`); `_op13_demand` samples `rng.integers(2400, 2601)` and skips Sunday via `day_of_week >= 6` (`supply_chain.py:1359-1365`).
- **Verdict:** MATCH

### Finding F2: Demand fulfillment from `rations_theatre`, not from Op9
- **Thesis says:** Troops receive at the theatre of operations (Op13); backorders accumulate when rations don't arrive within LTj=48h (`thesis.txt:5099-5112, 6380-6381`).
- **Repo does:** Demand pulls from `self.rations_theatre` (`supply_chain.py:1384-1389`); unfulfilled demand goes through `_enqueue_backorder` and `_delayed_backorder_check` after LTj=48h (`supply_chain.py:1391-1397, 697-708`).
- **Verdict:** MATCH

### Finding F3: SPT backorder scheduling + 60-cap + contingent priority
- **Thesis says:** Backorders scheduled by SPT (shortest processing time); queue cap = 60; R24 contingent orders have priority (`thesis.txt:5103-5112`).
- **Repo does:** `_backorder_priority_key` (line 641-656) sorts by quantity ascending (= SPT on size); `BACKORDER_QUEUE_CAP=60` enforced; contingent orders flagged via `OrderRecord.contingent`.
- **Verdict:** MATCH (priority mechanism for contingent orders needs spot-check, but SPT + cap are correct).

### Finding F4: R24 contingent demand code path
- **Thesis says:** Contingent orders tagged, given priority over regular (`thesis.txt:5110-5112`).
- **Repo does:** `_risk_R24` increments `_contingent_demand_pending`; `_op13_demand` adds it to the next demand and flags `OrderRecord.contingent=True` (`supply_chain.py:1662, 1369-1381`).
- **Verdict:** MATCH at the bookkeeping level; see B11 for the temporal-profile ambiguity.

### Finding F5: Op13 represented as a "demand sink" with PT=Q=ROP=0
- **Thesis says:** Op13 is "Military personnel in operations waiting for the delivery of combat rations" — a sink, not a processing step (`thesis.txt:4211-4222`).
- **Repo does:** `OPERATIONS[13]` has `pt=0, q=0, rop=0`; the demand generator `_op13_demand` consumes from `rations_theatre`.
- **Verdict:** MATCH

---

## Cross-cutting: the "−29% better RMSE than thesis" claim

- **The claim:** Repo RMSE 62,055 vs thesis 87,918 (THESIS_FIDELITY_AUDIT.md:18, 117).
- **What the thesis RMSE measures:** RMSE between historical Pt (8 yrs of real Colombian-army deliveries) and Garrido's MATLAB ECS_simulated (avg of 3 stochastic runs with current-level risks).
- **What the repo RMSE measures:** `report_table_6_10_reproduction.py` runs `deterministic_baseline=True, risks_enabled=False` (lines 36, 40) and computes RMSE between Pt and the deterministic year-by-year Python output.
- **Verdict:** UNVERIFIABLE as a like-for-like comparison. The two RMSEs use **different simulation conditions** (stochastic-with-risks vs deterministic-no-risks) and different time-window semantics (B1+E1+E2+E3 divergences). The arithmetic 87,918→62,055 ≡ −29.4% is correct, but the comparison is methodologically invalid: the repo's deterministic run by construction produces near-constant year-over-year output, so its RMSE-vs-Pt is essentially RMSE(constant, Pt) — a function of Pt's variance, not of model fidelity.
- **To make the claim verifiable:** re-run with `risk_occurrence_mode="thesis_periodic"`, `warmup_trigger="op9_arrival"`, `risks_enabled=True`, `risk_level="current"`, 3 seeds averaged, identical post-warm-up window. The repo has all the knobs but the claimed number did not come from that configuration.

---

## Top-3 highest-impact BUGs (ranked)

1. **B1 — `legacy_renewal` default makes all uniform risks ~2× too frequent.** `supply_chain.py:142` (constructor default) + `supply_chain.py:1483-1484` (sampling logic) + `run_static.py:28,64` (entry point uses default). Biases the stochastic baseline, all RL training, and every downstream "vs thesis" comparison.
2. **B3 — R13 period 4× too long in default mode.** `supply_chain.py:1494-1496` returns `op2_rop=672` (monthly) instead of `HOURS_PER_WEEK=168`. Gives 14 R13 events/yr vs thesis's 58. Inverts the bias direction of B1 (raw-material side under-stressed, assembly side over-stressed).
3. **E2 — `run_static.py` does not pass `warmup_trigger` or `risk_occurrence_mode`.** `run_static.py:28-35, 64-71`. The canonical entry point never runs the thesis protocol; only `scripts/report_table_6_10_reproduction.py` does. So the "Phase 1/Phase 2 PASSED" messages printed by the documented entry point are computed under non-thesis settings.

(Secondary BUGs: B9 R3 fires 2× per horizon in default mode; B12 default R14 defect routing is wrong; E1 default warmup_trigger is wrong; E4 priming is non-thesis.)

---

## What the repo got RIGHT (credit where due)

The 17 MATCHes above show that **the static structure is faithful**: all 13 operations, all 9 risks with their thesis parameters (a/b/n/p), all 5 inventory-buffer levels, all 3 shift configurations, the 60-backorder cap, λ=320.5, RATIONS_PER_BATCH=5000, 8064-h year, 161,280-h horizon, 6000-order cap, the entire Table 6.10 validation data, and the 838.8-h deterministic warm-up estimate are all correctly transcribed. The Container-not-Store and unbounded-WDC/SB/CSSU choices follow the thesis exactly.

The bugs are concentrated in **WHICH KNOB IS DEFAULT**, not in the underlying capability: every divergence has a thesis-faithful alternative that's already implemented (`risk_occurrence_mode="thesis_periodic"`, `warmup_trigger="op9_arrival"`, `r14_defect_mode="thesis_strict_op6"`, `priming_enabled=False`). The repo *can* reproduce the thesis — it just doesn't, by default.
