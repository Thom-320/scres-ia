# Explore — In <HOME>/Projects/research/scres-ia, explore VERY THOROUGHLY the CSSU (shared scarce

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a9006eeeb987f3b5a`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-01T21:25:45.478Z`
- **effort**: `low` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-a9006eeeb987f3b5a.jsonl`
- **turns**: 49 · **assistant messages**: 2

## Task given

```text
In <HOME>/Projects/research/scres-ia, explore VERY THOROUGHLY the CSSU (shared scarce resource) machinery, because I need to know exactly what it would take to go from TWO claimants to THREE asymmetric claimants.

Focus on:
1. supply_chain/cssu_allocation.py and supply_chain/supply_chain.py — how is the two-product share of Op5-Op7 implemented? Is the allocation a scalar in [0,1] (`validate_allocation_a`) or a vector? What exactly is hard-coded to two products? Show me the concrete symbols/functions/fields (product names, allocation param, `reallocate_unused` fungibility flag, forfeited counters, per-claimant fill accounting).
2. Where do per-claimant demand/risk/fill get computed and recorded — is there a `claimant_fill` dict anywhere, and what produces it? (see supply_chain/service_first_metric.py `service_first_key_v2` which takes `claimant_fill`).
3. What is the exact generalisation cost to N claimants? List the specific call sites that assume 2.
4. Any existing scripts under scripts/ that sweep CSSU allocation (e.g. contention headroom runs), and how they parameterise the split. Names like run_*contention*, run_*cssu*.
5. Whether switching/setup/dwell costs exist anywhere in the model already (search for setup, changeover, switch cost, dwell, min run length).

Report: concrete file:line references, the exact shape of the allocation interface, and an honest estimate of what's already generic vs genuinely hard-coded to 2. Do not write any files.
```

## Final answer

I have what I need.

## Summary

**Important correction to the premise:** the CSSU shared scarce resource is *not* a share of Op5–Op7 production. It is the **daily Op9→Op10 dispatch pool** (`min(rations_sb.level, _cssu_daily_capacity())`) split across two downstream *destinations* (CSSU A and CSSU B). Op5–Op7 is the aggregate assembly line and knows nothing about A/B. The "two claimants" are two consumption sites competing for a conserved daily convoy capacity.

---

### 1. The allocation interface — scalar, not a vector

`<HOME>/Projects/research/scres-ia/supply_chain/cssu_allocation.py`

- `CSSU_IDS = ("A", "B")` — line 14. The single source of truth for the claimant set, a 2-tuple.
- `ALLOCATION_LEVELS = (0.25, 0.50, 0.75)` — line 18, default static grid only.
- `SERVICE_RULES = ("SPT_FULL", "FIFO_PARTIAL", "R24_AGE_PARTIAL")` — line 19.
- `validate_allocation_a(value) -> float` — lines 22-27. **Scalar in [0,1]**, one number, interpreted as A's share; B gets the residual. There is no simplex validator anywhere in the repo.
- `AllocationResult` — lines 30-39: fields are literally `dispatched_a`, `dispatched_b`, `unused`, `available`. Named fields, not a dict/vector.
- `stable_cssu_destination(...)` — lines 42-45: `CSSU_IDS[digest[0] & 1]` — a **single bit**, hard 50/50, RNG-free. This is where "asymmetric claimants" is most structurally blocked: demand mass is split by a fair coin, and there is no weight parameter.
- `allocate_shared_capacity(...)` — lines 48-94. `cap_a = available * allocation_a; cap_b = available - cap_a` (79-80); `sent = {"A": ..., "B": ...}` (81). The `reallocate_unused` fungibility spare loop (83-89) is the one genuinely N-generic piece: it iterates `sorted(CSSU_IDS, key=lambda x: (-(demand[x]-sent[x]), x))` — largest-unmet-first, works for any N unchanged.
- `reallocate_unused: bool = True` — the FUNGIBILITY knob, documented at lines 63-69 as *the* mechanism knob (H_PI = 0.1515 non-fungible vs exactly 0 fungible).

### 2. Simulator state and per-claimant accounting

`<HOME>/Projects/research/scres-ia/supply_chain/supply_chain.py`

Constructor params (lines 278-286, validated 418-426, stored 488-495):
`cssu_topology_mode` ∈ {`"aggregate"`, `"split_v1"`}, `cssu_allocation_a: float = 0.50`, `cssu_service_rule`, `cssu_daily_capacity`, `cssu_reallocate_unused: bool = True`.

Per-claimant dicts, all literal `{"A": 0.0, "B": 0.0}` (lines 919-926):
`cssu_in_transit`, `cssu_inbound_in_transit`, `cssu_inventory`, `cssu_outbound_in_transit`, `cssu_delivered`, `cssu_demanded`, `cssu_dispatched`.

Counters: `cssu_allocation_live_epochs` / `cssu_allocation_moot_epochs` (926-927), `cssu_forfeited_epochs` / `cssu_forfeited_rations` (931-932), `cssu_local_down_count` keyed `(op_id, cssu)` over `("A","B")` (933-937).

Dispatch core `_dispatch_split_cssu_day` — **lines 4900-5033**. Two branches:
- `SPT_FULL` (4939-4975): indivisible whole-order lane. Share implemented as a *frequency over days* — `preferred = "A" if draw < allocation_a else "B"` (4952-4955). This is **binary by construction**; generalising needs an inverse-CDF over a share vector. `budgets = {"A": 0.0, "B": 0.0}` at 4965 and 4975. Forfeit accounting at 4971-4973.
- Partial rules (4976-5000): calls `allocate_shared_capacity`, then `nominal_a`/`nominal_b` (4986-4987) and the awkward `unmet` comprehension at **4992-4995** which branches on `c == "A"` to pick `dispatched_a` vs `dispatched_b` — pure 2-ary. `budgets = {"A": allocation.dispatched_a, "B": allocation.dispatched_b}` (5000).
- Drain loop `for cssu in ("A","B")` at 5007.

Liveness predicate `cssu_allocation_is_live()` — **4866-4898**. `both_feasible = all(... for cssu in ("A","B"))` (4882-4886); `return bool(both_feasible and 0.25 <= draw < 0.75)` (4890) — **the 0.25/0.75 literals are the old three-point grid baked in**; and the partial branch (4895-4898) tests `requested["A"] > available*α` **and** `requested["B"] > available*(1-α)` — a 2-ary conjunction that must become "≥2 claimants simultaneously exhaust their nominal shares".

Dynamic action path: `set_cssu_allocation_action(allocation_a: float, service_rule, activation_delay_hours=24.0)` (4783-4815), `_activate_due_cssu_action` (4817-4826), routed from `step()` at 2075-2088 via pseudo-action keys `{"cssu_allocation_a","cssu_service_rule"}` (`_PSEUDO_ACTION_KEYS`, lines 1400-1401). Not a gym action space — `env.py` / `arm_runner.py` contain no CSSU references at all.

Other 2-hardcodings: `OrderRecord.cssu_destination` (133) and `RiskEvent.affected_cssu` (177), validated against `{"A","B"}` at 1652, 1753, 3568, 5050, 5231; risk targeting `rng.choice(("A","B"))` at 5930, 5984, 6030; panel export names 8 explicit `cssu_A_*`/`cssu_B_*` keys at 2254-2265; observation loop `for cssu in ("A","B")` at 3074 emitting 13 per-claimant features each (3102-3126). Note `get_observation()` (3132-3170) is the 15-dim RL vector and contains **zero** CSSU features — it uses aggregate `rations_cssu` only.

### 3. `claimant_fill` — where it comes from

`<HOME>/Projects/research/scres-ia/supply_chain/service_first_metric.py:91-100` — `claimant_fills(sim)` reads `sim.cssu_demanded` / `sim.cssu_delivered` and returns `{name: delivered/demanded}`. **This function is already fully N-generic** (it iterates the mapping, no A/B literals). Returns `{}` when there is no partition.

`service_first_key_v2(panel, claimant_fill)` — 103-112, `worst = min(fills.values())`, also N-generic. Consumers: `scripts/run_contention_service_first_v2.py:96,102` (`n_claimants`), `scripts/run_service_first_v2_audit.py:64,74`, `scripts/run_garrido_q2_des288_v1.py:152-169`.

The only 2-assumption in the metric layer is a *test/gate assertion*: `scripts/run_contention_service_first_v2.py:204` — `"passed": claimants == {2}`.

### 4. Sweep scripts

- `scripts/run_contention_headroom_v1.py` — `SHARES = tuple(round(0.1+0.1*i,2) for i in range(9))` (line 48), constructs `cssu_topology_mode="split_v1", cssu_allocation_a=float(share)` at 77, crosses share × regime × fungibility, reads `cssu_allocation_live_epochs` at 88. Grid over a **1-D scalar**.
- `scripts/run_contention_service_first_v2.py` — same 9-point SHARES (47), `claimant_fills` at 96, argmax over shares (133-136).
- `scripts/run_cssu_liveness_scope.py` — `SHARES = (0.1,0.3,0.5,0.7,0.9)` (38), maps out where the lever is live at all.
- `scripts/run_cssu_liveness_gate.py` — actuator/latency gate.
- `scripts/run_dra1_static_frontier.py` — full static frontier over `ALLOCATION_LEVELS × SERVICE_RULES` (213-215, 234-236).

Every one of these iterates a scalar `share`; with a simplex they become nested/simplex-mesh sweeps (9 points → ~45 for a 3-simplex at 0.1 resolution).

### 5. Switching / setup / dwell costs

**None exist in the real DES.** `grep` for `changeover|setup_cost|setup_time|min_run_length|switch_cost|switchover` returns nothing in `supply_chain/supply_chain.py`. What exists:

- `research/paper2_exhaustive_search/toy_F8_multi_product.py` — a **standalone toy**, explicitly "NOT the real MFSC" (docstring lines 1-15). Two products, one shared line, switching costs `S ∈ {0,1,2}` idle periods (lines 187-216), DP that encodes changeover cost into the transition (447). This is the closest existing prior art for setup commitment and it was written as a pre-implementation bound check.
- `supply_chain/program_m_shared_lift.py` — irrevocable weekly reservation commitment (`DESTINATIONS = ("A","B")`, `ACTION_NAMES = ("RESERVE_A","RESERVE_B")`, lines 29-30; `PAYLOAD_CAPACITY_RATIONS`, `COMMITMENT_VEHICLE_HOURS` at 33-34). This is a *commitment*, not a setup cost — reservations cost the same whether loaded or empty, so it is a lumpy allocation rather than a changeover penalty. Also hard 2-ary (`warning_A`/`warning_B` at 128-129, 434-435, 499-500).
- `dwell` appears only as **regime-persistence dwell times** in the risk/context generators (`program_f.py:95-114`, `program_g.py:60-66`, `program_h_belief.py:17-25`, `config.py:714-716`) — unrelated to the resource lever.
- The 24 h `activation_delay_hours` in `set_cssu_allocation_action` (4788) is the only friction on changing the split, and it is free — no cost, no minimum dwell.

---

## Honest assessment: what it takes to go from 2 → 3 asymmetric claimants

**Already generic (no change needed):**
- `allocate_shared_capacity`'s spare-reallocation loop (cssu_allocation.py:83-89) — sorted largest-unmet-first over `CSSU_IDS`.
- `claimant_fills` and the whole `service_first_key_v2` family (service_first_metric.py:91-131).
- The demand/delivery event tapes `cssu_demand_events` / `cssu_delivery_events` (508-509) — `(time, destination, qty)` triples, destination-agnostic.
- `get_cssu_observation`'s per-claimant loop body — it already builds keys with f-strings (3102-3126); only the iterable is fixed.
- `_take_down_cssu` / `_bring_up_cssu` / `cssu_local_down_count` — keyed by `(op, name)`, generic apart from the literal construction at 936 and the `{"A","B"}` membership guard at 3568.

**Genuinely hard-coded to 2 — the real cost:**

1. **The allocation is a scalar, everywhere.** `validate_allocation_a` returns a float; `cssu_allocation_a` is a float attribute; `AllocationResult` has `dispatched_a`/`dispatched_b` fields; the `step()` action key is a scalar; `get_cssu_observation["allocation_a"]` is one number. This needs a parallel `allocation_shares: Mapping[str,float]` interface with a simplex validator (sums to 1, all ≥0) and `AllocationResult.dispatched: dict[str,float]`. Keeping `validate_allocation_a` as a 2-claimant shim is cheap and preserves every existing artifact bitwise.

2. **`stable_cssu_destination` is a single bit** (`digest[0] & 1`). Three *asymmetric* claimants need a weighted inverse-CDF over a uniform derived from the digest — and asymmetric demand mass is a **new parameter that does not exist today** (right now the split is exactly 50/50 by construction). This is the single biggest conceptual addition: "asymmetric" is currently unrepresentable, independent of N.

3. **`cssu_allocation_is_live()` (4866-4898)** — both the `0.25 <= draw < 0.75` literal and the 2-way `and` need replacing with a general "at least two claimants simultaneously bind" test. Getting this wrong silently corrupts every `live_epochs` / `moot_epochs` number the liveness scripts publish.

4. **`SPT_FULL` preferred-destination rule (4952-4955)** — `draw < α ? A : B`. Must become a cumulative-share lookup plus a fallback *ordering* over the remaining claimants (today `candidates[0]` at 4962 is trivially "the other one"; with 3 you must decide the tie-break policy, and it is a real modelling choice, not a refactor).

5. **The forfeit accounting comprehension at 4992-4995** — branches on `c == "A"`; collapses to a clean `sum(max(0, requested[c] - dispatched[c]))` once `AllocationResult` carries a dict.

6. **Panel/observation schema (2254-2265, 3102-3126)** — 8 + 26 hard-named keys. Any downstream consumer that reads `cssu_A_inventory` etc. by name (contracts JSON, tests, notebooks) will need a schema-version bump. This is the most tedious part: `contracts/` and `research/paper2_exhaustive_search/*.json` freeze these names.

7. **Risk targeting** (5930, 5984, 6030 `rng.choice(("A","B"))`) and `affected_cssu` validation (1652, 1753, 3568, 5050, 5231) — mechanical, but `rng.choice` over 3 elements **changes the RNG consumption pattern and breaks CRN with every existing artifact**. You almost certainly want the 3-claimant mode behind a new `cssu_topology_mode="split_v2"` so `split_v1` stays bitwise reproducible (there is a test for exactly this discipline: `tests/test_cssu_allocation_physics.py:63` "aggregate mode is bitwise identical to default").

8. **Sweep scripts** — the scalar `SHARES` grids become simplex meshes; `scripts/run_contention_service_first_v2.py:204`'s `claimants == {2}` gate must be relaxed.

**Rough shape of the work:** the *physics* is a day or two — `cssu_allocation.py` is 94 lines and its core loop already generalises. The expensive parts are (a) inventing the asymmetry parameterisation, which has no existing home; (b) the `SPT_FULL` frequency rule and the liveness predicate, which are genuine modelling decisions rather than mechanical edits; and (c) the artifact/schema/CRN custody discipline this repo clearly enforces — a `split_v2` mode plus regenerated contracts, rather than mutating `split_v1` in place.

**Adjacent note:** if the goal is asymmetric claimants *with* switching costs, `toy_F8_multi_product.py` is the existing bound-check pattern for setup commitment and it is already asymmetric-capable (product-specific regimes), but it is a toy at `research/paper2_exhaustive_search/` and shares no code with the DES.
