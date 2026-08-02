"""G3c v2: one temporal mechanism and an executable full-payload null test.

G3c v2 contains minimum dwell only.  Switching cost remains implementation code for a
possible future, separately preregistered contract; it is not a G3c factor or a G3c test arm.

The null test compares two complete simulations on the same burned unit-test tape.  It hashes
the canonical scientific payload (orders, exogenous risk events, action traces, ledgers and
metrics), not the artifact envelope.  This is the executable resolution of blocker 3 rather than
an assertion that the default is inert.
"""
from __future__ import annotations

import copy
import warnings

import pytest

from supply_chain.arm_runner import canonical_payload_sha256
from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.g3c_temporal import (
    G3C_MIN_DWELL_LEVELS_DAYS,
    g3c_arm_grid,
    validate_min_dwell_days,
)
from supply_chain.scientific_payload import (
    canonical_scientific_payload,
    scientific_payload_sha256,
)
from supply_chain.supply_chain import MFSCSimulation

warnings.filterwarnings("ignore")

RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
SEED = 5_200_001          # burned block; these are unit tests, not an experiment
WEEKS = 8
STEP = 24.0


def build(**kw) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=SEED, horizon=float(WEEKS * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(RISKS),
        cssu_topology_mode="split_v1", cssu_allocation_a=0.5,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"], **kw)


def run(**kw) -> dict:
    """Alternate the split every day so minimum dwell has something to constrain."""
    sim = build(**kw)
    alphas, done, i = [], False, 0
    while not done:
        target = 0.9 if (i // 2) % 2 == 0 else 0.1
        action = ({"cssu_allocation_a": target}
                  if sim._pending_cssu_action is None
                  and abs(sim.cssu_allocation_a - target) > 1e-9 else None)
        _, _, done, _ = sim.step(action=action, step_hours=STEP)
        alphas.append(float(sim.cssu_allocation_a))
        i += 1

    panel = compute_episode_metrics(sim)
    payload = canonical_scientific_payload(sim, panel)
    return {
        "panel": panel,
        "scientific_payload": payload,
        "scientific_payload_sha256": scientific_payload_sha256(payload),
        "alphas": alphas,
        "cssu_delivered": dict(sim.cssu_delivered),
        "cssu_demanded": dict(sim.cssu_demanded),
        "switches": sim.cssu_switch_count,
        "blocked": sim.cssu_blocked_by_dwell_count,
        "cost_paid": sim.cssu_switch_cost_paid,
        "cost_unpaid": sim.cssu_switch_cost_unpaid,
    }


@pytest.fixture(scope="module")
def default_run() -> dict:
    return run()


def test_null_arm_is_identical_to_the_shipped_defaults(default_run):
    """The blocker-3 check: explicit dwell=1 equals the shipped science payload."""
    explicit = run(cssu_min_dwell_days=1)
    assert explicit["scientific_payload_sha256"] == default_run["scientific_payload_sha256"]
    assert explicit["scientific_payload"] == default_run["scientific_payload"]


def test_dwell_actually_binds_when_switched_on(default_run):
    """A knob that changes nothing when enabled would make the factorial vacuous."""
    coupled = run(cssu_min_dwell_days=7)
    assert coupled["blocked"] > 0
    assert coupled["switches"] < default_run["switches"]


def test_g3c_has_one_mechanism_and_no_switch_cost_factor():
    """Blocker 1: the frozen G3c grid contains minimum dwell only."""
    assert G3C_MIN_DWELL_LEVELS_DAYS == (1, 3, 7)
    arms = g3c_arm_grid()
    assert [arm["min_dwell_days"] for arm in arms] == [1, 3, 7]
    assert {arm["mechanism"] for arm in arms} == {"min_dwell"}
    assert all("switch_cost_rations" not in arm for arm in arms)


def test_a_held_request_is_not_a_lost_request():
    """Dwell delays a switch; it does not cancel the pending decision."""
    coupled = run(cssu_min_dwell_days=7)
    assert coupled["switches"] > 0, "if dwell cancelled every request there would be no switches"


@pytest.mark.parametrize("bad", [0.5, 2, 4, 7.5])
def test_unregistered_min_dwell_levels_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_min_dwell_days(bad)
    assert validate_min_dwell_days(1) == 1


def test_canonical_hash_ignores_provenance_but_not_science():
    a = {"metric": 1.0, "created_at": "x", "self_sha256": "aa", "module_manifest": {"m": 1}}
    b = {"metric": 1.0, "created_at": "y", "self_sha256": "bb", "module_manifest": {"m": 2}}
    assert canonical_payload_sha256(a) == canonical_payload_sha256(b)
    assert canonical_payload_sha256(a) != canonical_payload_sha256({"metric": 2.0})


def test_scientific_payload_changes_when_a_physical_order_changes(default_run):
    changed = copy.deepcopy(default_run["scientific_payload"])
    assert changed["orders"]
    changed["orders"][0]["quantity"] = float(changed["orders"][0]["quantity"]) + 1.0
    assert scientific_payload_sha256(changed) != default_run["scientific_payload_sha256"]


# Frozen on 2026-08-02 from the verified-correct coupling code, on the burned unit-test tape
# (SEED 5_200_001, 8 weeks, alternating 0.9/0.1 requests, dwell=1 / cost=0).
GOLDEN_NULL_PAYLOAD_SHA256 = (
    "be9d1bc227d498cb093f654014b791066ea945ad5c71cfc7cf74b2d9a4df9c37")


def test_null_arm_matches_the_frozen_golden_hash(default_run):
    """The null test that can ACTUALLY fail on a code defect.

    `test_null_arm_is_identical_to_the_shipped_defaults` compares `run()` with
    `run(cssu_min_dwell_days=1)`. Both execute the SAME coupling code, so a defect inside that
    code breaks both equally and they stay identical to each other -- the check compares
    something with itself. Verified by injecting an off-by-one into the dwell computation
    (`- 1.0` dropped, so dwell=1 would block): all ten tests still passed.

    Anchoring to a hash frozen from known-good code removes the self-reference: any change to
    realized orders, risk events, action traces, ledgers or metrics moves this value.
    """
    assert default_run["scientific_payload_sha256"] == GOLDEN_NULL_PAYLOAD_SHA256, (
        "the null arm no longer reproduces the frozen science. Either a real defect was "
        "introduced, or the physics changed deliberately -- in which case re-freeze the golden "
        "hash in a commit that says why, never silently.")
