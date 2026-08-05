"""OFAT must re-run its INCUMBENT once the design is exhausted, not its last proposal.

The bug: the exhausted branch guarded on `"idx" not in locals()`, but `del idx` fires once per
CONTEXT rather than once per step. From step 1 onwards `idx` was already bound, so the guard was
False and the arm silently repeated whatever it had proposed last -- inside the comparator the
headline contrast is measured against.

The test drives the real `search()` with a budget deliberately longer than the design (17 OFAT
proposals for this factor set) and asserts the tail. It fails on the pre-fix code, which is the
only reason it is worth having.
"""
from __future__ import annotations

import importlib.util as iu
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = iu.spec_from_file_location("q2", ROOT / "scripts" / "run_garrido_q2_des288_v1.py")
q2 = iu.module_from_spec(SPEC)
SPEC.loader.exec_module(q2)

CONTEXT = "R1r"
SEED = 5_300_001
#: Every factor level is one proposal, and the design has as many as the levels sum to.
DESIGN_LENGTH = sum(len(levels) for levels in q2.FACTORS.values())


#: Distinct weights so no two configurations tie, and the LAST factor swept prefers its FIRST
#: level. Both properties are load-bearing: ties make "the incumbent" ambiguous, and a surface
#: that is monotone increasing in every factor would make the final proposal coincide with the
#: incumbent by accident -- the test would then pass on the buggy code and prove nothing.
WEIGHTS = {"buffer_hours": 1.0, "shifts": 0.37, "op9_rop": 0.11, "op12_rop": -0.29}


def _score_scalar(cfg) -> float:
    return sum(w * q2.FACTORS[name].index(cfg[name]) / (len(q2.FACTORS[name]) - 1)
               for name, w in WEIGHTS.items())


def synthetic_surface(configs):
    """A deterministic surface with a unique optimum, so the incumbent is unambiguous."""
    rng = np.random.default_rng(11)
    rows = []
    for cfg in configs:
        score = _score_scalar(cfg)
        rows.append({
            "config": dict(cfg), "context": CONTEXT, "seed": SEED,
            "service_key": [score, score, -score, score],
            "claimant_fills": {}, "demanded_by_claimant": {}, "delivered_by_claimant": {},
            "cssu_total_demanded": 1.0, "cssu_total_delivered": 1.0,
            "drivers": list(rng.random(4)),
            "panel": {k: 0.0 for k in (
                "n_orders", "n_served", "n_lost", "flow_fill_rate", "fill_rate",
                "backorder_qty_final", "service_loss_auc_ration_hours",
                "ret_excel_visible_clipped_0_1", "ret_excel", "delivered_rations",
                "demanded_rations")},
        })
    return {(CONTEXT, SEED): rows}


@pytest.fixture(scope="module")
def exhausted_run():
    configs = tuple(q2.CONFIGS)
    surface = synthetic_surface(configs)
    budget = DESIGN_LENGTH + 6                      # six steps past the end of the design
    return q2.search(
        strategy="ofat", surface=surface, configs=configs, contexts=(CONTEXT,),
        seed=SEED, budget=budget, rng=np.random.default_rng(0),
    ), configs, budget


def test_the_tail_repeats_a_single_configuration(exhausted_run):
    """Once exhausted, OFAT has nothing new to propose, so every remaining step is the same run."""
    run, _configs, budget = exhausted_run
    visited = run["per_context"][CONTEXT]["visited_sequence"]
    assert len(visited) == budget
    tail = visited[DESIGN_LENGTH:]
    assert len(set(tail)) == 1, f"the exhausted tail wandered across {len(set(tail))} configs"


def test_the_repeated_configuration_is_the_incumbent_not_the_last_proposal(exhausted_run):
    """THE regression. The pre-fix code repeated the final proposal, which is only the incumbent
    by coincidence when the last level swept happens to be the best one."""
    run, configs, _budget = exhausted_run
    visited = run["per_context"][CONTEXT]["visited_sequence"]
    tail_config = configs[visited[DESIGN_LENGTH]]
    incumbent = configs[max(visited[:DESIGN_LENGTH], key=lambda i: _score_scalar(configs[i]))]
    last_proposal = configs[visited[DESIGN_LENGTH - 1]]

    # Without this the test would be vacuous: on a surface where the final sweep ends on its best
    # level, the buggy code repeats the last proposal and that IS the incumbent.
    assert last_proposal != incumbent, "the fixture no longer separates the two behaviours"

    assert tail_config == incumbent, (
        f"exhausted OFAT re-ran {tail_config}; its incumbent was {incumbent} and its last "
        f"proposal was {last_proposal} -- repeating the proposal is the defect")


def test_every_proposal_moves_exactly_one_coordinate(exhausted_run):
    """The property that makes this arm OFAT at all, and the one an earlier defect broke."""
    run, _configs, _budget = exhausted_run
    changes = run["ofat_coordinate_changes"]
    assert changes, "no proposals were recorded"
    assert set(changes) <= {0, 1}, f"a proposal moved {max(changes)} coordinates at once"
