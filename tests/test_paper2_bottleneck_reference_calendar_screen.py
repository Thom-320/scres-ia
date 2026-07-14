import json
from hashlib import sha256

import numpy as np
import pytest

from scripts.search_paper2_bottleneck_reference_calendar import (
    CONTRACT_PATH,
    _bootstrap,
    _evaluate_locked_tape,
    candidate_calendars,
)
from scripts.watch_paper2_reference_screen import snapshot
from scripts.verify_paper2_bottleneck_reference_calendar import (
    VERIFICATION_SCHEMA,
)
from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
    calendar_index,
)
from supply_chain.paper2_bottleneck import CONTEXTS


def test_frozen_single_excursion_family_is_complete_and_feasible():
    candidates = candidate_calendars()
    contract = json.loads(CONTRACT_PATH.read_text())

    assert len(candidates) == contract["candidate_family"]["candidate_count"] == 507
    assert len(set(candidates)) == 507
    assert candidates[0] == (0,) * 24
    assert all(calendar_index(sequence) >= 0 for sequence in candidates)
    assert all(
        not (
            sequence[index] != sequence[index - 1]
            and sequence[index - 1] != sequence[index - 2]
        )
        for sequence in candidates
        for index in range(2, len(sequence))
    )
    for sequence in candidates[1:]:
        non_m = [index for index, action in enumerate(sequence) if action != 0]
        assert len(non_m) >= 2
        assert non_m == list(range(min(non_m), max(non_m) + 1))
        assert len({sequence[index] for index in non_m}) == 1


def test_locked_context_rotation_preserves_excluded_seed_position():
    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    rows = _contract_seed_rows(primary, "locked_bound")
    assert len(rows) == 119
    assert rows[0] == {
        "seed": 1_110_002,
        "context_0": CONTEXTS[1],
        "split": "locked",
    }
    assert rows[-1] == {
        "seed": 1_110_120,
        "context_0": CONTEXTS[2],
        "split": "locked",
    }


def test_bootstrap_is_frozen_and_deterministic():
    values = np.linspace(0.0, 0.03, 119)
    first = _bootstrap(values)
    second = _bootstrap(values.copy())
    assert first == second
    assert first["mean"] == float(values.mean())
    assert first["bootstrap_resamples"] == 10_000
    assert first["bootstrap_seed"] == 20260713
    with pytest.raises(ValueError, match="finite and non-negative"):
        _bootstrap(np.linspace(-0.01, 0.03, 119))


def test_locked_tape_evaluation_is_paired_and_resource_relaxed():
    row = _evaluate_locked_tape(
        0,
        1_110_002,
        CONTEXTS[1],
        (0,) * 24,
    )
    assert row["seed"] == 1_110_002
    assert len(row["exogenous_hashes"]) == 2
    assert row["oracle_minus_reference_lower_bound"] >= 0.0
    assert {
        policy["total_token_hours"]
        for policy in row["policies"].values()
    } == {4032.0}
    assert all(
        "reserve_units_issued" in policy
        and "reserve_units_replenished" in policy
        and "reserve_replenishment_requests" in policy
        and "reserve_inventory_terminal" in policy
        for policy in row["policies"].values()
    )


def test_independent_watcher_distinguishes_prestart_complete_and_failure(tmp_path):
    prestart = snapshot(tmp_path, watcher_started="2026-07-13T00:00:00+00:00")
    assert prestart["state"] == "watching_prestart"

    result = tmp_path / "result.json"
    result.write_text('{"result":true}\n')
    result_sha = sha256(result.read_bytes()).hexdigest()
    (tmp_path / "progress.json").write_text(json.dumps({
        "stage": "complete",
        "output_sha256": result_sha,
    }))
    (tmp_path / "pid.json").write_text(json.dumps({"scientific_pid": 999_999_999}))
    complete = snapshot(tmp_path, watcher_started="2026-07-13T00:00:00+00:00")
    assert complete["state"] == "completed_unverified"

    (tmp_path / "progress.json").write_text(json.dumps({
        "stage": "complete",
        "output_sha256": "0" * 64,
    }))
    failed = snapshot(tmp_path, watcher_started="2026-07-13T00:00:00+00:00")
    assert failed["state"] == "failed_or_incomplete"


def test_verification_schema_is_distinct_from_scientific_result():
    assert VERIFICATION_SCHEMA == "paper2_bottleneck_reference_calendar_verification_v1"
