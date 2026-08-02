"""The custody check must catch a real re-use, and must NOT dress a replay up as virginity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain import seed_custody as sc


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "status": "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED",
        "blocks": [
            {"id": "burned_block", "start": 1000, "end": 1099, "status": "BURNED"},
            {"id": "h3_vps", "start": 2000, "end": 2029, "status": "USED_PENDING_SOURCE_AUDIT"},
            {"id": "future", "start": 3000, "end": 3099, "status": "RESERVED_NOT_OPENED"},
        ],
    }))
    return path


@pytest.fixture()
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "run_a"
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"seeds": [5001, 5002]}))
    return tmp_path / "results"


def _check(seeds, registry, results, **kw):
    return sc.check_seeds(seeds, registry_path=registry, results_root=results, **kw)


def test_clean_seeds_are_only_no_known_collision_never_virgin(registry, results):
    out = _check([9001, 9002], registry, results)
    assert out["status"] == sc.NO_KNOWN_COLLISION
    # The registry declares itself incomplete, so the API must never promise virginity.
    assert out["registry_is_complete"] is False
    assert "NOT proof of virginity" in out["note"]


def test_registry_collision_is_caught(registry, results):
    out = _check([1050], registry, results)
    assert out["status"] == sc.COLLISION
    assert out["registry_conflicts"][0]["id"] == "burned_block"


def test_reserved_but_unopened_block_is_not_a_collision(registry, results):
    """A reserved range has not been consumed; treating it as used would block its own contract."""
    assert _check([3050], registry, results)["status"] == sc.NO_KNOWN_COLLISION


def test_sealed_artifact_seeds_are_caught_even_when_registry_is_silent(registry, results):
    """The registry is incomplete by its own admission, so the artifact scan must still run."""
    out = _check([5001], registry, results)
    assert out["status"] == sc.COLLISION
    assert out["sealed_artifact_overlap"] == [5001]
    assert out["registry_conflicts"] == []


def test_declared_replay_is_not_applicable_not_passed(registry, results):
    """The H3-prime case: re-running a used block on purpose is neither a pass nor a failure."""
    f = sc.custody_falsifier([2000, 2015], replay_of="h3_vps",
                             registry_path=registry, results_root=results)
    assert f["not_applicable"] is True
    assert f["passed"] is None


def test_replay_declaration_does_not_excuse_an_undeclared_block(registry, results):
    """Naming one block must not launder a collision with a different one."""
    out = _check([2000, 1050], registry, results, replay_of="h3_vps")
    assert out["status"] == sc.COLLISION


def test_replay_declaration_that_does_not_match_is_reported(registry, results):
    """Claiming a replay of a block these seeds do not touch must not silently grant immunity."""
    out = _check([9001], registry, results, replay_of="h3_vps")
    assert out["status"] == sc.NO_KNOWN_COLLISION
    assert "does not apply" in out["note"]


def test_scanner_sees_development_and_test_seed_keys(tmp_path: Path, registry):
    """G3-obs splits its block under these keys; the six copied scanners never looked at them."""
    root = tmp_path / "r"
    (root / "x").mkdir(parents=True)
    (root / "x" / "result.json").write_text(
        json.dumps({"development_seeds": [7001], "test_seeds": [7002]}))
    assert sc.seeds_used_by_sealed_artifacts(root) == {7001, 7002}


def test_exclude_lets_a_runner_overwrite_its_own_artifact(tmp_path: Path):
    root = tmp_path / "r"
    (root / "x").mkdir(parents=True)
    own = root / "x" / "result.json"
    own.write_text(json.dumps({"seeds": [8001]}))
    assert sc.seeds_used_by_sealed_artifacts(root) == {8001}
    assert sc.seeds_used_by_sealed_artifacts(root, exclude=own) == set()


def test_real_registry_marks_the_h3_vps_block_as_used(tmp_path: Path):
    """Against the repo's actual registry: the block the H3-prime audit replays is NOT virgin."""
    out = sc.check_seeds([6_000_091, 6_000_120], results_root=tmp_path)
    assert out["status"] == sc.COLLISION
    assert any(c["id"] == "garrido_h3_vps" for c in out["registry_conflicts"])
