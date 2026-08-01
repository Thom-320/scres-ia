import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed_registry_is_fail_closed_before_submission_receipt():
    registry = json.loads((ROOT / "research/seed_custody_registry.json").read_text())

    assert registry["schema_version"] == "seed_custody_registry_v1"
    assert registry["scientific_execution_authorized"] is False
    assert registry["new_seed_opening"] is False
    assert registry["status"] == "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED"

    blocks = registry["blocks"]
    ranges = []
    for block in blocks:
        assert block["start"] <= block["end"]
        assert block["status"] in {
            "BURNED",
            "USED_DEVELOPMENT_NOT_VIRGIN",
            "ARTIFACT_PRESENT_PENDING_CANONICAL_CUSTODY",
            "ARTIFACT_PRESENT_PENDING_MERGE",
            "USED_PENDING_SOURCE_AUDIT",
            "RESERVED_NOT_OPENED",
        }
        ranges.append((block["start"], block["end"], block["id"]))

    for index, (start, end, block_id) in enumerate(ranges):
        for other_start, other_end, other_id in ranges[index + 1 :]:
            assert end < other_start or other_end < start, (block_id, other_id)

    g3a = next(block for block in blocks if block["id"] == "g3a_v2_development")
    assert g3a["status"] == "RESERVED_NOT_OPENED"
    assert g3a["start"] == 7_700_001
    assert g3a["end"] == 7_700_120
