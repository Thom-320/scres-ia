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
            # Opened but produced no sealed artifact. Distinct from the others on
            # purpose: the block is NOT virgin, and there is nothing to audit either.
            "ATTEMPTED_NO_SEALED_ARTIFACT",
            # Consumed by a prospective confirmation that ran to completion. Strictly
            # MORE specific than BURNED -- it says both that the seeds are spent and
            # that the confirmation they were opened for finished -- so admitting it
            # permits nothing BURNED would not. Introduced by ed16b9e for the
            # grid-transfer confirmation and never added here, which is why this
            # guardrail sat red instead of guarding.
            "BURNED_CONFIRMATION_COMPLETE",
            # Opened in violation of its own preregistration's exit rule, by explicit PI decision
            # recorded in the block's `source`. Admitted here so the registry can SAY what
            # happened -- a state the registry cannot express is worse than one it can -- but its
            # presence is a red flag and never a routine state. Blocks in it are spent and carry
            # no prospective-confirmation value.
            "BURNED_OPENED_AGAINST_PREREGISTRATION",
            # A confirmation is running on this block right now. Admitted so the registry can say
            # so between opening and sealing; a block cannot be virgin and in flight at once.
            "OPEN_CONFIRMATION_IN_PROGRESS",
            "RESERVED_NOT_OPENED",
        }
        ranges.append((block["start"], block["end"], block["id"]))

    for index, (start, end, block_id) in enumerate(ranges):
        for other_start, other_end, other_id in ranges[index + 1 :]:
            assert end < other_start or other_end < start, (block_id, other_id)

    g3a = next(block for block in blocks if block["id"] == "g3a_v2_development")
    assert g3a["start"] == 7_700_001
    assert g3a["end"] == 7_700_120

    # This block is the last virgin one in the project and its opening is gated by
    # `submission_a_receipt_required_before_g3a_open`. It may leave RESERVED_NOT_OPENED ONLY with a
    # recorded PI exception that names this block and the rule it lifts, plus the contract the
    # block was opened under. An undocumented opening still fails here, which is the property this
    # guardrail exists for -- weakening it to "any status is fine now" would have thrown that away.
    if g3a["status"] != "RESERVED_NOT_OPENED":
        exceptions = [
            entry for entry in registry.get("pi_exceptions", [])
            if entry.get("block") == "g3a_v2_development"
            and "submission_a_receipt_required_before_g3a_open" in entry.get("lifts", [])
            and entry.get("authorisation")
            and entry.get("at")
        ]
        assert exceptions, (
            "g3a_v2_development left RESERVED_NOT_OPENED with no recorded PI exception lifting "
            "submission_a_receipt_required_before_g3a_open"
        )
        assert g3a.get("opened_by_contract"), "opened without naming the contract it opened under"
        assert g3a.get("authorisation") == exceptions[-1]["authorisation"]
        # The exception is per-opening. The header must NOT have been flipped to a general
        # authorisation, which the assertions at the top of this test already require.
        assert "this opening only" in exceptions[-1].get("scope", "")
