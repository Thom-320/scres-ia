from copy import deepcopy
import json
from pathlib import Path

from scripts.verify_paper2_terminal_return import validate_readiness


ROOT = Path(__file__).resolve().parent.parent
SEARCH = ROOT / "research" / "paper2_exhaustive_search"


def _readiness():
    return json.loads((SEARCH / "terminal_return_readiness.json").read_text())


def test_current_terminal_readiness_is_valid_and_explicitly_incomplete():
    validation = validate_readiness(_readiness())

    assert validation["passed"] is True
    assert validation["all_outputs_terminal_ready"] is False
    assert validation["terminal_ready_ids"] == [1, 2, 3, 4, 5]
    assert validation["nonterminal_ids"] == [6, 7, 8, 9, 10, 11, 12, 13]


def test_unhashed_artifact_cannot_be_promoted_by_flipping_terminal_ready():
    readiness = deepcopy(_readiness())
    row = readiness["required_outputs"][5]
    row["terminal_ready"] = True
    readiness["summary"]["terminal_ready_count"] += 1
    readiness["summary"]["nonterminal_output_ids"].remove(6)

    validation = validate_readiness(readiness)

    assert validation["passed"] is False
    assert any("unhashed artifact" in failure for failure in validation["failures"])
