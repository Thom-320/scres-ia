"""The Paper 2 artifacts must still be able to name the code that produced them.

WHY THIS FILE EXISTS. On 2026-08-07 `supply_chain/supply_chain.py` and `supply_chain/arm_runner.py`
were edited after `grid_transfer_confirmation_v2` -- the project's headline confirmation -- had been
sealed against them. The suite stayed green for a whole day. Four tests mention `module_manifest`
(`test_grid_transfer_confirmation.py`, `test_garrido_surface_cache_custody.py`,
`test_g3c_temporal_coupling_physics.py`, `test_seed_custody_module.py`) and every one of them asserts
only the SHAPE of the dict -- `{"modules": {"physics": "frozen"}}` and similar. Nothing compared a
single hash.

THE RULE THIS ENCODES. An artifact the manuscript cites may drift from its manifest only while a
sealed equivalence certificate covers exactly the files that drifted. A file that moves without
appearing in that certificate fails here immediately, which is what should have happened on the 7th.

marked `release_paper2`: this is a submission gate, not an archaeological test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain.seed_custody import module_manifest

ROOT = Path(__file__).resolve().parent.parent
CERTIFICATE = ROOT / "results/frozen_path_equivalence/result.json"

#: Cache roots whose slices carry the manifests the cited artifacts were sealed against.
CITED_CACHES = {
    "grid_transfer_confirmation_v2": "results/surface_cache/garrido_transfer_confirmation_v2_ext",
    "search_ladder_v5": "results/surface_cache/wrap288_v1",
}

pytestmark = pytest.mark.release_paper2


def _certificate() -> dict:
    assert CERTIFICATE.exists(), (
        f"{CERTIFICATE} is missing. Run scripts/verify_frozen_path_equivalence_v1.py before "
        "citing any Paper 2 number.")
    return json.loads(CERTIFICATE.read_text())


def _first_slice(cache_root: str) -> dict:
    slices = sorted((ROOT / cache_root).glob("*/*.json"))
    assert slices, f"no cache slices under {cache_root}"
    return json.loads(slices[0].read_text())


@pytest.mark.parametrize("name,cache_root", sorted(CITED_CACHES.items()))
def test_every_drifted_module_is_covered_by_the_equivalence_certificate(name, cache_root):
    """Recompute the hashes. Any mismatch must be a file the certificate already cleared."""
    stored = _first_slice(cache_root)["module_manifest"]
    live = module_manifest(tuple(stored["modules"]), script=stored["entry_script"])

    drifted = {mod for mod, digest in stored["modules"].items()
               if live["modules"].get(mod) != digest}
    if stored["entry_script_sha256"] != live["entry_script_sha256"]:
        drifted.add(stored["entry_script"])
    if not drifted:
        return

    cert = _certificate()
    assert cert["claim_status"] == "FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_SOURCE", (
        f"{name} drifted on {sorted(drifted)} and the equivalence certificate says "
        f"{cert['claim_status']!r}")
    assert cert["falsifiers"]["all_passed"] is True, (
        f"{name} drifted and the certificate's own falsifiers did not all pass")
    uncovered = sorted(drifted - set(cert["declared_drift"]))
    assert not uncovered, (
        f"{name} drifted on {uncovered}, which the equivalence certificate does not cover. "
        "Re-run scripts/verify_frozen_path_equivalence_v1.py and declare the new drift, or the "
        "artifact can no longer name the code that produced it.")


def test_the_certificate_actually_reran_the_simulator():
    """A certificate that checked nothing would clear everything.

    Guards the degenerate pass: zero cells, one context, or a comparator that never fired. Those
    are exactly the shapes a rushed re-run produces when it is asked to make a red test green.
    """
    cert = _certificate()
    per_cache = cert["by_cache"]
    assert set(per_cache) == set(CITED_CACHES), (
        f"the certificate covers {sorted(per_cache)}, the manuscript cites {sorted(CITED_CACHES)}")
    for name, row in per_cache.items():
        assert row["n_cells"] >= 100, f"{name}: only {row['n_cells']} cells re-evaluated"
        assert len(row["contexts"]) >= 6, f"{name}: sample spans {row['contexts']}"
        assert row["n_differing"] == 0, f"{name}: {row['n_differing']} cells did not reproduce"
    for name, control in cert["mutation_controls"].items():
        assert control["value_mutation_detected"], f"{name}: comparator missed a planted value"
        assert control["panel_mutation_detected"], f"{name}: comparator missed a planted panel key"
        assert control["clean_cell_still_matches"], f"{name}: the unperturbed cell did not match"


def test_no_undeclared_drift_anywhere_in_the_certificate():
    """The certificate names two edits. A third would mean it certifies less than it appears to."""
    cert = _certificate()
    assert cert["manifest_drift"]["undeclared_drift"] == [], (
        f"undeclared drift: {cert['manifest_drift']['undeclared_drift']}")
