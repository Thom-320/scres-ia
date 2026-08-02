"""Seed custody: one place that answers "may this run use these seeds?".

Six runners each carried a private copy of `seeds_used_by_sealed_artifacts`, and the meta-learner
checked a hand-maintained `PRIOR_SEEDS` tuple that had drifted out of date. A correction in one
copy left the other five stale, which is the most dangerous kind of duplication in this repo: a
virginity falsifier that silently checks the wrong thing still prints PASS.

The API deliberately does NOT have a function called `is_virgin`. `research/seed_custody_registry.json`
declares itself `BASELINE_INVENTORY_INCOMPLETE` and states in its own rules that
`no_result_file_is_not_virginity_evidence` and `untracked_results_are_not_virginity_evidence`.
Absence of a recorded collision is therefore NOT proof of virginity, and the returned status says
so: the strongest positive verdict available is `NO_KNOWN_COLLISION`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

REGISTRY_PATH = Path("research/seed_custody_registry.json")
RESULTS_ROOT = Path("results")
SEED_KEYS = ("seeds", "crn_seeds", "seed_block", "development_seeds", "test_seeds")

#: Registry statuses that mean the range has NOT been consumed. Everything else is a collision.
UNOPENED_STATUSES = frozenset({"RESERVED_NOT_OPENED"})

#: Verdicts. `NO_KNOWN_COLLISION` is the strongest positive one -- see the module docstring.
NO_KNOWN_COLLISION = "NO_KNOWN_COLLISION"
COLLISION = "COLLISION"
DECLARED_REPLAY = "DECLARED_REPLAY"


def load_registry(path: Path | str = REGISTRY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def seeds_used_by_sealed_artifacts(root: Path | str = RESULTS_ROOT,
                                   exclude: Path | str | None = None) -> set[int]:
    """Every seed recorded in a sealed artifact under `root`.

    Hoisted verbatim in behaviour from the six duplicated copies, with the addition of
    `development_seeds`/`test_seeds`, which the original missed -- G3-obs writes its split under
    those keys, so the old scanner would not have seen them.
    """
    used: set[int] = set()
    skip = Path(exclude).resolve() if exclude is not None else None
    for path in Path(root).rglob("result.json"):
        if skip is not None and path.resolve() == skip:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        _collect_seeds(payload, used)
    return used


def _collect_seeds(node: Any, out: set[int], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in SEED_KEYS and isinstance(value, list):
                out.update(int(v) for v in value[:5000] if isinstance(v, (int, float)))
            else:
                _collect_seeds(value, out, depth + 1)
    elif isinstance(node, list):
        for item in node[:50]:
            _collect_seeds(item, out, depth + 1)


def registry_conflicts(seeds: Iterable[int],
                       registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Registry blocks that overlap `seeds` and are not merely reserved."""
    wanted = set(int(s) for s in seeds)
    hits: list[dict[str, Any]] = []
    for block in registry.get("blocks", []):
        lo, hi = int(block["start"]), int(block["end"])
        overlap = sorted(s for s in wanted if lo <= s <= hi)
        if overlap and str(block.get("status")) not in UNOPENED_STATUSES:
            hits.append({"id": block.get("id"), "status": block.get("status"),
                         "range": [lo, hi], "n_overlapping": len(overlap),
                         "source": block.get("source")})
    return hits


def check_seeds(seeds: Iterable[int], *,
                registry_path: Path | str = REGISTRY_PATH,
                results_root: Path | str = RESULTS_ROOT,
                exclude: Path | str | None = None,
                replay_of: str | None = None) -> dict[str, Any]:
    """The single custody check every runner should call.

    `replay_of` names a registry block this run deliberately re-executes -- a custody audit, not
    a fresh experiment. In that case a collision with THAT block is expected and the verdict is
    `DECLARED_REPLAY`; the caller must then report its virginity falsifier as NOT_APPLICABLE
    rather than as passed or failed. Naming the block is required so that "it's just a replay"
    cannot be claimed after seeing a collision.
    """
    seeds = sorted(int(s) for s in seeds)
    registry = load_registry(registry_path)
    conflicts = registry_conflicts(seeds, registry)
    artifact_seeds = seeds_used_by_sealed_artifacts(results_root, exclude)
    artifact_overlap = sorted(set(seeds) & artifact_seeds)

    replay_conflicts = [c for c in conflicts if c["id"] == replay_of] if replay_of else []
    other_conflicts = [c for c in conflicts if c["id"] != replay_of]

    if replay_of is not None:
        if not replay_conflicts:
            status = COLLISION if other_conflicts or artifact_overlap else NO_KNOWN_COLLISION
            note = (f"declared replay_of={replay_of!r} but these seeds do not fall in that "
                    f"registry block; the declaration does not apply")
        else:
            status = COLLISION if other_conflicts else DECLARED_REPLAY
            note = (f"deliberate re-execution of registry block {replay_of!r}; these seeds are "
                    "NOT virgin and are not meant to be")
    elif other_conflicts or artifact_overlap:
        status, note = COLLISION, "seeds already consumed by a registry block or sealed artifact"
    else:
        status = NO_KNOWN_COLLISION
        note = ("no recorded collision. This is NOT proof of virginity: the registry declares "
                "itself incomplete and its own rules state that a missing result file is not "
                "virginity evidence")

    return {
        "status": status,
        "note": note,
        "seeds": seeds[:200],
        "n_seeds": len(seeds),
        "replay_of": replay_of,
        "registry_status": registry.get("status"),
        "registry_is_complete": False,
        "registry_conflicts": conflicts,
        "sealed_artifact_overlap": artifact_overlap[:200],
        "n_sealed_artifact_overlap": len(artifact_overlap),
    }


def custody_falsifier(seeds: Iterable[int], *, replay_of: str | None = None,
                      **kwargs: Any) -> dict[str, Any]:
    """Falsifier-shaped wrapper: `{passed, not_applicable, evidence}`.

    A declared replay returns `not_applicable=True` and `passed=None`, so it is counted in
    neither column. That distinction cost a real overclaim once ("eight falsifiers pass" when one
    could not fail), and encoding it here stops the next runner from repeating it.
    """
    result = check_seeds(seeds, replay_of=replay_of, **kwargs)
    if result["status"] == DECLARED_REPLAY:
        return {"passed": None, "not_applicable": True,
                "evidence": {"why_it_can_fail": "it cannot: this is a declared custody replay of "
                                                f"registry block {replay_of!r}, so the seeds are "
                                                "used on purpose. Counted in neither column.",
                             **result}}
    return {"passed": result["status"] == NO_KNOWN_COLLISION, "not_applicable": False,
            "evidence": {"why_it_can_fail": "a seed already consumed by a registry block or a "
                                            "sealed artifact would make this run a re-use rather "
                                            "than an independent one",
                         **result}}
