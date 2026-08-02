"""The shared arm-comparison instrument, so the same defect cannot be copy-pasted again.

The four 2026-07-30 runners were derived from one another with `sed`, and every one of them
carried the same set of contract violations. Two forensic audits found them; this module
exists so a fifth runner inherits the fix instead of the bug.

What it owns, and what each replaces:

* **the scored population** — one list feeds every moment. The old runners passed `APj`/`RPj`
  over ~277 orders while `ret` came from the ledger's ~217, so share denominators and
  numerators disagreed by 28%.
* **the population rate** — `scored_orders_per_year` on the *observed order span*, in thesis
  years. The old runners counted over 8,736 h and divided by `1.0 "year"` while the reference
  used 8,064 h/year. Garrido's own sheets exclude the warm-up (`min(OPTj)` is 823-1,225 h), so
  BOTH sides use `(last - first)/8064` — since 2026-07-31 on our side too, the earlier
  `horizon - warmup` differing from it by up to 2.2%. See
  `contracts/paper_b_v2_amendment_2026-07-31.json`.
* **`d_k`** — delegated to `fidelity_moments.discrepancies()`, which owns the definition and
  the degenerate-moment guard. Four independent re-implementations is how the `249.8 SD`
  error happened.
* **the verdict** — `non_dominated()` plus `epsilon_stability()`, never `sum(d_k)`. The master
  contract says the output is the NON-DOMINATED SET, never a winner, and forbids collapsing
  it with weights. `sum_dk` is reported as a descriptor and is not admissible as a ranking.
* **sealing** — every payload, INCLUDING a falsifier halt, carries provenance, the contract
  hash and its own `self_sha256`. The old halt path wrote a two-key file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .episode_metrics import (
    compute_order_level_ret_excel_request_snapshot_ledger as _ledger,
)
from .fidelity_moments import (
    EPSILON_BAND,
    EPSILON,
    MOMENT_NAMES,
    MomentReference,
    discrepancies,
    epsilon_stability,
    non_dominated,
)
from .provenance import calibration_stamp
from .scientific_payload import scientific_payload_sha256 as _scientific_payload_sha256

THESIS_YEAR_HOURS = 8064.0


def scored_orders(sim: Any) -> list:
    """The one population. Everything downstream is derived from this list."""
    return [
        o for o in sim.orders
        if not bool(getattr(o, "metrics_excluded", False))
        and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)
    ]


def episode_moments(sim: Any) -> dict[str, float]:
    """Six moments from one episode, one population, thesis year basis.

    The rate uses the **observed order span**, `max(OPTj) - min(OPTj)` over the scored
    population, because that is the estimator `fidelity_reference_v4` applies to Garrido's
    sheets -- the only rows he ships are already warm-up filtered, so his window can only be
    read off his own orders.

    CORRECTED 2026-07-31. This used `horizon - warmup`, which ends at the horizon while his
    ends at the last order. Measured on 12 roots x 2 families, the two conventions differ by
    **1.5% (R1r) and 2.2% (R2r)** -- larger than the moment's own gap to the reference in
    R1r, so the convention was deciding the `d_k`, not the data. See
    `results/metric_audit/fidelity_comparison_v4/`.

    One asymmetry remains and is disclosed rather than corrected: `n / span` counts `n`
    orders across `n - 1` gaps, an upward bias of `n/(n-1)` -- about 0.45% on our ~220
    orders against 0.05% on his ~2,100. It is the same estimator on both sides, so it
    cancels in direction and is an order of magnitude below the gap it replaces.
    """
    orders = scored_orders(sim)
    if not orders:
        raise ValueError("no scored orders; horizon is shorter than the warm-up")
    horizon = float(sim.env.now)

    # The ledger applies its own visibility filter (not lost, OATj present). Score the
    # SAME rows on every moment rather than mixing two populations.
    book = _ledger(orders, current_time=horizon)
    visible_ids = {id(o) for o in orders
                   if not bool(getattr(o, "lost", False))
                   and getattr(o, "OATj", None) is not None}
    population = [o for o in orders if id(o) in visible_ids]
    ret = [float(v) for v in book["ret_values"]]
    if len(ret) != len(population):  # pragma: no cover - guards a silent mismatch
        raise ValueError(
            f"ledger returned {len(ret)} values for {len(population)} visible orders; "
            "the population and the ledger filter have diverged")

    apj = [float(getattr(o, "APj", 0.0) or 0.0) for o in population]
    rpj = [float(getattr(o, "RPj", 0.0) or 0.0) for o in population]
    pos = sorted(v for v in rpj if v > 0.0)
    n = len(ret)
    # The observed span of the SAME population the moments are computed on.
    opt = [float(getattr(o, "OPTj", 0.0)) for o in population]
    window_hours = max(max(opt) - min(opt), 1e-9) if len(opt) > 1 else max(
        horizon - float(sim.warmup_time), 1e-9)
    return {
        "autotomy_share": sum(1 for v in apj if v > 0.0) / n,
        "ret_mean": sum(ret) / n,
        "ret_above_one_share": sum(1 for v in ret if v > 1.0) / n,
        "rpj_mean": (sum(pos) / len(pos)) if pos else 0.0,
        "rpj_p95": float(np.percentile(pos, 95)) if pos else 0.0,
        "scored_orders_per_year": n / (window_hours / THESIS_YEAR_HOURS),
    }


def aggregate(rows: Sequence[Mapping[str, float]],
              reference: Mapping[str, MomentReference]) -> dict[str, Any]:
    """Mean, SE and `d_k` for one arm-family cell. `d_k` is not recomputed here."""
    mean = {m: float(np.mean([r[m] for r in rows])) for m in MOMENT_NAMES}
    se = {m: float(np.std([r[m] for r in rows], ddof=1) / np.sqrt(len(rows)))
          for m in MOMENT_NAMES}
    return {
        "moments": mean, "moment_se": se, "n_episodes": len(rows),
        "discrepancies": discrepancies(mean, se, reference),
        # Descriptor only. The master contract forbids collapsing dominance with weights,
        # so this may never be used to rank or select an arm.
        "sum_dk_DESCRIPTIVE_NOT_A_RANKING": float(
            sum(v for v in discrepancies(mean, se, reference).values()
                if v == v)),  # NaN-safe
    }


def verdict(cells: Mapping[str, Mapping[str, float]],
            epsilons: Sequence[float] = EPSILON_BAND) -> dict[str, Any]:
    """The contract's own output: a non-dominated set plus its epsilon sensitivity."""
    d_only = {name: c["discrepancies"] for name, c in cells.items()}
    stability = epsilon_stability(d_only, epsilons)
    return {
        "non_dominated_set": non_dominated(d_only, EPSILON),
        "epsilon_declared": EPSILON,
        "epsilon_stability": stability,
        "set_is_epsilon_stable": bool(stability["stable"]),
        "discriminates": len(non_dominated(d_only, EPSILON)) < len(d_only),
        "selection_rule": ("the non-dominated set IS the output; no arm is chosen here and "
                           "sum_dk may not be used to rank"),
    }


def build_reference(blob: Mapping[str, Any],
                    family: str) -> dict[str, MomentReference]:
    return {m: MomentReference(**{k: v[k] for k in ("mean", "spread", "n_sheets")})
            for m, v in blob["reference_by_family"][family].items()
            if m in MOMENT_NAMES}


def seal_and_write(payload: dict[str, Any], path: Path, *,
                   contract: Path, reference: Path,
                   stamp_extra: Mapping[str, Any] | None = None) -> str:
    """Seal EVERY payload, halts included, and return the digest.

    The old halt path wrote `{"claim_status": ..., "summary": ...}` and nothing else: no
    contract hash, no provenance, no seal. A halted run is still a result and still has to
    be attributable.
    """
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload["contract_path"] = str(contract)
    payload["contract_sha256"] = sha256(Path(contract).read_bytes()).hexdigest()
    payload["reference_path"] = str(reference)
    payload["reference_sha256"] = json.loads(
        Path(reference).read_text()).get("self_sha256")
    payload["calibration_provenance"] = calibration_stamp(**dict(stamp_extra or {}))
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return payload["self_sha256"]


def run_falsifiers(checks: Mapping[str, Callable[[], tuple[bool, Any]]]) -> dict[str, Any]:
    """Run every declared falsifier and record its evidence, not just its boolean.

    Three 2026-07-30 falsifiers reported `True` for quantities that could not vary. Storing
    the evidence alongside the verdict makes a tautological check visible in the artifact
    instead of only in an audit.
    """
    out: dict[str, Any] = {}
    for name, fn in checks.items():
        ok, evidence = fn()
        out[name] = {"passed": bool(ok), "evidence": evidence}
    out["all_passed"] = all(v["passed"] for k, v in out.items() if k != "all_passed")
    return out


#: Keys excluded from the canonical scientific payload hash: they change on every run or every
#: commit without the science changing. `self_sha256` covers the whole envelope INCLUDING these,
#: which is why it can never be used to compare two runs of different code -- the H3' audit hit
#: exactly that wall.
VOLATILE_KEYS = frozenset({
    "created_at", "elapsed_seconds", "self_sha256", "calibration_provenance",
    "contract_path", "reference_path", "module_manifest", "audit_status", "replay_of",
})


def canonical_payload_sha256(payload: Mapping[str, Any], *,
                             extra_exclude: frozenset[str] = frozenset()) -> str:
    """Hash of the SCIENTIFIC content only: events, actions, ledgers, metrics, verdicts.

    Two runs of different code that produce identical science must produce the same value here,
    while `self_sha256` must differ because the provenance genuinely changed. Without this split
    a null-arm identity check is untestable: you cannot demand that a hash ignore the fact that
    the program was just modified.
    """
    body = {k: v for k, v in payload.items()
            if k not in VOLATILE_KEYS and k not in extra_exclude}
    return _scientific_payload_sha256(body)
