#!/usr/bin/env python3
"""Fig-5 surrogate study — dataset dump (Track B, preregistration 2026-07-23).

Regenerates the 48 burned campaigns' EXACT 4^8 weekly-count frontiers
(`simulate_full_des_frontier`, label = early_ret_complete_cohort) and stores
one compressed npz per campaign under results/fig5_surrogate_v1/frontiers/.

Campaign identity comes from the frozen Pareto merge
(results/q_r1/comparator_v2_frozen_pareto_c256_v1/pareto_merged/result.json):
each pareto_pairs row gives (history_root, campaign_index, persistence_mode
-> kappa, retained_prior).  Campaigns are rebuilt identically to the burned
path via scripts.c6_perbatch_ceiling.rebuild_campaign (torch-free scheduler
read from the same frozen contract).

Calendar encoding (implicit by row index): row i of the 65,536-row label
array corresponds to full_action_calendars(8)[i], i.e. the base-4 digits of i
with week 0 as the MOST significant digit (itertools.product order):
    index = sum(calendar[w] * 4**(7-w) for w in range(8)).

Self-checks recorded in manifest.json:
  * identity check (all 48 campaigns): replaying the frozen retained AND
    reset calendars through the rebuilt frontier reproduces the Pareto's
    stored early_ret_complete_cohort values to <= 1e-9;
  * spot replay (3 random campaigns x 5 random calendars): single-calendar
    re-evaluation agrees with the stored frontier labels to <= 1e-9;
  * sha256 for every written file.

Labels are stored twice: 'labels' (float32, per preregistration) and
'labels_f64' (float64) so the 1e-9 replay tolerance and exact argmax/regret
grading are checked against full precision, not the f32 quantization.

Usage:
    OMP_NUM_THREADS=1 .venv/bin/python scripts/dump_fig5_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.c6_perbatch_ceiling import (  # noqa: E402
    CAMPAIGNS_PER_HISTORY,
    DOMINANT_SHARE,
    MODE_TO_KAPPA,
    REGIME_PERSISTENCE,
    _count_scheduler,
    rebuild_campaign,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    full_action_calendars,
    simulate_full_des_frontier,
)

PARETO_PATH = (
    _ROOT
    / "results/q_r1/comparator_v2_frozen_pareto_c256_v1/pareto_merged/result.json"
)
OUT_DIR = _ROOT / "results/fig5_surrogate_v1/frontiers"
LABEL = "early_ret_complete_cohort"
TOL = 1e-9
SPOT_SEED = 20260723


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _calendar_index(calendar: np.ndarray) -> int:
    idx = 0
    for w in range(8):
        idx = idx * 4 + int(calendar[w])
    return idx


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pareto = json.loads(PARETO_PATH.read_text())
    pairs = pareto["pareto_pairs"]
    assert len(pairs) == 48, f"expected 48 pareto rows, got {len(pairs)}"

    scheduler = _count_scheduler()
    calendars = full_action_calendars(8)
    cal_path = OUT_DIR / "calendars.npz"
    np.savez_compressed(cal_path, calendars=calendars)

    files: list[dict] = []
    identity_checks: list[dict] = []
    stored: dict[str, np.ndarray] = {}
    stored_camp: dict[str, object] = {}

    for row in sorted(
        pairs, key=lambda r: (r["history_root"], r["persistence_mode"])
    ):
        root = int(row["history_root"])
        mode = str(row["persistence_mode"])
        kappa = MODE_TO_KAPPA[mode]
        index = int(row["campaign_index"])
        camp = rebuild_campaign(root, kappa, index)
        t0 = time.time()
        metrics = simulate_full_des_frontier(
            skeleton=camp.skeleton,
            scheduler=scheduler,
            calendars=calendars,
            include_q_r1_metrics=True,
        )
        elapsed = time.time() - t0
        labels64 = np.asarray(metrics[LABEL], dtype=np.float64)
        assert labels64.shape == (65536,)

        # Identity check vs the frozen Pareto record (retained + reset arms).
        checks = {}
        for arm in ("retained", "reset"):
            cal = np.asarray(row[f"{arm}_calendar"], dtype=np.uint8)
            idx = _calendar_index(cal)
            regen = float(labels64[idx])
            frozen = float(row[arm][LABEL])
            diff = abs(regen - frozen)
            assert diff <= TOL, (
                f"identity check failed root={root} mode={mode} arm={arm}: "
                f"regen={regen!r} frozen={frozen!r}"
            )
            checks[arm] = {"index": idx, "abs_diff": diff}

        tag = f"r{root}_k{kappa:.2f}_i{index}"
        meta = {
            "history_root": root,
            "campaign_index": index,
            "persistence_mode": mode,
            "kappa": kappa,
            "retained_prior": float(row["retained_prior"]),
            "initial_regime": camp.initial_regime,
            "last_regime": camp.last_regime,
            "regime_persistence": REGIME_PERSISTENCE,
            "dominant_share": DOMINANT_SHARE,
            "campaigns_per_history": CAMPAIGNS_PER_HISTORY,
            "label": LABEL,
            "label_min": float(labels64.min()),
            "label_max": float(labels64.max()),
            "label_argmax_index": int(labels64.argmax()),
            "frontier_seconds": round(elapsed, 3),
            "identity_check_vs_frozen_pareto": checks,
        }
        npz_path = OUT_DIR / f"campaign_{tag}.npz"
        np.savez_compressed(
            npz_path,
            labels=labels64.astype(np.float32),
            labels_f64=labels64,
        )
        meta_path = OUT_DIR / f"campaign_{tag}.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        files.append({"npz": npz_path.name, "meta": meta_path.name, **{
            k: meta[k] for k in (
                "history_root", "campaign_index", "kappa", "retained_prior",
                "initial_regime", "label_max",
            )
        }})
        identity_checks.append({"tag": tag, **checks})
        stored[tag] = labels64
        stored_camp[tag] = camp
        print(f"[dump] {tag}: max={labels64.max():.6f} ({elapsed:.1f}s)",
              flush=True)

    # Spot replay: 3 random campaigns x 5 random calendars, single-calendar
    # re-evaluation vs stored labels.
    rng = np.random.default_rng(SPOT_SEED)
    tags = sorted(stored)
    spot = []
    for tag in rng.choice(tags, size=3, replace=False):
        camp = stored_camp[tag]
        for idx in rng.choice(65536, size=5, replace=False):
            idx = int(idx)
            single = simulate_full_des_frontier(
                skeleton=camp.skeleton,
                scheduler=scheduler,
                calendars=calendars[idx][None, :],
                include_q_r1_metrics=True,
            )
            value = float(np.asarray(single[LABEL], dtype=np.float64)[0])
            diff = abs(value - float(stored[tag][idx]))
            assert diff <= TOL, (
                f"spot replay failed {tag} idx={idx}: {value!r} vs "
                f"{float(stored[tag][idx])!r}"
            )
            spot.append({"tag": tag, "calendar_index": idx, "abs_diff": diff})
    print(f"[dump] spot replay OK: {len(spot)} single-calendar re-evals "
          f"within {TOL}", flush=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preregistration": "docs/FIG5_SURROGATE_PREREGISTRATION_2026-07-23.md",
        "source_pareto": str(PARETO_PATH.relative_to(_ROOT)),
        "label": LABEL,
        "calendar_encoding": (
            "row i == full_action_calendars(8)[i]; base-4 digits of i, week 0 "
            "most significant: index = sum(cal[w] * 4**(7-w))"
        ),
        "n_campaigns": len(files),
        "tolerance": TOL,
        "spot_replay_seed": SPOT_SEED,
        "spot_replay": spot,
        "identity_checks_vs_frozen_pareto": identity_checks,
        "files": files,
        "sha256": {
            p.name: _sha256(p)
            for p in sorted(OUT_DIR.iterdir())
            if p.name != "manifest.json"
        },
        "total_seconds": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"[dump] wrote {len(files)} campaigns + manifest in "
          f"{manifest['total_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
