#!/usr/bin/env python3
"""Stage 1 of the oracle metric: grade every controller we already have calendars for.

Everything here is an exact table lookup into the 48 enumerated 4^8 frontiers, so the
ceiling carries no estimation error and no policy is re-simulated. Burned development
campaigns only (roots 7570801-24); no claim, no sealed seed.

Usage:
    .venv/bin/python scripts/report_oracle_capture_metric.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.oracle_capture import (  # noqa: E402
    BOOT_SEED,
    Campaign,
    best_static_calendar,
    capture_ratios,
    clustered,
    constant_action_indices,
    load_campaigns,
    pooled_capture,
)

FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"
PARETO = ROOT / "results/q_r1/comparator_v2_frozen_pareto_c256_v1/pareto_merged/result.json"
SCREEN = ROOT / "results/q_r1/service_aware_screen_v1"
OUT = ROOT / "results/oracle_capture_v1"


def index_to_calendar(index: int) -> list[int]:
    return [(index // 4 ** (7 - w)) % 4 for w in range(8)]


def main() -> int:
    campaigns = load_campaigns(FRONTIERS)
    by_key: dict[tuple, Campaign] = {c.key: c for c in campaigns}
    rng = np.random.default_rng(BOOT_SEED)

    # ---- bars ---------------------------------------------------------------
    static_row, static_mean = best_static_calendar(campaigns)
    static_calendar = index_to_calendar(static_row)
    constants = constant_action_indices()

    bars = {
        "uninformed_expectation": {c.key: float(c.labels.mean()) for c in campaigns},
        "best_static_open_loop": {c.key: float(c.labels[static_row]) for c in campaigns},
    }
    for action, row in constants.items():
        bars[f"constant_action_{action}"] = {c.key: float(c.labels[row]) for c in campaigns}

    # ---- policies -----------------------------------------------------------
    policies: dict[str, dict[tuple, list[int]]] = {}
    policies["frozen_c256_mpc_reset"] = {
        c.key: index_to_calendar(c.frozen_indices["reset"]) for c in campaigns}
    policies["frozen_c256_mpc_retained"] = {
        c.key: index_to_calendar(c.frozen_indices["retained"]) for c in campaigns}
    policies["best_static_open_loop"] = {c.key: static_calendar for c in campaigns}
    for action, row in constants.items():
        policies[f"constant_action_{action}"] = {
            c.key: index_to_calendar(row) for c in campaigns}

    for path in sorted(glob.glob(str(SCREEN / "shard_c*/result.json"))):
        d = json.loads(Path(path).read_text())
        for row in d["rows"]:
            key = (row["history_root"], row["campaign_index"], row["persistence_mode"])
            if key not in by_key:
                continue
            cfg = row["config_id"]
            # keep the wf prefix: fta0.10 exists at floor 0.70 AND 0.80, so the
            # post-"legacy_" suffix alone collides and would silently merge configs
            suffix = cfg.split("_c256_")[-1].replace("_unone_expected_tol0.0000", "")
            tag = "service_aware_" + suffix
            policies.setdefault(tag, {})[key] = list(row["variant_calendar"])

    # ---- grading ------------------------------------------------------------
    ceilings = np.array([c.ceiling for c in campaigns])
    report: dict[str, dict] = {}
    for name, calendars in sorted(policies.items()):
        entry: dict[str, dict] = {}
        for bar_name, bar_values in bars.items():
            etas = capture_ratios(campaigns, calendars, bar_values)
            entry[bar_name] = {
                "per_campaign": clustered(etas, np.random.default_rng(BOOT_SEED)),
                "pooled": pooled_capture(campaigns, calendars, bar_values,
                                         np.random.default_rng(BOOT_SEED)),
            }
        raw = {c.key: c.value_of(calendars[c.key]) for c in campaigns if c.key in calendars}
        entry["absolute"] = {
            "mean_label": float(np.mean(list(raw.values()))),
            "mean_regret_vs_ceiling": float(np.mean(
                [by_key[k].ceiling - v for k, v in raw.items()])),
            "exact_optimum_hits": int(sum(
                1 for k, v in raw.items() if by_key[k].ceiling - v <= 1e-9)),
            "n_campaigns": len(raw),
        }
        # how much of the movement vs the retained arm is inside an exact value tie:
        # the objective cannot distinguish those calendars, only the service ledger can
        ties = moved = 0
        for c in campaigns:
            if c.key not in calendars:
                continue
            row = c.frozen_indices["retained"]
            if calendars[c.key] != index_to_calendar(row):
                moved += 1
                if abs(c.value_of(calendars[c.key]) - float(c.labels[row])) <= 1e-12:
                    ties += 1
        entry["vs_retained_arm"] = {"campaigns_with_different_calendar": moved,
                                    "of_which_exact_value_ties": ties}
        report[name] = entry

    out = {
        "schema": "oracle_capture_metric_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM_METHODOLOGICAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": "Garrido, meeting 2026-07-22: explicit learning metric = % of the "
                        "post-hoc clairvoyant maximum captured; learning confirmed only if "
                        "it beats the best static policy",
        "objective": "early_ret_complete_cohort",
        "oracle": {
            "method": "exhaustive enumeration of all 4^8 = 65,536 weekly calendars per "
                      "campaign; exact, not estimated; valid upper bound for ANY policy",
            "n_campaigns": len(campaigns),
            "ceiling_mean": float(ceilings.mean()),
            "ceiling_min": float(ceilings.min()),
            "ceiling_max": float(ceilings.max()),
        },
        "bars": {
            "best_static_open_loop": {
                "calendar": static_calendar, "frontier_row": static_row,
                "mean_label": static_mean,
                "definition": "single calendar maximizing the mean exact label across the "
                              "48 campaigns: knows the distribution, not the campaign",
            },
            "uninformed_expectation": {
                "definition": "mean label over all 65,536 calendars = an arbitrary "
                              "discretionary calendar in expectation (Baseline-0 analogue)",
            },
        },
        "policies": report,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle_capture_metric.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"oracle ceiling mean {ceilings.mean():.4f} "
          f"[{ceilings.min():.4f}, {ceilings.max():.4f}] over {len(campaigns)} campaigns")
    print(f"best static open-loop calendar {static_calendar} mean {static_mean:.4f}\n")
    print(f"{'policy':38} {'pooled vs static':>18} {'per-campaign':>16} "
          f"{'opt':>4} {'ties':>5}")
    for name, entry in sorted(report.items(), key=lambda kv:
                              -kv[1]["best_static_open_loop"]["pooled"]["pooled_ratio"]):
        s = entry["best_static_open_loop"]
        print(f"{name[:38]:38} {s['pooled']['pooled_ratio']:+.4f}/{s['pooled']['lcb95']:+.4f} "
              f"{s['per_campaign']['mean']:+.4f} (n{s['per_campaign']['n_campaigns']:>2}) "
              f"{entry['absolute']['exact_optimum_hits']:>4} "
              f"{entry['vs_retained_arm']['of_which_exact_value_ties']:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
