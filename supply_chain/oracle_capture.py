"""Clairvoyant-headroom diagnostic: what fraction of an exact ceiling a policy captures.

Requested by Garrido (meeting 2026-07-22) as an explicit way to measure progress over
training.  The instrument has two deliberately separate uses:

1. controller headroom accounting, including structured controllers that are not trained;
2. training-progress curves for learned policies.

For every already-run campaign the exact clairvoyant maximum is known by exhaustive
enumeration of all 4^8 = 65,536 weekly-allocation calendars, so a realized calendar is
graded by table lookup with no re-simulation and no numerical error in the ceiling.

Definition. For campaign i with exact label array L_i over the 65,536 calendars:

    C_i     = max_k L_i[k]                     clairvoyant ceiling (post-hoc only)
    B_i     = reference bar (a static policy)
    V_i     = L_i[index(calendar chosen by the policy)]
    eta_i   = (V_i - B_i) / (C_i - B_i)         capture ratio in [.., 1]

eta = 1 means the policy matched a decision-maker who knew the whole future; eta = 0 means
it did no better than the static bar; eta < 0 means it did worse. Campaign-level ratios are
aggregated clustered on history_root (the resampling unit), with a one-sided LCB95.

Beating the static bar demonstrates state-dependent decision value.  It is not, by itself,
evidence that a controller was trained or that it retained knowledge between campaigns.
Those claims require the corresponding treatment and matched information rights.

The ceiling is a valid upper bound for ANY policy in this action space, including a policy
with privileged information, which is what makes it a fair grading device rather than a
comparator: no controller can exceed it, so eta is a bounded efficiency, not a win rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

WEEKS = 8
ACTIONS = 4
N_CALENDARS = ACTIONS**WEEKS
BOOT_SEED, BOOT_DRAWS = 20260724, 10_000


def calendar_index(calendar) -> int:
    """Row of the frontier table for a weekly-count calendar (week 0 most significant)."""
    idx = 0
    for week, action in enumerate(calendar):
        a = int(action)
        if not 0 <= a < ACTIONS:
            raise ValueError(f"action {a} outside the frozen 4-action contract")
        idx += a * ACTIONS ** (WEEKS - 1 - week)
    return idx


@dataclass(frozen=True)
class Campaign:
    """One graded campaign: exact labels plus its frozen identity."""

    history_root: int
    campaign_index: int
    persistence_mode: str
    kappa: float
    retained_prior: float
    initial_regime: str
    labels: np.ndarray  # float64, shape (65536,)
    frozen_indices: dict[str, int]  # 'retained' / 'reset' rows, abs_diff 0 vs the Pareto

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.history_root, self.campaign_index, self.persistence_mode)

    @property
    def ceiling(self) -> float:
        return float(self.labels.max())

    def value_of(self, calendar) -> float:
        return float(self.labels[calendar_index(calendar)])


def load_campaigns(frontier_dir: Path) -> list[Campaign]:
    """Load the 48 exact frontiers and verify the stored calendar-index convention."""
    table = np.load(frontier_dir / "calendars.npz")["calendars"]
    if table.shape != (N_CALENDARS, WEEKS):
        raise AssertionError(f"calendar table shape drift: {table.shape}")
    # self-check: the index formula must reproduce the stored table, not be assumed
    probe = np.random.default_rng(0).integers(0, N_CALENDARS, 512)
    for row in probe:
        if calendar_index(table[row]) != int(row):
            raise AssertionError("calendar index convention mismatch vs calendars.npz")

    campaigns: list[Campaign] = []
    for npz in sorted(frontier_dir.glob("campaign_*.npz")):
        meta = json.loads(npz.with_suffix(".json").read_text())
        if meta["label"] != "early_ret_complete_cohort":
            raise AssertionError(f"unexpected label in {npz.name}: {meta['label']}")
        labels = np.load(npz)["labels_f64"]
        if labels.shape != (N_CALENDARS,):
            raise AssertionError(f"label shape drift in {npz.name}")
        identity = meta["identity_check_vs_frozen_pareto"]
        for arm in ("retained", "reset"):
            if float(identity[arm]["abs_diff"]) > 1e-9:
                raise AssertionError(f"{npz.name} {arm} identity check failed")
        if abs(float(labels.max()) - float(meta["label_max"])) > 1e-9:
            raise AssertionError(f"{npz.name} ceiling disagrees with its sidecar")
        campaigns.append(Campaign(
            history_root=int(meta["history_root"]),
            campaign_index=int(meta["campaign_index"]),
            persistence_mode=str(meta["persistence_mode"]),
            kappa=float(meta["kappa"]),
            retained_prior=float(meta["retained_prior"]),
            initial_regime=str(meta["initial_regime"]),
            labels=labels,
            frozen_indices={a: int(identity[a]["index"]) for a in ("retained", "reset")},
        ))
    if len(campaigns) != 48:
        raise AssertionError(f"expected the 48 burned campaigns, found {len(campaigns)}")
    return campaigns


def best_static_calendar(campaigns: list[Campaign]) -> tuple[int, float]:
    """The single calendar maximizing the mean exact label across campaigns.

    This is the strongest possible static (open-loop) policy: it is allowed full knowledge
    of the campaign distribution but none of the individual campaign, so beating it is
    evidence of state-dependent decision-making rather than of a good fixed guess.
    """
    stacked = np.vstack([c.labels for c in campaigns])
    means = stacked.mean(axis=0)
    row = int(means.argmax())
    return row, float(means[row])


def constant_action_indices() -> dict[int, int]:
    """Frontier rows of the four constant-action calendars (the discretionary anchors)."""
    return {a: calendar_index([a] * WEEKS) for a in range(ACTIONS)}


def capture_ratios(campaigns: list[Campaign], calendars: dict[tuple, list[int]],
                   bar_values: dict[tuple, float]) -> dict[tuple, float]:
    """eta per campaign for one policy, given its calendar and the bar value per campaign."""
    out: dict[tuple, float] = {}
    for c in campaigns:
        if c.key not in calendars:
            continue
        span = c.ceiling - bar_values[c.key]
        if span <= 1e-12:
            continue  # degenerate campaign: the bar already sits at the ceiling
        out[c.key] = (c.value_of(calendars[c.key]) - bar_values[c.key]) / span
    return out


def pooled_capture(campaigns: list[Campaign], calendars: dict[tuple, list[int]],
                   bar_values: dict[tuple, float], rng) -> dict[str, float]:
    """Portfolio capture: sum(V - B) / sum(C - B) over ALL campaigns.

    The per-campaign ratio is undefined where the bar already sits at the ceiling (in this
    burned set the best static calendar is exactly optimal in 27 of 48 campaigns, so those
    are dropped from the per-campaign mean). This pooled form keeps every campaign. A
    zero-headroom campaign contributes zero to the denominator, but contributes V-B to the
    numerator. It therefore neither helps nor hurts only when the evaluated policy also
    matches the ceiling; otherwise it correctly penalizes a regression below the static bar.
    """
    rows = []
    for c in campaigns:
        if c.key not in calendars:
            continue
        rows.append((c.history_root,
                     c.value_of(calendars[c.key]) - bar_values[c.key],
                     c.ceiling - bar_values[c.key]))
    by_root: dict[int, list[tuple[float, float]]] = {}
    for root, num, den in rows:
        by_root.setdefault(root, []).append((num, den))
    roots = sorted(by_root)

    def ratio(sample) -> float:
        num = sum(n for r in sample for n, _ in by_root[r])
        den = sum(d for r in sample for _, d in by_root[r])
        return num / den if den > 1e-12 else float("nan")

    boot = np.array([ratio(rng.choice(roots, len(roots), True)) for _ in range(BOOT_DRAWS)])
    zero_headroom_rows = [(n, d) for _, n, d in rows if d <= 1e-12]
    return {
        "pooled_ratio": ratio(roots),
        "lcb95": float(np.nanquantile(boot, 0.05)),
        "ucb95": float(np.nanquantile(boot, 0.95)),
        "n_campaigns": len(rows),
        "n_zero_headroom_campaigns": len(zero_headroom_rows),
        "zero_headroom_numerator": float(sum(n for n, _ in zero_headroom_rows)),
        "n_zero_headroom_regressions": int(
            sum(1 for n, _ in zero_headroom_rows if n < -1e-12)
        ),
    }


def clustered(values: dict[tuple, float], rng) -> dict[str, float]:
    """Root-clustered mean with one-sided LCB95; roots are the resampling unit."""
    by_root: dict[int, list[float]] = {}
    for (root, _idx, _mode), v in values.items():
        by_root.setdefault(root, []).append(v)
    roots = sorted(by_root)
    per_root = np.array([np.mean(by_root[r]) for r in roots])
    boot = np.array([rng.choice(per_root, len(per_root), True).mean()
                     for _ in range(BOOT_DRAWS)])
    return {
        "n_campaigns": len(values),
        "n_roots": len(roots),
        "mean": float(per_root.mean()),
        "lcb95": float(np.quantile(boot, 0.05)),
        "ucb95": float(np.quantile(boot, 0.95)),
        "median": float(np.median(list(values.values()))),
    }
