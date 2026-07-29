#!/usr/bin/env python3
"""Sprint 1 for preventive Track B: can observed risk history predict common risks?

This is deliberately *not* an RL training script. It builds a supervised
belief-learning dataset from frozen-policy rollouts and the DES risk ledger:

    features at week t  ->  risk Ri starts in the next H weeks?

The goal is to decide whether the proposed PPO+MLP/Real-KAN auxiliary risk head
has a learnable signal before we touch SB3 internals. Evaluation is grouped by
episode to avoid treating neighbouring weeks as independent.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SOURCE = Path("outputs/experiments/track_b_risk_event_counterfactual_fixed_rng_2026-07-04")
DEFAULT_OUTPUT = Path("outputs/experiments/track_b_risk_belief_predictor_2026-07-04")
TARGET_RISKS = ("R11", "R13", "R24")
HORIZONS = (1, 2, 4, 8)
STEP_HOURS = 168.0

OBS_FEATURES_NO_FORECAST = (
    "fill_rate_obs",
    "rolling_fill_rate_4w",
    "backlog_age_norm",
    "op10_queue_pressure_norm",
    "op12_queue_pressure_norm",
    "new_demanded",
    "new_backorder_qty",
    "pending_backorder_qty",
)
FORECAST_FEATURES = ("forecast_48h", "forecast_168h")
REGIME_FEATURES = (
    "regime_nominal",
    "regime_strained",
    "regime_pre_disruption",
    "regime_disrupted",
    "regime_recovery",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-risks", nargs="+", default=list(TARGET_RISKS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--test-episode-mod", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def build_event_index(risk_rows: Iterable[dict[str, str]]) -> dict[tuple[str, int, int], dict[str, list[int]]]:
    index: dict[tuple[str, int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in risk_rows:
        risk = str(row["risk_id"])
        if risk == "R14":
            continue
        key = (str(row["policy"]), int(row["seed"]), int(row["episode"]))
        index[key][risk].append(int(float(row["start_step"])))
    for by_risk in index.values():
        for steps in by_risk.values():
            steps.sort()
    return index


def memory_features_for_step(event_steps: dict[str, list[int]], target_risks: list[str], step: int) -> dict[str, float]:
    feats: dict[str, float] = {}
    for risk in target_risks:
        prior = [s for s in event_steps.get(risk, []) if s < step]
        weeks_since = 104.0 if not prior else float(min(104, step - prior[-1]))
        count_8w = sum(1 for s in prior if step - s <= 8)
        count_26w = sum(1 for s in prior if step - s <= 26)
        # Simple exponentially decayed event memory; half-life around 8 weeks.
        ewma = sum(float(np.exp(-(step - s) / 8.0)) for s in prior)
        feats[f"mem_weeks_since_last_{risk}"] = weeks_since / 104.0
        feats[f"mem_count_{risk}_8w"] = min(count_8w / 8.0, 1.0)
        feats[f"mem_count_{risk}_26w"] = min(count_26w / 26.0, 1.0)
        feats[f"mem_ewma_{risk}_8w"] = min(ewma / 8.0, 1.0)
    return feats


def future_label(event_steps: dict[str, list[int]], risk: str, step: int, horizon: int) -> int:
    return int(any(step < s <= step + horizon for s in event_steps.get(risk, [])))


def build_dataset(
    step_rows: list[dict[str, str]],
    risk_rows: list[dict[str, str]],
    *,
    target_risks: list[str],
    horizons: list[int],
) -> list[dict[str, object]]:
    event_index = build_event_index(risk_rows)
    out: list[dict[str, object]] = []
    for row in step_rows:
        if row.get("condition", "full") != "full":
            continue
        policy = str(row["policy"])
        seed = int(row["seed"])
        episode = int(row["episode"])
        step = int(row["step"])
        key = (policy, seed, episode)
        event_steps = event_index.get(key, {})
        memory = memory_features_for_step(event_steps, target_risks, step)
        sample: dict[str, object] = {
            "policy": policy,
            "seed": seed,
            "episode": episode,
            "step": step,
            "group": f"{policy}_s{seed}_e{episode}",
        }
        for name in OBS_FEATURES_NO_FORECAST + FORECAST_FEATURES + REGIME_FEATURES:
            sample[name] = f(row, name)
        sample.update(memory)
        for risk in target_risks:
            for horizon in horizons:
                sample[f"y_{risk}_{horizon}w"] = future_label(event_steps, risk, step, horizon)
        out.append(sample)
    return out


def split_train_test(rows: list[dict[str, object]], *, test_episode_mod: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, row in enumerate(rows):
        # Deterministic grouped split by episode identity; all weeks from the
        # same seed/episode go to the same side.
        seed = int(row["seed"])
        episode = int(row["episode"])
        if (seed * 31 + episode) % int(test_episode_mod) == 0:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def metric_or_none(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_prob))


def train_eval(
    rows: list[dict[str, object]],
    *,
    feature_sets: dict[str, list[str]],
    target_risks: list[str],
    horizons: list[int],
    max_iter: int,
    test_episode_mod: int,
) -> list[dict[str, object]]:
    train_idx, test_idx = split_train_test(rows, test_episode_mod=test_episode_mod)
    results: list[dict[str, object]] = []
    for feature_set, features in feature_sets.items():
        x_all = np.asarray([[float(row.get(feat, 0.0)) for feat in features] for row in rows], dtype=np.float32)
        for policy in sorted({str(row["policy"]) for row in rows}):
            policy_idx = np.asarray([i for i, row in enumerate(rows) if row["policy"] == policy], dtype=int)
            p_train = np.intersect1d(train_idx, policy_idx)
            p_test = np.intersect1d(test_idx, policy_idx)
            if len(p_train) == 0 or len(p_test) == 0:
                continue
            for risk in target_risks:
                for horizon in horizons:
                    target = f"y_{risk}_{horizon}w"
                    y_all = np.asarray([int(row[target]) for row in rows], dtype=int)
                    y_train = y_all[p_train]
                    y_test = y_all[p_test]
                    base_rate = float(np.mean(y_test)) if len(y_test) else 0.0
                    if len(np.unique(y_train)) < 2:
                        results.append(
                            {
                                "policy": policy,
                                "feature_set": feature_set,
                                "risk_id": risk,
                                "horizon_weeks": horizon,
                                "n_train": len(p_train),
                                "n_test": len(p_test),
                                "test_base_rate": base_rate,
                                "auc": "",
                                "average_precision": "",
                                "brier": "",
                                "status": "single_class_train",
                            }
                        )
                        continue
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=int(max_iter),
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    )
                    model.fit(x_all[p_train], y_train)
                    y_prob = model.predict_proba(x_all[p_test])[:, 1]
                    auc = metric_or_none(roc_auc_score, y_test, y_prob)
                    ap = metric_or_none(average_precision_score, y_test, y_prob)
                    brier = float(brier_score_loss(y_test, y_prob))
                    results.append(
                        {
                            "policy": policy,
                            "feature_set": feature_set,
                            "risk_id": risk,
                            "horizon_weeks": horizon,
                            "n_train": len(p_train),
                            "n_test": len(p_test),
                            "test_base_rate": base_rate,
                            "auc": "" if auc is None else auc,
                            "average_precision": "" if ap is None else ap,
                            "brier": brier,
                            "status": "ok",
                        }
                    )
    return results


def write_verdict(path: Path, results: list[dict[str, object]]) -> None:
    ok = [r for r in results if r["status"] == "ok" and r["auc"] != ""]
    lines = [
        "# Track B risk-belief predictor smoke — 2026-07-04",
        "",
        "Este smoke responde una pregunta previa al entrenamiento PPO auxiliar:",
        "",
        "> Con historia observada de riesgos frecuentes, ¿hay señal predictiva suficiente para justificar una cabeza auxiliar?",
        "",
        "No entrena políticas RL. Sólo entrena clasificadores logísticos por política/riesgo/horizonte, con split agrupado por episodio.",
        "",
        "## Mejores AUC por riesgo",
        "",
        "| Politica | Riesgo | Mejor feature set | Horizonte | AUC | Base rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for policy in sorted({str(r["policy"]) for r in ok}):
        for risk in sorted({str(r["risk_id"]) for r in ok}):
            group = [r for r in ok if r["policy"] == policy and r["risk_id"] == risk]
            if not group:
                continue
            best = max(group, key=lambda r: float(r["auc"]))
            lines.append(
                "| {policy} | {risk} | {fs} | {h} | {auc:.3f} | {base:.3f} |".format(
                    policy=policy,
                    risk=risk,
                    fs=best["feature_set"],
                    h=int(best["horizon_weeks"]),
                    auc=float(best["auc"]),
                    base=float(best["test_base_rate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            "- Si `memory_only` ya tiene AUC útil, la memoria histórica por riesgo es una señal defendible sin forecast privilegiado.",
            "- Si sólo gana `with_forecast_regime`, el modelo está aprovechando señales privilegiadas del benchmark, no memoria aprendida.",
            "- Si los AUC quedan cerca de 0.5, no hay base para meter una cabeza auxiliar todavía.",
            "",
            "El siguiente paso, si este smoke muestra señal, es crear `PPO+MLP-belief` y `PPO+RealKAN-belief` usando estas variables como observación adicional y una cabeza auxiliar de predicción.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_rows = read_csv(args.source_dir / "step_ledger_full.csv")
    risk_rows = read_csv(args.source_dir / "risk_event_ledger.csv")
    target_risks = [str(r) for r in args.target_risks]
    horizons = [int(h) for h in args.horizons]
    dataset = build_dataset(step_rows, risk_rows, target_risks=target_risks, horizons=horizons)
    write_csv(args.output_dir / "risk_belief_dataset.csv", dataset)

    mem_features = [
        f"{prefix}_{risk}{suffix}"
        for risk in target_risks
        for prefix, suffix in (
            ("mem_weeks_since_last", ""),
            ("mem_count", "_8w"),
            ("mem_count", "_26w"),
            ("mem_ewma", "_8w"),
        )
    ]
    # Names above are easier to build explicitly because count/ewma include risk in the middle.
    mem_features = []
    for risk in target_risks:
        mem_features.extend(
            [
                f"mem_weeks_since_last_{risk}",
                f"mem_count_{risk}_8w",
                f"mem_count_{risk}_26w",
                f"mem_ewma_{risk}_8w",
            ]
        )
    feature_sets = {
        "memory_only": mem_features,
        "operational_no_forecast": list(OBS_FEATURES_NO_FORECAST) + mem_features,
        "with_forecast_regime": list(OBS_FEATURES_NO_FORECAST) + list(FORECAST_FEATURES) + list(REGIME_FEATURES) + mem_features,
    }
    results = train_eval(
        dataset,
        feature_sets=feature_sets,
        target_risks=target_risks,
        horizons=horizons,
        max_iter=int(args.max_iter),
        test_episode_mod=int(args.test_episode_mod),
    )
    write_csv(args.output_dir / "risk_belief_predictor_metrics.csv", results)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "target_risks": target_risks,
        "horizons": horizons,
        "n_rows": len(dataset),
        "feature_sets": feature_sets,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_verdict(args.output_dir / "verdict.md", results)
    print(f"Wrote risk-belief predictor smoke to {args.output_dir}")
    for row in results:
        if row["status"] == "ok" and row["feature_set"] == "memory_only":
            print(
                f"{row['policy']} {row['risk_id']} h{row['horizon_weeks']} "
                f"memory_only auc={row['auc']} base={row['test_base_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
