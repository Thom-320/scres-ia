#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "None", None) else 0.0


def load_summary(input_path: Path) -> list[dict[str, str]]:
    if input_path.is_dir():
        input_path = input_path / "summary.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {input_path}")
    return read_rows(input_path)


def ret_bucket(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "ret_low"
    if value <= q2:
        return "ret_mid"
    return "ret_high"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def transactions(rows: list[dict[str, str]]) -> list[set[str]]:
    ret_values = [as_float(row, "mean_ret") for row in rows]
    q1 = percentile(ret_values, 1 / 3)
    q2 = percentile(ret_values, 2 / 3)
    items: list[set[str]] = []
    for row in rows:
        risk_overrides = json.loads(row.get("risk_overrides") or "{}")
        tx = {
            f"family={row.get('family')}",
            f"shifts={row.get('shifts')}",
            f"inventory_period={row.get('inventory_replenishment_period')}",
            ret_bucket(as_float(row, "mean_ret"), q1, q2),
        }
        for risk_id, level in risk_overrides.items():
            tx.add(f"{risk_id}={level}")
        items.append(tx)
    return items


def support(transactions_: list[set[str]], itemset: Iterable[str]) -> float:
    required = set(itemset)
    if not transactions_:
        return 0.0
    return sum(1 for tx in transactions_ if required.issubset(tx)) / len(transactions_)


def apriori_rules(
    rows: list[dict[str, str]], *, min_support: float, min_confidence: float
) -> list[dict[str, Any]]:
    txs = transactions(rows)
    universe = sorted(set().union(*txs)) if txs else []
    consequents = ("ret_high", "ret_mid", "ret_low")
    rules: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for antecedent in itertools.combinations(
            [item for item in universe if item not in consequents], size
        ):
            antecedent_support = support(txs, antecedent)
            if antecedent_support <= 0.0:
                continue
            for consequent in consequents:
                rule_items = (*antecedent, consequent)
                rule_support = support(txs, rule_items)
                confidence = rule_support / antecedent_support
                if rule_support >= min_support and confidence >= min_confidence:
                    rules.append(
                        {
                            "antecedent": " & ".join(antecedent),
                            "consequent": consequent,
                            "support": rule_support,
                            "confidence": confidence,
                            "antecedent_support": antecedent_support,
                        }
                    )
    return sorted(rules, key=lambda row: (-row["confidence"], -row["support"]))


def kw_wilcoxon(rows: list[dict[str, str]], *, group_key: str) -> list[dict[str, Any]]:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required for --kw-wilcoxon analysis") from exc

    groups: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get(group_key, ""))
        groups.setdefault(group, []).append(as_float(row, "mean_ret"))
    non_empty = {key: vals for key, vals in groups.items() if vals}
    results: list[dict[str, Any]] = []
    if len(non_empty) >= 2:
        h_stat, p_value = stats.kruskal(*non_empty.values())
        results.append(
            {
                "test": "kruskal",
                "group_key": group_key,
                "comparison": ",".join(sorted(non_empty)),
                "statistic": float(h_stat),
                "p_value": float(p_value),
            }
        )
    for left, right in itertools.combinations(sorted(non_empty), 2):
        x = non_empty[left]
        y = non_empty[right]
        if len(x) == len(y):
            stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
            test_name = "wilcoxon_signed_rank"
        else:
            stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
            test_name = "mann_whitney_u"
        results.append(
            {
                "test": test_name,
                "group_key": group_key,
                "comparison": f"{left} vs {right}",
                "statistic": float(stat),
                "p_value": float(p_value),
            }
        )
    return results


def binomial_capacity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required for --binomial analysis") from exc

    by_cfi: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_cfi.setdefault(row["Cfi"], []).append(row)
    wins = 0
    trials = 0
    for run_rows in by_cfi.values():
        by_shift: dict[str, list[float]] = {}
        for row in run_rows:
            by_shift.setdefault(row.get("shifts", ""), []).append(
                as_float(row, "mean_ret")
            )
        if "3" in by_shift and "1" in by_shift:
            wins += int(
                sum(by_shift["3"]) / len(by_shift["3"])
                > sum(by_shift["1"]) / len(by_shift["1"])
            )
            trials += 1
    if trials == 0:
        return []
    result = stats.binomtest(wins, trials, p=0.5, alternative="greater")
    return [
        {
            "test": "binomial_capacity_s3_gt_s1",
            "wins": wins,
            "trials": trials,
            "p_value": float(result.pvalue),
        }
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze outputs from scripts/run_thesis_factorial.py."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Run dir or summary.csv."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apriori", action="store_true")
    parser.add_argument("--kw-wilcoxon", action="store_true")
    parser.add_argument("--binomial", action="store_true")
    parser.add_argument(
        "--group-key",
        default="inventory_replenishment_period",
        help="CSV column used for KW/pairwise grouping.",
    )
    parser.add_argument("--minsup", type=float, default=0.10)
    parser.add_argument("--minconf", type=float, default=0.90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_summary(args.input)
    output_rows: list[dict[str, Any]] = []
    if args.apriori:
        output_rows.extend(
            apriori_rules(rows, min_support=args.minsup, min_confidence=args.minconf)
        )
    if args.kw_wilcoxon:
        output_rows.extend(kw_wilcoxon(rows, group_key=args.group_key))
    if args.binomial:
        output_rows.extend(binomial_capacity(rows))
    if not (args.apriori or args.kw_wilcoxon or args.binomial):
        raise ValueError(
            "Select at least one analysis: --apriori, --kw-wilcoxon, or --binomial."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, output_rows)
    print(json.dumps({"output": str(args.output), "rows": len(output_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
