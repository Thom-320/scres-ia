"""
Post-run analyzer for Track B 500k benchmark.

Reads the completed benchmark artifacts and produces a final comparison
report against static baselines and the 100k smoke.

Usage:
    python scripts/analyze_track_b_500k.py
    python scripts/analyze_track_b_500k.py --run-dir outputs/track_b_benchmarks/track_b_ret_seq_k020_500k
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_RUN_DIR = REPO / "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k"
SMOKE_DIR = REPO / "outputs/benchmarks/track_b_smoke_initial_2026-03-31"
DOE_DIR = REPO / "outputs/doe/track_b_minimal_doe_initial_2026-03-30"
FAMILY_A_DIR = REPO / "outputs/paper_benchmarks/paper_ret_seq_k020_500k"
OUTPUT_DIR = REPO / "outputs/track_b_benchmarks/track_b_500k_analysis"


def load_policy_summary(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict], policy: str) -> dict | None:
    for r in rows:
        if r.get("policy") == policy:
            return r
    return None


def safe_float(val: str | None, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def analyze(run_dir: Path) -> dict:
    policy_csv = run_dir / "policy_summary.csv"

    if not policy_csv.exists():
        return {"error": f"No policy_summary.csv in {run_dir}. Run not finished?"}

    rows = load_policy_summary(policy_csv)

    # Extract PPO and baselines
    ppo = pick(rows, "ppo")
    s2 = pick(rows, "s2_d1.00")
    s3 = pick(rows, "s3_d1.00")
    s3_d2 = pick(rows, "s3_d2.00")

    if ppo is None:
        return {"error": "No 'ppo' policy found in policy_summary.csv"}

    def metrics(row: dict | None) -> dict:
        if row is None:
            return {"reward": None, "fill": None, "backorder": None, "ret": None}
        return {
            "reward": safe_float(row.get("reward_total_mean")),
            "fill": safe_float(row.get("fill_rate_mean")),
            "backorder": safe_float(row.get("backorder_rate_mean")),
            "ret": safe_float(
                row.get("order_level_ret_mean_mean", row.get("order_level_ret_mean"))
            ),
            "rolling_fill_4w": safe_float(row.get("rolling_fill_rate_4w_mean")),
            "s1_pct": safe_float(row.get("pct_steps_S1_mean")),
            "s2_pct": safe_float(row.get("pct_steps_S2_mean")),
            "s3_pct": safe_float(row.get("pct_steps_S3_mean")),
        }

    ppo_m = metrics(ppo)
    s2_m = metrics(s2)
    s3_m = metrics(s3)
    s3_d2_m = metrics(s3_d2)

    # Best static
    statics = [("s2_d1.00", s2_m), ("s3_d1.00", s3_m), ("s3_d2.00", s3_d2_m)]
    best_static_name, best_static = max(
        [(n, m) for n, m in statics if m["fill"] is not None],
        key=lambda x: x[1]["fill"],
    )

    result = {
        "run_dir": str(run_dir),
        "ppo": ppo_m,
        "s2_d1.00": s2_m,
        "s3_d1.00": s3_m,
        "s3_d2.00": s3_d2_m,
        "best_static": {"name": best_static_name, **best_static},
        "delta_ppo_vs_s2": {
            "fill_pp": (ppo_m["fill"] - s2_m["fill"]) * 100 if s2_m["fill"] else None,
            "reward": ppo_m["reward"] - s2_m["reward"] if s2_m["reward"] else None,
        },
        "delta_ppo_vs_best_static": {
            "fill_pp": (
                (ppo_m["fill"] - best_static["fill"]) * 100
                if best_static["fill"]
                else None
            ),
            "reward": (
                ppo_m["reward"] - best_static["reward"]
                if best_static["reward"]
                else None
            ),
        },
        "ppo_beats_s2": ppo_m["fill"] > s2_m["fill"] if s2_m["fill"] else None,
        "ppo_beats_best_static": ppo_m["fill"] > best_static["fill"],
        "shift_collapse": ppo_m["s1_pct"] > 80
        or ppo_m["s2_pct"] > 80
        or ppo_m["s3_pct"] > 80,
    }

    # Compare with smoke if available
    if SMOKE_DIR.exists() and (SMOKE_DIR / "policy_summary.csv").exists():
        smoke_rows = load_policy_summary(SMOKE_DIR / "policy_summary.csv")
        smoke_ppo = pick(smoke_rows, "ppo")
        if smoke_ppo:
            smoke_m = metrics(smoke_ppo)
            result["smoke_100k_ppo"] = smoke_m
            result["delta_500k_vs_smoke"] = {
                "fill_pp": (
                    (ppo_m["fill"] - smoke_m["fill"]) * 100 if smoke_m["fill"] else None
                ),
                "reward": (
                    ppo_m["reward"] - smoke_m["reward"] if smoke_m["reward"] else None
                ),
            }

    # Compare with Family A if available
    if FAMILY_A_DIR.exists() and (FAMILY_A_DIR / "comparison_table.csv").exists():
        with open(FAMILY_A_DIR / "comparison_table.csv") as f:
            fa_rows = list(csv.DictReader(f))
        if fa_rows:
            fa = fa_rows[0]
            result["family_a_comparison"] = {
                "ppo_fill": safe_float(fa.get("ppo_fill_rate_mean")),
                "s2_fill": safe_float(fa.get("static_s2_fill_rate_mean")),
                "ppo_beats_s2": fa.get("ppo_beats_static_s2"),
                "note": "Family A: thesis-faithful, no downstream control",
            }

    return result


def write_report(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Markdown
    md_lines = ["# Track B 500k Analysis\n"]

    if "error" in result:
        md_lines.append(f"**ERROR:** {result['error']}\n")
        md_lines.append(
            "Run `python scripts/analyze_track_b_500k.py` after the benchmark completes.\n"
        )
    else:
        ppo = result["ppo"]
        s2 = result["s2_d1.00"]
        best = result["best_static"]
        d_s2 = result["delta_ppo_vs_s2"]
        d_best = result["delta_ppo_vs_best_static"]

        md_lines.append("## Main Result\n")
        md_lines.append(
            "| Policy | Reward | Fill Rate | Backorder | Order ReT | Shifts S1/S2/S3 |"
        )
        md_lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for name, m in [
            ("**PPO**", ppo),
            ("s2_d1.00", s2),
            (f"best_static ({best['name']})", best),
        ]:
            md_lines.append(
                f"| {name} | {m['reward']:.2f} | {m['fill']:.4f} | "
                f"{m['backorder']:.4f} | {m.get('ret', 0):.4f} | "
                f"{m.get('s1_pct', 0):.1f}/{m.get('s2_pct', 0):.1f}/{m.get('s3_pct', 0):.1f} |"
            )

        md_lines.append("\n## Deltas")
        md_lines.append(
            f"- PPO vs S2: fill {d_s2['fill_pp']:+.2f}pp, reward {d_s2['reward']:+.2f}"
        )
        md_lines.append(
            f"- PPO vs best static: fill {d_best['fill_pp']:+.2f}pp, reward {d_best['reward']:+.2f}"
        )
        md_lines.append(f"- PPO beats S2: **{result['ppo_beats_s2']}**")
        md_lines.append(
            f"- PPO beats best static: **{result['ppo_beats_best_static']}**"
        )
        md_lines.append(f"- Shift collapse: {result['shift_collapse']}")

        if "smoke_100k_ppo" in result:
            d_smoke = result["delta_500k_vs_smoke"]
            md_lines.append("\n## vs Smoke 100k")
            md_lines.append(f"- 500k fill vs 100k fill: {d_smoke['fill_pp']:+.2f}pp")
            md_lines.append(f"- 500k reward vs 100k reward: {d_smoke['reward']:+.2f}")

        if "family_a_comparison" in result:
            fa = result["family_a_comparison"]
            md_lines.append("\n## vs Family A (Track A)")
            md_lines.append(f"- Family A PPO fill: {fa['ppo_fill']:.4f}")
            md_lines.append(f"- Family A PPO beats S2: {fa['ppo_beats_s2']}")
            md_lines.append(f"- Track B PPO fill: {ppo['fill']:.4f}")
            md_lines.append("- **Track B opened the headroom that Track A lacked.**")

        md_lines.append("\n## Verdict")
        if result["ppo_beats_best_static"] and not result["shift_collapse"]:
            md_lines.append(
                "**PASS.** PPO beats best static without shift collapse. Track B is validated."
            )
        elif result["ppo_beats_s2"] and not result["shift_collapse"]:
            md_lines.append(
                "**PARTIAL.** PPO beats S2 but not best static. Modest improvement."
            )
        else:
            md_lines.append(
                "**FAIL.** PPO does not beat baselines or shows shift collapse."
            )

    with open(output_dir / "analysis.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Analysis written to {output_dir}/")
    print("  analysis.json")
    print("  analysis.md")


def main():
    parser = argparse.ArgumentParser(description="Analyze Track B 500k benchmark")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Path to the Track B 500k run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the analysis",
    )
    args = parser.parse_args()

    result = analyze(args.run_dir)
    write_report(result, args.output_dir)

    # Print summary to stdout
    if "error" not in result:
        ppo = result["ppo"]
        d = result["delta_ppo_vs_s2"]
        print(f"\nPPO: reward={ppo['reward']:.2f} fill={ppo['fill']:.4f}")
        print(f"vs S2: fill {d['fill_pp']:+.2f}pp, reward {d['reward']:+.2f}")
        print(f"Beats S2: {result['ppo_beats_s2']}")
        print(f"Beats best static: {result['ppo_beats_best_static']}")
        print(f"Shift collapse: {result['shift_collapse']}")
    else:
        print(f"\n{result['error']}")
        print("Re-run after benchmark completes.")


if __name__ == "__main__":
    main()
