#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "docs/meeting_packages/garrido_2026-03-11"
CONTROL_REWARD_DIR = ROOT / "outputs/benchmarks/control_reward"
LOCAL_ROBUSTNESS_DIR = ROOT / "outputs/benchmarks/control_reward_local_robustness"
ARTIFACT_ROOT = ROOT / "docs/artifacts/control_reward"
LONG_INCREASED_DIR = ARTIFACT_ROOT / "control_reward_500k_increased_stopt"
LONG_SEVERE_DIR = ARTIFACT_ROOT / "control_reward_500k_severe_stopt"
FIGURE_SOURCE_DIR = ROOT / "outputs/figures/control_reward_paper"
WINNING_COMBO = {"w_bo": "4.0", "w_cost": "0.02", "w_disr": "0.0"}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def find_policy_row(
    rows: list[dict[str, str]], *, phase: str, policy: str, combo: dict[str, str]
) -> dict[str, str]:
    for row in rows:
        if (
            row["phase"] == phase
            and row["policy"] == policy
            and row["w_bo"] == combo["w_bo"]
            and row["w_cost"] == combo["w_cost"]
            and row["w_disr"] == combo["w_disr"]
        ):
            return row
    raise ValueError(f"Missing row for {phase=} {policy=} {combo=}")


def find_comparison_row(
    rows: list[dict[str, str]], combo: dict[str, str]
) -> dict[str, str]:
    for row in rows:
        if (
            row["w_bo"] == combo["w_bo"]
            and row["w_cost"] == combo["w_cost"]
            and row["w_disr"] == combo["w_disr"]
        ):
            return row
    raise ValueError(f"Missing comparison row for {combo=}")


def render_table(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    output_path: Path,
    col_widths: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.4 + 0.48 * len(rows)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#5a5a5a")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#d9e6f2")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 1:
            cell.set_facecolor("#f7f7f7")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_static_baseline_table() -> Path:
    policy_rows = load_csv_rows(CONTROL_REWARD_DIR / "policy_summary.csv")
    data_rows: list[list[str]] = []
    for policy in ("static_s1", "static_s2", "static_s3"):
        row = find_policy_row(
            policy_rows, phase="static_screen", policy=policy, combo=WINNING_COMBO
        )
        data_rows.append(
            [
                policy,
                f"{float(row['reward_total_mean']):.2f}",
                f"{100.0 * float(row['fill_rate_mean']):.1f}%",
                f"{100.0 * float(row['backorder_rate_mean']):.1f}%",
                (
                    f"S1 {float(row['pct_steps_S1_mean']):.0f}% / "
                    f"S2 {float(row['pct_steps_S2_mean']):.0f}% / "
                    f"S3 {float(row['pct_steps_S3_mean']):.0f}%"
                ),
            ]
        )

    output_path = PACKAGE_DIR / "01_static_shift_baselines.png"
    render_table(
        title="Static shift-control baselines under control_v1 (w_bo=4.0, w_cost=0.02)",
        headers=["Policy", "Reward", "Fill rate", "Backorder", "Shift mix"],
        rows=data_rows,
        output_path=output_path,
        col_widths=[0.16, 0.15, 0.16, 0.16, 0.37],
    )
    return output_path


def build_best_regime_table() -> Path:
    increased_rows = load_csv_rows(LONG_INCREASED_DIR / "comparison_table.csv")
    severe_rows = load_csv_rows(LONG_SEVERE_DIR / "comparison_table.csv")
    increased_row = find_comparison_row(increased_rows, WINNING_COMBO)
    severe_row = find_comparison_row(severe_rows, WINNING_COMBO)

    data_rows = [
        [
            "500k increased + stoPT",
            "static_s2",
            "No",
            f"{float(increased_row['ppo_reward_mean']):.2f}",
            f"{float(increased_row['best_static_reward_mean']):.2f}",
            f"{100.0 * float(increased_row['ppo_fill_rate_mean']):.1f}%",
            f"{100.0 * float(increased_row['best_static_fill_rate_mean']):.1f}%",
            (
                f"S1 {float(increased_row['ppo_pct_steps_S1_mean']):.0f}% / "
                f"S2 {float(increased_row['ppo_pct_steps_S2_mean']):.0f}% / "
                f"S3 {float(increased_row['ppo_pct_steps_S3_mean']):.0f}%"
            ),
        ],
        [
            "500k severe + stoPT",
            severe_row["best_static_policy"],
            "Yes" if severe_row["ppo_beats_best_static"] == "True" else "No",
            f"{float(severe_row['ppo_reward_mean']):.2f}",
            f"{float(severe_row['best_static_reward_mean']):.2f}",
            f"{100.0 * float(severe_row['ppo_fill_rate_mean']):.1f}%",
            f"{100.0 * float(severe_row['best_static_fill_rate_mean']):.1f}%",
            (
                f"S1 {float(severe_row['ppo_pct_steps_S1_mean']):.0f}% / "
                f"S2 {float(severe_row['ppo_pct_steps_S2_mean']):.0f}% / "
                f"S3 {float(severe_row['ppo_pct_steps_S3_mean']):.0f}%"
            ),
        ],
    ]

    output_path = PACKAGE_DIR / "03_best_regime_summary.png"
    render_table(
        title="500k × 5-seed adaptive-control summary under stochastic PT",
        headers=[
            "Scenario",
            "Best static",
            "PPO > best static",
            "PPO reward",
            "Best static",
            "PPO fill",
            "Best static",
            "PPO shift mix",
        ],
        rows=data_rows,
        output_path=output_path,
        col_widths=[0.20, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10, 0.24],
    )
    return output_path


def summarize_long_run_results() -> str:
    increased_rows = load_csv_rows(LONG_INCREASED_DIR / "comparison_table.csv")
    severe_rows = load_csv_rows(LONG_SEVERE_DIR / "comparison_table.csv")
    increased_row = find_comparison_row(increased_rows, WINNING_COMBO)
    severe_row = find_comparison_row(severe_rows, WINNING_COMBO)
    return "\n".join(
        [
            (
                "- increased + stochastic_pt: PPO is competitive but not superior in "
                f"reward ({float(increased_row['ppo_reward_mean']):.2f} vs "
                f"{float(increased_row['best_static_reward_mean']):.2f}); service is effectively "
                f"matched ({100.0 * float(increased_row['ppo_fill_rate_mean']):.1f}% vs "
                f"{100.0 * float(increased_row['best_static_fill_rate_mean']):.1f}%)."
            ),
            (
                "- severe + stochastic_pt: PPO beats the best static baseline in reward "
                f"({float(severe_row['ppo_reward_mean']):.2f} vs "
                f"{float(severe_row['best_static_reward_mean']):.2f}) while maintaining comparable "
                f"service ({100.0 * float(severe_row['ppo_fill_rate_mean']):.1f}% vs "
                f"{100.0 * float(severe_row['best_static_fill_rate_mean']):.1f}%)."
            ),
        ]
    )


def write_meeting_order() -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"""**Garrido Meeting Package**

Generated: {now}

**Show in this order**

1. `01_static_shift_baselines.png`
   This is the first slide. It proves that the environment is differentiated before any learning claim: shift control materially changes service levels.

2. `02_ppo_training_reward_curve.png`
   Use this only as a PPO diagnostic. Do not call it cross-validation loss. Say: “for PPO we track training reward and held-out evaluation, not supervised cross-validation loss.”

3. `03_best_regime_summary.png`
   This is now the main results slide. Say clearly that under `increased + stochastic_pt` PPO is competitive but not superior, while under `severe + stochastic_pt` PPO beats the best static baseline.

4. `04_policy_comparison.png`
   Use this only if someone asks what “winning regime” means operationally.

5. `05_action_mix.png`
   Use this if Garrido or David asks whether PPO collapsed to a trivial static policy.

**One paragraph to say out loud**

The DES is already validated and the RL interface is operational. Before claiming anything about PPO, we verified that the static shift baselines are clearly differentiated, which confirms that shift allocation is a meaningful control lever. We also separated the training objective from the reporting metric: `ReT_thesis` is retained as an evaluation metric, while `control_v1` is used for learning because direct control needs an operational reward. The completed 500k-step stochastic-PT runs now show a clean pattern: under increased risk PPO is competitive with the best static baseline, and under severe risk adaptive switching produces a reward advantage without collapse.

**Observability line**

We treat the exposed observation as a practical Gymnasium-compatible operational snapshot and a useful Markovian approximation for control, while explicitly acknowledging a remaining partial-observability caveat.

**Long-run benchmark outcome**

{summarize_long_run_results()}
"""
    output_path = PACKAGE_DIR / "00_meeting_order.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_david_note() -> Path:
    content = """**Note for David: why there is no cross-validation loss plot here**

The current method is PPO over a Gymnasium environment, not a supervised learner trained on fixed folds. Because of that, there is no standard cross-validation loss curve analogous to ANN/GNN validation loss. The evaluation equivalents we can show right now are:

- PPO training reward vs timesteps
- held-out evaluation against fixed baselines on the same seeds
- fill rate, backorder rate, and shift-mix diagnostics

So the correct sentence in the meeting is:

> For PPO, we do not evaluate with cross-validation loss; we evaluate with training reward diagnostics plus out-of-sample policy comparison on fixed seeds and operational metrics.
"""
    output_path = PACKAGE_DIR / "06_note_for_david.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def copy_existing_figures() -> list[Path]:
    copied: list[Path] = []
    mapping = {
        "figure_1_training_reward_curve.png": "02_ppo_training_reward_curve.png",
        "figure_2_policy_comparison.png": "04_policy_comparison.png",
        "figure_3_action_mix.png": "05_action_mix.png",
    }
    for src_name, dst_name in mapping.items():
        src = FIGURE_SOURCE_DIR / src_name
        if src.exists():
            dst = PACKAGE_DIR / dst_name
            dst.write_bytes(src.read_bytes())
            copied.append(dst)
    return copied


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    build_static_baseline_table()
    build_best_regime_table()
    write_meeting_order()
    write_david_note()
    copied = copy_existing_figures()
    print(f"Wrote package to {PACKAGE_DIR}")
    for path in copied:
        print(f"Copied {path.name}")


if __name__ == "__main__":
    main()
