"""Compare the preserved original MATD3 with residual MATD3."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent / "results"
BASELINE = ROOT / "matd3_distributed_v1"
RESIDUAL = ROOT / "matd3_residual_v1"


def _rewards(folder: Path) -> np.ndarray:
    with (folder / "training_episodes.csv").open(encoding="utf-8") as stream:
        return np.asarray([float(row["reward"]) for row in csv.DictReader(stream)], dtype=np.float64)


def plot_learning(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)
    for label, values, color in (
        ("Original MATD3", _rewards(BASELINE), "#0072B2"),
        ("Residual MATD3", _rewards(RESIDUAL), "#D55E00"),
    ):
        episodes = np.arange(1, len(values) + 1)
        window = min(50, max(5, len(values) // 20))
        ax.plot(episodes, values, color=color, alpha=0.16, linewidth=0.7)
        if len(values) >= window:
            rolling = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(values) + 1), rolling, color=color, linewidth=2.2, label=f"{label} (rolling {window})")
    ax.axhline(0.0, color="#555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set(title="Original MATD3 vs residual MATD3", xlabel="Episode", ylabel="Team return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_table(path: Path) -> None:
    with (BASELINE / "metrics.json").open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    with (RESIDUAL / "metrics.json").open(encoding="utf-8") as stream:
        residual = json.load(stream)
    labels = [
        ("mean_return", "平均回报 ↑"),
        ("iae_mean_mol_m3", "平均浓度绝对误差 ↓"),
        ("settling_time_mean_s", "平均稳定时间 (s) ↓"),
        ("peak_abs_error_mol_m3", "峰值绝对误差 ↓"),
        ("steady_error_std_mol_m3", "稳态误差标准差 ↓"),
        ("action_variation_normalized", "动作变化率 ↓"),
    ]
    lines = [
        "# Original MATD3 vs residual MATD3",
        "",
        "Same environment, 80,000 training steps, 4 vectorized environments, seed=42, and 30 evaluation episodes.",
        "Residual MATD3 starts from the preserved baseline checkpoint. The baseline actors are frozen; zero-initialized bounded residual heads learn corrections with max normalized amplitude 0.15.",
        "",
        "| 版本 | " + " | ".join(label for _, label in labels) + " |",
        "|---|" + "---:|" * len(labels),
        "| Original MATD3 | " + " | ".join(f"{baseline[key]:.3f}" for key, _ in labels) + " |",
        "| Residual MATD3 | " + " | ".join(f"{residual[key]:.3f}" for key, _ in labels) + " |",
        "",
        "Higher return is better; errors, settling time, and action variation are better when lower.",
        "This is a single-seed comparison; repeat with multiple seeds before making a final superiority claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    plot_learning(RESIDUAL / "comparison_learning_curve.png")
    write_table(RESIDUAL / "comparison_metrics.md")
    print(f"Wrote comparison outputs under {RESIDUAL}")
