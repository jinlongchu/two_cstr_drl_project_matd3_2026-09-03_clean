"""Create two comparison figures for PPO, SAC and DDPG runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG, PPO, SAC

from train_ppo_three_segment import C2_START, RECYCLE_START, TARGET_SCHEDULE, evaluate


COLORS = {"PPO": "#1f77b4", "SAC": "#d62728", "DDPG": "#2ca02c"}


def rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.arange(len(values)), values
    return np.arange(window - 1, len(values)), np.convolve(values, np.ones(window) / window, mode="valid")


def read_column(path: Path, name: str) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        return np.asarray([float(row[name]) for row in csv.DictReader(stream)])


def plot_learning(root: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=170, constrained_layout=True)
    for algorithm in ("PPO", "SAC", "DDPG"):
        directory = root / f"{algorithm.lower()}_three_segment_targets_v3"
        episodes = read_column(directory / "training_episodes.csv", "reward")
        ex, er = rolling(episodes, min(50, max(5, len(episodes) // 20)))
        ax.plot(ex + 1, er, color=COLORS[algorithm], linewidth=2.2, label=algorithm)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    ax.set(title="PPO / SAC / DDPG episode-return learning curves", xlabel="Episode", ylabel="Return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def stack_band(trajectories, key: str, index: int, valid_mask=None):
    values = np.stack([item[key] for item in trajectories])[:, :, index]
    if valid_mask is not None:
        values = np.where(np.stack([item[valid_mask] for item in trajectories]), values, np.nan)
    valid = np.any(np.isfinite(values), axis=0)
    median = np.full(values.shape[1], np.nan)
    lo = np.full(values.shape[1], np.nan)
    hi = np.full(values.shape[1], np.nan)
    median[valid] = np.nanmedian(values[:, valid], axis=0)
    lo[valid] = np.nanmin(values[:, valid], axis=0)
    hi[valid] = np.nanmax(values[:, valid], axis=0)
    return median, lo, hi


def plot_effect(root: Path, output: Path) -> None:
    model_specs = {
        "PPO": (PPO, root / "ppo_three_segment_targets_v3" / "ppo_three_segment.zip"),
        "SAC": (SAC, root / "sac_three_segment_targets_v3" / "sac_three_segment.zip"),
        "DDPG": (DDPG, root / "ddpg_three_segment_targets_v3" / "ddpg_three_segment.zip"),
    }
    trajectories = {}
    for algorithm, (cls, path) in model_specs.items():
        model = cls.load(path)
        trajectories[algorithm], _ = evaluate(model, 30, seed=1042)

    first = trajectories["PPO"][0]
    t_state = np.arange(first["states"].shape[0])
    t_action = np.arange(first["actions"].shape[0])
    fig = plt.figure(figsize=(14, 8.5), dpi=170, constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    def draw(ax, median, lo, hi, time, color, label):
        valid = np.isfinite(median)
        ax.fill_between(time, lo, hi, where=valid, color=color, alpha=0.08, linewidth=0)
        ax.plot(time, median, color=color, linewidth=2, label=label)

    for algorithm, runs in trajectories.items():
        c2_valid = np.stack([item["c2_valid"] for item in runs])
        med, lo, hi = stack_band(runs, "states", 0)
        draw(axes[0], med, lo, hi, t_state, COLORS[algorithm], f"{algorithm} $C_1$")
        med, lo, hi = stack_band(runs, "states", 2, "c2_valid")
        draw(axes[0], med, lo, hi, t_state, COLORS[algorithm], f"{algorithm} $C_2$")
        med, lo, hi = stack_band(runs, "states", 1)
        draw(axes[1], med, lo, hi, t_state, COLORS[algorithm], f"{algorithm} $T_1$")
        med, lo, hi = stack_band(runs, "states", 3, "c2_valid")
        draw(axes[1], med, lo, hi, t_state, COLORS[algorithm], f"{algorithm} $T_2$")
        med, lo, hi = stack_band(runs, "actions", 0)
        draw(axes[2], med * 1e4, lo * 1e4, hi * 1e4, t_action, COLORS[algorithm], f"{algorithm} $F$")
        med, lo, hi = stack_band(runs, "actions", 1)
        draw(axes[2], med * 1e4, lo * 1e4, hi * 1e4, t_action, COLORS[algorithm], f"{algorithm} $L$")
        med, lo, hi = stack_band(runs, "actions", 2)
        draw(axes[3], med, lo, hi, t_action, COLORS[algorithm], f"{algorithm} $T_{{c1}}$")
        med, lo, hi = stack_band(runs, "actions", 3)
        draw(axes[3], med, lo, hi, t_action, COLORS[algorithm], f"{algorithm} $T_{{c2}}$")

    axes[0].plot(t_state, np.repeat(TARGET_SCHEDULE[0][0], len(t_state)), color="#111", linestyle="--", linewidth=1.4, label="$C_1^*$")
    target_c1 = np.array([TARGET_SCHEDULE[min(int(t // 60), 2)][0] for t in t_state])
    target_c2 = np.array([TARGET_SCHEDULE[min(int(t // 60), 2)][1] if t >= C2_START else np.nan for t in t_state])
    axes[0].lines[-1].set_ydata(target_c1)
    axes[0].plot(t_state, target_c2, color="#111", linestyle=":", linewidth=1.4, label="$C_2^*$")
    for ax in axes:
        ax.axvline(C2_START, color="#555", linestyle=":", linewidth=1)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[0].axvline(60, color="#777", linestyle=":", linewidth=1)
    axes[0].axvline(120, color="#777", linestyle=":", linewidth=1)
    axes[2].axvline(RECYCLE_START, color="#555", linestyle=":", linewidth=1)
    axes[0].set(title="CSTR concentration", xlabel="Time (s)", ylabel="Concentration (mol/m³)")
    axes[1].set(title="CSTR temperature", xlabel="Time (s)", ylabel="Temperature (K)")
    axes[2].set(title="Flow actions", xlabel="Time (s)", ylabel=r"Flow ($10^{-4}$ m³/s)")
    axes[3].set(title="CSTR cooling actions", xlabel="Time (s)", ylabel="Cooling temperature (K)")
    fig.suptitle("PPO / SAC / DDPG realistic two-CSTR comparison", y=1.01)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    plot_learning(args.root, args.root / "three_algorithm_learning_comparison_v3.png")
    print("learning comparison figure written; effect figures remain per algorithm")


if __name__ == "__main__":
    main()
