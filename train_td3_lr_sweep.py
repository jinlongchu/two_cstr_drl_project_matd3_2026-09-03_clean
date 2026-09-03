"""TD3 learning-rate sweep for the realistic two-CSTR environment."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from train_improved_ddpg import ENV_KWARGS, metrics
from train_multiseed import DEFAULT_SEEDS
from train_ppo_three_segment import evaluate, plot_effect
from two_cstr_env import TwoCSTRStageOffsetEnv


LEARNING_RATES = (5e-5, 1e-4, 3e-4, 1e-3, 3e-3)


class EpisodeCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
        return True

    def _on_training_end(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(zip(range(1, len(self.rewards) + 1), self.rewards, self.lengths))


def lr_label(lr: float) -> str:
    return f"lr_{lr:.0e}".replace("-", "m")


def make_env(seed: int):
    def factory():
        env = Monitor(TwoCSTRStageOffsetEnv(**ENV_KWARGS))
        env.reset(seed=seed)
        return env

    return factory


def train_one(lr: float, seed: int, timesteps: int, eval_episodes: int, root: Path, reuse_root: Path | None) -> dict:
    directory = root / lr_label(lr) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    reused = False
    source = reuse_root / "td3" / f"seed_{seed}" if reuse_root is not None and np.isclose(lr, 3e-4) else None
    if source is not None and (source / "metrics.json").exists() and (source / "trajectories.npy").exists():
        # The 3e-4 point is exactly the TD3 baseline already trained under the
        # same environment, seeds, horizon, and evaluation protocol.
        for name in ("training_episodes.csv", "trajectories.npy", "metrics.json"):
            (directory / name).write_bytes((source / name).read_bytes())
        row = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        row.update({"learning_rate": lr, "seed": seed, "reused_baseline": True})
        (directory / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row

    env = DummyVecEnv([make_env(seed)])
    noise = NormalActionNoise(np.zeros(4, dtype=np.float32), 0.1 * np.ones(4, dtype=np.float32))
    model = TD3(
        "MlpPolicy", env, seed=seed, learning_rate=lr, buffer_size=100_000,
        learning_starts=1_000, batch_size=256, gamma=0.99, tau=0.005,
        train_freq=4, gradient_steps=1, action_noise=noise,
        policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
        policy_kwargs={"net_arch": [64, 64]}, verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=EpisodeCallback(directory / "training_episodes.csv"), progress_bar=False)
    model.save(directory / "model")
    env.close()
    trajectories, _ = evaluate(model, eval_episodes, seed=seed + 1000)
    np.save(directory / "trajectories.npy", np.asarray(trajectories, dtype=object), allow_pickle=True)
    row = {"algorithm": "TD3", "learning_rate": lr, "seed": seed, "timesteps": timesteps, "reused_baseline": reused, **metrics(trajectories)}
    (directory / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def read_rewards(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        return np.asarray([float(row["reward"]) for row in csv.DictReader(stream)])


METRICS = ("mean_return", "iae_mean_mol_m3", "settling_time_mean_s", "peak_abs_error_mol_m3", "steady_error_std_mol_m3", "action_variation_normalized")
LABELS = {
    "mean_return": "Mean return ↑",
    "iae_mean_mol_m3": "Mean concentration error ↓",
    "settling_time_mean_s": "Settling time (s) ↓",
    "peak_abs_error_mol_m3": "Peak error ↓",
    "steady_error_std_mol_m3": "Steady error std. ↓",
    "action_variation_normalized": "Action variation ↓",
}


def summarize(rows: list[dict], root: Path) -> list[dict]:
    summary = []
    for lr in LEARNING_RATES:
        selected = [row for row in rows if np.isclose(row["learning_rate"], lr)]
        item = {"learning_rate": lr}
        for key in METRICS:
            values = [row[key] for row in selected]
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std"] = float(np.std(values, ddof=1))
        summary.append(item)
    # Equal-weight rank across the six reported criteria; return is maximized,
    # all other criteria are minimized.
    for key in METRICS:
        ordered = sorted(summary, key=lambda item: item[f"{key}_mean"], reverse=key == "mean_return")
        for rank, item in enumerate(ordered, 1):
            item.setdefault("rank_sum", 0.0)
            item["rank_sum"] += rank
    best_rank = min(item["rank_sum"] for item in summary)
    for item in summary:
        item["mean_rank"] = item["rank_sum"] / len(METRICS)
        item["recommended"] = bool(item["rank_sum"] == best_rank)
        del item["rank_sum"]
    (root / "td3_learning_rate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["learning_rate"] + [x for key in METRICS for x in (f"{key}_mean", f"{key}_std")] + ["mean_rank", "recommended"]
    with (root / "td3_learning_rate_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# TD3 learning-rate sweep", "", "Mean ± sample standard deviation over seeds 11, 22, and 33; each evaluation uses 30 deterministic episodes.", "", "| Learning rate | " + " | ".join(LABELS.values()) + " | Mean rank | Recommended |", "|---:|" + "---:|" * (len(METRICS) + 2)]
    for item in summary:
        values = [f"{item[key + '_mean']:.3f} ± {item[key + '_std']:.3f}" for key in METRICS]
        lines.append("| " + f"{item['learning_rate']:.0e}" + " | " + " | ".join(values) + f" | {item['mean_rank']:.2f} | {'Yes' if item['recommended'] else ''} |")
    (root / "td3_learning_rate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def plot_sweep(rows: list[dict], root: Path, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), dpi=170, constrained_layout=True)
    axes = axes.ravel()
    palette = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(LEARNING_RATES)))
    for color, lr in zip(palette, LEARNING_RATES):
        curves = [read_rewards(root / lr_label(lr) / f"seed_{seed}" / "training_episodes.csv") for seed in DEFAULT_SEEDS]
        window = min(50, max(5, min(map(len, curves)) // 20))
        smooth = [np.convolve(curve, np.ones(window) / window, mode="valid") for curve in curves]
        n = min(map(len, smooth))
        data = np.stack([curve[:n] for curve in smooth])
        x = np.arange(1, n + 1)
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        axes[0].plot(x, mean, color=color, linewidth=2, label=f"{lr:.0e}")
        axes[0].fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    axes[0].axhline(0, color="#555", linestyle="--", linewidth=1)
    axes[0].set(title="Episode-return learning curves", xlabel="Episode", ylabel="Return")
    axes[0].legend(title="Learning rate", frameon=False, ncol=2)

    summary = summarize(rows, root)
    for ax, key in zip(axes[1:], METRICS):
        x = np.asarray([item["learning_rate"] for item in summary])
        y = np.asarray([item[f"{key}_mean"] for item in summary])
        err = np.asarray([item[f"{key}_std"] for item in summary])
        ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.8, capsize=3, color="#1f77b4")
        for item in summary:
            if item["recommended"]:
                ax.scatter(item["learning_rate"], item[f"{key}_mean"], s=80, facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=3)
        ax.set_xscale("log")
        ax.set_title(LABELS[key])
        ax.set_xlabel("Learning rate")
        ax.grid(True, alpha=0.25)
    fig.suptitle("TD3 learning-rate hyperparameter sweep — three-segment two-CSTR task", fontsize=15)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "td3_learning_rate_sweep_v1")
    parser.add_argument("--reuse-root", type=Path, default=Path(__file__).parent / "results" / "multiseed_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(lr, seed) for lr in LEARNING_RATES for seed in DEFAULT_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_one, lr, seed, args.timesteps, args.eval_episodes, args.output_dir, args.reuse_root) for lr, seed in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: (LEARNING_RATES.index(next(lr for lr in LEARNING_RATES if np.isclose(lr, row["learning_rate"]))), row["seed"]))
    (args.output_dir / "td3_learning_rate_per_seed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = summarize(rows, args.output_dir)
    plot_sweep(rows, args.output_dir, args.output_dir / "td3_learning_rate_sweep.png")
    best_lr = next(item["learning_rate"] for item in summary if item["recommended"])
    trajectories = []
    for seed in DEFAULT_SEEDS:
        path = args.output_dir / lr_label(best_lr) / f"seed_{seed}" / "trajectories.npy"
        trajectories.extend(np.load(path, allow_pickle=True).tolist())
    plot_effect(trajectories, args.output_dir / "td3_best_learning_rate_effect.png", f"TD3 (lr={best_lr:.0e})")
    print(json.dumps({"recommended_learning_rate": best_lr}, indent=2))


if __name__ == "__main__":
    main()
