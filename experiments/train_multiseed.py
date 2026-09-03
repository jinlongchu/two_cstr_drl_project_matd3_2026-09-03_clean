"""Train the five two-CSTR controllers with three random seeds.

The script produces one episode-return learning-curve comparison, one effect
figure per algorithm, and a mean +/- standard-deviation table across seeds.
No step-reward figure is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from train_improved_ddpg import (
    ENV_KWARGS,
    AdaptiveOUNoise,
    EventPrioritizedReplayBuffer,
    HistoryObservationWrapper,
    ResidualActionWrapper,
    evaluate_model,
    metrics,
)
from train_ppo_three_segment import evaluate, plot_effect
from two_cstr_env import TwoCSTRStageOffsetEnv


ALGORITHMS = ("PPO", "SAC", "DDPG", "TD3", "Improved DDPG")
DEFAULT_SEEDS = (11, 22, 33)
COLORS = {"PPO": "#1f77b4", "SAC": "#d62728", "DDPG": "#2ca02c", "TD3": "#9467bd", "Improved DDPG": "#ff7f0e"}


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


def make_env(seed: int, improved: bool):
    def factory():
        env = TwoCSTRStageOffsetEnv(**ENV_KWARGS)
        if improved:
            env = ResidualActionWrapper(env)
            env = HistoryObservationWrapper(env, history=5)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return factory


def build_model(algorithm: str, env, seed: int):
    if algorithm == "PPO":
        return PPO(
            "MlpPolicy", env, seed=seed, n_steps=1024, batch_size=256,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, policy_kwargs={"net_arch": [64, 64]}, verbose=0,
        )
    if algorithm == "Improved DDPG":
        return DDPG(
            "MlpPolicy", env, seed=seed, learning_rate=3e-4,
            buffer_size=20_000, learning_starts=2_000, batch_size=64,
            gamma=0.99, tau=0.002, train_freq=4, gradient_steps=1,
            action_noise=AdaptiveOUNoise(4),
            replay_buffer_class=EventPrioritizedReplayBuffer,
            policy_kwargs={"net_arch": [128, 128]}, verbose=0,
        )
    common = dict(
        policy="MlpPolicy", env=env, seed=seed, learning_rate=3e-4,
        buffer_size=100_000, learning_starts=1_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=4, gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]}, verbose=0,
    )
    if algorithm == "SAC":
        return SAC(ent_coef="auto", **common)
    noise = NormalActionNoise(np.zeros(4, dtype=np.float32), 0.1 * np.ones(4, dtype=np.float32))
    if algorithm == "DDPG":
        return DDPG(action_noise=noise, **common)
    return TD3(action_noise=noise, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, **common)


def train_one(algorithm: str, seed: int, timesteps: int, eval_episodes: int, root: Path) -> dict:
    improved = algorithm == "Improved DDPG"
    run_dir = root / algorithm.lower().replace(" ", "_") / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env(seed, improved)])
    model = build_model(algorithm, env, seed)
    callback = EpisodeCallback(run_dir / "training_episodes.csv")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(run_dir / "model")
    env.close()
    if improved:
        trajectories = evaluate_model(model, eval_episodes, seed + 1000, improved=True)
    else:
        trajectories, _ = evaluate(model, eval_episodes, seed=seed + 1000)
    row = {"algorithm": algorithm, "seed": seed, "timesteps": timesteps, **metrics(trajectories)}
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    np.save(run_dir / "trajectories.npy", np.asarray(trajectories, dtype=object), allow_pickle=True)
    return row


def read_rewards(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        return np.asarray([float(row["reward"]) for row in csv.DictReader(stream)])


def plot_learning(all_rows: list[dict], root: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=170, constrained_layout=True)
    for algorithm in ALGORITHMS:
        curves = []
        for seed in DEFAULT_SEEDS:
            path = root / algorithm.lower().replace(" ", "_") / f"seed_{seed}" / "training_episodes.csv"
            values = read_rewards(path)
            window = min(50, max(5, len(values) // 20))
            if len(values) >= window:
                values = np.convolve(values, np.ones(window) / window, mode="valid")
            curves.append(values)
        n = min(map(len, curves))
        data = np.stack([curve[:n] for curve in curves])
        x = np.arange(1, n + 1)
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        ax.plot(x, mean, color=COLORS[algorithm], linewidth=2.1, label=algorithm)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[algorithm], alpha=0.12, linewidth=0)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    ax.set(title="Five two-CSTR controllers — 3-seed learning curves", xlabel="Episode", ylabel="Return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def aggregate_table(rows: list[dict], root: Path) -> None:
    keys = ["mean_return", "iae_mean_mol_m3", "settling_time_mean_s", "peak_abs_error_mol_m3", "steady_error_std_mol_m3", "action_variation_normalized"]
    summary = []
    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        mean_row = {key: float(np.mean([row[key] for row in selected])) for key in keys}
        std_row = {key: float(np.std([row[key] for row in selected], ddof=1)) for key in keys}
        summary.append({"algorithm": algorithm, **{f"{key}_mean": mean_row[key] for key in keys}, **{f"{key}_std": std_row[key] for key in keys}})
    (root / "comparison_metrics_3seeds.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "comparison_metrics_3seeds.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["algorithm"] + [item for key in keys for item in (f"{key}_mean", f"{key}_std")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    labels = {
        "mean_return": "平均回报 ↑",
        "iae_mean_mol_m3": "平均浓度绝对误差 ↓",
        "settling_time_mean_s": "平均稳定时间 (s) ↓",
        "peak_abs_error_mol_m3": "峰值绝对误差 ↓",
        "steady_error_std_mol_m3": "稳态误差标准差 ↓",
        "action_variation_normalized": "归一化动作变化率 ↓",
    }
    lines = ["# 5 algorithms × 3 seeds comparison", "", "Values are mean ± sample standard deviation across seeds; each seed is evaluated over the same 30 deterministic episodes.", "", "| 算法 | " + " | ".join(labels.values()) + " |", "|---|" + "---:|" * len(keys)]
    for item in summary:
        cells = []
        for key in keys:
            cells.append(f"{item[key + '_mean']:.3f} ± {item[key + '_std']:.3f}")
        lines.append("| " + item["algorithm"] + " | " + " | ".join(cells) + " |")
    (root / "comparison_metrics_3seeds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "multiseed_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(algorithm, seed) for algorithm in ALGORITHMS for seed in DEFAULT_SEEDS]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_one, algorithm, seed, args.timesteps, args.eval_episodes, args.output_dir) for algorithm, seed in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: (ALGORITHMS.index(row["algorithm"]), row["seed"]))
    (args.output_dir / "per_seed_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_learning(rows, args.output_dir, args.output_dir / "multiseed_learning_curves.png")
    for algorithm in ALGORITHMS:
        trajectories = []
        for seed in DEFAULT_SEEDS:
            run_dir = args.output_dir / algorithm.lower().replace(" ", "_") / f"seed_{seed}"
            trajectories.extend(np.load(run_dir / "trajectories.npy", allow_pickle=True).tolist())
        plot_effect(trajectories, args.output_dir / f"{algorithm.lower().replace(' ', '_')}_multiseed_effect.png", algorithm)
    aggregate_table(rows, args.output_dir)
    print(f"Completed {len(rows)} runs; results written to {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
