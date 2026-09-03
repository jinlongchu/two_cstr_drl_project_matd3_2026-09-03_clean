"""Train and evaluate PPO on the fixed-target two-CSTR task.

Outputs are written to ``experiments/results/ppo_fixed_target``:
``learning_curve.png`` and ``ppo_fixed_target_effect.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from two_cstr_env import TwoCSTRFixedTargetEnv


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, output_csv: Path):
        super().__init__()
        self.output_csv = output_csv
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
        return True

    def _on_training_end(self) -> None:
        with self.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(zip(range(1, len(self.rewards) + 1), self.rewards, self.lengths))


def make_env(seed: int = 0):
    def _factory():
        env = TwoCSTRFixedTargetEnv(target=(92.0, 306.0), horizon=60, dt=1.0)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _factory


def evaluate(model: PPO, episodes: int, seed: int = 1000):
    trajectories = []
    returns = []
    for episode in range(episodes):
        env = TwoCSTRFixedTargetEnv(target=(92.0, 306.0), horizon=60, dt=1.0)
        obs, _ = env.reset(seed=seed + episode)
        states = [env._state.copy()]
        actions = []
        rewards = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(info["state"].copy())
            actions.append(info["action_physical"].copy())
            rewards.append(float(reward))
            done = terminated or truncated
        trajectories.append(
            {
                "states": np.asarray(states),
                "actions": np.asarray(actions),
                "rewards": np.asarray(rewards),
            }
        )
        returns.append(float(np.sum(rewards)))
    return trajectories, np.asarray(returns)


def plot_learning_curve(csv_path: Path, output_path: Path) -> None:
    data = []
    with csv_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            data.append(float(row["reward"]))
    rewards = np.asarray(data)
    episodes = np.arange(1, len(rewards) + 1)
    window = min(50, max(5, len(rewards) // 20)) if len(rewards) else 5
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid") if len(rewards) >= window else rewards
    rolling_x = episodes[window - 1 :] if len(rewards) >= window else episodes
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    ax.plot(episodes, rewards, color="#9bb8d6", alpha=0.32, linewidth=0.8, label="Episode reward")
    ax.plot(rolling_x, rolling, color="#1f77b4", linewidth=2.0, label=f"Rolling mean ({window})")
    # The reward is the negative tracking/action cost, so zero is the
    # theoretical upper bound (perfect tracking with no action movement).
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("PPO learning curve — two CSTR fixed target")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_effect(trajectories, output_path: Path) -> None:
    states = np.stack([item["states"] for item in trajectories])
    actions = np.stack([item["actions"] for item in trajectories])
    t_state = np.arange(states.shape[1])
    t_action = np.arange(actions.shape[1])
    target_c, target_t = 92.0, 306.0
    # Keep the reference-paper layout (process variables + manipulated
    # variables), but expose both tanks: C1/T1 are the upstream response and
    # C2/T2 are the downstream product response.
    fig = plt.figure(figsize=(13.0, 7.2), dpi=160)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.34, wspace=0.25)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]

    def band(ax, values, color, label, time):
        median = np.median(values, axis=0)
        lo = np.min(values, axis=0)
        hi = np.max(values, axis=0)
        ax.fill_between(time, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(time, median, color=color, linewidth=2, label=label)

    band(axes[0], states[:, :, 0], "#ff7f0e", r"$C_1$ (CSTR1)", t_state)
    band(axes[0], states[:, :, 2], "#d62728", r"$C_2$ (CSTR2)", t_state)
    axes[0].axhline(target_c, color="#333333", linestyle="--", linewidth=1.3, label=r"$C_2^*$")
    axes[0].set_ylabel("Concentration (mol/m³)")
    axes[0].set_title("Concentration response: CSTR1 → CSTR2")
    axes[0].set_xlabel("Time (s)")

    band(axes[1], states[:, :, 1], "#2ca02c", r"$T_1$ (CSTR1)", t_state)
    band(axes[1], states[:, :, 3], "#1f77b4", r"$T_2$ (CSTR2)", t_state)
    axes[1].axhline(target_t, color="#333333", linestyle="--", linewidth=1.3, label=r"$T_2^*$")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Temperature response: CSTR1 → CSTR2")
    axes[1].set_xlabel("Time (s)")

    for idx, (color, label) in enumerate([("#ff7f0e", r"$T_{c1}$"), ("#2ca02c", r"$T_{c2}$")]):
        median = np.median(actions[:, :, idx], axis=0)
        lo = np.min(actions[:, :, idx], axis=0)
        hi = np.max(actions[:, :, idx], axis=0)
        axes[2].fill_between(t_action, lo, hi, color=color, alpha=0.12, linewidth=0)
        axes[2].plot(t_action, median, color=color, linewidth=2, label=label)
    axes[2].set_ylabel("Cooling temperature (K)")
    axes[2].set_title("Manipulated variables: CSTR1 and CSTR2 cooling")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("PPO control response — median and min/max over evaluation episodes", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "ppo_fixed_target")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    monitor_csv = args.output_dir / "training_episodes.csv"
    env = DummyVecEnv([make_env(args.seed)])
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )
    callback = EpisodeRewardCallback(monitor_csv)
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.output_dir / "ppo_fixed_target")
    env.close()

    trajectories, returns = evaluate(model, args.eval_episodes, seed=args.seed + 1000)
    plot_learning_curve(monitor_csv, args.output_dir / "learning_curve.png")
    plot_effect(trajectories, args.output_dir / "ppo_fixed_target_effect.png")
    summary = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "target": {"C2": 92.0, "T2": 306.0},
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean([np.max(np.abs(t["states"][-1, [2, 3]] - np.array([92.0, 306.0]))) < np.array([2.0, 2.0]).max() for t in trajectories])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
