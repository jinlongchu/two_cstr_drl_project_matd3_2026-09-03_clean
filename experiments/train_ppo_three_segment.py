"""Train PPO on the three-segment staged two-CSTR concentration task."""

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

from two_cstr_env import TwoCSTRStageOffsetEnv


TARGET_SCHEDULE = ((96.0, 94.5), (94.0, 92.0), (95.0, 93.0))
C2_START = 10
RECYCLE_START = 20
HORIZON = 180


class RewardCallback(BaseCallback):
    def __init__(self, episode_csv: Path, step_csv: Path):
        super().__init__()
        self.episode_csv = episode_csv
        self.step_csv = step_csv
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.step_rewards: list[float] = []
        self.step_timesteps: list[int] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
        self.step_rewards.append(float(rewards[0]))
        self.step_timesteps.append(int(self.num_timesteps))
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True

    def _on_training_end(self) -> None:
        with self.episode_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(
                zip(range(1, len(self.episode_rewards) + 1), self.episode_rewards, self.episode_lengths)
            )
        with self.step_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestep", "reward"])
            writer.writerows(zip(self.step_timesteps, self.step_rewards))


def make_env(seed: int = 0):
    def _factory():
        env = TwoCSTRStageOffsetEnv(
            target_schedule=TARGET_SCHEDULE,
            c2_start_offset=C2_START,
            recycle_start_offset=RECYCLE_START,
            horizon=HORIZON,
            dt=1.0,
            initial_concentration_noise=0.8,
            initial_temperature_noise=1.5,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _factory


def evaluate(model: PPO, episodes: int, seed: int = 1000):
    trajectories = []
    returns = []
    for episode in range(episodes):
        env = TwoCSTRStageOffsetEnv(
            target_schedule=TARGET_SCHEDULE,
            c2_start_offset=C2_START,
            recycle_start_offset=RECYCLE_START,
            horizon=HORIZON,
            dt=1.0,
            initial_concentration_noise=0.8,
            initial_temperature_noise=1.5,
        )
        obs, reset_info = env.reset(seed=seed + episode)
        states = [reset_info["state"].copy()]
        targets = [reset_info["target"].copy()]
        c2_valid = [False]
        actions = []
        commands = []
        rewards = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(info["state"].copy())
            targets.append(info["target"].copy())
            c2_valid.append(bool(info["c2_output_valid"]))
            available_action = [np.nan if value is None else float(value) for value in info["action_physical_available"]]
            actions.append(np.asarray(available_action, dtype=np.float64))
            available_command = [np.nan if value is None else float(value) for value in info["action_command_available"]]
            commands.append(np.asarray(available_command, dtype=np.float64))
            rewards.append(float(reward))
            done = terminated or truncated
        trajectories.append(
            {
                "states": np.asarray(states),
                "targets": np.asarray(targets),
                "c2_valid": np.asarray(c2_valid, dtype=bool),
                "actions": np.asarray(actions),
                "commands": np.asarray(commands),
                "rewards": np.asarray(rewards),
            }
        )
        returns.append(float(np.sum(rewards)))
    return trajectories, np.asarray(returns)


def _rolling(values: np.ndarray, window: int):
    if len(values) < window:
        return np.arange(len(values)), values
    return np.arange(window - 1, len(values)), np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curve(csv_path: Path, output_path: Path, algorithm: str = "PPO") -> None:
    rewards = np.asarray([float(row["reward"]) for row in csv.DictReader(csv_path.open(encoding="utf-8"))])
    episodes = np.arange(1, len(rewards) + 1)
    window = min(50, max(5, len(rewards) // 20)) if len(rewards) else 5
    rolling_x, rolling = _rolling(rewards, window)
    if len(rewards) >= window:
        rolling_x = rolling_x + 1
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.plot(episodes, rewards, color="#9bb8d6", alpha=0.32, linewidth=0.8, label="Episode return")
    ax.plot(rolling_x, rolling, color="#1f77b4", linewidth=2.0, label=f"Rolling mean ({window})")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"{algorithm} episode-return learning curve — three-segment targets")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_step_reward_curve(csv_path: Path, output_path: Path, algorithm: str = "PPO") -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    timesteps = np.asarray([int(row["timestep"]) for row in rows])
    rewards = np.asarray([float(row["reward"]) for row in rows])
    window = min(300, max(20, len(rewards) // 30)) if len(rewards) else 20
    rolling_x, rolling = _rolling(rewards, window)
    if len(rewards) >= window:
        rolling_x = timesteps[window - 1 :]
    else:
        rolling_x = timesteps
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.plot(timesteps, rewards, color="#b8cbe0", alpha=0.25, linewidth=0.55, label="Step reward")
    ax.plot(rolling_x, rolling, color="#2ca02c", linewidth=1.8, label=f"Rolling mean ({window})")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Reward at step")
    ax.set_title(f"{algorithm} step-reward learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_effect(trajectories, output_path: Path, algorithm: str = "PPO") -> None:
    states = np.stack([item["states"] for item in trajectories])
    targets = np.stack([item["targets"] for item in trajectories])
    c2_valid = np.stack([item["c2_valid"] for item in trajectories])
    actions = np.stack([item["actions"] for item in trajectories])
    commands = np.stack([item["commands"] for item in trajectories])
    t_state = np.arange(states.shape[1])
    t_action = np.arange(actions.shape[1])
    fig = plt.figure(figsize=(13.0, 8.0), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    def band(ax, values, color, label, time, linestyle="-"):
        valid_columns = np.any(np.isfinite(values), axis=0)
        median = np.full(values.shape[1], np.nan)
        lo = np.full(values.shape[1], np.nan)
        hi = np.full(values.shape[1], np.nan)
        median[valid_columns] = np.nanmedian(values[:, valid_columns], axis=0)
        lo[valid_columns] = np.nanmin(values[:, valid_columns], axis=0)
        hi[valid_columns] = np.nanmax(values[:, valid_columns], axis=0)
        ax.fill_between(time, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(time, median, color=color, linewidth=2, linestyle=linestyle, label=label)

    c2 = np.where(c2_valid, states[:, :, 2], np.nan)
    t2 = np.where(c2_valid, states[:, :, 3], np.nan)
    c2_target = np.where(c2_valid, targets[:, :, 1], np.nan)
    band(axes[0], states[:, :, 0], "#ff7f0e", r"$C_1$ (CSTR1)", t_state)
    band(axes[0], c2, "#d62728", r"$C_2$ (CSTR2)", t_state)
    band(axes[0], targets[:, :, 0], "#7b2cbf", r"$C_1^*$", t_state)
    band(axes[0], c2_target, "#003049", r"$C_2^*$ (from 10 s)", t_state)
    axes[0].axvline(10, color="#555555", linestyle=":", linewidth=1)
    axes[0].axvline(60, color="#777777", linestyle=":", linewidth=1)
    axes[0].axvline(120, color="#777777", linestyle=":", linewidth=1)
    axes[0].set_title("CSTR concentration")
    axes[0].set_ylabel("Concentration (mol/m³)")
    axes[0].set_xlabel("Time (s)")

    band(axes[1], states[:, :, 1], "#2ca02c", r"$T_1$ (CSTR1)", t_state)
    band(axes[1], t2, "#1f77b4", r"$T_2$ (CSTR2)", t_state)
    axes[1].axvline(10, color="#555555", linestyle=":", linewidth=1)
    axes[1].set_title("CSTR temperature")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_xlabel("Time (s)")

    band(axes[2], actions[:, :, 0] * 1.0e4, "#9467bd", r"$F$", t_action)
    band(axes[2], actions[:, :, 1] * 1.0e4, "#8c564b", r"$L$", t_action)
    axes[2].axvline(20, color="#555555", linestyle=":", linewidth=1, label="L available from 20 s")
    axes[2].set_title("Flow actions")
    axes[2].set_ylabel(r"Flow ($10^{-4}$ m³/s)")
    axes[2].set_xlabel("Time (s)")

    band(axes[3], actions[:, :, 2], "#ff7f0e", r"$T_{c1}$ actual", t_action)
    band(axes[3], actions[:, :, 3], "#2ca02c", r"$T_{c2}$ actual", t_action)
    band(axes[3], commands[:, :, 2], "#ff7f0e", r"$T_{c1}$ command", t_action, linestyle="--")
    band(axes[3], commands[:, :, 3], "#2ca02c", r"$T_{c2}$ command", t_action, linestyle="--")
    axes[3].axvline(10, color="#555555", linestyle=":", linewidth=1, label="Tc2 available from 10 s")
    axes[3].set_title("CSTR cooling actions")
    axes[3].set_ylabel("Cooling temperature (K)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{algorithm} three-segment two-CSTR response — median and min/max", y=1.02)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "ppo_three_segment_realistic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = args.output_dir / "training_episodes.csv"
    step_csv = args.output_dir / "training_steps.csv"
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
    callback = RewardCallback(episode_csv, step_csv)
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.output_dir / "ppo_three_segment")
    env.close()

    trajectories, returns = evaluate(model, args.eval_episodes, seed=args.seed + 1000)
    plot_learning_curve(episode_csv, args.output_dir / "learning_curve.png")
    plot_effect(trajectories, args.output_dir / "ppo_three_segment_effect.png")

    final_states = np.stack([item["states"][-1] for item in trajectories])
    handoff_errors = np.asarray(
        [abs(item["states"][C2_START, 2] - item["states"][C2_START, 0]) for item in trajectories]
    )
    final_target = np.asarray(TARGET_SCHEDULE[-1])
    success = np.logical_and(
        np.abs(final_states[:, 0] - final_target[0]) < 2.0,
        np.abs(final_states[:, 2] - final_target[1]) < 2.0,
    )
    summary = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "target_schedule": [list(pair) for pair in TARGET_SCHEDULE],
        "c2_start_offset_s": C2_START,
        "recycle_start_offset_s": RECYCLE_START,
        "horizon_s": HORIZON,
        "initial_concentration_noise": 0.8,
        "initial_temperature_noise": 1.5,
        "cooling_valve_tau_s": 5.0,
        "feed_disturbance": {
            "rho": 0.98,
            "concentration_std": 0.08,
            "temperature_std": 0.12,
            "concentration_bound": 0.8,
            "temperature_bound": 1.2,
        },
        "sensor_noise_std": {"concentration": 0.05, "temperature": 0.12},
        "actuator_noise": {"flow_fraction": 0.015, "cooling_temperature_std": 0.20},
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate_final_tolerance_2": float(np.mean(success)),
        "mean_final_C1": float(np.mean(final_states[:, 0])),
        "mean_final_C2": float(np.mean(final_states[:, 2])),
        "max_handoff_error": float(np.max(handoff_errors)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
