"""Train PPO on the staged two-CSTR concentration task.

The environment has no CSTR2 concentration output for t < 10 s, performs the
exact handoff C2(10) = C1(10), and enables the CSTR2-to-CSTR1 recycle at t=20 s.
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

from two_cstr_env import TwoCSTRStageOffsetEnv


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
        env = TwoCSTRStageOffsetEnv(
            target=(94.0, 92.0),
            c2_start_offset=10,
            recycle_start_offset=20,
            horizon=120,
            dt=1.0,
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
            target=(94.0, 92.0),
            c2_start_offset=10,
            recycle_start_offset=20,
            horizon=120,
            dt=1.0,
        )
        obs, reset_info = env.reset(seed=seed + episode)
        states = [reset_info["state"].copy()]
        c2_valid = [bool(reset_info["c2_output_valid"])]
        recycle_valid = [False]
        actions = []
        rewards = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(info["state"].copy())
            c2_valid.append(bool(info["c2_output_valid"]))
            recycle_valid.append(bool(info["recycle_active"]))
            available_action = [np.nan if value is None else float(value) for value in info["action_physical_available"]]
            actions.append(np.asarray(available_action, dtype=np.float64))
            rewards.append(float(reward))
            done = terminated or truncated
        trajectories.append(
            {
                "states": np.asarray(states),
                "c2_valid": np.asarray(c2_valid, dtype=bool),
                "recycle_valid": np.asarray(recycle_valid, dtype=bool),
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
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("PPO learning curve — staged two-CSTR environment")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_effect(trajectories, output_path: Path) -> None:
    states = np.stack([item["states"] for item in trajectories])
    c2_valid = np.stack([item["c2_valid"] for item in trajectories])
    actions = np.stack([item["actions"] for item in trajectories])
    t_state = np.arange(states.shape[1])
    # Actions are selected from the state at integer time t and held over
    # [t, t+1], so plot their command time rather than the post-step index.
    t_action = np.arange(actions.shape[1])
    target_c1, target_c2 = 94.0, 92.0
    fig = plt.figure(figsize=(13.0, 8.0), dpi=160)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    def band(ax, values, color, label, time):
        valid_columns = np.any(np.isfinite(values), axis=0)
        median = np.full(values.shape[1], np.nan)
        lo = np.full(values.shape[1], np.nan)
        hi = np.full(values.shape[1], np.nan)
        median[valid_columns] = np.nanmedian(values[:, valid_columns], axis=0)
        lo[valid_columns] = np.nanmin(values[:, valid_columns], axis=0)
        hi[valid_columns] = np.nanmax(values[:, valid_columns], axis=0)
        ax.fill_between(time, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(time, median, color=color, linewidth=2, label=label)

    # CSTR2 has no valid concentration before t=10 s; NaNs prevent a line
    # from being drawn in that interval.
    c2 = np.where(c2_valid, states[:, :, 2], np.nan)
    t2 = np.where(c2_valid, states[:, :, 3], np.nan)
    band(axes[0], states[:, :, 0], "#ff7f0e", r"$C_1$ (CSTR1)", t_state)
    band(axes[0], c2, "#d62728", r"$C_2$ (CSTR2)", t_state)
    axes[0].axhline(target_c1, color="#ff7f0e", linestyle="--", linewidth=1.2, label=r"$C_1^*$")
    axes[0].plot([10, 120], [target_c2, target_c2], color="#333333", linestyle="--", linewidth=1.2, label=r"$C_2^*$ (from 10 s)")
    axes[0].axvline(10, color="#555555", linestyle=":", linewidth=1)
    axes[0].set_title("Concentration: CSTR1 → CSTR2")
    axes[0].set_ylabel("Concentration (mol/m³)")
    axes[0].set_xlabel("Time (s)")

    band(axes[1], states[:, :, 1], "#2ca02c", r"$T_1$ (CSTR1)", t_state)
    band(axes[1], t2, "#1f77b4", r"$T_2$ (CSTR2)", t_state)
    axes[1].axvline(10, color="#555555", linestyle=":", linewidth=1)
    axes[1].set_title("Calculated temperature (no temperature target)")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_xlabel("Time (s)")

    # Scale flows by 1e4 so F and L are readable on the same axis.
    band(axes[2], actions[:, :, 0] * 1.0e4, "#9467bd", r"$F$", t_action)
    band(axes[2], actions[:, :, 1] * 1.0e4, "#8c564b", r"$L$", t_action)
    axes[2].axvline(20, color="#555555", linestyle=":", linewidth=1, label="L available from 20 s")
    axes[2].set_title("Flow actions")
    axes[2].set_ylabel(r"Flow ($10^{-4}$ m³/s)")
    axes[2].set_xlabel("Time (s)")

    band(axes[3], actions[:, :, 2], "#ff7f0e", r"$T_{c1}$", t_action)
    band(axes[3], actions[:, :, 3], "#2ca02c", r"$T_{c2}$", t_action)
    axes[3].axvline(10, color="#555555", linestyle=":", linewidth=1, label="CSTR2 starts")
    axes[3].set_title("Cooling actions")
    axes[3].set_ylabel("Cooling temperature (K)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("PPO staged two-CSTR response — median and min/max over evaluations", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "ppo_stage_offset")
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
    model.save(args.output_dir / "ppo_stage_offset")
    env.close()

    trajectories, returns = evaluate(model, args.eval_episodes, seed=args.seed + 1000)
    plot_learning_curve(monitor_csv, args.output_dir / "learning_curve.png")
    plot_effect(trajectories, args.output_dir / "ppo_stage_offset_effect.png")

    target_c1, target_c2 = 94.0, 92.0
    final_states = np.stack([item["states"][-1] for item in trajectories])
    handoff_errors = np.asarray(
        [abs(item["states"][10, 2] - item["states"][10, 0]) for item in trajectories]
    )
    success = np.logical_and(
        np.abs(final_states[:, 0] - target_c1) < 2.0,
        np.abs(final_states[:, 2] - target_c2) < 2.0,
    )
    summary = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "target": {"C1": 94.0, "C2": 92.0},
        "c2_start_offset_s": 10,
        "recycle_start_offset_s": 20,
        "horizon_s": 120,
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
