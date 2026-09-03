"""Train SAC or DDPG on the same realistic two-CSTR task used by PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from train_ppo_three_segment import (
    C2_START,
    HORIZON,
    RECYCLE_START,
    TARGET_SCHEDULE,
    evaluate,
    plot_effect,
    plot_learning_curve,
)
from two_cstr_env import TwoCSTRStageOffsetEnv


class RewardCallback(BaseCallback):
    def __init__(self, episode_csv: Path, step_csv: Path):
        super().__init__()
        self.episode_csv = episode_csv
        self.step_csv = step_csv
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_rewards = []
        self.step_timesteps = []

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
        import csv

        with self.episode_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(zip(range(1, len(self.episode_rewards) + 1), self.episode_rewards, self.episode_lengths))
        with self.step_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestep", "reward"])
            writer.writerows(zip(self.step_timesteps, self.step_rewards))


def make_env(seed: int):
    def factory():
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

    return factory


def evaluate_and_summarize(model, algorithm: str, episodes: int, seed: int, output_dir: Path):
    trajectories, returns = evaluate(model, episodes, seed=seed)
    plot_learning_curve(output_dir / "training_episodes.csv", output_dir / "learning_curve.png", algorithm)
    plot_effect(trajectories, output_dir / f"{algorithm.lower()}_three_segment_effect.png", algorithm)
    final_states = np.stack([item["states"][-1] for item in trajectories])
    handoff_errors = np.asarray(
        [abs(item["states"][C2_START, 2] - item["states"][C2_START, 0]) for item in trajectories]
    )
    target = np.asarray(TARGET_SCHEDULE[-1])
    success = np.logical_and(
        np.abs(final_states[:, 0] - target[0]) < 2.0,
        np.abs(final_states[:, 2] - target[1]) < 2.0,
    )
    summary = {
        "algorithm": algorithm,
        "timesteps": int(model.num_timesteps),
        "eval_episodes": episodes,
        "target_schedule": [list(pair) for pair in TARGET_SCHEDULE],
        "c2_start_offset_s": C2_START,
        "recycle_start_offset_s": RECYCLE_START,
        "horizon_s": HORIZON,
        "initial_concentration_noise": 0.8,
        "initial_temperature_noise": 1.5,
        "cooling_valve_tau_s": 5.0,
        "feed_disturbance": {"rho": 0.98, "concentration_std": 0.08, "temperature_std": 0.12},
        "sensor_noise_std": {"concentration": 0.05, "temperature": 0.12},
        "actuator_noise": {"flow_fraction": 0.015, "cooling_temperature_std": 0.20},
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate_final_tolerance_2": float(np.mean(success)),
        "mean_final_C1": float(np.mean(final_states[:, 0])),
        "mean_final_C2": float(np.mean(final_states[:, 2])),
        "max_handoff_error": float(np.max(handoff_errors)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["SAC", "DDPG", "TD3"], required=True)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env(args.seed)])
    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,
        # Update every four environment steps: equal interaction budget
        # across algorithms while keeping off-policy training practical.
        train_freq=args.train_freq,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        verbose=1,
    )
    if args.algorithm == "SAC":
        model = SAC(ent_coef="auto", **common)
    else:
        action_noise = NormalActionNoise(
            mean=np.zeros(4, dtype=np.float32), sigma=0.1 * np.ones(4, dtype=np.float32)
        )
        if args.algorithm == "DDPG":
            model = DDPG(action_noise=action_noise, **common)
        else:
            model = TD3(
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                **common,
            )
    callback = RewardCallback(args.output_dir / "training_episodes.csv", args.output_dir / "training_steps.csv")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.output_dir / f"{args.algorithm.lower()}_three_segment")
    env.close()
    evaluate_and_summarize(model, args.algorithm, args.eval_episodes, args.seed + 1000, args.output_dir)


if __name__ == "__main__":
    main()
