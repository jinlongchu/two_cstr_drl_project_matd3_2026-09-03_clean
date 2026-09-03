"""Train and evaluate an environment-specific improved DDPG controller.

The controller keeps a single DDPG critic (so it remains DDPG rather than
turning into TD3) and adds: adaptive OU exploration, event-prioritized replay,
five-step observation history, and a residual action policy around a simple
concentration-error baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from train_ppo_three_segment import (
    C2_START,
    HORIZON,
    RECYCLE_START,
    TARGET_SCHEDULE,
    plot_effect,
    plot_learning_curve,
)
from two_cstr_env import TwoCSTRStageOffsetEnv


ENV_KWARGS = dict(
    target_schedule=TARGET_SCHEDULE,
    c2_start_offset=C2_START,
    recycle_start_offset=RECYCLE_START,
    horizon=HORIZON,
    dt=1.0,
    initial_concentration_noise=0.8,
    initial_temperature_noise=1.5,
)


class ResidualActionWrapper(gym.ActionWrapper):
    """Map policy actions to bounded residual corrections around a baseline."""

    def __init__(self, env: gym.Env, residual_scale=(0.35, 0.35, 0.55, 0.55)):
        super().__init__(env)
        self.residual_scale = np.asarray(residual_scale, dtype=np.float64)
        self.action_space = spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32), dtype=np.float32)

    @staticmethod
    def _baseline(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        previous = obs[6:10].copy()
        e1 = obs[0] - obs[4]
        e2 = obs[2] - obs[5] if obs[10] > 0.0 else 0.0
        # Concentration above target calls for colder coolant
        # (lower normalized Tc) to slow the exothermic reaction.
        baseline = previous.copy()
        baseline[0] = np.clip(previous[0] - 0.12 * e1, -1.0, 1.0)
        baseline[1] = np.clip(previous[1] - 0.08 * e2, -1.0, 1.0)
        baseline[2] = np.clip(previous[2] - 0.55 * e1, -1.0, 1.0)
        baseline[3] = np.clip(previous[3] - 0.55 * e2, -1.0, 1.0)
        if obs[11] < 0.0:
            baseline[1] = -1.0
        if obs[10] < 0.0:
            baseline[3] = -1.0
        return baseline

    def action(self, action: np.ndarray) -> np.ndarray:
        obs = self.env.unwrapped._observation()
        baseline = self._baseline(obs)
        return np.clip(baseline + self.residual_scale * np.asarray(action, dtype=np.float64), -1.0, 1.0).astype(np.float32)


class HistoryObservationWrapper(gym.ObservationWrapper):
    """Stack recent observations so DDPG can infer lag and disturbances."""

    def __init__(self, env: gym.Env, history: int = 5):
        super().__init__(env)
        self.history = int(history)
        if self.history < 2:
            raise ValueError("history must be at least 2")
        self._frames: list[np.ndarray] = []
        shape = (self.history * int(np.prod(env.observation_space.shape)),)
        self.observation_space = spaces.Box(low=-np.ones(shape, dtype=np.float32), high=np.ones(shape, dtype=np.float32), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation, dtype=np.float32)
        self._frames.append(frame.copy())
        self._frames = self._frames[-self.history :]
        while len(self._frames) < self.history:
            self._frames.insert(0, frame.copy())
        return np.concatenate(self._frames).astype(np.float32)

    def reset(self, **kwargs):
        self._frames = []
        return super().reset(**kwargs)


class AdaptiveOUNoise(ActionNoise):
    """Correlated exploration whose standard deviation decays over training."""

    def __init__(self, size: int, sigma_start: float = 0.20, sigma_end: float = 0.025, decay_steps: int = 90_000):
        self.size = size
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps
        self.theta = 0.15
        self.dt = 1.0
        self.noise_prev = np.zeros(size, dtype=np.float32)
        self.calls = 0

    def reset(self) -> None:
        self.noise_prev.fill(0.0)

    def __call__(self) -> np.ndarray:
        progress = min(self.calls / self.decay_steps, 1.0)
        sigma = self.sigma_end + (self.sigma_start - self.sigma_end) * (1.0 - progress)
        self.noise_prev += self.theta * (-self.noise_prev) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.calls += 1
        return self.noise_prev.astype(np.float32)


class EventPrioritizedReplayBuffer(ReplayBuffer):
    """Prioritize high-error and target-transition samples.

    SB3's standard DDPG does not expose TD errors to the replay buffer.  We
    therefore use an online proxy priority based on reward magnitude and the
    environment's explicit target/handoff event flags.  Sampling is genuinely
    non-uniform and uses the same replay tensors as SB3.
    """

    def __init__(self, *args, alpha: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.priorities = np.ones(self.buffer_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        index = self.pos
        event = any(bool(info.get("target_switch_event", False) or info.get("c2_handoff", False) or info.get("recycle_event", False)) for info in infos)
        base = float(np.mean(np.abs(np.asarray(reward)))) + 0.05
        priority = base * (4.0 if event else 1.0)
        super().add(obs, next_obs, action, reward, done, infos)
        self.priorities[index] = priority
        self.max_priority = max(self.max_priority, float(priority))

    def sample(self, batch_size: int, env=None):
        size = self.buffer_size if self.full else self.pos
        # A cumulative distribution is materially faster than np.random.choice
        # with a probability vector for this long-horizon environment.
        weights = np.maximum(self.priorities[:size], 1e-6) ** self.alpha
        cdf = np.cumsum(weights)
        draws = np.random.random(batch_size) * cdf[-1]
        indices = np.searchsorted(cdf, draws, side="right")
        if size < batch_size:
            indices = np.random.randint(0, size, size=batch_size)
        return self._get_samples(indices, env=env)


class EpisodeCallback(BaseCallback):
    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.rewards = []
        self.lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
        return True

    def _on_training_end(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "reward", "length"])
            writer.writerows(zip(range(1, len(self.rewards) + 1), self.rewards, self.lengths))


def make_env(seed: int, improved: bool = True):
    def factory():
        env: gym.Env = TwoCSTRStageOffsetEnv(**ENV_KWARGS)
        if improved:
            env = ResidualActionWrapper(env)
            env = HistoryObservationWrapper(env, history=5)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return factory


def evaluate_model(model, episodes: int, seed: int, improved: bool):
    trajectories = []
    for episode in range(episodes):
        env = make_env(seed + episode, improved)()
        obs, reset_info = env.reset(seed=seed + episode)
        states = [reset_info["state"].copy()]
        targets = [reset_info["target"].copy()]
        c2_valid = [False]
        actions, commands, rewards = [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(info["state"].copy())
            targets.append(info["target"].copy())
            c2_valid.append(bool(info["c2_output_valid"]))
            actions.append(np.asarray([np.nan if v is None else float(v) for v in info["action_physical_available"]]))
            commands.append(np.asarray([np.nan if v is None else float(v) for v in info["action_command_available"]]))
            rewards.append(float(reward))
            done = terminated or truncated
        trajectories.append({"states": np.asarray(states), "targets": np.asarray(targets), "c2_valid": np.asarray(c2_valid), "actions": np.asarray(actions), "commands": np.asarray(commands), "rewards": np.asarray(rewards)})
        env.close()
    return trajectories


def metrics(trajectories) -> dict[str, float]:
    iae, settling, peak_error, steady_std, action_variation = [], [], [], [], []
    for item in trajectories:
        states, targets = item["states"], item["targets"]
        valid = item["c2_valid"]
        e1 = np.abs(states[:, 0] - targets[:, 0])
        e2 = np.where(valid, np.abs(states[:, 2] - targets[:, 1]), np.nan)
        iae.append(float(np.nanmean(e1 + e2)))
        peak_error.append(float(max(np.nanmax(e1[60:]), np.nanmax(e2[60:]))))
        steady_std.append(float(np.nanmean([np.std(e1[-30:]), np.nanstd(e2[-30:])])))
        settle_times = []
        for switch in (60, 120):
            found = HORIZON - switch
            for i in range(switch, HORIZON - 9):
                window1 = e1[i : i + 10]
                window2 = e2[i : i + 10]
                if np.all(window1 <= 0.5) and np.all(np.isfinite(window2)) and np.all(window2 <= 0.5):
                    found = i - switch
                    break
            settle_times.append(found)
        settling.append(float(np.mean(settle_times)))
        act = item["actions"]
        diffs = np.abs(np.diff(act, axis=0))
        action_variation.append(float(np.nanmean(diffs / np.asarray([3e-4, 2e-4, 40.0, 40.0]))))
    return {
        "mean_return": float(np.mean([np.sum(item["rewards"]) for item in trajectories])),
        "iae_mean_mol_m3": float(np.mean(iae)),
        "settling_time_mean_s": float(np.mean(settling)),
        "peak_abs_error_mol_m3": float(np.mean(peak_error)),
        "steady_error_std_mol_m3": float(np.mean(steady_std)),
        "action_variation_normalized": float(np.mean(action_variation)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "improved_ddpg")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args.seed, improved=True)])
    noise = AdaptiveOUNoise(4)
    model = DDPG(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=args.buffer_size, learning_starts=2_000,
        batch_size=args.batch_size, gamma=0.99, tau=0.002, train_freq=4, gradient_steps=1,
        action_noise=noise, replay_buffer_class=EventPrioritizedReplayBuffer,
        policy_kwargs={"net_arch": [128, 128]}, seed=args.seed, verbose=1,
    )
    callback = EpisodeCallback(args.output_dir / "training_episodes.csv")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.output_dir / "improved_ddpg")
    env.close()

    improved_trajectories = evaluate_model(model, args.eval_episodes, args.seed + 1000, improved=True)
    plot_learning_curve(args.output_dir / "training_episodes.csv", args.output_dir / "learning_curve.png", "Improved DDPG")
    plot_effect(improved_trajectories, args.output_dir / "improved_ddpg_effect.png", "Improved DDPG")

    models = {
        "PPO": (PPO, Path(__file__).parent / "results" / "ppo_three_segment_targets_v3" / "ppo_three_segment.zip", False),
        "SAC": (SAC, Path(__file__).parent / "results" / "sac_three_segment_targets_v3" / "sac_three_segment.zip", False),
        "DDPG": (DDPG, Path(__file__).parent / "results" / "ddpg_three_segment_targets_v3" / "ddpg_three_segment.zip", False),
        "TD3": (TD3, Path(__file__).parent / "results" / "td3_three_segment_targets_v3_optimized" / "td3_three_segment.zip", False),
        "Improved DDPG": (DDPG, args.output_dir / "improved_ddpg.zip", True),
    }
    comparison = {}
    for name, (cls, path, is_improved) in models.items():
        loaded = cls.load(path)
        comparison[name] = metrics(evaluate_model(loaded, args.eval_episodes, args.seed + 1000, improved=is_improved))
    (args.output_dir / "comparison_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["algorithm", *next(iter(comparison.values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in comparison.items():
            writer.writerow({"algorithm": name, **row})
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
