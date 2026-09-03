"""Train AgileRL MATD3 on the distributed two-CSTR environment.

The two actors control separate actuator groups, while MATD3 learns a
centralized pair of critics from synchronized joint transitions:

    cstr1 -> [F, Tc1]
    cstr2 -> [L, Tc2]

Only episode-return learning curves and process-response figures are written;
no per-step reward figure is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from agilerl.algorithms import MATD3
from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.utils import make_multi_agent_vect_envs

from two_cstr_env import TwoCSTRParallelEnv


AGENTS = ("cstr1", "cstr2")
TARGET_SCHEDULE = ((96.0, 94.5), (94.0, 92.0), (95.0, 93.0))
C2_START = 10
RECYCLE_START = 20
HORIZON = 180


def env_kwargs() -> dict:
    return {
        "target_schedule": TARGET_SCHEDULE,
        "c2_start_offset": C2_START,
        "recycle_start_offset": RECYCLE_START,
        "horizon": HORIZON,
        "dt": 1.0,
        "initial_concentration_noise": 0.8,
        "initial_temperature_noise": 1.5,
        "cooling_valve_tau_s": 5.0,
        "feed_disturbance_rho": 0.98,
        "sensor_concentration_std": 0.05,
        "sensor_temperature_std": 0.12,
        "flow_actuator_noise_fraction": 0.015,
        "cooling_actuator_noise_std": 0.20,
    }


def build_agent(env, seed: int, batch_size: int, num_envs: int, device: str) -> MATD3:
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    return MATD3.population(
        size=1,
        observation_space=observation_spaces,
        action_space=action_spaces,
        agent_ids=list(env.agents),
        net_config={
            "latent_dim": 64,
            "encoder_config": {"hidden_size": [64]},
            "head_config": {"hidden_size": [64]},
        },
        device=device,
        vect_noise_dim=int(num_envs),
        batch_size=int(batch_size),
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        learn_step=4,
        policy_freq=2,
        expl_noise=0.10,
        O_U_noise=True,
    )[0]


def write_episode_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "reward", "length"])
        writer.writeheader()
        writer.writerows(records)


def train(args: argparse.Namespace) -> tuple[MATD3, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_multi_agent_vect_envs(
        env=TwoCSTRParallelEnv,
        num_envs=args.num_envs,
        **env_kwargs(),
    )
    agent = build_agent(env, args.seed, args.batch_size, args.num_envs, device)
    memory = ReplayBuffer(max_size=args.buffer_size, device=device)
    observations, infos = env.reset(seed=args.seed)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_records: list[dict] = []
    total_steps = 0
    next_report = args.report_every
    agent.set_training_mode(True)

    while total_steps < args.timesteps:
        processed_action, raw_action = agent.get_action(obs=observations, infos=infos)
        next_observations, rewards, terminations, truncations, next_infos = env.step(processed_action)
        done = {
            agent_id: np.logical_or(terminations[agent_id], truncations[agent_id])
            for agent_id in AGENTS
        }
        transition = MultiAgentTransition(
            obs=observations,
            action=raw_action,
            reward=rewards,
            next_obs=next_observations,
            done=done,
            batch_size=[args.num_envs],
        ).to_tensordict()
        memory.add(transition)

        episode_returns += np.asarray(rewards["cstr1"], dtype=np.float64)
        episode_lengths += 1
        done_mask = np.asarray(done["cstr1"], dtype=bool)
        for index in np.flatnonzero(done_mask):
            episode_records.append(
                {
                    "episode": len(episode_records) + 1,
                    "reward": float(episode_returns[index]),
                    "length": int(episode_lengths[index]),
                }
            )
            episode_returns[index] = 0.0
            episode_lengths[index] = 0
        agent.reset_action_noise(np.flatnonzero(done_mask).tolist())

        total_steps += args.num_envs
        if len(memory) >= args.batch_size:
            # learn_step=4 and num_envs=4 give one MATD3 update per vector step.
            for _ in range(max(1, args.num_envs // agent.learn_step)):
                agent.learn(memory.sample(args.batch_size))

        observations, infos = next_observations, next_infos
        if total_steps >= next_report:
            recent = episode_records[-10:]
            recent_return = float(np.mean([item["reward"] for item in recent])) if recent else float("nan")
            print(
                f"steps={total_steps}/{args.timesteps} episodes={len(episode_records)} "
                f"recent_return={recent_return:.3f} replay={len(memory)}",
                flush=True,
            )
            next_report += args.report_every

    write_episode_csv(args.output_dir / "training_episodes.csv", episode_records)
    agent.save_checkpoint(args.output_dir / "matd3_distributed.pt")
    env.close()
    return agent, args.output_dir


def _as_available(values) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def evaluate(agent: MATD3, episodes: int, seed: int) -> list[dict]:
    env = TwoCSTRParallelEnv(**env_kwargs())
    agent.set_training_mode(False)
    trajectories = []
    for episode in range(episodes):
        observations, infos = env.reset(seed=seed + episode)
        initial_info = infos["cstr1"]["base_info"]
        states = [np.asarray(initial_info["state"], dtype=np.float64).copy()]
        targets = [np.asarray(initial_info["target"], dtype=np.float64).copy()]
        c2_valid = [False]
        actions = []
        commands = []
        rewards = []
        done = False
        while not done:
            processed_action, _ = agent.get_action(obs=observations, infos=infos)
            scalar_actions = {
                agent_id: np.asarray(processed_action[agent_id]).reshape(-1, 2)[0]
                for agent_id in AGENTS
            }
            observations, reward, terminations, truncations, infos = env.step(scalar_actions)
            base_info = infos["cstr1"]["base_info"]
            states.append(np.asarray(base_info["state"], dtype=np.float64).copy())
            targets.append(np.asarray(base_info["target"], dtype=np.float64).copy())
            c2_valid.append(bool(base_info["c2_output_valid"]))
            actions.append(_as_available(base_info["action_physical_available"]))
            commands.append(_as_available(base_info["action_command_available"]))
            rewards.append(float(reward["cstr1"]))
            done = bool(terminations["cstr1"] or truncations["cstr1"])
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
    env.close()
    agent.set_training_mode(True)
    return trajectories


def calculate_metrics(trajectories: list[dict]) -> dict[str, float]:
    iae, settling, peak_error, steady_std, action_variation = [], [], [], [], []
    for item in trajectories:
        states = item["states"]
        targets = item["targets"]
        valid = item["c2_valid"]
        e1 = np.abs(states[:, 0] - targets[:, 0])
        e2 = np.where(valid, np.abs(states[:, 2] - targets[:, 1]), np.nan)
        iae.append(float(np.nanmean(e1 + e2)))
        peak_error.append(float(max(np.nanmax(e1[60:]), np.nanmax(e2[60:]))))
        steady_std.append(float(np.nanmean([np.std(e1[-30:]), np.nanstd(e2[-30:])])))
        settle_times = []
        for switch in (60, 120):
            found = HORIZON - switch
            for index in range(switch, HORIZON - 9):
                w1 = e1[index : index + 10]
                w2 = e2[index : index + 10]
                if np.all(w1 <= 0.5) and np.all(np.isfinite(w2)) and np.all(w2 <= 0.5):
                    found = index - switch
                    break
            settle_times.append(found)
        settling.append(float(np.mean(settle_times)))
        diffs = np.abs(np.diff(item["actions"], axis=0))
        action_variation.append(float(np.nanmean(diffs / np.asarray([3e-4, 2e-4, 40.0, 40.0]))))
    return {
        "mean_return": float(np.mean([np.sum(item["rewards"]) for item in trajectories])),
        "iae_mean_mol_m3": float(np.mean(iae)),
        "settling_time_mean_s": float(np.mean(settling)),
        "peak_abs_error_mol_m3": float(np.mean(peak_error)),
        "steady_error_std_mol_m3": float(np.mean(steady_std)),
        "action_variation_normalized": float(np.mean(action_variation)),
    }


def _band(ax, values: np.ndarray, color: str, label: str, time: np.ndarray, linestyle: str = "-") -> None:
    valid_columns = np.any(np.isfinite(values), axis=0)
    median = np.full(values.shape[1], np.nan)
    low = np.full(values.shape[1], np.nan)
    high = np.full(values.shape[1], np.nan)
    median[valid_columns] = np.nanmedian(values[:, valid_columns], axis=0)
    low[valid_columns] = np.nanmin(values[:, valid_columns], axis=0)
    high[valid_columns] = np.nanmax(values[:, valid_columns], axis=0)
    ax.fill_between(time, low, high, color=color, alpha=0.14, linewidth=0)
    ax.plot(time, median, color=color, linewidth=2.0, linestyle=linestyle, label=label)


def plot_effect(trajectories: list[dict], output_path: Path) -> None:
    states = np.stack([item["states"] for item in trajectories])
    targets = np.stack([item["targets"] for item in trajectories])
    valid = np.stack([item["c2_valid"] for item in trajectories])
    actions = np.stack([item["actions"] for item in trajectories])
    commands = np.stack([item["commands"] for item in trajectories])
    t_state = np.arange(states.shape[1])
    t_action = np.arange(actions.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), dpi=170, constrained_layout=True)
    axes = axes.ravel()

    c2 = np.where(valid, states[:, :, 2], np.nan)
    t2 = np.where(valid, states[:, :, 3], np.nan)
    c2_target = np.where(valid, targets[:, :, 1], np.nan)
    _band(axes[0], states[:, :, 0], "#0072B2", r"$C_1$ (CSTR1)", t_state)
    _band(axes[0], c2, "#D55E00", r"$C_2$ (CSTR2)", t_state)
    _band(axes[0], targets[:, :, 0], "#6A3D9A", r"$C_1^*$", t_state, "--")
    _band(axes[0], c2_target, "#009E73", r"$C_2^*$", t_state, "--")
    axes[0].axvline(C2_START, color="#555", linestyle=":", linewidth=1)
    axes[0].axvline(60, color="#777", linestyle=":", linewidth=1)
    axes[0].axvline(120, color="#777", linestyle=":", linewidth=1)
    axes[0].set(title="CSTR concentration", xlabel="Time (s)", ylabel="Concentration (mol/m³)")

    _band(axes[1], states[:, :, 1], "#E69F00", r"$T_1$ (CSTR1)", t_state)
    _band(axes[1], t2, "#56B4E9", r"$T_2$ (CSTR2)", t_state)
    axes[1].axvline(C2_START, color="#555", linestyle=":", linewidth=1)
    axes[1].set(title="CSTR temperature", xlabel="Time (s)", ylabel="Temperature (K)")

    _band(axes[2], actions[:, :, 0] * 1.0e4, "#CC79A7", r"$F$", t_action)
    _band(axes[2], actions[:, :, 1] * 1.0e4, "#A65628", r"$L$", t_action)
    axes[2].axvline(RECYCLE_START, color="#555", linestyle=":", linewidth=1)
    axes[2].set(title="Flow actions", xlabel="Time (s)", ylabel=r"Flow ($10^{-4}$ m³/s)")

    _band(axes[3], actions[:, :, 2], "#0072B2", r"$T_{c1}$ actual", t_action)
    _band(axes[3], actions[:, :, 3], "#009E73", r"$T_{c2}$ actual", t_action)
    _band(axes[3], commands[:, :, 2], "#0072B2", r"$T_{c1}$ command", t_action, "--")
    _band(axes[3], commands[:, :, 3], "#009E73", r"$T_{c2}$ command", t_action, "--")
    axes[3].axvline(C2_START, color="#555", linestyle=":", linewidth=1)
    axes[3].set(title="CSTR cooling actions", xlabel="Time (s)", ylabel="Cooling temperature (K)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("MATD3 distributed two-CSTR response — median and min/max", y=1.02)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_learning(path: Path, output_path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    episodes = np.arange(1, len(rewards) + 1)
    window = min(50, max(5, len(rewards) // 20)) if len(rewards) else 5
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=170)
    ax.plot(episodes, rewards, color="#8FBBD9", alpha=0.35, linewidth=0.8, label="Episode return")
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(rewards) + 1), rolling, color="#0072B2", linewidth=2.1, label=f"Rolling mean ({window})")
    ax.axhline(0.0, color="#555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set(title="MATD3 episode-return learning curve", xlabel="Episode", ylabel="Team return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "matd3_distributed_v1",
    )
    args = parser.parse_args()
    agent, output_dir = train(args)
    trajectories = evaluate(agent, args.eval_episodes, args.seed + 1000)
    np.save(output_dir / "trajectories.npy", np.asarray(trajectories, dtype=object), allow_pickle=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(calculate_metrics(trajectories), indent=2),
        encoding="utf-8",
    )
    plot_learning(output_dir / "training_episodes.csv", output_dir / "learning_curve.png")
    plot_effect(trajectories, output_dir / "matd3_distributed_effect.png")
    print(f"Completed MATD3 run; results written to {output_dir}")


if __name__ == "__main__":
    main()
