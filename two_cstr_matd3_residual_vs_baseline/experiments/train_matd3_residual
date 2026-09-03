"""Residual MATD3 fine-tuning from the preserved MATD3 baseline.

The baseline actors remain frozen and provide the main control action.  Two
small residual heads learn bounded corrections around that action:

    a = clip(a_baseline + alpha * delta_a, -1, 1)

The residual heads are zero-initialized, so the initial deterministic policy
is exactly the saved baseline.  The environment, reward, observations,
exploration noise, and MATD3 twin-critic update are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from agilerl.algorithms import MATD3
from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.utils import make_multi_agent_vect_envs
from torch import nn

from train_matd3 import (
    AGENTS,
    HORIZON,
    build_agent,
    calculate_metrics,
    env_kwargs,
    plot_effect,
    plot_learning,
    write_episode_csv,
    _as_available,
)
from two_cstr_env import TwoCSTRParallelEnv


BASELINE_CHECKPOINT = Path(__file__).parent / "results" / "matd3_distributed_v1" / "matd3_distributed.pt"


class ResidualActor(nn.Module):
    def __init__(self, input_dim: int = 13, action_dim: int = 2, max_delta: float = 0.15) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.body = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        # Zero residual at initialization: deterministic behaviour starts at
        # the already-trained baseline policy.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.body(obs))


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - tau).add_(tau * source_param)


def _as_tensor_obs(agent: MATD3, observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return agent.preprocess_observation(observations)


def select_action(
    agent: MATD3,
    residuals: dict[str, ResidualActor],
    observations: dict[str, np.ndarray],
    explore: bool,
) -> dict[str, np.ndarray]:
    states = _as_tensor_obs(agent, observations)
    actions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for agent_id in AGENTS:
            base = agent.actors[agent_id](states[agent_id])
            correction = residuals[agent_id](states[agent_id])
            action = torch.clamp(base + correction, -1.0, 1.0)
            if explore:
                action = torch.clamp(action + agent.action_noise(agent_id), -1.0, 1.0)
            actions[agent_id] = action.cpu().numpy()
    return actions


def _joint_residual_action(
    agent: MATD3,
    residuals: dict[str, ResidualActor],
    states: dict[str, torch.Tensor],
    detach_other: str | None = None,
) -> torch.Tensor:
    outputs = []
    for agent_id in AGENTS:
        base = agent.actors[agent_id](states[agent_id])
        residual = residuals[agent_id](states[agent_id])
        action = torch.clamp(base + residual, -1.0, 1.0)
        if detach_other == agent_id:
            action = action.detach()
        outputs.append(action)
    return torch.cat(outputs, dim=1)


def update(
    agent: MATD3,
    residuals: dict[str, ResidualActor],
    target_residuals: dict[str, ResidualActor],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: dict[str, tuple[torch.optim.Optimizer, torch.optim.Optimizer]],
    batch,
    update_count: int,
    device: str,
    max_delta: float,
) -> tuple[float, float]:
    obs = {agent_id: batch["obs", agent_id].to(device) for agent_id in AGENTS}
    next_obs = {agent_id: batch["next_obs", agent_id].to(device) for agent_id in AGENTS}
    states = agent.preprocess_observation(obs)
    next_states = agent.preprocess_observation(next_obs)
    data_action = torch.cat([batch["action", agent_id].to(device) for agent_id in AGENTS], dim=1)
    rewards = {agent_id: batch["reward", agent_id].to(device).reshape(-1, 1) for agent_id in AGENTS}
    dones = {agent_id: batch["done", agent_id].to(device).reshape(-1, 1) for agent_id in AGENTS}

    with torch.no_grad():
        next_parts = []
        for agent_id in AGENTS:
            base = agent.actor_targets[agent_id](next_states[agent_id])
            correction = target_residuals[agent_id](next_states[agent_id])
            next_parts.append(torch.clamp(base + correction, -1.0, 1.0))
        next_action = torch.cat(next_parts, dim=1)

    critic_loss_total = 0.0
    current_qs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for agent_id in AGENTS:
        q1 = agent.critics_1[agent_id](states, data_action)
        q2 = agent.critics_2[agent_id](states, data_action)
        with torch.no_grad():
            next_q1 = agent.critic_targets_1[agent_id](next_states, next_action)
            next_q2 = agent.critic_targets_2[agent_id](next_states, next_action)
            target = rewards[agent_id] + (1.0 - dones[agent_id]) * agent.gamma * torch.minimum(next_q1, next_q2)
        loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        opt1, opt2 = critic_optimizers[agent_id]
        opt1.zero_grad()
        opt2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.critics_1[agent_id].parameters(), 5.0)
        nn.utils.clip_grad_norm_(agent.critics_2[agent_id].parameters(), 5.0)
        opt1.step()
        opt2.step()
        critic_loss_total += float(loss.detach().cpu())
        current_qs[agent_id] = (q1, q2)

    actor_loss_value = float("nan")
    if update_count % agent.policy_freq == 0:
        # Both agents optimize the cooperative team objective, while each
        # critic receives the same centralized joint action.
        joint_action = _joint_residual_action(agent, residuals, states)
        actor_loss = sum(-agent.critics_1[agent_id](states, joint_action).mean() for agent_id in AGENTS)
        # A very small residual regularizer prevents unnecessarily large
        # corrections when the baseline already tracks well.
        residual_penalty = sum(
            (residuals[agent_id](states[agent_id]) / max_delta).pow(2).mean() for agent_id in AGENTS
        )
        actor_loss = actor_loss + 0.005 * residual_penalty
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(list(residuals["cstr1"].parameters()) + list(residuals["cstr2"].parameters()), 5.0)
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.detach().cpu())
        for agent_id in AGENTS:
            _soft_update(residuals[agent_id], target_residuals[agent_id], agent.tau)
            _soft_update(agent.critics_1[agent_id], agent.critic_targets_1[agent_id], agent.tau)
            _soft_update(agent.critics_2[agent_id], agent.critic_targets_2[agent_id], agent.tau)

    return actor_loss_value, critic_loss_total / len(AGENTS)


def evaluate_residual(agent, residuals, episodes: int, seed: int) -> list[dict]:
    env = TwoCSTRParallelEnv(**env_kwargs())
    trajectories = []
    for episode in range(episodes):
        observations, infos = env.reset(seed=seed + episode)
        initial_info = infos["cstr1"]["base_info"]
        states = [np.asarray(initial_info["state"], dtype=np.float64).copy()]
        targets = [np.asarray(initial_info["target"], dtype=np.float64).copy()]
        c2_valid = [False]
        actions, commands, rewards = [], [], []
        done = False
        while not done:
            action = select_action(agent, residuals, {k: np.asarray(v)[None, :] for k, v in observations.items()}, False)
            scalar_action = {agent_id: action[agent_id][0] for agent_id in AGENTS}
            observations, reward, terminations, truncations, infos = env.step(scalar_action)
            base_info = infos["cstr1"]["base_info"]
            states.append(np.asarray(base_info["state"], dtype=np.float64).copy())
            targets.append(np.asarray(base_info["target"], dtype=np.float64).copy())
            c2_valid.append(bool(base_info["c2_output_valid"]))
            actions.append(_as_available(base_info["action_physical_available"]))
            commands.append(_as_available(base_info["action_command_available"]))
            rewards.append(float(reward["cstr1"]))
            done = bool(terminations["cstr1"] or truncations["cstr1"])
        trajectories.append({
            "states": np.asarray(states),
            "targets": np.asarray(targets),
            "c2_valid": np.asarray(c2_valid, dtype=bool),
            "actions": np.asarray(actions),
            "commands": np.asarray(commands),
            "rewards": np.asarray(rewards),
        })
    env.close()
    return trajectories


def train(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_multi_agent_vect_envs(env=TwoCSTRParallelEnv, num_envs=args.num_envs, **env_kwargs())
    agent = MATD3.load(str(args.baseline_checkpoint), device=device)
    agent.set_training_mode(True)
    for actor in agent.actors.values():
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
    residuals = {
        agent_id: ResidualActor(max_delta=args.max_residual).to(device) for agent_id in AGENTS
    }
    target_residuals = {
        agent_id: ResidualActor(max_delta=args.max_residual).to(device) for agent_id in AGENTS
    }
    for agent_id in AGENTS:
        target_residuals[agent_id].load_state_dict(residuals[agent_id].state_dict())
    actor_optimizer = torch.optim.Adam(
        list(residuals["cstr1"].parameters()) + list(residuals["cstr2"].parameters()), lr=args.lr_residual
    )
    critic_optimizers = {
        agent_id: (
            torch.optim.Adam(agent.critics_1[agent_id].parameters(), lr=args.lr_critic),
            torch.optim.Adam(agent.critics_2[agent_id].parameters(), lr=args.lr_critic),
        )
        for agent_id in AGENTS
    }
    memory = ReplayBuffer(max_size=args.buffer_size, device=device)
    observations, infos = env.reset(seed=args.seed)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_records = []
    total_steps, update_count, next_report = 0, 0, args.report_every

    while total_steps < args.timesteps:
        action = select_action(agent, residuals, observations, True)
        next_observations, rewards, terminations, truncations, next_infos = env.step(action)
        done = {agent_id: np.logical_or(terminations[agent_id], truncations[agent_id]) for agent_id in AGENTS}
        transition = MultiAgentTransition(
            obs=observations, action=action, reward=rewards,
            next_obs=next_observations, done=done, batch_size=[args.num_envs],
        ).to_tensordict()
        memory.add(transition)
        episode_returns += np.asarray(rewards["cstr1"], dtype=np.float64)
        episode_lengths += 1
        done_mask = np.asarray(done["cstr1"], dtype=bool)
        for index in np.flatnonzero(done_mask):
            episode_records.append({"episode": len(episode_records) + 1, "reward": float(episode_returns[index]), "length": int(episode_lengths[index])})
            episode_returns[index] = 0.0
            episode_lengths[index] = 0
        agent.reset_action_noise(np.flatnonzero(done_mask).tolist())
        total_steps += args.num_envs
        if len(memory) >= args.batch_size:
            for _ in range(max(1, args.num_envs // 4)):
                update_count += 1
                update(agent, residuals, target_residuals, actor_optimizer, critic_optimizers, memory.sample(args.batch_size), update_count, device, args.max_residual)
        observations, infos = next_observations, next_infos
        if total_steps >= next_report:
            recent = episode_records[-10:]
            recent_return = float(np.mean([item["reward"] for item in recent])) if recent else float("nan")
            print(f"steps={total_steps}/{args.timesteps} episodes={len(episode_records)} recent_return={recent_return:.3f} replay={len(memory)}", flush=True)
            next_report += args.report_every

    write_episode_csv(args.output_dir / "training_episodes.csv", episode_records)
    torch.save({"residuals": {k: v.state_dict() for k, v in residuals.items()}, "baseline_checkpoint": str(args.baseline_checkpoint), "max_residual": args.max_residual}, args.output_dir / "matd3_residual.pt")
    env.close()
    return agent, residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-residual", type=float, default=5e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--max-residual", type=float, default=0.15)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "matd3_residual_v1")
    args = parser.parse_args()
    agent, residuals = train(args)
    trajectories = evaluate_residual(agent, residuals, args.eval_episodes, args.seed + 1000)
    np.save(args.output_dir / "trajectories.npy", np.asarray(trajectories, dtype=object), allow_pickle=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(calculate_metrics(trajectories), indent=2), encoding="utf-8")
    plot_learning(args.output_dir / "training_episodes.csv", args.output_dir / "learning_curve.png")
    plot_effect(trajectories, args.output_dir / "matd3_residual_effect.png")
    print(f"Completed residual MATD3 run; results written to {args.output_dir}")


if __name__ == "__main__":
    main()
