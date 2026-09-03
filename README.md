# Two-CSTR realistic DRL experiment

This folder contains the runnable two-CSTR series-with-recycle environment and
the PPO, SAC, DDPG, TD3, and improved-DDPG experiments.

## Environment

- Actions: `[F, L, Tc1, Tc2]` (fresh flow, recycle flow, two cooling temperatures).
- CSTR2 output starts at `t=10 s`, with `C2(10)=C1(10)` exactly.
- Recycle flow `L` becomes active at `t=20 s`.
- Episode length: `180 s`; targets: `(96,94.5)`, `(94,92)`, `(95,93)`.
- Realism: 5 s cooling-valve first-order lag, AR(1) feed disturbances, sensor noise, actuator noise, and randomized initial state.

## Install and run one algorithm

```bash
pip install -r requirements-experiments.txt
python experiments/train_ppo_three_segment.py --timesteps 150000 --eval-episodes 30 --seed 42 --output-dir experiments/results/ppo_three_segment_targets_v3
python experiments/train_sac_ddpg.py --algorithm SAC --timesteps 150000 --eval-episodes 30 --seed 42 --output-dir experiments/results/sac_three_segment_targets_v3
python experiments/train_sac_ddpg.py --algorithm DDPG --timesteps 150000 --eval-episodes 30 --seed 42 --output-dir experiments/results/ddpg_three_segment_targets_v3
python experiments/plot_algorithm_comparison.py --root experiments/results
```

The scripts intentionally generate episode-return curves only; step-reward
figures are not produced.

## Distributed PettingZoo environment (MATD3)

`TwoCSTRParallelEnv` keeps the same two-CSTR physics and exposes two
simultaneous cooperative agents for CTDE algorithms such as MATD3:

- `cstr1` action: `[F, Tc1]`
- `cstr2` action: `[L, Tc2]`
- `t < 10 s`: CSTR2 output and `Tc2` are unavailable;
- `t < 20 s`: recycle `L` is unavailable;
- `info[agent]["global_state"]`: normalized physical state for a centralized
  critic; local actor observations remain separate.

Install the optional multi-agent dependencies with
`pip install -r requirements-experiments.txt`.  A quick interface check is:

```bash
python experiments/check_matd3_env.py
```

Train MATD3 with four parallel episodes per environment step:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/train_matd3.py --timesteps 80000 --eval-episodes 30 \
  --num-envs 4 --batch-size 256 --output-dir experiments/results/matd3_distributed_v1
```

The output directory contains `learning_curve.png`,
`matd3_distributed_effect.png`, `metrics.json`, and the MATD3 checkpoint.

## Three-seed comparison

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/train_multiseed.py --timesteps 150000 --eval-episodes 30 --workers 3 \
  --output-dir experiments/results/multiseed_v1
```

This trains PPO, SAC, DDPG, TD3, and improved DDPG with seeds 11, 22, and 33.
The aggregate learning curve is `results/multiseed_v1/multiseed_learning_curves.png`;
the five independent effect figures and mean +/- standard-deviation table are
stored in the same directory.

## TD3 learning-rate sweep

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/train_td3_lr_sweep.py --timesteps 150000 --eval-episodes 30 --workers 3 \
  --output-dir experiments/results/td3_learning_rate_sweep_v1 \
  --reuse-root experiments/results/multiseed_v1
```

The sweep compares learning rates `5e-5`, `1e-4`, `3e-4`, `1e-3`, and `3e-3`
over the same three seeds. The tuning figure and CSV/Markdown tables are in
`results/td3_learning_rate_sweep_v1`.
