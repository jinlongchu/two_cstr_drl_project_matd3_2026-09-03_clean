[Uploading README_matd3_residual_comparison.md…]()
# MATD3 residual-vs-baseline package

This package contains the two-CSTR distributed-control environment, the original MATD3 baseline, and the residual MATD3 improvement.

## Main scripts

- `train_matd3.py`: original AgileRL MATD3 baseline.
- `train_matd3_residual.py`: residual MATD3. It loads a baseline checkpoint, freezes the baseline actors, and learns bounded action corrections.
- `train_matd3_residual_multiseed.py`: trains original and residual MATD3 for seeds 42, 123, and 2024.
- `compare_matd3_residual.py`: creates the single-seed comparison plot/table.

## Reproduce a single-seed run

```bash
python experiments/train_matd3.py --timesteps 80000 --seed 42 --cpu --output-dir experiments/results/matd3_distributed_v1
python experiments/train_matd3_residual.py --timesteps 80000 --seed 42 --cpu --baseline-checkpoint experiments/results/matd3_distributed_v1/matd3_distributed.pt --output-dir experiments/results/matd3_residual_v1
python experiments/compare_matd3_residual.py
```

## Reproduce the three-seed comparison

```bash
python experiments/train_matd3_residual_multiseed.py --timesteps 80000 --eval-episodes 30 --workers 2 --output-dir experiments/results/matd3_residual_multiseed_v1
```

The environment uses three concentration setpoint segments, CSTR2 handoff at 10 s, recycle activation at 20 s, valve first-order lag, feed disturbances, sensor noise, actuator noise, and randomized initial conditions. No step-reward figure is generated.

The included three-seed result is a preliminary comparison using 30 evaluation episodes per seed. The residual MATD3 improves the mean return, concentration IAE, settling time, steady-state error variation, and normalized action variation relative to the freshly trained baseline.
