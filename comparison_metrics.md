# Two-CSTR controller comparison

All values are means over 30 deterministic evaluation episodes under the same
three-segment target schedule and environment configuration.

| Algorithm | Mean return (higher) | Mean abs. concentration error (lower) | Settling time (s, lower) | Peak abs. error (lower) | Steady error std. (lower) | Normalized action variation (lower) |
|---|---:|---:|---:|---:|---:|---:|
| PPO | -4.464 | 0.713 | 29.217 | 2.724 | 0.042 | 0.0210 |
| SAC | -8.434 | 1.271 | 48.400 | 2.093 | 0.390 | 0.0475 |
| DDPG | -2.063 | 0.503 | 7.417 | 1.981 | 0.073 | 0.0452 |
| TD3 | -2.734 | 0.592 | 7.700 | 2.018 | 0.051 | 0.0185 |
| **Improved DDPG** | **-2.059** | **0.500** | 7.833 | **1.168** | **0.035** | 0.0268 |

The improved controller uses adaptive OU exploration, event-prioritized replay,
five-step observation history, and residual actions around a concentration-error
baseline. It improves peak error, steady-state variability, and action smoothness
relative to baseline DDPG, while matching its overall return and mean tracking
error. Settling time is nearly unchanged (slightly higher by 0.42 s).
