# Original MATD3 vs residual MATD3

Same environment, 80,000 training steps, 4 vectorized environments, seed=42, and 30 evaluation episodes.
Residual MATD3 starts from the preserved baseline checkpoint. The baseline actors are frozen; zero-initialized bounded residual heads learn corrections with max normalized amplitude 0.15.

| 版本 | 平均回报 ↑ | 平均浓度绝对误差 ↓ | 平均稳定时间 (s) ↓ | 峰值绝对误差 ↓ | 稳态误差标准差 ↓ | 动作变化率 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original MATD3 | -2.382 | 0.526 | 6.600 | 1.949 | 0.047 | 0.036 |
| Residual MATD3 | -2.056 | 0.451 | 4.683 | 1.912 | 0.028 | 0.039 |

Higher return is better; errors, settling time, and action variation are better when lower.
This is a single-seed comparison; repeat with multiple seeds before making a final superiority claim.
