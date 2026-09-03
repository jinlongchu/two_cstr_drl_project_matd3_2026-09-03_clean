# Original MATD3 vs residual MATD3 — three random seeds

Each seed uses a freshly trained original MATD3 and a residual MATD3 fine-tuned from that seed's baseline checkpoint. Values are mean ± sample standard deviation across seeds; each run uses 30 evaluation episodes.

| 版本 | 平均回报 ↑ | 平均浓度绝对误差 ↓ | 平均稳定时间 (s) ↓ | 峰值绝对误差 ↓ | 稳态误差标准差 ↓ | 归一化动作变化率 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original MATD3 | -4.738 ± 3.666 | 0.888 ± 0.483 | 35.589 ± 47.155 | 1.938 ± 0.075 | 0.060 ± 0.012 | 0.042 ± 0.010 |
| Residual MATD3 | -3.757 ± 2.644 | 0.714 ± 0.376 | 27.906 ± 36.513 | 1.912 ± 0.200 | 0.044 ± 0.012 | 0.038 ± 0.011 |

Higher return is better; errors, settling time, and action variation are better when lower.
