# TD3 learning-rate sweep

Mean ± sample standard deviation over seeds 11, 22, and 33; each evaluation uses 30 deterministic episodes.

| Learning rate | Mean return ↑ | Mean concentration error ↓ | Settling time (s) ↓ | Peak error ↓ | Steady error std. ↓ | Action variation ↓ | Mean rank | Recommended |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-05 | -4.094 ± 1.023 | 0.776 ± 0.199 | 16.806 ± 9.161 | 2.386 ± 0.169 | 0.054 ± 0.008 | 0.021 ± 0.003 | 3.83 |  |
| 1e-04 | -4.044 ± 0.863 | 0.751 ± 0.106 | 16.283 ± 4.901 | 2.491 ± 0.272 | 0.080 ± 0.051 | 0.025 ± 0.005 | 4.00 |  |
| 3e-04 | -2.182 ± 0.254 | 0.540 ± 0.032 | 8.256 ± 2.817 | 2.089 ± 0.046 | 0.065 ± 0.002 | 0.027 ± 0.002 | 3.17 |  |
| 1e-03 | -1.893 ± 0.719 | 0.479 ± 0.030 | 5.033 ± 1.418 | 1.927 ± 0.390 | 0.054 ± 0.021 | 0.044 ± 0.018 | 1.83 | Yes |
| 3e-03 | -2.069 ± 0.917 | 0.495 ± 0.090 | 7.033 ± 2.991 | 1.806 ± 0.755 | 0.041 ± 0.003 | 0.051 ± 0.019 | 2.17 |  |
