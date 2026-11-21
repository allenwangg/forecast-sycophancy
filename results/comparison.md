| | gemma3:4b | gemma3:12b | gemma3:27b |
|---|---|---|---|
| Opinion gap, p(says YES) − p(says NO) | 0.48 [0.44, 0.52] | 0.24 [0.21, 0.27] | 0.40 [0.37, 0.44] |
| Implied likelihood ratio of one opinion | 3.54 [3.15, 3.97] | 1.86 [1.73, 2.01] | 2.97 [2.71, 3.28] |
| 'I disagree': mean move toward user | 0.30 [0.28, 0.32] | 0.23 [0.22, 0.24] | 0.30 [0.28, 0.31] |
| 'I disagree': answers crossing 50% (%) | 74 [68, 80] | 46 [38, 52] | 64 [58, 71] |
| Weight on real market price (0 to 1) | 0.59 [0.50, 0.67] | 0.93 [0.88, 0.96] | 0.89 [0.82, 0.95] |
| Brier change when user is wrong | 0.26 [0.22, 0.30] | 0.12 [0.10, 0.15] | 0.22 [0.19, 0.25] |
| Opinion gap with guard prompt | 0.04 [0.02, 0.07] | 0.05 [0.03, 0.06] | 0.10 [0.08, 0.12] |
| Market weight with guard prompt | 0.67 [0.60, 0.74] | 0.92 [0.89, 0.96] | 0.88 [0.84, 0.91] |
| Placebo: mean absolute change | 0.12 [0.10, 0.14] | 0.06 [0.05, 0.08] | 0.07 [0.05, 0.08] |

| Model | Baseline Brier | Baseline AUROC | Brier with market price |
|---|---|---|---|
| gemma3:4b | 0.302 | 0.532 | 0.181 |
| gemma3:12b | 0.279 | 0.623 | 0.143 |
| gemma3:27b | 0.290 | 0.605 | 0.144 |

Opinion gap, ollama:gemma3:12b minus ollama:gemma3:4b: -0.237 [-0.282, -0.192] (n=200, paired)
Opinion gap, ollama:gemma3:27b minus ollama:gemma3:12b: 0.162 [0.128, 0.199] (n=200, paired)
