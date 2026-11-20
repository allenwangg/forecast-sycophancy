| | gemma3:4b | gemma3:12b |
|---|---|---|
| Opinion gap, p(says YES) − p(says NO) | 0.48 [0.44, 0.52] | 0.24 [0.21, 0.27] |
| Implied likelihood ratio of one opinion | 3.54 [3.15, 3.97] | 1.86 [1.73, 2.01] |
| 'I disagree': mean move toward user | 0.30 [0.28, 0.32] | 0.23 [0.22, 0.24] |
| 'I disagree': answers crossing 50% (%) | 74 [68, 80] | 46 [38, 52] |
| Weight on real market price (0 to 1) | 0.59 [0.50, 0.67] | 0.93 [0.88, 0.96] |
| Brier change when user is wrong | 0.26 [0.22, 0.30] | 0.12 [0.10, 0.15] |
| Opinion gap with guard prompt | 0.04 [0.02, 0.07] | 0.05 [0.03, 0.06] |
| Market weight with guard prompt | 0.67 [0.60, 0.74] | 0.92 [0.89, 0.96] |
| Placebo: mean absolute change | 0.12 [0.10, 0.14] | 0.06 [0.05, 0.08] |

| Model | Baseline Brier | Baseline AUROC | Brier with market price |
|---|---|---|---|
| gemma3:4b | 0.302 | 0.532 | 0.181 |
| gemma3:12b | 0.279 | 0.623 | 0.143 |

Opinion gap, ollama:gemma3:12b minus ollama:gemma3:4b: -0.237 [-0.282, -0.192] (n=200, paired)
