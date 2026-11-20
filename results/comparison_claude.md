| | claude-code:opus | claude-code:fable |
|---|---|---|
| Opinion gap, p(says YES) − p(says NO) | 0.08 [0.07, 0.09] | 0.02 [0.01, 0.03] |
| Implied likelihood ratio of one opinion | 1.31 [1.26, 1.35] | 1.09 [1.06, 1.12] |
| 'I disagree': mean move toward user | 0.02 [0.01, 0.02] | 0.00 [0.00, 0.00] |
| 'I disagree': answers crossing 50% (%) | 3 [1, 6] | 0 [0, 0] |
| Weight on real market price (0 to 1) | 0.86 [0.83, 0.89] | 0.85 [0.79, 0.91] |
| Brier change when user is wrong | 0.05 [0.04, 0.06] | 0.02 [0.01, 0.02] |
| Opinion gap with guard prompt | 0.05 [0.04, 0.05] | 0.03 [0.02, 0.04] |
| Market weight with guard prompt | 0.87 [0.83, 0.90] | 0.89 [0.84, 0.93] |
| Placebo: mean absolute change | 0.04 [0.03, 0.05] | 0.03 [0.03, 0.04] |

| Model | Baseline Brier | Baseline AUROC | Brier with market price |
|---|---|---|---|
| claude-code:opus | 0.238 | 0.759 | 0.148 |
| claude-code:fable | 0.222 | 0.786 | 0.149 |

Opinion gap, claude-code:fable minus claude-code:opus: -0.058 [-0.072, -0.045] (n=200, paired)
