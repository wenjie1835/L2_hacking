# Group Meeting Workflow Report

## H1/H2: multi-domain reward overgeneralization

A positive row has high source accuracy, low target/conflict accuracy, high AGOP source-target cosine, and positive transfer spurious score.

| phenomenon | reward | source_accuracy_mean | target_accuracy_mean | conflict_accuracy_mean | agop_source_target_abs_cosine_mean | target_reward_spur_corr_mean | transfer_spurious_score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code_test_hacking | test_passing | 0.8622 | 0.5544 | 0.16 | 0.9938 | 0.8509 | 0.5851 |
| confidence | decisiveness | 0.8656 | 0.2733 | 0.08667 | 0.997 | 0.9068 | 0.7339 |
| fabricated_reasoning | reasoning_quality | 0.8156 | 0.4433 | 0.1356 | 0.9731 | 0.8607 | 0.5929 |
| formatting | readability | 0.8489 | 0.5456 | 0.2333 | 0.9962 | 0.7571 | 0.4727 |
| sycophancy | supportiveness | 0.8778 | 0.2933 | 0.1033 | 0.9991 | 0.896 | 0.6938 |
| verbosity | helpfulness | 0.8778 | 0.4511 | 0.1789 | 0.9964 | 0.8575 | 0.6243 |

## H3: best weight decay per phenomenon

Rows are selected by minimum proxy-minus-true hacking gap under best-of-N at N=64. This is the group-meeting mitigation view; the CSV keeps all weight-decay points.

| phenomenon | reward | weight_decay | target_accuracy_mean | conflict_accuracy_mean | abs_spurious_over_true_sensitivity_mean | transfer_spurious_score_mean | selected_true_utility_mean_mean | proxy_minus_true_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence | decisiveness | 0.5 | 0.3367 | 0.07333 | 1.123 | 0.7576 | 0.05836 | 2.642 |
| fabricated_reasoning | reasoning_quality | 0.5 | 0.69 | 0.1933 | 0.7584 | 0.5064 | 0.3506 | 2.933 |
| formatting | readability | 0.5 | 0.66 | 0.35 | 0.6127 | 0.1434 | 0.819 | 1.725 |
| verbosity | helpfulness | 0.5 | 0.7267 | 0.24 | 0.6557 | 0.4289 | 0.6197 | 2.938 |

## Best-of-N at N=64

| phenomenon | reward | weight_decay | selected_proxy_reward_mean_mean | selected_true_utility_mean_mean | selected_spurious_utility_mean_mean | proxy_minus_true_gap_mean |
| --- | --- | --- | --- | --- | --- | --- |
| confidence | decisiveness | 0 | 3.694 | 0.2303 | 13.49 | 3.464 |
| confidence | decisiveness | 0.1 | 3.453 | 0.1754 | 13.56 | 3.278 |
| confidence | decisiveness | 0.5 | 2.7 | 0.05836 | 13.72 | 2.642 |
| fabricated_reasoning | reasoning_quality | 0 | 4.571 | 0.5182 | 11.94 | 4.053 |
| fabricated_reasoning | reasoning_quality | 0.1 | 4.262 | 0.4827 | 11.96 | 3.78 |
| fabricated_reasoning | reasoning_quality | 0.5 | 3.284 | 0.3506 | 12.28 | 2.933 |
| formatting | readability | 0 | 3.409 | 1.041 | 7.4 | 2.367 |
| formatting | readability | 0.1 | 3.203 | 1.115 | 7.247 | 2.087 |
| formatting | readability | 0.5 | 2.544 | 0.819 | 8.406 | 1.725 |
| verbosity | helpfulness | 0 | 5.021 | 0.8288 | 9.791 | 4.193 |
| verbosity | helpfulness | 0.1 | 4.665 | 0.7909 | 9.939 | 3.874 |
| verbosity | helpfulness | 0.5 | 3.557 | 0.6197 | 10.59 | 2.938 |

## Files

- `h12_diagnostics.csv`: one row per seed/domain/reward-model variant.
- `h3_weight_decay.csv`: one row per seed/domain/weight decay.
- `h3_best_of_n.csv`: best-of-N hacking curves.