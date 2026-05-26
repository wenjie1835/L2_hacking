# Group Meeting Workflow Report

## H1/H2: multi-domain reward overgeneralization

A positive row has high source accuracy, low target/conflict accuracy, high AGOP source-target cosine, and positive transfer spurious score.

| phenomenon | reward | source_accuracy_mean | target_accuracy_mean | conflict_accuracy_mean | agop_source_target_abs_cosine_mean | target_reward_spur_corr_mean | transfer_spurious_score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code_test_hacking | test_passing | 0.8233 | 0.6121 | 0.3406 | 0.9496 | 0.5246 | 0.2209 |
| confidence | decisiveness | 0.8567 | 0.2785 | 0.1052 | 0.9161 | 0.9042 | 0.6638 |
| fabricated_reasoning | reasoning_quality | 0.8415 | 0.3904 | 0.1423 | 0.7789 | 0.8779 | 0.4966 |
| formatting | readability | 0.8367 | 0.6154 | 0.2962 | 0.7143 | 0.6128 | 0.2099 |
| sycophancy | supportiveness | 0.8525 | 0.3715 | 0.1431 | 0.9625 | 0.8888 | 0.6547 |
| verbosity | helpfulness | 0.8431 | 0.4454 | 0.1504 | 0.9607 | 0.8817 | 0.6428 |

## H3: best weight decay per phenomenon

Rows are selected by minimum proxy-minus-true hacking gap under best-of-N at N=64. This is the group-meeting mitigation view; the CSV keeps all weight-decay points.

| phenomenon | reward | weight_decay | target_accuracy_mean | conflict_accuracy_mean | abs_spurious_over_true_sensitivity_mean | transfer_spurious_score_mean | selected_true_utility_mean_mean | proxy_minus_true_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence | decisiveness | 1 | 0.4831 | 0.09813 | 0.9688 | 0.7796 | 0.2033 | 2.198 |
| fabricated_reasoning | reasoning_quality | 1 | 0.6581 | 0.1894 | 0.7673 | 0.6404 | 0.3726 | 2.349 |
| formatting | readability | 1 | 0.7956 | 0.5719 | 0.4864 | 0.06878 | 1.048 | 1.381 |
| verbosity | helpfulness | 1 | 0.5994 | 0.145 | 0.8283 | 0.7238 | 0.341 | 2.21 |

## Best-of-N at N=64

| phenomenon | reward | weight_decay | selected_proxy_reward_mean_mean | selected_true_utility_mean_mean | selected_spurious_utility_mean_mean | proxy_minus_true_gap_mean |
| --- | --- | --- | --- | --- | --- | --- |
| confidence | decisiveness | 0 | 7.495 | 0.6827 | 11.93 | 6.812 |
| confidence | decisiveness | 0.001 | 7.476 | 0.6844 | 11.92 | 6.791 |
| confidence | decisiveness | 0.01 | 7.309 | 0.7012 | 11.82 | 6.608 |
| confidence | decisiveness | 0.1 | 6.024 | 0.7807 | 11.33 | 5.243 |
| confidence | decisiveness | 0.5 | 3.582 | 0.3819 | 13.26 | 3.2 |
| confidence | decisiveness | 1 | 2.401 | 0.2033 | 13.78 | 2.198 |
| fabricated_reasoning | reasoning_quality | 0 | 8.788 | 1.035 | 8.764 | 7.753 |
| fabricated_reasoning | reasoning_quality | 0.001 | 8.767 | 1.033 | 8.76 | 7.735 |
| fabricated_reasoning | reasoning_quality | 0.01 | 8.588 | 1.031 | 8.784 | 7.557 |
| fabricated_reasoning | reasoning_quality | 0.1 | 7.174 | 0.9996 | 9.28 | 6.174 |
| fabricated_reasoning | reasoning_quality | 0.5 | 4.204 | 0.4699 | 12.48 | 3.734 |
| fabricated_reasoning | reasoning_quality | 1 | 2.721 | 0.3726 | 12.67 | 2.349 |
| formatting | readability | 0 | 7.678 | 1.718 | 5.318 | 5.961 |
| formatting | readability | 0.001 | 7.659 | 1.718 | 5.32 | 5.941 |
| formatting | readability | 0.01 | 7.488 | 1.716 | 5.308 | 5.772 |
| formatting | readability | 0.1 | 6.153 | 1.703 | 5.29 | 4.45 |
| formatting | readability | 0.5 | 3.624 | 1.452 | 6.357 | 2.172 |
| formatting | readability | 1 | 2.43 | 1.048 | 8.417 | 1.381 |
| verbosity | helpfulness | 0 | 8.222 | 1.083 | 8.117 | 7.14 |
| verbosity | helpfulness | 0.001 | 8.201 | 1.078 | 8.122 | 7.122 |
| verbosity | helpfulness | 0.01 | 8.011 | 1.069 | 8.295 | 6.942 |
| verbosity | helpfulness | 0.1 | 6.62 | 0.6109 | 10.75 | 6.009 |
| verbosity | helpfulness | 0.5 | 3.887 | 0.457 | 11.15 | 3.43 |
| verbosity | helpfulness | 1 | 2.551 | 0.341 | 11.38 | 2.21 |

## Files

- `h12_diagnostics.csv`: one row per seed/domain/reward-model variant.
- `h3_weight_decay.csv`: one row per seed/domain/weight decay.
- `h3_best_of_n.csv`: best-of-N hacking curves.