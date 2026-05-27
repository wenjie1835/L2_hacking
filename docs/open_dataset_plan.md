# Open-Dataset Experiment Plan

## Motivation

The synthetic workflow is useful because the true and spurious latent features
are known, but it is not enough for a NeurIPS-scale claim. The open-dataset
workflow tests the same mechanism on real preference and reward-hacking
benchmarks. The main paper-level workflow should use
`src/transformer_rm_workflow.py`, which trains a GPU transformer reward model.
The NumPy workflow in `src/real_data_workflow.py` remains a cheap baseline and
schema check.

## Data Sources

| Role | Dataset | Use |
| --- | --- | --- |
| Source RM training | `HuggingFaceH4/ultrafeedback_binarized` | General helpfulness/honesty preference pairs. |
| Source RM training | `Anthropic/hh-rlhf` | Helpful/harmless assistant preference pairs. |
| OOD preference eval | `allenai/reward-bench` | Chat, safety, code, and reasoning preference checks. |
| Reward-hacking eval | `ktolnos/rh-bench` | Clean-vs-hacking pairs across sycophancy, reward tampering, deception, evaluation gaming, output style gaming, and environment exploitation. |
| Reward-hacking eval | `meg-tong/sycophancy-eval` | Sycophancy-specific benchmark rows when the Hugging Face mirror exposes pair-style fields. |

## Hypotheses

H1: Reward models trained on broad source preference data will prefer hacking
responses over clean responses on multiple open reward-hacking categories.

H2: The source-domain AGOP direction and hacking-domain AGOP direction will stay
aligned, and the hacking-domain direction will concentrate on proxy/style
features instead of task-grounded quality.

H3: Moderate weight decay may reduce proxy/style sensitivity and best-of-N
hacking selection, but it should only be called a mitigation if it improves
clean-vs-hacking accuracy without destroying source preference accuracy.

## Primary Metrics

| Metric | File | Desired Direction |
| --- | --- | --- |
| Source preference accuracy | `source_eval.csv` | Stay high. |
| RewardBench OOD accuracy | `ood_preference_eval.csv` | Stay high. |
| Clean-vs-hacking accuracy | `hacking_eval.csv` | Increase with mitigation. |
| Hacking failure rate | `hacking_eval.csv` | Decrease with mitigation. |
| Source-target AGOP cosine | `agop_diagnostics.csv` | High for H2 evidence. |
| Reward-style correlation | `hacking_eval.csv` | Decrease with mitigation. |
| Best-of-N hacking-selected rate | `best_of_n.csv` | Decrease with mitigation. |

## Complete Run

```bash
python src/transformer_rm_workflow.py \
  --output-dir results/transformer_rm_full \
  --model-name distilroberta-base \
  --seeds 0 1 2 \
  --weight-decays 0 1e-5 1e-4 1e-3 1e-2 \
  --max-train-pairs-per-source 12000 \
  --max-eval-pairs-per-dataset 4000 \
  --max-hacking-pairs-per-category 1200 \
  --epochs 2 \
  --device cuda
```

Use the lightweight GPU run first on a new server:

```bash
python src/transformer_rm_workflow.py --quick --device cuda
```

It uses one seed, smaller datasets, and fewer weight-decay values. It writes
`device_info.json`; check that the `device` field is `cuda`.
