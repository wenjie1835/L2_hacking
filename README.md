# Weight Decay vs Reward Hacking

This repository tests the hypothesis:

> If reward hacking is partly caused by representational feature mixing / superposition-like entanglement, then changing weight decay in the reward model may reduce reliance on spurious features and delay reward hacking under optimization pressure.

There are now three workflows:

- `src/transformer_rm_workflow.py`: the main GPU transformer reward-model workflow.
- `src/real_data_workflow.py`: a CPU hashed-feature baseline for cheap diagnostics.
- `src/group_meeting_workflow.py` and `src/run_experiment.py`: lightweight synthetic controls where the true and spurious features are known.

## Main GPU Transformer RM Experiment

The main workflow uses public datasets instead of the simplified synthetic latent world:

- RM training source domains:
  - `HuggingFaceH4/ultrafeedback_binarized`
  - `Anthropic/hh-rlhf`
- OOD preference evaluation:
  - `allenai/reward-bench`
- Reward-hacking evaluation:
  - `ktolnos/rh-bench`
  - optional additional hacking sources can be passed with `--hacking-sources`

The reward model uses the common architecture for open-source RMs: a pretrained Transformer backbone plus a scalar reward head. The script implements this as `AutoModelForSequenceClassification(num_labels=1)` and trains it with Bradley-Terry preference loss:

```text
loss = -log sigmoid(r(chosen) - r(rejected))
```

By default it uses `distilroberta-base` for a lightweight GPU run. You can switch to `roberta-base`, `microsoft/deberta-v3-small`, or another sequence-classification compatible backbone with `--model-name`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight single-seed GPU experiment:

```bash
python src/transformer_rm_workflow.py --quick --device cuda
```

Run the full experiment:

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

The workflow writes:

- `device_info.json`: CUDA/GPU and precision metadata.
- `dataset_manifest.csv`: loaded datasets, splits, counts, licenses, and any load errors.
- `train_history.csv`: pairwise RM training loss and accuracy.
- `source_eval.csv`: held-out source preference accuracy.
- `ood_preference_eval.csv`: RewardBench-style OOD preference accuracy.
- `hacking_eval.csv`: clean-vs-hacking pair accuracy by reward-hacking category.
- `agop_diagnostics.csv`: source-target AGOP cosine in token-embedding gradient space.
- `best_of_n.csv`: adversarial clean-vs-many-hacking selection curves.
- `summary_by_weight_decay.csv`: aggregate weight-decay trend.
- `report.md`: tables for paper/slide triage.

### How To Read The Full Experiment

H1/H2 evidence should look like:

- the RM learns source preferences but fails on multiple clean-vs-hacking categories;
- source and hacking-domain AGOP directions have high absolute cosine;
- hacking-domain AGOP mass is concentrated in style/proxy groups such as confidence, sycophancy, reasoning theater, test gaming, or tampering/deception terms.

H3 evidence for a successful weight-decay mitigation should require all three:

- clean-vs-hacking accuracy increases or hacking failure rate decreases;
- best-of-N hacking-selected rate decreases as N grows;
- source-target AGOP transfer or reward-style correlation decreases without destroying source preference accuracy.

## CPU Open-Dataset Baseline

For a cheaper diagnostic that does not require GPU, run:

```bash
python src/real_data_workflow.py --quick
```

This baseline uses hashed text n-gram features plus explicit style/proxy features and a small NumPy bottleneck MLP. It is useful for fast debugging, but the transformer workflow above is the main architecture for paper-level experiments.

## What it does

1. Creates synthetic response vectors with:
   - true features: determine the oracle preference;
   - spurious features: correlated with the true feature during training, but broken or reversed during evaluation;
   - nuisance features: irrelevant noise.
2. Trains pairwise reward models with Bradley-Terry loss while scanning `weight_decay`.
3. Evaluates:
   - IID preference accuracy;
   - OOD preference accuracy;
   - conflict-set accuracy, where the true and spurious features disagree;
   - spurious sensitivity;
   - true sensitivity;
   - a representation-mixing proxy based on linear probes;
   - best-of-N reward hacking curves.
4. Writes CSVs and plots to `results/`.

## Quick start

```bash
cd L2_hacking
python src/run_experiment.py --device cpu --quick
```

For a more stable run:

```bash
python src/run_experiment.py --device cpu --train-pairs 50000 --epochs 12 --seeds 0 1 2 3 4
```

## Group Meeting Workflow

For a small workflow that matches the NeurIPS-paper story, run:

```bash
python src/group_meeting_workflow.py --quick
```

This creates `results/group_meeting_smoke/` with:

- `h12_diagnostics.csv`: H1/H2 diagnostics across six reward-hacking phenomena and three small reward-model variants.
- `h3_weight_decay.csv`: fixed-size reward model sweep over `weight_decay`.
- `h3_best_of_n.csv`: best-of-N optimization pressure curves.
- `report.md`: compact tables for slides.

Use this version for a slightly more stable group-meeting table:

```bash
python src/group_meeting_workflow.py \
  --output-dir results/group_meeting_full \
  --train-pairs 1200 \
  --test-pairs 800 \
  --epochs 70 \
  --seeds 0 1 \
  --weight-decays 0 1e-3 1e-2 1e-1 5e-1 1.0
```

The workflow separates the claims:

- H1/H2 uses a stronger domain-shift suite to show reward-sensitive AGOP directions stay shared across source and target while their true-utility meaning changes.
- H3 uses a cleaner mitigation suite where the true feature is available but proxy shortcuts still tempt the reward model, then checks whether weight decay improves best-of-N selected true utility.

## Output files

- `results/summary_metrics.csv`: one row per `(seed, weight_decay)`.
- `results/best_of_n_metrics.csv`: best-of-N curves.
- `results/summary_by_weight_decay.csv`: mean/std across seeds.
- `results/best_of_n_by_weight_decay.csv`: mean/std across seeds.
- `results/*.png`: diagnostic plots.

## How to read the result

Evidence in favor of the hypothesis would look like this:

- moderate weight decay improves OOD/conflict accuracy;
- moderate weight decay lowers spurious sensitivity;
- best-of-N proxy reward still rises, but oracle true utility collapses later or less severely;
- probe-based representation mixing decreases.

A null or negative result is also useful:

- no OOD improvement;
- spurious sensitivity unchanged;
- best-of-N hacking curves unchanged;
- only IID accuracy drops as weight decay increases.

## Notes

This does not prove or disprove the mechanistic superposition story by itself. It is a cheap screening test. If the signal is positive, the next step is to repeat the experiment with a real text reward model and SAE-based feature attribution.
