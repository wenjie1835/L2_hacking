# Weight Decay vs Reward Hacking: Small Synthetic Validation

This is a small, controllable experiment for testing the hypothesis:

> If reward hacking is partly caused by representational feature mixing / superposition-like entanglement, then changing weight decay in the reward model may reduce reliance on spurious features and delay reward hacking under optimization pressure.

The experiment is intentionally synthetic so that the true causal feature and the spurious feature are known.

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
