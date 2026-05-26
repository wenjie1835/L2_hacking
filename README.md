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
cd reward_hacking_wd_experiment
python src/run_experiment.py --device cpu --quick
```

For a more stable run:

```bash
python src/run_experiment.py --device cpu --train-pairs 50000 --epochs 12 --seeds 0 1 2 3 4
```

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
