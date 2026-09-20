# How Certain Are We About the HPO Winner?
### Statistical Uncertainty in Hyperparameter Selection

**AISTATS 2027 Submission**

---

## Core Claim

> Hyperparameter optimization returns a single "winner" — but validation performance is noisy. The same data, resampled, often yields a different winner. We show that this *winner-selection uncertainty* is predictive of post-selection generalization error, independent of validation score and budget.

---

## Project Structure

```
src/
  benchmarks/    # HPO task definitions (SVM, XGBoost, MLP on real datasets)
  searchers/     # Random search, Bayesian optimization, Hyperband
  uncertainty/   # Selection uncertainty estimators (bootstrap, subsampling)
  analysis/      # Regression, correlation, plotting utilities

experiments/     # Entry-point scripts for each experiment
results/
  raw/           # JSON/CSV dumps of all runs
  figures/       # Paper-ready plots
  tables/        # LaTeX tables

paper/           # LaTeX source (coming)
tests/           # Unit tests
```

---

## Experimental Protocol

For each (benchmark task × HPO algorithm × budget) triple:

1. Run full HPO; record winner `ĥ` and its validation score
2. Bootstrap-resample the train/val split **B=200** times; rerun HPO each time
3. Estimate:
   - **P(ĥ = h*)** — exact winner stability
   - **P(ĥ ∈ H_ε)** — ε-optimal-set coverage (ε = 2% of val loss range)
   - **Selection entropy** — H(distribution over winners)
4. Evaluate the original winner on held-out test data
5. Record **generalization gap** = `L_test(ĥ) − L̂_val(ĥ)`
6. Test: does selection uncertainty predict generalization gap?

---

## Benchmarks

| Task | Dataset | Model | Search Space dim |
|------|---------|-------|-----------------|
| SVM-RBF | sklearn datasets (iris, wine, breast_cancer, digits) | SVM | 2 (C, gamma) |
| XGBoost | OpenML cc18 tasks | XGBoost | 5 |
| MLP | OpenML cc18 tasks | MLP | 4 |

---

## Reproducing

```bash
pip install -r requirements.txt

# Main experiment
python experiments/run_main_experiment.py --budget 30 --bootstrap 200

# Analysis + figures
python experiments/analyze_results.py
```

---

## Key Files

- `src/uncertainty/estimators.py` — the core uncertainty estimators
- `src/benchmarks/tasks.py` — benchmark task definitions
- `experiments/run_main_experiment.py` — main entry point
- `experiments/analyze_results.py` — produces paper figures

