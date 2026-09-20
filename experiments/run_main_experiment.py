"""
Main Experiment: Winner-Selection Uncertainty vs. Generalization Gap
=====================================================================

For each (task × dataset × searcher × budget) combination:
  1. Estimate selection uncertainty via bootstrap resampling
  2. Record generalization gap
  3. Save results to results/raw/

Usage:
    python experiments/run_main_experiment.py [--budget 30] [--bootstrap 200] [--jobs -1]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.benchmarks.tasks import get_task
from src.benchmarks.datasets import load_dataset, list_datasets
from src.searchers.searchers import get_searcher
from src.uncertainty.estimators import estimate_selection_uncertainty

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

TASKS = ["svm_rbf", "xgboost", "mlp"]

DATASETS = ["breast_cancer", "wine", "digits"]  # iris too small for stability signal

SEARCHERS = ["random_search", "bayesian_opt", "successive_halving"]

BUDGETS = [20, 40]  # two budget levels for budget-sensitivity analysis

SEEDS = [42, 123, 999]  # multiple seeds for robustness


def run_single(
    task_name: str,
    dataset_name: str,
    searcher_name: str,
    budget: int,
    seed: int,
    B: int,
    n_jobs: int,
    out_dir: Path,
) -> dict:
    """Run one (task, dataset, searcher, budget, seed) combination."""

    run_id = f"{task_name}__{dataset_name}__{searcher_name}__b{budget}__s{seed}"
    out_file = out_dir / f"{run_id}.json"

    if out_file.exists():
        print(f"  [SKIP] {run_id} already done.")
        with open(out_file) as f:
            return json.load(f)

    print(f"\n[RUN] {run_id}")
    t0 = time.time()

    task     = get_task(task_name)
    data     = load_dataset(dataset_name, seed=seed)
    searcher = get_searcher(searcher_name)

    est = estimate_selection_uncertainty(
        task=task,
        data=data,
        searcher=searcher,
        budget=budget,
        B=B,
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
    )

    result = est.summary()
    result["seed"]    = seed
    result["run_id"]  = run_id
    result["elapsed"] = round(time.time() - t0, 1)

    # Also save bootstrap winner vectors for 2D visualization (SVM only)
    if task_name == "svm_rbf":
        result["bootstrap_winners"] = est.bootstrap_winners
        result["original_winner"]   = est.original_winner

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Done in {result['elapsed']}s | "
          f"WS={result['winner_stability']:.3f} | "
          f"OSC={result['eps_coverage']:.3f} | "
          f"SE={result['selection_entropy']:.3f} | "
          f"gap={result['gen_gap']:.4f}")
    return result


def main(args):
    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(TASKS, DATASETS, SEARCHERS, BUDGETS, SEEDS))
    print(f"\nTotal runs: {len(combos)} "
          f"({len(TASKS)} tasks × {len(DATASETS)} datasets × "
          f"{len(SEARCHERS)} searchers × {len(BUDGETS)} budgets × {len(SEEDS)} seeds)")

    all_results = []
    for task_name, dataset_name, searcher_name, budget, seed in tqdm(combos, desc="Experiments"):
        try:
            result = run_single(
                task_name, dataset_name, searcher_name, budget, seed,
                B=args.bootstrap, n_jobs=args.jobs, out_dir=out_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {task_name}/{dataset_name}/{searcher_name}: {e}")
            continue

    # Aggregate to CSV
    df = pd.DataFrame(all_results)
    csv_path = Path("results/main_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path} ({len(df)} rows)")
    print(df[["task", "dataset", "searcher", "budget",
              "val_loss", "test_loss", "gen_gap",
              "winner_stability", "eps_coverage", "selection_entropy"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",    type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--jobs",      type=int, default=-1)
    args = parser.parse_args()
    main(args)
