"""
Winner-Selection Uncertainty Estimators
========================================
This is the core contribution of the paper.

Given an HPO procedure and a dataset, we estimate the distribution over
selected configurations P(ĥ = h) by bootstrap resampling the development
data and rerunning the full HPO procedure.

From this distribution we compute three uncertainty measures:

1. **Winner stability** (WS):
   P(ĥ = ĥ_original) — probability the same exact winner is recovered.
   For continuous spaces, we use an ε-ball in log-hyperparameter space.

2. **ε-optimal-set coverage** (OSC):
   P(ĥ ∈ H_ε) — probability bootstrap winner falls within ε of the
   original winner's TRUE loss (approximated by its mean across resamples).

3. **Selection entropy** (SE):
   H(distribution over ε-discretized winner region) — entropy of the
   winner distribution, normalized by log(B).

These are returned as an UncertaintyEstimate dataclass.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from src.benchmarks.tasks import HPOTask
from src.benchmarks.datasets import DataSplit
from src.searchers.searchers import Searcher, SearchResult


# ---------------------------------------------------------------------------
# Config distance utilities
# ---------------------------------------------------------------------------

def config_to_log_vec(config: Dict[str, float], task: HPOTask) -> np.ndarray:
    """Convert config to log-scaled vector for distance computations."""
    vec = []
    for name, hp in task.search_space.items():
        v = config[name]
        if hp.log_scale:
            vec.append(np.log(max(v, 1e-12)))
        else:
            vec.append(v)
    return np.array(vec)


def config_distance(
    c1: Dict[str, float],
    c2: Dict[str, float],
    task: HPOTask,
    normalize: bool = True,
) -> float:
    """
    Euclidean distance in (log-)parameter space, normalized by space diameter.
    """
    v1 = config_to_log_vec(c1, task)
    v2 = config_to_log_vec(c2, task)

    if normalize:
        # Normalize by search space diameter
        low_vec = config_to_log_vec(
            {n: hp.low for n, hp in task.search_space.items()}, task
        )
        high_vec = config_to_log_vec(
            {n: hp.high for n, hp in task.search_space.items()}, task
        )
        diameter = np.linalg.norm(high_vec - low_vec)
        if diameter < 1e-12:
            return 0.0
        return float(np.linalg.norm(v1 - v2) / diameter)

    return float(np.linalg.norm(v1 - v2))


# ---------------------------------------------------------------------------
# Main result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyEstimate:
    # Identity
    task_name:     str
    dataset_name:  str
    searcher_name: str
    budget:        int
    B:             int  # number of bootstrap resamples

    # Original run
    original_winner: Dict[str, float]
    original_val_loss: float

    # Bootstrap distribution
    bootstrap_winners:    List[Dict[str, float]]  # length B
    bootstrap_val_losses: List[float]             # length B

    # Derived: test performance of original winner
    test_loss: float = float("nan")

    # Uncertainty measures (filled by compute_measures)
    winner_stability:     float = float("nan")  # WS
    eps_optimal_coverage: float = float("nan")  # OSC
    selection_entropy:    float = float("nan")  # SE
    mean_winner_distance: float = float("nan")  # mean dist to original winner

    # Generalization gap
    @property
    def generalization_gap(self) -> float:
        """Test loss minus (original) validation loss. Key outcome variable."""
        return self.test_loss - self.original_val_loss

    def compute_measures(
        self,
        task: HPOTask,
        eps_config: float = 0.10,   # ε-ball radius in normalized config space
        eps_loss: float   = 0.02,   # ε for ε-optimal-set (fraction of val loss range)
    ):
        """
        Fill in winner_stability, eps_optimal_coverage, selection_entropy.
        Call after test_loss has been set.
        """
        B = len(self.bootstrap_winners)
        if B == 0:
            return

        # --- Winner stability: fraction of bootstrap runs whose winner
        #     falls within eps_config of the original winner in config space
        distances = np.array([
            config_distance(w, self.original_winner, task)
            for w in self.bootstrap_winners
        ])
        self.winner_stability = float(np.mean(distances < eps_config))
        self.mean_winner_distance = float(np.mean(distances))

        # --- ε-optimal-set coverage:
        #     loss range across bootstrap val losses
        boot_losses = np.array(self.bootstrap_val_losses)
        loss_range = boot_losses.max() - boot_losses.min() + 1e-8
        threshold = self.original_val_loss + eps_loss * loss_range
        self.eps_optimal_coverage = float(np.mean(boot_losses <= threshold))

        # --- Selection entropy: discretize config space into bins
        #     using winner distances; compute entropy over distance quantiles
        n_bins = min(10, B // 5)
        if n_bins < 2:
            self.selection_entropy = 0.0
            return
        hist, _ = np.histogram(distances, bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(n_bins)
        self.selection_entropy = float(entropy / max_entropy if max_entropy > 0 else 0.0)

    def summary(self) -> Dict:
        return {
            "task":               self.task_name,
            "dataset":            self.dataset_name,
            "searcher":           self.searcher_name,
            "budget":             self.budget,
            "B":                  self.B,
            "val_loss":           self.original_val_loss,
            "test_loss":          self.test_loss,
            "gen_gap":            self.generalization_gap,
            "winner_stability":   self.winner_stability,
            "eps_coverage":       self.eps_optimal_coverage,
            "selection_entropy":  self.selection_entropy,
            "mean_winner_dist":   self.mean_winner_distance,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def _single_bootstrap_run(
    task: HPOTask,
    data: DataSplit,
    searcher: Searcher,
    budget: int,
    seed: int,
) -> Tuple[Dict[str, float], float]:
    """One bootstrap resample + HPO run. Returns (winner_config, val_loss)."""
    rng = np.random.default_rng(seed)
    boot_data = data.bootstrap_resample(rng)
    result = searcher.search(task, boot_data, budget, rng)
    return result.winner_config, result.winner_val_loss


def estimate_selection_uncertainty(
    task: HPOTask,
    data: DataSplit,
    searcher: Searcher,
    budget: int,
    B: int = 200,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True,
) -> UncertaintyEstimate:
    """
    Main estimation function.

    1. Run HPO once on the original data split.
    2. Bootstrap-resample development data B times, rerun HPO each time.
    3. Evaluate original winner on test set.
    4. Compute uncertainty measures.
    """
    rng = np.random.default_rng(seed)

    # --- Original run ---
    original_result = searcher.search(task, data, budget, rng)

    # --- Bootstrap runs (parallelized) ---
    seeds = rng.integers(0, 2**31, size=B).tolist()

    if verbose:
        print(f"  Running {B} bootstrap resamples ({searcher.name}, budget={budget})...")

    boot_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_bootstrap_run)(task, data, searcher, budget, s)
        for s in seeds
    )

    boot_winners  = [r[0] for r in boot_results]
    boot_losses   = [r[1] for r in boot_results]

    # --- Test evaluation of original winner ---
    test_loss = task.evaluate_test(
        original_result.winner_config,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
    )

    # --- Assemble estimate ---
    est = UncertaintyEstimate(
        task_name=task.name,
        dataset_name=data.name,
        searcher_name=searcher.name,
        budget=budget,
        B=B,
        original_winner=original_result.winner_config,
        original_val_loss=original_result.winner_val_loss,
        bootstrap_winners=boot_winners,
        bootstrap_val_losses=boot_losses,
        test_loss=test_loss,
    )
    est.compute_measures(task)

    return est
