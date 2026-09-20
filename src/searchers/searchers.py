"""
HPO Searchers
=============
All searchers expose a unified interface:

    result = searcher.search(task, data_split, budget, rng)

Returns a SearchResult with:
  - winner config
  - winner val loss
  - full trajectory [(config, val_loss), ...]
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.benchmarks.tasks import HPOTask, SearchSpace
from src.benchmarks.datasets import DataSplit


@dataclass
class SearchResult:
    winner_config: Dict[str, float]
    winner_val_loss: float
    trajectory: List[Tuple[Dict[str, float], float]]  # (config, val_loss)

    @property
    def n_evals(self) -> int:
        return len(self.trajectory)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Searcher:
    name: str

    def search(
        self,
        task: HPOTask,
        data: DataSplit,
        budget: int,
        rng: np.random.Generator,
    ) -> SearchResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

class RandomSearch(Searcher):
    name = "random_search"

    def search(self, task, data, budget, rng) -> SearchResult:
        trajectory = []
        best_loss = float("inf")
        best_config = None

        for _ in range(budget):
            config = {
                name: hp.sample(rng)
                for name, hp in task.search_space.items()
            }
            loss = task.evaluate(
                config, data.X_train, data.y_train, data.X_val, data.y_val
            )
            trajectory.append((config, loss))
            if loss < best_loss:
                best_loss = loss
                best_config = config

        return SearchResult(
            winner_config=best_config,
            winner_val_loss=best_loss,
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Bayesian Optimization (TPE via Optuna)
# ---------------------------------------------------------------------------

class BayesianOptimization(Searcher):
    name = "bayesian_opt"

    def search(self, task, data, budget, rng) -> SearchResult:
        trajectory = []
        seed = int(rng.integers(0, 2**31))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            config = {}
            for name, hp in task.search_space.items():
                if hp.log_scale:
                    config[name] = trial.suggest_float(name, hp.low, hp.high, log=True)
                else:
                    config[name] = trial.suggest_float(name, hp.low, hp.high)
            loss = task.evaluate(
                config, data.X_train, data.y_train, data.X_val, data.y_val
            )
            trajectory.append((config, loss))
            return loss

        study.optimize(objective, n_trials=budget, show_progress_bar=False)

        best = study.best_trial
        best_config = {
            name: best.params[name]
            for name in task.search_space
        }

        return SearchResult(
            winner_config=best_config,
            winner_val_loss=best.value,
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Successive Halving (a simple version)
# ---------------------------------------------------------------------------

class SuccessiveHalving(Searcher):
    """
    Simplified successive halving (SHA):
    - Start with `budget` random configs, each evaluated once
    - Keep top 1/eta fraction, promote to next round
    - Repeat until 1 config remains
    
    Budget here counts total evaluations.
    eta=3 gives: n -> n/3 -> n/9 -> ... -> 1
    """
    name = "successive_halving"

    def __init__(self, eta: int = 3):
        self.eta = eta

    def search(self, task, data, budget, rng) -> SearchResult:
        eta = self.eta
        # Number of initial configs: largest power of eta <= budget
        n_init = eta ** int(np.floor(np.log(budget) / np.log(eta)))
        n_init = max(n_init, eta)

        trajectory = []

        # Initial random configs
        configs = [
            {name: hp.sample(rng) for name, hp in task.search_space.items()}
            for _ in range(n_init)
        ]

        # Evaluate all initial configs
        losses = []
        for cfg in configs:
            loss = task.evaluate(cfg, data.X_train, data.y_train, data.X_val, data.y_val)
            losses.append(loss)
            trajectory.append((cfg, loss))

        # Successive halving rounds
        current_configs = configs
        current_losses = losses
        while len(current_configs) > 1:
            n_keep = max(1, len(current_configs) // eta)
            ranked = np.argsort(current_losses)[:n_keep]
            current_configs = [current_configs[i] for i in ranked]
            current_losses = [current_losses[i] for i in ranked]
            # Re-evaluate survivors (simulates additional resources)
            new_losses = []
            for cfg in current_configs:
                loss = task.evaluate(cfg, data.X_train, data.y_train, data.X_val, data.y_val)
                new_losses.append(loss)
                trajectory.append((cfg, loss))
            current_losses = new_losses

        best_idx = int(np.argmin(current_losses))
        return SearchResult(
            winner_config=current_configs[best_idx],
            winner_val_loss=current_losses[best_idx],
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SEARCHER_REGISTRY = {
    "random_search":      RandomSearch,
    "bayesian_opt":       BayesianOptimization,
    "successive_halving": SuccessiveHalving,
}

def get_searcher(name: str) -> Searcher:
    return SEARCHER_REGISTRY[name]()
