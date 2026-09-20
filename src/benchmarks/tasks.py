"""
HPO Benchmark Tasks
====================
Each task exposes a unified interface:
    task.evaluate(config, X_train, y_train, X_val, y_val) -> float (val loss)
    task.search_space  -> dict of (name -> (low, high, log_scale))
    task.name

We use real sklearn datasets split into train/val/test.
The key design choice: evaluation is cheap enough to run B=200 bootstrap resamples.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

@dataclass
class HParam:
    """Single hyperparameter specification."""
    low: float
    high: float
    log_scale: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        if self.log_scale:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.low, self.high))


SearchSpace = Dict[str, HParam]


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

class HPOTask:
    name: str
    search_space: SearchSpace

    def evaluate(
        self,
        config: Dict[str, float],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Returns validation LOSS (lower is better)."""
        raise NotImplementedError

    def evaluate_test(
        self,
        config: Dict[str, float],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Returns test LOSS (lower is better)."""
        return self.evaluate(config, X_train, y_train, X_test, y_test)


# ---------------------------------------------------------------------------
# Task 1: SVM with RBF kernel
# ---------------------------------------------------------------------------

class SVMTask(HPOTask):
    """
    Search space: C in [1e-3, 1e3] (log), gamma in [1e-4, 1e1] (log).
    2D landscape — useful for visualization of winner distribution.
    """
    name = "svm_rbf"
    search_space: SearchSpace = {
        "C":     HParam(1e-3, 1e3,  log_scale=True),
        "gamma": HParam(1e-4, 1e1,  log_scale=True),
    }

    def evaluate(self, config, X_train, y_train, X_val, y_val) -> float:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=config["C"],
                gamma=config["gamma"],
                kernel="rbf",
                max_iter=5000,
            ))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        return 1.0 - accuracy_score(y_val, preds)


# ---------------------------------------------------------------------------
# Task 2: XGBoost
# ---------------------------------------------------------------------------

class XGBoostTask(HPOTask):
    """
    5D search space representative of real-world HPO complexity.
    """
    name = "xgboost"
    search_space: SearchSpace = {
        "learning_rate":  HParam(1e-3, 0.5,  log_scale=True),
        "max_depth":      HParam(2,    10,    log_scale=False),
        "subsample":      HParam(0.5,  1.0,   log_scale=False),
        "colsample":      HParam(0.5,  1.0,   log_scale=False),
        "reg_lambda":     HParam(1e-2, 10.0,  log_scale=True),
    }

    def evaluate(self, config, X_train, y_train, X_val, y_val) -> float:
        n_classes = len(np.unique(y_train))
        clf = XGBClassifier(
            learning_rate=config["learning_rate"],
            max_depth=int(round(config["max_depth"])),
            subsample=config["subsample"],
            colsample_bytree=config["colsample"],
            reg_lambda=config["reg_lambda"],
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            objective="multi:softmax" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            random_state=0,
        )
        clf.fit(X_train, y_train, verbose=False)
        preds = clf.predict(X_val)
        return 1.0 - accuracy_score(y_val, preds)


# ---------------------------------------------------------------------------
# Task 3: MLP
# ---------------------------------------------------------------------------

class MLPTask(HPOTask):
    """
    4D search space: learning rate, hidden layer size, alpha (L2), momentum.
    """
    name = "mlp"
    search_space: SearchSpace = {
        "learning_rate_init": HParam(1e-4, 0.1,   log_scale=True),
        "hidden_layer_size":  HParam(16,   256,   log_scale=True),
        "alpha":              HParam(1e-5, 1e-1,  log_scale=True),
        "momentum":           HParam(0.5,  0.99,  log_scale=False),
    }

    def evaluate(self, config, X_train, y_train, X_val, y_val) -> float:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(int(round(config["hidden_layer_size"])),),
                learning_rate_init=config["learning_rate_init"],
                alpha=config["alpha"],
                momentum=config["momentum"],
                max_iter=300,
                random_state=0,
            ))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        return 1.0 - accuracy_score(y_val, preds)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "svm_rbf":  SVMTask,
    "xgboost":  XGBoostTask,
    "mlp":      MLPTask,
}

def get_task(name: str) -> HPOTask:
    return TASK_REGISTRY[name]()
