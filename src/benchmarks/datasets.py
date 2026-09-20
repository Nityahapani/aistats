"""
Dataset Loading
===============
Loads sklearn/OpenML datasets and returns consistent train/val/test splits.

Design: we use a fixed outer test set (20%) and a development pool (80%).
The val split is carved from the development pool and is the object of
bootstrap resampling in the uncertainty experiments.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_iris,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    name:    str

    @property
    def X_dev(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_val])

    @property
    def y_dev(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val])

    def bootstrap_resample(self, rng: np.random.Generator, val_frac: float = 0.3) -> "DataSplit":
        """
        Resample the development set; return a new DataSplit with the same test set.
        This is the key operation for estimating winner-selection uncertainty.
        """
        X_dev, y_dev = self.X_dev, self.y_dev
        n = len(X_dev)
        idx = rng.integers(0, n, size=n)  # bootstrap resample
        X_boot, y_boot = X_dev[idx], y_dev[idx]
        # Re-split into train/val
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_boot, y_boot, test_size=val_frac, random_state=int(rng.integers(0, 2**31)),
            stratify=y_boot if len(np.unique(y_boot)) > 1 else None,
        )
        return DataSplit(
            X_train=X_tr, y_train=y_tr,
            X_val=X_v,   y_val=y_v,
            X_test=self.X_test, y_test=self.y_test,
            name=self.name,
        )


def _make_split(X: np.ndarray, y: np.ndarray, name: str, seed: int = 42) -> DataSplit:
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Outer hold-out test set — never touched during HPO
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y,
    )
    # Development: 70% train, 30% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.30, random_state=seed + 1, stratify=y_dev,
    )
    return DataSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        name=name,
    )


DATASET_LOADERS = {
    "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
    "wine":          lambda: load_wine(return_X_y=True),
    "digits":        lambda: load_digits(return_X_y=True),
    "iris":          lambda: load_iris(return_X_y=True),
}


def load_dataset(name: str, seed: int = 42) -> DataSplit:
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS)}")
    X, y = DATASET_LOADERS[name]()
    return _make_split(np.array(X, dtype=float), np.array(y), name, seed=seed)


def list_datasets():
    return list(DATASET_LOADERS.keys())
