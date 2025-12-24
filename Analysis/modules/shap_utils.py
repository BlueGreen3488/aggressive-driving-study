# -*- coding: utf-8 -*-
"""
Utilities for computing SHAP values using tree-based explainers.

Backward-compatibility policy:
- Public function signature and return values are preserved.
- SHAP outputs are normalized to a consistent ndarray format.
"""

from __future__ import annotations

from typing import Tuple, Any
import numpy as np
import pandas as pd
import shap


def compute_shap(
    model,
    X_train: "pd.DataFrame",
    X_test: "pd.DataFrame",
    background_size: int = 5000,
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values for a trained tree-based model.

    Parameters
    ----------
    model
        Trained tree-based model (e.g., XGBoost regressor).
    X_train
        Training feature matrix (used for background sampling).
    X_test
        Test feature matrix to be explained.
    background_size
        Maximum number of background samples.

    Returns
    -------
    explainer
        SHAP TreeExplainer instance.
    shap_values
        SHAP values as a numpy array with shape [n_samples, n_features].
        For multiclass outputs, values are summed across classes.
    """
    # Basic input validation (non-breaking)
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)!r}.")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"X_test must be a pandas DataFrame, got {type(X_test)!r}.")
    if len(X_train) == 0:
        raise ValueError("X_train is empty; cannot construct SHAP background.")
    if background_size <= 0:
        raise ValueError("background_size must be a positive integer.")

    # Background sampling (deterministic)
    bg_n = min(len(X_train), int(background_size))
    background = X_train.sample(n=bg_n, random_state=42)

    # Build explainer
    try:
        explainer = shap.TreeExplainer(
            model,
            background,
            feature_perturbation="interventional",
        )
    except TypeError as e:
        raise TypeError(
            "Failed to construct TreeExplainer with "
            "feature_perturbation='interventional'. "
            "Please check your SHAP version."
        ) from e

    # Compute SHAP values (support both old and new APIs)
    if hasattr(explainer, "shap_values"):
        sv = explainer.shap_values(X_test)
    else:
        sv = explainer(X_test).values

    # Normalize output shape:
    # - regression: [n, p]
    # - classification: list of [n, p] -> sum over classes
    if isinstance(sv, list):
        sv = np.sum(np.stack(sv, axis=0), axis=0)

    sv = np.asarray(sv, dtype=float)

    if sv.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape {sv.shape}; expected 2D array [n_samples, n_features]."
        )

    return explainer, sv
