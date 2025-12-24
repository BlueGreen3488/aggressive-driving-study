# -*- coding: utf-8 -*-
"""
Model training and evaluation utilities based on XGBoost.

This module provides:
- A reproducible train/validation/test split with optional early stopping
- k-fold cross-validation using R² as the evaluation metric

The implementation is intentionally conservative and transparent, prioritizing
interpretability and methodological clarity over aggressive hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------
# Single model training with optional validation split
# ---------------------------------------------------------------------
def train_model(
    X,
    y,
    *,
    test_size: float = 0.1,
    valid_size: Optional[float] = None,
    random_state: int = 42,
    early_stopping_rounds: int = 20,
    eval_metric: str = "rmse",
    verbose: bool = True,
    **xgb_kwargs: Any,
) -> Tuple[
    xgb.XGBRegressor,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
]:
    """
    Train an XGBoost regression model with reproducible data splitting.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.ndarray
        Target variable.
    test_size
        Fraction of samples reserved for the test set.
    valid_size
        Fraction of samples reserved for the validation set (relative to full data).
        If None, no explicit validation set is used.
    random_state
        Random seed controlling all data splits.
    early_stopping_rounds
        Number of rounds for early stopping based on validation loss.
    eval_metric
        Evaluation metric passed to XGBoost.
    verbose
        Whether to print training logs.
    **xgb_kwargs
        Additional keyword arguments passed to `xgb.XGBRegressor`.

    Returns
    -------
    model
        Trained XGBoost regressor.
    X_train, X_valid, X_test
        Feature subsets used for training, validation (if any), and testing.
    y_train, y_valid, y_test
        Target subsets corresponding to the feature splits.
    """
    # ----------------------
    # Train / test split
    # ----------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # ----------------------
    # Optional validation split
    # ----------------------
    if valid_size is not None and valid_size > 0.0:
        valid_size_adj = valid_size / (1.0 - test_size)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp,
            y_temp,
            test_size=valid_size_adj,
            random_state=random_state,
        )
        eval_set = [(X_valid, y_valid)]
    else:
        X_train, y_train = X_temp, y_temp
        X_valid, y_valid = None, None
        eval_set = None

    # ----------------------
    # Model definition
    # ----------------------
    model = xgb.XGBRegressor(
        random_state=random_state,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds if eval_set else None,
        **xgb_kwargs,
    )

    # ----------------------
    # Model fitting
    # ----------------------
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=verbose,
    )

    return (
        model,
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )


# ---------------------------------------------------------------------
# k-fold cross-validation (R²)
# ---------------------------------------------------------------------
def cross_val_r2(
    X,
    y,
    *,
    cv: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
    eval_metric: str = "rmse",
    **xgb_kwargs: Any,
) -> Tuple[np.ndarray, float, float]:
    """
    Perform k-fold cross-validation using R² as the performance metric.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.ndarray
        Target variable.
    cv
        Number of folds.
    random_state
        Random seed controlling fold shuffling.
    early_stopping_rounds
        Early stopping patience within each fold.
    eval_metric
        Evaluation metric passed to XGBoost.
    **xgb_kwargs
        Additional keyword arguments passed to `xgb.XGBRegressor`.

    Returns
    -------
    scores
        Array of R² scores across folds.
    mean_score
        Mean R² across folds.
    std_score
        Standard deviation of R² across folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBRegressor(
            random_state=random_state,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            **xgb_kwargs,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        y_pred = model.predict(X_valid)
        scores.append(r2_score(y_valid, y_pred))

    scores = np.asarray(scores, dtype=float)
    return scores, float(scores.mean()), float(scores.std())
