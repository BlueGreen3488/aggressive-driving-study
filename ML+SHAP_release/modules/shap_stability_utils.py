# -*- coding: utf-8 -*-
"""
Utilities for evaluating the stability of SHAP-based feature importance rankings.

Core function
-------------
shap_stability_check(...)

Two SHAP computation modes are supported:
1) use_fast_shap=True:
   Use XGBoost native pred_contribs (fast, recommended).
2) use_fast_shap=False:
   Use TreeExplainer via compute_shap (slower, but fully consistent
   with the main analysis pipeline).

Backward-compatibility policy:
- Public function signatures, defaults, and return keys are preserved.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Internal: stratified sampling (robust version)
# ---------------------------------------------------------------------
def _stratified_sample(
    X: pd.DataFrame,
    n: int,
    strata_cols: Optional[List[str]],
    seed: int,
) -> pd.DataFrame:
    cols = [c for c in (strata_cols or []) if c in X.columns]
    n = min(len(X), int(n))
    if n <= 0:
        raise ValueError("explain_n must be a positive integer.")

    # No valid strata columns -> simple random sampling
    if len(cols) == 0:
        return X.sample(n=n, random_state=seed)

    frac = n / len(X)
    parts = []

    # Keep missing-value strata and avoid empty groups
    for _, g in X.groupby(cols, dropna=False, observed=True):
        take = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(len(g), take), random_state=seed))

    out = pd.concat(parts, axis=0)
    return out.sample(n=min(len(out), n), random_state=seed)


# ---------------------------------------------------------------------
# Internal: SHAP matrices
# ---------------------------------------------------------------------
def _shap_matrix_fast(model, Xslice: pd.DataFrame) -> np.ndarray:
    """
    Fast SHAP using XGBoost native pred_contribs.
    """
    import xgboost as xgb

    if not hasattr(model, "get_booster"):
        raise TypeError(
            "Fast SHAP requires an XGBoost model exposing `get_booster()`."
        )

    booster = model.get_booster()

    # Use trees up to the best iteration if available
    try:
        it_range: Tuple[int, int] = (0, int(model.best_iteration_) + 1)
    except Exception:
        # XGBoost convention: non-positive range -> use all trees
        it_range = (0, 0)

    dmat = xgb.DMatrix(Xslice)
    contribs = booster.predict(
        dmat,
        pred_contribs=True,
        approx_contribs=True,
        iteration_range=it_range,
    )

    # Last column is the bias term
    return contribs[:, :-1]


def _shap_matrix_treeexplainer(
    model,
    X_train: pd.DataFrame,
    Xslice: pd.DataFrame,
    background_n: int,
) -> np.ndarray:
    """
    SHAP via TreeExplainer, using the existing compute_shap utility.
    Import is local to avoid hard dependency at module import time.
    """
    try:
        from modules.shap_utils import compute_shap
    except ImportError as e:
        raise ImportError(
            "Failed to import compute_shap from modules.shap_utils. "
            "Please ensure the project package structure is available."
        ) from e

    _, shap_values = compute_shap(
        model, X_train, Xslice, background_size=background_n
    )
    return shap_values


def _get_shap_values(
    model,
    X_train: pd.DataFrame,
    Xslice: pd.DataFrame,
    use_fast_shap: bool,
    background_n: int,
) -> np.ndarray:
    if use_fast_shap:
        return _shap_matrix_fast(model, Xslice)
    return _shap_matrix_treeexplainer(model, X_train, Xslice, background_n)


# ---------------------------------------------------------------------
# Internal: importance and ranking
# ---------------------------------------------------------------------
def _importance_and_rank(
    shap_vals: np.ndarray,
    feature_names: np.ndarray,
):
    imp = np.abs(shap_vals).mean(axis=0)
    rank_idx = np.argsort(-imp)  # descending
    return imp, rank_idx, feature_names[rank_idx]


# ---------------------------------------------------------------------
# Public API (signature unchanged)
# ---------------------------------------------------------------------
def shap_stability_check(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    explain_n: int = 20000,
    repeats: int = 5,
    strata_cols: Optional[List[str]] = None,
    use_fast_shap: bool = True,
    background_n: int = 800,
    topk: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate the stability of SHAP-based feature importance rankings.

    Procedure
    ---------
    - Draw a stratified sample of size `explain_n` from X_test as the baseline.
    - Repeat R times:
        * re-sample explain_n rows
        * compute SHAP values
        * compute feature importance and ranking
        * compare with the baseline ranking using Spearman rho
        * compute Top-K overlap
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if explain_n <= 0:
        raise ValueError("explain_n must be a positive integer.")
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer.")
    if topk <= 0:
        raise ValueError("topk must be a positive integer.")

    feature_names = np.array(X_test.columns)

    # Baseline
    X_base = _stratified_sample(
        X_test, explain_n, strata_cols, seed=random_seed
    )
    S_base = _get_shap_values(
        model, X_train, X_base, use_fast_shap, background_n
    )
    imp_base, rank_base, names_base = _importance_and_rank(
        S_base, feature_names
    )

    rhos: List[float] = []
    overlaps: List[float] = []
    base_topk = set(names_base[:topk])

    # Repeated resampling
    for r in range(1, repeats + 1):
        X_r = _stratified_sample(
            X_test, explain_n, strata_cols, seed=random_seed + r
        )
        S_r = _get_shap_values(
            model, X_train, X_r, use_fast_shap, background_n
        )
        _, rank_r, names_r = _importance_and_rank(
            S_r, feature_names
        )

        # Spearman rank correlation (index-based)
        rho, _ = spearmanr(rank_base, rank_r)
        rhos.append(float(rho))

        # Top-K overlap
        topk_r = set(names_r[:topk])
        overlaps.append(len(topk_r & base_topk) / float(topk))

    # Baseline importance table
    imp_table = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_SHAP": imp_base,
        }
    ).sort_values("mean_abs_SHAP", ascending=False, ignore_index=True)

    result: Dict[str, Any] = {
        "rho_list": rhos,
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "topk_overlap_list": overlaps,
        "topk_overlap_mean": float(np.mean(overlaps)),
        "topk": int(topk),
        "explain_n": int(min(len(X_test), explain_n)),
        "repeats": int(repeats),
        "used_strata_cols": [c for c in (strata_cols or []) if c in X_test.columns],
        "use_fast_shap": bool(use_fast_shap),
        "background_n": int(background_n),
        "importance_table": imp_table,
    }
    return result
