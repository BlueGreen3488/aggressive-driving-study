# -*- coding: utf-8 -*-
"""
Utilities for computing and summarizing SHAP interaction values.

Backward-compatibility policy:
- Function names, signatures, defaults, and return types are preserved.
- No behavioural changes that would break existing pipelines.

This module is designed for tree-based explainers that support
`shap_interaction_values`, e.g., SHAP TreeExplainer for XGBoost.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import shap


def _validate_square_tensor(t: np.ndarray) -> None:
    """
    Validate that the tensor has shape [n_samples, n_features, n_features].
    """
    if not isinstance(t, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray, got {type(t)!r}.")
    if t.ndim != 3:
        raise ValueError(f"Expected [n_samples, n_features, n_features], got shape={t.shape}.")
    if t.shape[1] != t.shape[2]:
        raise ValueError("Last two dims must be square (n_features x n_features).")


def compute_interactions_chunked(
    explainer: Any,  # keep loose typing to avoid dependency on SHAP private classes
    X: "pd.DataFrame",
    *,
    chunk_size: int = 4096,
    max_rows: Optional[int] = None,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Compute SHAP interaction values in memory-safe chunks and (optionally) with row sampling.

    Parameters
    ----------
    explainer
        A SHAP explainer object providing a `shap_interaction_values(X)` method.
        Typically `shap.TreeExplainer(model)`.
    X
        Feature matrix (pandas DataFrame).
    chunk_size
        Number of rows per chunk.
    max_rows
        If provided and X is larger than this, a deterministic sample is drawn.
    random_state
        Random seed used when sampling rows.

    Returns
    -------
    np.ndarray
        SHAP interaction tensor with shape [n_samples, n_features, n_features].
        For multi-output explainers returning a list, the outputs are summed.
    """
    if not hasattr(explainer, "shap_interaction_values"):
        raise TypeError(
            "The provided explainer does not expose `shap_interaction_values`. "
            "Please pass a tree-based SHAP explainer (e.g., shap.TreeExplainer)."
        )
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)!r}.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer if provided.")

    # Optional sampling (deterministic)
    if max_rows is not None and len(X) > max_rows:
        X_use = X.sample(n=max_rows, random_state=random_state)
    else:
        X_use = X

    n = len(X_use)
    if n == 0:
        raise ValueError("X contains no rows after optional sampling.")
    p = X_use.shape[1]
    if p == 0:
        raise ValueError("X contains no feature columns.")

    out_chunks: List[np.ndarray] = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        X_chunk = X_use.iloc[start:stop, :]

        sv_int_chunk = explainer.shap_interaction_values(X_chunk)

        # Some explainers return a list for multi-output / multiclass settings.
        if isinstance(sv_int_chunk, list):
            sv_int_chunk = np.sum(np.stack(sv_int_chunk, axis=0), axis=0)

        _validate_square_tensor(sv_int_chunk)

        # Defensive shape check (features must match)
        if sv_int_chunk.shape[1] != p or sv_int_chunk.shape[2] != p:
            raise ValueError(
                "Interaction tensor feature dimension does not match X. "
                f"Expected p={p}, got shape={sv_int_chunk.shape}."
            )

        out_chunks.append(sv_int_chunk)

    sv_int = np.concatenate(out_chunks, axis=0) if len(out_chunks) > 1 else out_chunks[0]
    _validate_square_tensor(sv_int)
    return sv_int


def summarize_interactions(
    sv_int: np.ndarray,
    feature_names: List[str],
    *,
    agg: str = "mean_abs",  # 'mean_abs' or 'mean_signed'
) -> "pd.DataFrame":
    """
    Reduce the interaction tensor to a DataFrame with a score per unordered feature pair.

    Parameters
    ----------
    sv_int
        Interaction tensor of shape [n_samples, n_features, n_features].
    feature_names
        Feature names aligned with sv_int dimensions.
    agg
        Aggregation method:
        - "mean_abs": score = mean(|interaction|), also return signed mean
        - "mean_signed": score = mean(interaction), signed_mean equals score

    Returns
    -------
    pd.DataFrame
        Columns: ["feat_i", "feat_j", "score", "signed_mean"]
        Includes diagonal pairs (i == j) for completeness; caller may drop them.
    """
    _validate_square_tensor(sv_int)

    n, p, _ = sv_int.shape
    if p != len(feature_names):
        raise ValueError("feature_names length must match sv_int dim=1/2.")

    if agg == "mean_abs":
        M = np.mean(np.abs(sv_int), axis=0)
        signed = np.mean(sv_int, axis=0)
    elif agg == "mean_signed":
        M = np.mean(sv_int, axis=0)
        signed = M.copy()
    else:
        raise ValueError("agg must be 'mean_abs' or 'mean_signed'.")

    rows: List[Tuple[str, str, float, float]] = []
    for i in range(p):
        for j in range(i, p):
            rows.append(
                (feature_names[i], feature_names[j], float(M[i, j]), float(signed[i, j]))
            )

    df = pd.DataFrame(rows, columns=["feat_i", "feat_j", "score", "signed_mean"])
    df["is_diag"] = (df["feat_i"] == df["feat_j"]).astype(int)

    # Keep original sorting behaviour: non-diagonal first, then by score desc
    df = df.sort_values(["is_diag", "score"], ascending=[True, False]).reset_index(drop=True)
    return df[["feat_i", "feat_j", "score", "signed_mean"]]


def topk_pairs_for_dependence_plots(
    df_pairs: "pd.DataFrame",
    k: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Return top-k off-diagonal feature pairs (for plotting pairwise dependence).

    Parameters
    ----------
    df_pairs
        Output DataFrame from `summarize_interactions`.
    k
        Number of pairs.

    Returns
    -------
    list of tuples
        [(feat_i, feat_j, score), ...]
    """
    if not isinstance(df_pairs, pd.DataFrame):
        raise TypeError(f"df_pairs must be a pandas DataFrame, got {type(df_pairs)!r}.")
    if k <= 0:
        return []

    required = {"feat_i", "feat_j", "score"}
    missing = required - set(df_pairs.columns)
    if missing:
        raise KeyError(f"df_pairs is missing required columns: {sorted(missing)}")

    df_off = df_pairs[df_pairs["feat_i"] != df_pairs["feat_j"]]
    top = df_off.head(k)
    return list(zip(top["feat_i"].tolist(), top["feat_j"].tolist(), top["score"].tolist()))


def export_interactions_csv(df_pairs: "pd.DataFrame", path: str) -> str:
    """
    Export interaction summary DataFrame to CSV and return the resolved path.
    """
    import pathlib

    if not isinstance(df_pairs, pd.DataFrame):
        raise TypeError(f"df_pairs must be a pandas DataFrame, got {type(df_pairs)!r}.")

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(p, index=False)
    return str(p.resolve())
