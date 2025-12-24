# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities.

This module is intentionally backward-compatible with the original project code:
- `load_and_preprocess()` keeps the same signature and return values.
- Column names with spaces (e.g., "traffic volume") are preserved to avoid
  breaking existing feature lists.

The function assumes an input CSV that already contains the modelling table
(road-segment × time-period level), including:
- time (datetime-like)
- gamma (target)
- car_id (or a configurable group column)
- semantic proportions class_0 ... class_18
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------
# Helpers (kept as private functions, similar to the original structure)
# ---------------------------------------------------------------------
def _make_daynight(ts: pd.Series) -> pd.Series:
    """
    Night: 17:00–next day 07:00 -> 1
    Day: otherwise -> 0
    """
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().any():
        raise ValueError("Column 'time' contains non-parseable datetime values.")
    hh = ts.dt.hour
    return ((hh >= 17) | (hh < 7)).astype(int)


def _residualize_oneway(s: pd.Series, g: pd.Series) -> pd.Series:
    """One-way fixed-effect residualization: s - E[s | g]."""
    return s - s.groupby(g).transform("mean")


def _tag_holiday_series(
    ts: pd.Series,
    holiday_dates: Optional[Iterable[Union[str, pd.Timestamp, _dt.date]]] = None,
) -> pd.Series:
    """
    Mark whether each timestamp falls on a holiday (date-based).

    If holiday_dates is None, a small project-specific default set is used.
    """
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().any():
        raise ValueError("Column 'time' contains non-parseable datetime values.")

    dates = ts.dt.date
    if holiday_dates is None:
        holiday_set = {
            _dt.date(2023, 1, 1),
            _dt.date(2023, 1, 2),
            _dt.date(2023, 1, 7),
        }
    else:
        holiday_set = {pd.to_datetime(d).date() for d in holiday_dates}

    return pd.Series(dates.isin(holiday_set), index=ts.index)


def _normalize_type_filter(tf) -> Optional[List[str]]:
    """Normalize road-type filter to a list of lowercase strings."""
    if tf is None:
        return None
    if isinstance(tf, (list, tuple, set)):
        return [str(t).strip().lower() for t in tf]
    return [str(tf).strip().lower()]


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")


def _feature_engineering_from_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive aggregated semantic components from class_0..class_18.
    Names are preserved for backward compatibility.
    """
    class_cols = [f"class_{i}" for i in range(19)]
    _validate_required_columns(df, class_cols)

    df["motorway"] = df["class_0"]
    df["sidewalk"] = df["class_1"]
    df["construction"] = df[["class_2", "class_3", "class_4", "class_5"]].sum(axis=1)
    df["traffic signal"] = df[["class_6", "class_7"]].sum(axis=1)
    df["vegetation"] = df["class_8"]
    df["terrain"] = df["class_9"]
    df["sky"] = df["class_10"]
    df["human"] = df[["class_11", "class_12", "class_18"]].sum(axis=1)
    df["vehicle"] = df[["class_13", "class_14", "class_15", "class_16", "class_17"]].sum(axis=1)

    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to the standardized names used in modelling.
    This mapping mirrors the original code logic.
    """
    rename_map: Dict[str, str] = {
        "speed_ratio": "flow efficiency",
        "continuous_driving_time": "fatigue",
        "track_count": "traffic volume",
        "building_area": "building density",
        "floor_average": "building height",
        "length_meter":"road length",
        "entropy": "diversity",
    }
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------
# Public API (signature preserved)
# ---------------------------------------------------------------------
def load_and_preprocess(
    path,
    dtype,
    feature_cols,
    *,
    group_col="car_id",
    fe_residualize=True,
    period="all",        # "day" | "night" | "all"
    cal="all",           # "workday" | "holiday" | "all"
    time_min="2023-01-01",
    time_max="2023-01-07 23:59:59",
    holiday_dates=None,
    type_filter=None,    # e.g. "motorway" or ["motorway", "trunk"]
    type_col="type",
):
    """
    Load a prepared CSV and return:
    - X: predictors (residualized if enabled)
    - X_display: unresidualized predictors (useful for plotting)
    - y: target (residualized if enabled)

    This function is backward-compatible with the original pipeline.
    """
    df = pd.read_csv(path, dtype=dtype)
    _validate_required_columns(df, ["time", "gamma", group_col])

    # Parse time early
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Column 'time' contains non-parseable datetime values.")

    # Optional road type filtering (before feature engineering is also fine)
    type_list = _normalize_type_filter(type_filter)
    if type_list is not None:
        if type_col not in df.columns:
            raise KeyError(
                f"Road type filtering requested but column '{type_col}' was not found."
            )
        df[type_col] = df[type_col].astype(str).str.strip().str.lower()
        df = df[df[type_col].isin(type_list)].copy()
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No samples remain after filtering {type_col} in {type_list}.")

    # Feature engineering + renaming
    df = _feature_engineering_from_classes(df)
    df = _rename_columns(df)

    # Stable ordering
    df = df.sort_values([group_col, "time"]).reset_index(drop=True)

    # Optional time window
    if time_min is not None:
        df = df[df["time"] >= pd.to_datetime(time_min)]
    if time_max is not None:
        df = df[df["time"] <= pd.to_datetime(time_max)]
    df = df.reset_index(drop=True)

    # Period filter
    if period not in ("day", "night", "all"):
        raise ValueError("`period` must be one of: 'day', 'night', 'all'.")
    if period == "all":
        mask_period = pd.Series(True, index=df.index)
    else:
        is_night = _make_daynight(df["time"]).astype(bool)
        mask_period = is_night if period == "night" else ~is_night

    # Calendar filter
    if cal not in ("workday", "holiday", "all"):
        raise ValueError("`cal` must be one of: 'workday', 'holiday', 'all'.")
    if cal == "all":
        mask_cal = pd.Series(True, index=df.index)
    else:
        is_holiday = _tag_holiday_series(df["time"], holiday_dates=holiday_dates).astype(bool)
        mask_cal = (~is_holiday) if cal == "workday" else is_holiday

    df = df.loc[mask_period & mask_cal].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No samples remain after applying period/calendar/time filters.")

    # Validate requested features exist
    _validate_required_columns(df, list(feature_cols))

    X_display = df.loc[:, feature_cols].copy()

    if fe_residualize:
        y = _residualize_oneway(df["gamma"], df[group_col])
        X = pd.DataFrame(index=df.index)
        for col in feature_cols:
            X[col] = _residualize_oneway(df[col], df[group_col])
    else:
        y = df["gamma"].copy()
        X = df.loc[:, feature_cols].copy()

    return X, X_display, y
