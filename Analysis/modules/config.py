# modules/config.py
"""
Project configuration for paper reproducibility.

Keep dataset schema (DTYPE), modeling features (FEATURE_COLS), and plotting
limits (DEPENDENCE_LIMITS) in one place to keep the main notebook concise.
"""

from __future__ import annotations


# --------- Fixed CSV schema (optional but recommended for reproducibility) ---------
DTYPE = {
    "car_id": "int64",
    "azimuth": "int64",
    "velocity": "float64",
    "altitude": "int64",
    "mileage": "float64",
    "lat": "float64",
    "lon": "float64",
    "sample_number": "int64",
    "time_interval": "int64",
    "acceleration": "float64",
    "jerk": "float64",
    "section": "category",
    "intersection": "category",
    "road": "category",
    "continuous_driving_time": "int64",
    "start_number": "int64",
    "end_number": "int64",
    "gamma": "float64",
    "behavior": "category",
    "time_period": "category",
    "direction": "float64",
    "trajectory_count": "int64",
    "length_meter": "float64",
    "avg_speed": "float64",
    "non_congested_speed": "float64",
    "speed_ratio": "float64",
    "jerk_mean_abs": "float64",
    "entropy": "float64",
    "track_count": "int64",
    "pm25": "int64",
    "region": "int64",
    "building_area": "float64",
    "floor_area": "float64",
}


# --------- Fixed model features used in the paper ---------
FEATURE_COLS = [
    "velocity",
    "road length",
    "flow efficiency",
    "diversity",
    "motorway",
    "sidewalk",
    "construction",
    "traffic signal",
    "vegetation",
    "sky",
    "human",
    "vehicle",
    "traffic volume",
    "pm25",
    "building density",
    "building height",
]


# --------- Dependence plot limits (paper-ready) ---------
# Feature name must match the column name used in X (training/SHAP input).
DEPENDENCE_LIMITS = {
    "velocity": {"xlim": (-5, 120), "ylim": (-0.6, 1.2)},
    "road length": {"xlim": (-50, 1200), "ylim": (-0.2, 0.2)},
    "flow efficiency": {"xlim": (-0.05, 2), "ylim": (-0.2, 0.2)},
    "diversity": {"xlim": (0.9, 2), "ylim": (-0.2, 0.2)},
    "motorway": {"xlim": (0.15, 0.5), "ylim": (-0.1, 0.1)},
    "sidewalk": {"xlim": (-0.01, 0.1), "ylim": (-0.1, 0.1)},
    "construction": {"xlim": (-0.05, 0.6), "ylim": (-0.2, 0.2)},
    "traffic signal": {"xlim": (-0.001, 0.02), "ylim": (-0.2, 0.2)},
    "vegetation": {"xlim": (-0.05, 0.7), "ylim": (-0.2, 0.2)},
    "sky": {"xlim": (-0.05, 0.5), "ylim": (-0.2, 0.2)},
    "human": {"xlim": (-0.005, 0.03), "ylim": (-0.2, 0.2)},
    "vehicle": {"xlim": (-0.01, 0.2), "ylim": (-0.2, 0.2)},
    "pm25": {"xlim": (0, 140), "ylim": (-0.1, 0.1)},
    "building density": {"xlim": (-500, 15000), "ylim": (-0.2, 0.2)},
    "building height": {"xlim": (-5, 30), "ylim": (-0.2, 0.2)},
    "traffic volume": {"xlim": (-1000, 30000), "ylim": (-0.4, 0.4)},
}


# --------- XGBoost parameter presets used in the paper ---------
# Note: keep keys consistent with model_utils.train_model / cross_val_r2.

XGB_PARAMS_DEFAULT = dict(
    n_estimators=4000,
    learning_rate=0.025,
    max_depth=7,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_lambda=6,
    reg_alpha=0.1,
)

# Only used for: period="night" AND no road-type filtering (type_filter is None/empty).
XGB_PARAMS_NIGHT_ALL = dict(
    n_estimators=4000,
    learning_rate=0.025,
    max_depth=6,
    min_child_weight=4,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.2,
    reg_lambda=8,
    reg_alpha=0.3,
)


def select_xgb_params(period: str, type_filter: str | None) -> dict:
    """
    Select XGBoost hyperparameters based on experiment condition.

    Rule (as used in the paper):
    - night + all roads (type_filter is None/empty): use XGB_PARAMS_NIGHT_ALL
    - otherwise: use XGB_PARAMS_DEFAULT
    """
    period = str(period).strip().lower()
    tf = None if type_filter is None else str(type_filter).strip()
    if period == "night" and not tf:
        return dict(XGB_PARAMS_NIGHT_ALL)
    return dict(XGB_PARAMS_DEFAULT)

