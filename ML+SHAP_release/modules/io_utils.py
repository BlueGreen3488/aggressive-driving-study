# modules/io_utils.py
"""
I/O utilities for reproducible outputs.

Output folder rule:
- period="day", type_filter="motorway" -> outputs/day+motorway/
- period="day", type_filter=None       -> outputs/day/
- period="night", type_filter="trunk"  -> outputs/night+trunk/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def _folder_name(period: str, type_filter: Optional[str]) -> str:
    period = str(period).strip()
    if not type_filter:
        return period
    return f"{period}+{str(type_filter).strip()}"


def get_out_dirs(
    base_dir: str | Path,
    period: str,
    type_filter: Optional[str],
    make: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    """
    Create and return (run_dir, fig_dir, tab_dir, model_dir).
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / _folder_name(period, type_filter)
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    model_dir = run_dir / "models"

    if make:
        for d in (fig_dir, tab_dir, model_dir):
            d.mkdir(parents=True, exist_ok=True)

    return run_dir, fig_dir, tab_dir, model_dir


def safe_feature_filename(feature: str) -> str:
    """
    Convert a feature name into a filesystem-friendly filename stem.
    Example: 'road length' -> 'road_length'
    """
    return feature.strip().replace(" ", "_")
