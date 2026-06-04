"""Utilities for loading uploaded data files."""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}")
