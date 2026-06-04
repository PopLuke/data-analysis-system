"""Data cleaning helper functions."""

import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain missing values."""
    return df.dropna()


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Detect outliers in a numeric column using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)
