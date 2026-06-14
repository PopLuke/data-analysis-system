"""Data cleaning helper functions."""

import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain missing values."""
    return df.dropna()


def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with column means."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns
    for col in numeric_cols:
        result[col] = result[col].fillna(result[col].mean())
    return result


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with column medians."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns
    for col in numeric_cols:
        result[col] = result[col].fillna(result[col].median())
    return result


def fill_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with the mode of each column."""
    result = df.copy()
    for col in result.columns:
        mode_values = result[col].mode(dropna=True)
        if not mode_values.empty:
            result[col] = result[col].fillna(mode_values.iloc[0])
    return result


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Detect outliers in a numeric column using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers from a numeric column using the IQR method."""
    mask = detect_outliers_iqr(df, column)
    return df.loc[~mask].copy()
