"""Data cleaning helper functions."""

import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()


def build_missing_value_summary(df: pd.DataFrame) -> list[dict]:
    """Build a JSON-friendly missing value summary for each column."""
    row_count = len(df)
    missing_counts = missing_value_summary(df)

    result = []
    for column, count in missing_counts.items():
        missing_count = int(count)
        missing_rate = 0 if row_count == 0 else round(missing_count / row_count, 4)
        result.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
            }
        )
    return result


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain missing values."""
    return df.dropna()


def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle missing values according to the selected strategy."""
    cleaned_df = df.copy()

    if strategy == "drop":
        return drop_missing_rows(cleaned_df).reset_index(drop=True)

    if strategy in {"mean", "median"}:
        numeric_columns = cleaned_df.select_dtypes(include="number").columns
        if strategy == "mean":
            fill_values = cleaned_df[numeric_columns].mean()
        else:
            fill_values = cleaned_df[numeric_columns].median()
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(fill_values)
        return cleaned_df

    if strategy == "mode":
        for column in cleaned_df.columns:
            mode_values = cleaned_df[column].mode(dropna=True)
            if not mode_values.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode_values.iloc[0])
        return cleaned_df

    raise ValueError(f"Unsupported missing value strategy: {strategy}")


def calculate_iqr_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Calculate IQR lower and upper bounds for a numeric column."""
    _validate_numeric_column(df, column)
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return float(lower_bound), float(upper_bound)


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Detect outliers in a numeric column using the IQR method."""
    lower_bound, upper_bound = calculate_iqr_bounds(df, column)
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove rows whose selected column is an IQR outlier."""
    outlier_mask = detect_outliers_iqr(df, column)
    return df.loc[~outlier_mask].reset_index(drop=True)


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column does not exist: {column}")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column is not numeric: {column}")
