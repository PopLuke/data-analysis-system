"""Machine learning analysis helper functions."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# K-Means 聚类
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame, columns: list[str], n_clusters: int = 3) -> pd.DataFrame:
    """Run K-Means clustering on selected numeric columns.

    Returns a DataFrame with an added ``cluster`` column.
    """
    features = df[columns].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(scaled)

    result = features.copy()
    result["cluster"] = labels
    return result


# ---------------------------------------------------------------------------
# 线性回归
# ---------------------------------------------------------------------------

def run_regression(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Train a Linear Regression model and return evaluation metrics + predictions.

    Returns
    -------
    dict with keys:
        metrics  – dict of R², MAE, RMSE
        coef     – list of (feature, coefficient) tuples
        intercept – float
        predictions – DataFrame with actual / predicted columns (test split)
    """
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R²": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }

    coef = list(zip(features, [round(c, 4) for c in model.coef_]))

    predictions = pd.DataFrame({
        "实际值": y_test.values,
        "预测值": y_pred.round(4),
    }).head(50)

    return {
        "metrics": metrics,
        "coef": coef,
        "intercept": round(float(model.intercept_), 4),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# 随机森林分类
# ---------------------------------------------------------------------------

def run_classification(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Train a Random Forest Classifier and return evaluation metrics.

    Returns
    -------
    dict with keys:
        metrics       – dict of accuracy
        report_df     – classification report as DataFrame
        feature_importance – list of (feature, importance) tuples, sorted desc
        predictions   – DataFrame with actual / predicted columns (test split)
    """
    data = df[features + [target]].dropna()
    X = data[features]

    # Encode target if it is not numeric
    le = LabelEncoder()
    y = le.fit_transform(data[target].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)

    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in le.classes_],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(4)

    importance = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    importance = [(f, round(float(v), 4)) for f, v in importance]

    predictions = pd.DataFrame({
        "实际类别": le.inverse_transform(y_test),
        "预测类别": le.inverse_transform(y_pred),
    }).head(50)

    return {
        "metrics": {"准确率": accuracy},
        "report_df": report_df,
        "feature_importance": importance,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# PCA 降维
# ---------------------------------------------------------------------------

def run_pca(df: pd.DataFrame) -> dict:
    """Run PCA on all numeric columns.

    Returns
    -------
    dict with keys:
        variance_ratio  – list of explained variance ratios per component
        cumulative      – list of cumulative explained variance
        components_df   – DataFrame of component loadings
        transformed_df  – first 2 principal components for every row (head 50)
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2:
        raise ValueError("PCA 需要至少 2 个数值列")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    n_components = min(numeric.shape[1], numeric.shape[0], 10)
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(scaled)

    variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cumulative = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]

    component_names = [f"PC{i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(
        pca.components_.round(4),
        index=component_names,
        columns=numeric.columns,
    )

    pc_cols = component_names[:2] if n_components >= 2 else component_names
    transformed_df = pd.DataFrame(
        transformed[:, :len(pc_cols)],
        columns=pc_cols,
    ).round(4).head(50)

    return {
        "variance_ratio": variance_ratio,
        "cumulative": cumulative,
        "components_df": components_df,
        "transformed_df": transformed_df,
        "n_components": n_components,
    }
