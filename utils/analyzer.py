"""Machine learning analysis helper functions."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_kmeans(df: pd.DataFrame, columns: list[str], n_clusters: int = 3) -> pd.DataFrame:
    """Run K-Means clustering on selected numeric columns."""
    features = df[columns].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(scaled_features)

    result = features.copy()
    result["cluster"] = labels
    return result
