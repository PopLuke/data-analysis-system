"""Machine learning analysis helper functions."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


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

def run_regression(df: pd.DataFrame, target: str, features: list[str]) -> dict:
        model = LinearRegression()
        X = df[features].dropna()
        y = df[target].loc[X.index]
        model.fit(X, y)
        return {
            'predictions': model.predict(X),
            'r2_score': model.score(X, y),
            'coefficients': dict(zip(features, model.coef_))
        }
    
def run_classification(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    model = RandomForestClassifier(n_estimators=100)
    X = df[features].dropna()
    y = df[target].loc[X.index]
    model.fit(X, y)
    return {
        'predictions': model.predict(X),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
def run_pca(df: pd.DataFrame, n_components: int = 2) -> list:
    numeric_cols = df.select_dtypes(include=['number']).dropna()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(numeric_cols)
    return reduced.tolist()
