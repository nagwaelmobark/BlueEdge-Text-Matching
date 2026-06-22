"""Model training utilities for BlueEdge text matching."""

import networkx as nx
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def add_split_blocks(
    df: pd.DataFrame,
    group_col_1: str = "group_id_1",
    group_col_2: str = "group_id_2"
) -> pd.DataFrame:
    """Create connected-component split blocks to avoid group leakage."""
    out = df.copy()

    graph = nx.Graph()
    graph.add_edges_from(out[[group_col_1, group_col_2]].values)

    components = list(nx.connected_components(graph))
    component_map = {
        group_id: component_index
        for component_index, component in enumerate(components)
        for group_id in component
    }

    out["split_block"] = out[group_col_1].map(component_map)
    return out


def group_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "split_block"
):
    """Perform group-based train/test split."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def check_group_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col_1: str = "group_id_1",
    group_col_2: str = "group_id_2"
) -> int:
    """Return the number of overlapping group IDs between train and test."""
    train_groups = set(train_df[group_col_1]) | set(train_df[group_col_2])
    test_groups = set(test_df[group_col_1]) | set(test_df[group_col_2])

    return len(train_groups & test_groups)


def get_default_models(random_state: int = 42) -> dict:
    """Return the default machine learning models used in the experiments."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5
        ),
    }


def scale_features(X_train, X_test):
    """Fit scaler on train data and transform both train and test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_models(models: dict, X_train, y_train) -> dict:
    """Train a dictionary of models."""
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models


def predict_models(models: dict, X_test) -> dict:
    """Generate predictions for all trained models."""
    predictions = {}

    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_test)

    return predictions
