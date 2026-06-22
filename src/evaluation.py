"""Evaluation utilities for BlueEdge text matching."""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_predictions(y_true, predictions: dict) -> pd.DataFrame:
    """Evaluate model predictions using accuracy, precision, recall, and F1."""
    results = []

    for model_name, y_pred in predictions.items():
        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
        })

    return pd.DataFrame(results)


def get_best_model_name(results_df: pd.DataFrame, metric: str = "F1") -> str:
    """Return the best model name according to a selected metric."""
    return results_df.loc[results_df[metric].idxmax(), "Model"]


def add_predictions_to_dataframe(
    test_df: pd.DataFrame,
    predictions,
    prediction_col: str = "predicted"
) -> pd.DataFrame:
    """Attach predictions to a copy of the test dataframe."""
    out = test_df.copy()
    out[prediction_col] = predictions
    return out


def recall_by_error_type(
    eval_df: pd.DataFrame,
    label_col: str = "is_duplicate",
    error_col: str = "error_type",
    prediction_col: str = "predicted"
) -> pd.Series:
    """Calculate recall for each positive duplicate error category."""
    positive_df = eval_df[eval_df[label_col] == 1]

    return positive_df.groupby(error_col).apply(
        lambda group: round((group[prediction_col] == 1).mean(), 3)
    )


def false_positive_rate_by_error_type(
    eval_df: pd.DataFrame,
    label_col: str = "is_duplicate",
    error_col: str = "error_type",
    prediction_col: str = "predicted"
) -> pd.Series:
    """Calculate false positive rate for each negative category."""
    negative_df = eval_df[eval_df[label_col] == 0]

    return negative_df.groupby(error_col).apply(
        lambda group: round((group[prediction_col] == 1).mean(), 3)
    )


def summarize_model_results(results_df: pd.DataFrame) -> str:
    """Create a simple text summary for model comparison results."""
    best_model = get_best_model_name(results_df, metric="F1")
    best_f1 = results_df.loc[results_df["Model"] == best_model, "F1"].values[0]

    return f"Best model by F1-score: {best_model} with F1 = {best_f1:.4f}"
