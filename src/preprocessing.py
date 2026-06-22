"""Preprocessing utilities for BlueEdge text matching."""

import re
import pandas as pd


def normalize_text(value: object) -> str:
    """Lowercase, remove punctuation-like symbols, and collapse spaces."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_normalized_columns(
    df: pd.DataFrame,
    col1: str = "name_1",
    col2: str = "name_2"
) -> pd.DataFrame:
    """Add normalized versions of the two name/text columns."""
    out = df.copy()
    out[f"{col1}_norm"] = out[col1].apply(normalize_text)
    out[f"{col2}_norm"] = out[col2].apply(normalize_text)
    return out
