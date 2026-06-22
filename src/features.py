"""Feature extraction utilities for BlueEdge text matching."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_COLUMNS = [
    "lev_sim",
    "jaro_winkler_sim",
    "fuzzy_ratio",
    "fuzzy_token_sort",
    "length_ratio",
    "tfidf_cosine",
]


def levenshtein_similarity(text_a: str, text_b: str) -> float:
    """Calculate normalized Levenshtein similarity."""
    if not text_a and not text_b:
        return 1.0

    distance = Levenshtein.distance(text_a, text_b)
    return 1 - distance / max(len(text_a), len(text_b), 1)


def length_ratio(text_a: str, text_b: str) -> float:
    """Calculate length similarity ratio between two strings."""
    len_a = len(text_a.replace(" ", ""))
    len_b = len(text_b.replace(" ", ""))

    if max(len_a, len_b) == 0:
        return 1.0

    return min(len_a, len_b) / max(len_a, len_b)


def add_similarity_features(
    df: pd.DataFrame,
    col1: str = "name_1_norm",
    col2: str = "name_2_norm"
) -> pd.DataFrame:
    """Add classical similarity features to the dataset."""
    out = df.copy()

    out["lev_sim"] = out.apply(
        lambda row: levenshtein_similarity(row[col1], row[col2]),
        axis=1
    )

    out["jaro_winkler_sim"] = out.apply(
        lambda row: JaroWinkler.similarity(row[col1], row[col2]),
        axis=1
    )

    out["fuzzy_ratio"] = out.apply(
        lambda row: fuzz.ratio(row[col1], row[col2]) / 100,
        axis=1
    )

    out["fuzzy_token_sort"] = out.apply(
        lambda row: fuzz.token_sort_ratio(row[col1], row[col2]) / 100,
        axis=1
    )

    out["length_ratio"] = out.apply(
        lambda row: length_ratio(row[col1], row[col2]),
        axis=1
    )

    return out


def fit_tfidf_vectorizer(
    train_df: pd.DataFrame,
    col1: str = "name_1_norm",
    col2: str = "name_2_norm"
) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on training text only to avoid leakage."""
    train_corpus = pd.concat([train_df[col1], train_df[col2]]).unique().tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        min_df=1
    )

    vectorizer.fit(train_corpus)
    return vectorizer


def calculate_tfidf_cosine(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    col1: str = "name_1_norm",
    col2: str = "name_2_norm"
) -> np.ndarray:
    """Calculate TF-IDF cosine similarity row by row."""
    vector_1 = vectorizer.transform(df[col1])
    vector_2 = vectorizer.transform(df[col2])

    similarities = np.array([
        cosine_similarity(vector_1[i], vector_2[i])[0, 0]
        for i in range(vector_1.shape[0])
    ])

    return similarities


def add_tfidf_feature(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    col1: str = "name_1_norm",
    col2: str = "name_2_norm"
) -> pd.DataFrame:
    """Add TF-IDF cosine similarity feature."""
    out = df.copy()
    out["tfidf_cosine"] = calculate_tfidf_cosine(out, vectorizer, col1, col2)
    return out
