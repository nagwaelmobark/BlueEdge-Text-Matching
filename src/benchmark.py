"""Benchmark utilities for BlueEdge text matching."""

import time
import tracemalloc
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd


Pair = Tuple[str, str]


def measure_runtime(
    pairs: Iterable[Pair],
    similarity_function: Callable[[str, str], float],
    repeats: int = 30,
    warmup: int = 5
) -> dict:
    """Measure average runtime for a similarity function."""
    pairs = list(pairs)

    for _ in range(warmup):
        for text_a, text_b in pairs:
            similarity_function(text_a, text_b)

    pass_times = []

    for _ in range(repeats):
        start_time = time.perf_counter()

        for text_a, text_b in pairs:
            similarity_function(text_a, text_b)

        pass_times.append(time.perf_counter() - start_time)

    time_per_record_us = np.array(pass_times) / max(len(pairs), 1) * 1e6

    return {
        "mean_us": round(float(time_per_record_us.mean()), 3),
        "std_us": round(float(time_per_record_us.std()), 3),
        "median_us": round(float(np.median(time_per_record_us)), 3),
        "min_us": round(float(time_per_record_us.min()), 3),
        "max_us": round(float(time_per_record_us.max()), 3),
        "n_pairs": len(pairs),
        "repeats": repeats,
    }


def measure_peak_memory(
    pairs: Iterable[Pair],
    similarity_function: Callable[[str, str], float]
) -> dict:
    """Measure peak memory usage with tracemalloc."""
    pairs = list(pairs)

    tracemalloc.start()

    for text_a, text_b in pairs:
        similarity_function(text_a, text_b)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_memory_kb": round(peak / 1024, 3),
        "peak_memory_mb": round(peak / 1024 / 1024, 6),
    }


def benchmark_similarity_function(
    pairs: Iterable[Pair],
    similarity_function: Callable[[str, str], float],
    method_name: str,
    repeats: int = 30,
    warmup: int = 5
) -> dict:
    """Run runtime and memory benchmark for one similarity method."""
    runtime_results = measure_runtime(
        pairs=pairs,
        similarity_function=similarity_function,
        repeats=repeats,
        warmup=warmup
    )

    memory_results = measure_peak_memory(
        pairs=pairs,
        similarity_function=similarity_function
    )

    return {
        "method": method_name,
        **runtime_results,
        **memory_results,
    }


def benchmark_to_dataframe(results: list) -> pd.DataFrame:
    """Convert benchmark results to a dataframe."""
    return pd.DataFrame(results)


def save_benchmark_results(
    results_df: pd.DataFrame,
    output_path: str = "results/tables/benchmark_results.csv"
) -> None:
    """Save benchmark results to CSV."""
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
