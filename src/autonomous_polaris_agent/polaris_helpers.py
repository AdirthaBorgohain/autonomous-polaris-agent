"""Polaris benchmark loading, slug resolution, and inspection helpers."""

import re
from typing import Any

import numpy as np
import polaris as po

# URL pattern: https://polarishub.io/benchmarks/owner/name or similar
POLARIS_BENCHMARK_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?polarishub\.io/benchmarks/([^/]+)/([^/?#]+)",
    re.IGNORECASE,
)
DEFAULT_OWNER = "polaris"


def resolve_slug(user_input: str) -> str:
    """Extract benchmark slug from a URL or return the input as slug.

    - https://polarishub.io/benchmarks/polaris/adme-fang-solu-1 -> polaris/adme-fang-solu-1
    - polaris/adme-fang-solu-1 -> polaris/adme-fang-solu-1
    - adme-fang-solu-1 -> polaris/adme-fang-solu-1 (default owner)

    Raises:
        ValueError: If input is empty or not a valid slug/URL.
    """
    s = (user_input or "").strip()
    if not s:
        raise ValueError("Empty benchmark slug or URL")

    match = POLARIS_BENCHMARK_URL_PATTERN.search(s)
    if match:
        owner, name = match.group(1), match.group(2)
        return f"{owner}/{name}"

    if "/" in s:
        return s

    return f"{DEFAULT_OWNER}/{s}"


def load_benchmark(slug: str, verify_checksum: str = "verify_unless_zarr") -> Any:
    """Load a Polaris benchmark by slug. Returns the benchmark object."""
    return po.load_benchmark(slug, verify_checksum=verify_checksum)


def inspect_benchmark(benchmark: Any) -> dict[str, Any]:
    """Inspect a loaded benchmark: sizes, target info, task type, suggested metric.

    Returns a dict with: train_size, test_size, target_cols, target_summary,
    task_type, suggested_metric, input_cols (modalities).
    """
    train, test = benchmark.get_train_test_split()
    train_size = len(train.inputs) if hasattr(train.inputs, "__len__") else sum(1 for _ in train)
    test_size = len(test.inputs) if hasattr(test.inputs, "__len__") else sum(1 for _ in test)

    target_cols = getattr(train, "target_cols", None) or []
    input_cols = getattr(train, "input_cols", None) or []
    targets = np.asarray(train.targets)

    if targets.size == 0:
        target_summary = {"dtype": str(targets.dtype), "note": "no targets"}
        task_type = "unknown"
    elif np.issubdtype(targets.dtype, np.floating) or np.issubdtype(targets.dtype, np.integer):
        if np.issubdtype(targets.dtype, np.floating) and targets.ndim == 1:
            target_summary = {
                "dtype": str(targets.dtype),
                "mean": float(np.nanmean(targets)),
                "std": float(np.nanstd(targets)) if targets.size > 1 else 0.0,
                "min": float(np.nanmin(targets)),
                "max": float(np.nanmax(targets)),
            }
        else:
            target_summary = {"dtype": str(targets.dtype), "shape": targets.shape}
        task_type = "regression"
    else:
        unique = np.unique(targets)
        target_summary = {
            "dtype": str(targets.dtype),
            "n_unique": int(len(unique)),
            "classes_sample": unique.tolist()[:10],
        }
        task_type = "classification"

    if task_type == "regression":
        suggested_metric = "mean_absolute_error"
    else:
        suggested_metric = "balanced_accuracy"

    return {
        "train_size": train_size,
        "test_size": test_size,
        "target_cols": target_cols,
        "input_cols": input_cols,
        "target_summary": target_summary,
        "task_type": task_type,
        "suggested_metric": suggested_metric,
    }


def evaluate_predictions(benchmark: Any, predictions: list[float]) -> dict[str, float]:
    """Run benchmark.evaluate(predictions) and return a dict of metric name -> score."""
    results = benchmark.evaluate(predictions)
    out = {}
    if hasattr(results, "results") and results.results is not None:
        df = results.results
        if hasattr(df, "columns") and "Metric" in df.columns and "Score" in df.columns:
            for _, row in df.iterrows():
                out[str(row["Metric"])] = float(row["Score"])
    return out if out else {"score": float("nan")}
