"""Thin Polaris benchmark loading and evaluation for Deep Agent tools."""

import json
from typing import Any

from autonomous_polaris_agent.polaris_helpers import (
    evaluate_predictions as _evaluate_predictions,
    get_benchmark_metadata as _get_benchmark_metadata,
    load_benchmark as _load_benchmark,
    resolve_slug,
)


def load_benchmark_metadata(slug: str) -> tuple[dict[str, Any], str]:
    """Load a Polaris benchmark and return (metadata, resolved_slug).

    Metadata includes n_train_datapoints, n_test_datapoints, target_types,
    main_metric, etc. Resolved slug is owner/name format.
    """
    resolved = resolve_slug(slug)
    benchmark = _load_benchmark(resolved)
    metadata = _get_benchmark_metadata(benchmark)
    return metadata, resolved


def evaluate_predictions_from_file(
    predictions_path: str,
    benchmark_slug: str,
    prediction_format: str = "labels",
) -> dict[str, float]:
    """Load predictions from a JSON file and evaluate against the benchmark.

    Args:
        predictions_path: Path to JSON file containing a list of predictions.
        benchmark_slug: Polaris benchmark slug (e.g. polaris/adme-fang-solu-1).
        prediction_format: 'labels' or 'probabilities' (use probabilities for roc_auc).

    Returns:
        Dict of metric name -> score.
    """
    with open(predictions_path) as f:
        predictions = json.load(f)
    benchmark = _load_benchmark(resolve_slug(benchmark_slug))
    return _evaluate_predictions(
        benchmark=benchmark,
        predictions=predictions,
        prediction_format=prediction_format,
    )
