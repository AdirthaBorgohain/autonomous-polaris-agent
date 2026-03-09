"""Polaris benchmark tools: load, evaluate."""

from typing import Any

from autonomous_polaris_agent.polaris_helpers import (
    evaluate_predictions as _evaluate_predictions,
    get_benchmark_metadata as _get_benchmark_metadata,
    load_benchmark as _load_benchmark,
    resolve_slug,
)


def load_benchmark(slug: str) -> tuple[Any, dict[str, Any]]:
    """Load a Polaris benchmark by slug and return (benchmark, metadata).

    Resolves URL or short slug to owner/name. Metadata is from benchmark.model_dump_json().
    """
    resolved = resolve_slug(slug)
    benchmark = _load_benchmark(resolved)
    metadata = _get_benchmark_metadata(benchmark)
    return benchmark, metadata


def evaluate_predictions(
    benchmark: Any,
    predictions: list[float],
    prediction_format: str = "labels",
) -> dict[str, float]:
    """Run benchmark.evaluate and return metric name -> score."""
    return _evaluate_predictions(
        benchmark=benchmark,
        predictions=predictions,
        prediction_format=prediction_format,
    )
