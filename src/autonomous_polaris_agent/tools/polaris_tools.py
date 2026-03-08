"""Polaris benchmark tools: load, inspect, evaluate."""

from typing import Any

from autonomous_polaris_agent.polaris_helpers import (
    evaluate_predictions as _evaluate_predictions,
    inspect_benchmark as _inspect_benchmark,
    load_benchmark as _load_benchmark,
    resolve_slug,
)


def load_benchmark(slug: str) -> tuple[Any, dict[str, Any]]:
    """Load a Polaris benchmark by slug and return (benchmark, inspection).

    Resolves URL or short slug to owner/name. Idempotent for same slug.
    """
    resolved = resolve_slug(slug)
    benchmark = _load_benchmark(resolved)
    inspection = _inspect_benchmark(benchmark)
    return benchmark, inspection


def inspect_benchmark(benchmark: Any) -> dict[str, Any]:
    """Return train/test sizes, target summary, modalities, task type, suggested metric."""
    return _inspect_benchmark(benchmark)


def evaluate_predictions(benchmark: Any, predictions: list[float]) -> dict[str, float]:
    """Run benchmark.evaluate(predictions) and return metric name -> score."""
    return _evaluate_predictions(benchmark, predictions)
