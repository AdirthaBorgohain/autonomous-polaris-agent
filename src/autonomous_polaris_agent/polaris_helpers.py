"""Polaris benchmark loading, slug resolution, and metadata helpers."""

import json
import re
from typing import Any

import polaris as po

# URL pattern: https://polarishub.io/benchmarks/owner/name or similar
POLARIS_BENCHMARK_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?polarishub\.io/benchmarks/([^/]+)/([^/?#]+)",
    re.IGNORECASE,
)
DEFAULT_OWNER = "polaris"


def _normalize_to_owner_name(slug: str) -> str:
    """Ensure slug has exactly two segments (owner/name). Polaris Hub expects owner/name."""
    parts = slug.strip().split("/")
    if len(parts) <= 2:
        return slug
    return "/".join(parts[-2:])


def resolve_slug(user_input: str) -> str:
    """Extract benchmark slug from a URL or return the input as slug.

    - https://polarishub.io/benchmarks/tdcommons/caco2-wang -> tdcommons/caco2-wang
    - https://polarishub.io/benchmarks/polaris/adme-fang-solu-1 -> polaris/adme-fang-solu-1
    - tdcommons/caco2-wang -> tdcommons/caco2-wang
    - polaris/adme-fang-solu-1 -> polaris/adme-fang-solu-1
    - polaris/tdcommons/caco2-wang -> tdcommons/caco2-wang (normalized to owner/name)
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
        return _normalize_to_owner_name(s)

    return f"{DEFAULT_OWNER}/{s}"


def load_benchmark(slug: str, verify_checksum: str = "verify_unless_zarr") -> Any:
    """Load a Polaris benchmark by slug. Returns the benchmark object."""
    return po.load_benchmark(slug, verify_checksum=verify_checksum)


def get_benchmark_metadata(benchmark: Any) -> dict[str, Any]:
    """Return benchmark metadata as a JSON-serializable dict for the agent.

    Uses polaris model_dump_json(); dataset is excluded via Field(exclude=True).
    Agent deduces task_type, metrics, strategy from this metadata.
    """
    return json.loads(benchmark.model_dump_json())


def evaluate_predictions(
    benchmark: Any,
    predictions: list[float],
    prediction_format: str = "labels",
) -> dict[str, float]:
    """Run benchmark.evaluate and return a dict of metric name -> score.

    For roc_auc benchmarks pass prediction_format='probabilities' (uses y_prob).
    For balanced_accuracy/accuracy pass prediction_format='labels' (uses y_pred).
    """
    if prediction_format == "probabilities":
        results = benchmark.evaluate(y_prob=predictions)
    else:
        results = benchmark.evaluate(y_pred=predictions)
    out = {}
    if hasattr(results, "results") and results.results is not None:
        df = results.results
        if hasattr(df, "columns") and "Metric" in df.columns and "Score" in df.columns:
            for _, row in df.iterrows():
                out[str(row["Metric"])] = float(row["Score"])
    return out if out else {"score": float("nan")}
