"""Deep Agent tools: load_benchmark, evaluate_predictions."""

import json
import tempfile
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.tools import tool

from autonomous_polaris_agent.benchmark import (
    evaluate_predictions_from_file,
    load_benchmark_metadata,
)


@tool
def load_benchmark(slug: str) -> str:
    """Load a Polaris benchmark by slug and return metadata.

    Use this to understand the dataset before writing your training code.
    Returns JSON with n_train_datapoints, n_test_datapoints, target_types,
    main_metric, and other benchmark info. The slug can be owner/name
    (e.g. polaris/adme-fang-solu-1) or a URL; it will be resolved automatically.
    """
    try:
        metadata, resolved = load_benchmark_metadata(slug)
        out = dict(metadata)
        out["resolved_slug"] = resolved
        return json.dumps(out, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_evaluate_predictions_tool(
    backend: Any | None = None,
):
    """Create evaluate_predictions tool, sandbox-aware when backend is a SandboxBackendProtocol.

    When backend implements SandboxBackendProtocol, predictions_path is fetched from the
    sandbox before evaluation. Otherwise the path is read from the host filesystem.
    """

    @tool
    def evaluate_predictions(
        predictions_path: str,
        benchmark_slug: str,
        prediction_format: str = "labels",
    ) -> str:
        """Evaluate predictions from a file against a Polaris benchmark.

        Call this after your training script writes predictions to a JSON file.
        predictions_path: path to JSON file (list of floats for regression, ints or floats for classification).
        benchmark_slug: e.g. polaris/adme-fang-solu-1.
        prediction_format: use 'probabilities' when main_metric is roc_auc/roc_auc_score/auroc; use 'labels' otherwise.
        Returns JSON with metric names and scores.
        """
        try:
            path_to_use = predictions_path
            if isinstance(backend, SandboxBackendProtocol):
                responses = backend.download_files([predictions_path])
                if not responses or responses[0].error or responses[0].content is None:
                    return json.dumps(
                        {"error": f"Could not read predictions file from sandbox: {predictions_path}"},
                    )
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".json",
                    delete=False,
                ) as f:
                    f.write(responses[0].content)
                    path_to_use = f.name
                try:
                    metrics = evaluate_predictions_from_file(
                        predictions_path=path_to_use,
                        benchmark_slug=benchmark_slug,
                        prediction_format=prediction_format,
                    )
                    return json.dumps(metrics, indent=2)
                finally:
                    import os as _os

                    _os.unlink(path_to_use)

            metrics = evaluate_predictions_from_file(
                predictions_path=path_to_use,
                benchmark_slug=benchmark_slug,
                prediction_format=prediction_format,
            )
            return json.dumps(metrics, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return evaluate_predictions


@tool
def evaluate_predictions(
    predictions_path: str,
    benchmark_slug: str,
    prediction_format: str = "labels",
) -> str:
    """Evaluate predictions from a file against a Polaris benchmark.

    Call this after your training script writes predictions to a JSON file.
    predictions_path: path to JSON file (list of floats for regression, ints or floats for classification).
    benchmark_slug: e.g. polaris/adme-fang-solu-1.
    prediction_format: use 'probabilities' when main_metric is roc_auc/roc_auc_score/auroc; use 'labels' otherwise.
    Returns JSON with metric names and scores.
    """
    try:
        metrics = evaluate_predictions_from_file(
            predictions_path=predictions_path,
            benchmark_slug=benchmark_slug,
            prediction_format=prediction_format,
        )
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

