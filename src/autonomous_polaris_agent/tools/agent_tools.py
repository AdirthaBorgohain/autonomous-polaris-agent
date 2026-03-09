"""LangChain tools and stateful executors for the ReAct modelling agent."""

import json
from typing import Any, Literal

from langchain_core.tools import tool

from autonomous_polaris_agent.console import (
    training_with_hyperparameter_tuning_display,
)
from autonomous_polaris_agent.state import PipelinePlan
from autonomous_polaris_agent.tools.ml_tools import (
    create_ensemble_predictions as _create_ensemble,
)
from autonomous_polaris_agent.tools.ml_tools import (
    evaluate_predictions as _evaluate_predictions,
)
from autonomous_polaris_agent.tools.ml_tools import (
    train_and_predict as _train_and_predict,
)


def get_modelling_tools():
    """Return list of tools for the modelling agent (for bind_tools)."""

    @tool
    def list_available_models() -> str:
        """List available fingerprints, model types, and pipeline options. Call this to see what you can use for train_and_predict."""
        return _catalog_json()

    @tool
    def train_and_predict(
        fingerprint: str,
        model_type: str,
        task_type: Literal["regression", "classification"],
        prediction_format: Literal["labels", "probabilities"] = "labels",
        hyperparameters: dict[str, Any] | None = None,
        use_cv: bool = False,
        tune_param_grid: dict[str, list[Any]] | None = None,
        scale_features: bool = False,
        robust_scale: bool = False,
        use_pca: bool = False,
        n_components: int | float | None = None,
        feature_selection: str | None = None,
        n_features_to_select: int | float | None = None,
        variance_threshold: float | None = None,
        rationale: str = "",
    ) -> str:
        """Train an ML pipeline on the benchmark and get test predictions. Determine task_type from the benchmark metadata (target_types, readme, etc.) before calling. Pass prediction_format='probabilities' when main_metric is roc_auc/roc_auc_score/auroc; use 'labels' for balanced_accuracy/accuracy. Use fingerprint (e.g. morgan, morgan_counts, rdkit_descriptors, maccs, atom_pair, morgan_rdkit_concat), model_type (e.g. ridge, lasso, random_forest, extra_trees, svr, kernel_ridge, xgboost, lightgbm), optional hyperparameters (including featurizer params morgan_radius, morgan_n_bits—see list_available_models), and pipeline options. When use_cv=True you must pass tune_param_grid. Returns run id, test metrics, and when use_cv was used also best_hyperparameters and cv_score."""
        return "Use execute_train_and_predict via tool node with state."

    @tool
    def evaluate_predictions(run_id: str) -> str:
        """Evaluate predictions for a completed run (e.g. run_0). Pass the model_id from the last train_and_predict result. Returns metric names and scores."""
        return "Use execute_evaluate_predictions via tool node with state."

    @tool
    def create_ensemble_predictions(
        run_ids: list[str],
        task_type: Literal["regression", "classification"],
        prediction_format: Literal["labels", "probabilities"] = "labels",
        strategy: Literal["mean", "median"] = "mean",
    ) -> str:
        """Combine predictions from previous runs (e.g. run_0, run_1) using mean or median. Pass prediction_format matching the runs (probabilities for roc_auc benchmarks, labels otherwise). Determine task_type from the benchmark metadata (target_types, readme, etc.) before calling. Call evaluate_predictions on the returned ensemble run_id to get its metrics."""
        return "Use execute_create_ensemble via tool node with state."

    @tool
    def finish_and_report() -> str:
        """Call when you are done running experiments and want to generate the final report. You must have at least one completed run."""
        return "OK"

    return [
        list_available_models,
        train_and_predict,
        evaluate_predictions,
        create_ensemble_predictions,
        finish_and_report,
    ]


def _catalog_json() -> str:
    """Return the full catalog as JSON string for list_available_models."""
    catalog = {
        "fingerprints": [
            {
                "id": "morgan",
                "description": "ECFP; radius 2-3, n_bits 2048; standard for structure-activity",
            },
            {
                "id": "rdkit_descriptors",
                "description": "Small interpretable descriptor set",
            },
            {
                "id": "maccs",
                "description": "167-bit MACCS keys; quick structural filter",
            },
            {
                "id": "atom_pair",
                "description": "Atom types + graph distance; n_bits 2048",
            },
            {
                "id": "morgan_counts",
                "description": "Morgan count vector; tunable radius, n_bits",
            },
            {
                "id": "morgan_rdkit_concat",
                "description": "Morgan + RDKit descriptors concatenated; tunable radius, n_bits for Morgan part",
            },
        ],
        "model_types": [
            {
                "id": "ridge",
                "params": ["alpha"],
                "description": "Linear L2; good baseline, small n",
            },
            {
                "id": "random_forest",
                "params": ["n_estimators", "max_depth"],
                "description": "Tree ensemble",
            },
            {
                "id": "lasso",
                "params": ["alpha"],
                "description": "L1 linear; regression only",
            },
            {
                "id": "elastic_net",
                "params": ["alpha", "l1_ratio"],
                "description": "L1+L2 linear; sparsity",
            },
            {
                "id": "svr",
                "params": ["C", "gamma", "epsilon"],
                "description": "Kernel SVM; use scale_features true",
            },
            {
                "id": "kernel_ridge",
                "params": ["alpha", "gamma", "kernel"],
                "description": "Kernel ridge regression; use scale_features true",
            },
            {
                "id": "extra_trees",
                "params": ["n_estimators", "max_depth"],
                "description": "Extremely randomized trees; same param style as random_forest",
            },
            {
                "id": "hist_gradient_boosting",
                "params": ["max_iter", "max_depth", "learning_rate"],
                "description": "Fast histogram-based GB; tune aggressively for competitive results",
            },
            {
                "id": "xgboost",
                "params": [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                ],
                "description": "Gradient boosting; often strong on tabular/benchmarks",
            },
            {
                "id": "lightgbm",
                "params": [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                ],
                "description": "Gradient boosting; fast and often strong on tabular/benchmarks",
            },
        ],
        "pipeline_options": [
            {
                "id": "scale_features",
                "type": "bool",
                "description": "StandardScaler; recommend for ridge/SVR",
            },
            {
                "id": "robust_scale",
                "type": "bool",
                "description": "RobustScaler instead of StandardScaler; use when outliers or non-normality",
            },
            {
                "id": "use_pca",
                "type": "bool",
                "description": "Dimensionality reduction before model",
            },
            {
                "id": "n_components",
                "type": "int or 0-1 float",
                "description": "PCA components; only if use_pca true",
            },
            {
                "id": "feature_selection",
                "type": "variance|kbest|rfe",
                "description": "Applied after scaling, before PCA. variance: drop low-variance features (set variance_threshold). kbest: keep top k by F-regression (set n_features_to_select). rfe: recursive elimination (set n_features_to_select; slower).",
            },
            {
                "id": "n_features_to_select",
                "type": "int or 0-1 float",
                "description": "For kbest/rfe: number or fraction of features to keep",
            },
            {
                "id": "variance_threshold",
                "type": "float",
                "description": "For feature_selection=variance: drop features with variance below this",
            },
        ],
        "tuning_options": [
            {
                "id": "use_cv",
                "type": "bool",
                "description": "When true, run RandomizedSearchCV on the training set to tune hyperparameters.",
            },
            {
                "id": "tune_param_grid",
                "type": "dict[str, list]",
                "description": 'Required when use_cv is true. Dict mapping param name to list of values to search (e.g. {"alpha": [0.1, 1.0, 10.0]}). Keys must be among the model\'s params (see model_types).',
            },
        ],
        "featurizer_params": [
            {
                "id": "morgan_radius",
                "type": "int",
                "description": "Morgan radius (e.g. 2, 3). Applies to morgan, morgan_counts, and the Morgan part of morgan_rdkit_concat. Pass in hyperparameters.",
            },
            {
                "id": "morgan_n_bits",
                "type": "int",
                "description": "Fingerprint size (e.g. 1024, 2048, 4096). Applies to morgan, morgan_counts, morgan_rdkit_concat (Morgan part), and atom_pair. Pass in hyperparameters.",
            },
        ],
    }
    return json.dumps(catalog, indent=2)


def execute_list_available_models(state: dict[str, Any]) -> tuple[str, None]:
    """Stateful executor for list_available_models."""
    return _catalog_json(), None


def execute_train_and_predict(
    state: dict[str, Any], **kwargs: Any
) -> tuple[str, dict[str, Any] | None]:
    """Stateful executor: run train_and_predict. Returns (message_str, state_updates or None)."""
    benchmark = state.get("benchmark")
    if not benchmark:
        return json.dumps({"error": "No benchmark loaded"}), None
    out_dir = state.get("output_dir") or "outputs"
    artifacts_dir = f"{out_dir}/artifacts"
    completed = list(state.get("completed_runs") or [])
    run_id = len(completed)
    plan_dict = {
        "fingerprint": kwargs.get("fingerprint", "morgan"),
        "model_type": kwargs.get("model_type", "ridge"),
        "hyperparameters": kwargs.get("hyperparameters") or {},
        "rationale": kwargs.get("rationale", ""),
        "use_cv": kwargs.get("use_cv", False),
        "tune_param_grid": kwargs.get("tune_param_grid"),
        "scale_features": kwargs.get("scale_features", False),
        "robust_scale": kwargs.get("robust_scale", False),
        "use_pca": kwargs.get("use_pca", False),
        "n_components": kwargs.get("n_components"),
        "feature_selection": kwargs.get("feature_selection"),
        "n_features_to_select": kwargs.get("n_features_to_select"),
        "variance_threshold": kwargs.get("variance_threshold"),
    }
    plan = PipelinePlan(**plan_dict)
    task_type = kwargs.get("task_type", "regression")
    if plan.use_cv and plan.tune_param_grid:
        training_with_hyperparameter_tuning_display()
    prediction_format = kwargs.get("prediction_format", "labels")
    try:
        preds, result = _train_and_predict(
            benchmark=benchmark,
            plan=plan,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            task_type=task_type,
            prediction_format=prediction_format,
        )
        metrics = _evaluate_predictions(
            benchmark=benchmark,
            predictions=preds,
            prediction_format=prediction_format,
        )
        result_dict = result.model_dump()
        result_dict["metrics"] = metrics
        result_dict["metric_name"] = next(iter(metrics.keys()), "")
        result_dict["metric_value"] = metrics.get(result_dict["metric_name"], 0.0)
        msg_payload = {
            "model_id": result.model_id,
            "description": result.description,
            "metrics": metrics,
            "predictions_saved": True,
            "next_step": "Use these metrics to decide your next pipeline (e.g. try a different model or representation if this one underperformed), or create_ensemble_predictions, or finish_and_report.",
        }
        if result_dict.get("best_hyperparameters") is not None:
            msg_payload["best_hyperparameters"] = result_dict["best_hyperparameters"]
        if result_dict.get("cv_score") is not None:
            msg_payload["cv_score"] = result_dict["cv_score"]
        msg = json.dumps(msg_payload)
        state_updates = {
            "completed_runs": completed + [result_dict],
            "last_predictions": preds,
            "iteration": run_id + 1,
        }
        return msg, state_updates
    except Exception as e:
        return json.dumps({"error": str(e), "model_id": f"run_{run_id}_error"}), None


def execute_evaluate_predictions(
    state: dict[str, Any], run_id: str
) -> tuple[str, None]:
    """Stateful executor: evaluate a run by run_id from completed_runs."""
    completed = list(state.get("completed_runs") or [])
    benchmark = state.get("benchmark")
    if not benchmark:
        return json.dumps({"error": "No benchmark loaded"}), None
    for r in completed:
        rid = r.get("model_id") if isinstance(r, dict) else getattr(r, "model_id", "")
        if rid == run_id:
            m = r.get("metrics") if isinstance(r, dict) else getattr(r, "metrics", {})
            if m:
                return json.dumps(m), None
            return json.dumps(
                {"note": "Run has no metrics yet", "run_id": run_id}
            ), None
    return json.dumps({"error": f"Run {run_id} not found in completed_runs"}), None


def execute_create_ensemble_predictions(
    state: dict[str, Any],
    run_ids: list[str],
    task_type: Literal["regression", "classification"],
    prediction_format: Literal["labels", "probabilities"] = "labels",
    strategy: Literal["mean", "median"] = "mean",
) -> tuple[str, dict[str, Any] | None]:
    """Stateful executor: create ensemble and evaluate. Returns (message_str, state_updates or None)."""
    benchmark = state.get("benchmark")
    out_dir = state.get("output_dir") or "outputs"
    artifacts_dir = f"{out_dir}/artifacts"
    completed = list(state.get("completed_runs") or [])
    if not benchmark:
        return json.dumps({"error": "No benchmark loaded"}), None
    try:
        preds, result = _create_ensemble(
            artifacts_dir=artifacts_dir,
            run_ids=run_ids,
            strategy=strategy,
            task_type=task_type,
            prediction_format=prediction_format,
        )
        metrics = _evaluate_predictions(
            benchmark=benchmark,
            predictions=preds,
            prediction_format=prediction_format,
        )
        result_dict = result.model_dump()
        result_dict["metrics"] = metrics
        result_dict["metric_name"] = next(iter(metrics.keys()), "")
        result_dict["metric_value"] = metrics.get(result_dict["metric_name"], 0.0)
        msg = json.dumps(
            {
                "model_id": result.model_id,
                "description": result.description,
                "metrics": metrics,
            }
        )
        state_updates = {
            "completed_runs": completed + [result_dict],
            "last_predictions": preds,
        }
        return msg, state_updates
    except Exception as e:
        return json.dumps({"error": str(e)}), None


def execute_finish_and_report(state: dict[str, Any]) -> tuple[str, None]:
    """Stateful executor: signal to generate report. Refuses until num_models runs completed."""
    parsed = state.get("parsed") or {}
    num_models = (
        parsed.get("num_models", 1)
        if isinstance(parsed, dict)
        else getattr(parsed, "num_models", 1)
    )
    completed_runs = state.get("completed_runs") or []
    if len(completed_runs) < num_models:
        need = num_models - len(completed_runs)
        return (
            json.dumps(
                {
                    "status": "refused",
                    "message": f"You must run exactly {num_models} pipelines. You have completed {len(completed_runs)}. Run {need} more train_and_predict before calling finish_and_report.",
                }
            ),
            None,
        )
    return json.dumps(
        {"status": "finish", "message": "Proceed to report generator."}
    ), None


def dispatch_tool(
    state: dict[str, Any], tool_name: str, tool_args: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Dispatch a tool call to the correct stateful executor. Returns (content_str, state_updates or None)."""
    if tool_name == "list_available_models":
        return execute_list_available_models(state=state)
    if tool_name == "train_and_predict":
        return execute_train_and_predict(state=state, **tool_args)
    if tool_name == "evaluate_predictions":
        return execute_evaluate_predictions(state=state, **tool_args)
    if tool_name == "create_ensemble_predictions":
        return execute_create_ensemble_predictions(state=state, **tool_args)
    if tool_name == "finish_and_report":
        return execute_finish_and_report(state=state)
    return json.dumps({"error": f"Unknown tool: {tool_name}"}), None
