"""ML pipeline tools: train, predict, evaluate, save model/plot."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from autonomous_polaris_agent.state import PipelinePlan, RunResult
from autonomous_polaris_agent.tools.featurizers import build_features

# Featurizer-only keys (used in build_features, not passed to sklearn)
FEATURIZER_HP_KEYS = frozenset({"radius", "morgan_radius", "n_bits", "morgan_n_bits"})

REGRESSORS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

# Valid constructor kwargs per model (sklearn); featurizer params excluded
RIDGE_KEYS = frozenset(
    {"alpha", "fit_intercept", "copy_X", "max_iter", "tol", "solver", "positive"}
)
RF_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    }
)
GB_KEYS = frozenset(
    {"n_estimators", "max_depth", "learning_rate", "min_samples_split", "random_state"}
)


def _model_hp_keys(model_type: str) -> frozenset[str]:
    key = (model_type or "ridge").lower().replace(" ", "_")
    if key == "random_forest":
        return RF_KEYS
    if key == "gradient_boosting":
        return GB_KEYS
    return RIDGE_KEYS


def _get_regressor(model_type: str, **kwargs) -> Any:
    key = (model_type or "ridge").lower().replace(" ", "_")
    cls = REGRESSORS.get(key, Ridge)
    allowed = _model_hp_keys(model_type)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _default_hyperparams(model_type: str, plan: PipelinePlan) -> dict[str, Any]:
    raw = dict(plan.hyperparameters) if plan.hyperparameters else {}
    # Drop featurizer-only and non-model keys before passing to sklearn
    allowed = _model_hp_keys(plan.model_type or "ridge")
    hp = {k: v for k, v in raw.items() if k in allowed}
    key = (plan.model_type or "ridge").lower().replace(" ", "_")
    if key == "ridge" and "alpha" not in hp:
        hp.setdefault("alpha", 1.0)
    if key == "random_forest" and "n_estimators" not in hp:
        hp.setdefault("n_estimators", 100)
    if key == "gradient_boosting" and "n_estimators" not in hp:
        hp.setdefault("n_estimators", 100)
    return hp


def train_and_predict(
    benchmark: Any,
    plan: PipelinePlan,
    artifacts_dir: str = "outputs/artifacts",
    run_id: int = 0,
    generate_plots: bool = True,
) -> tuple[list[float], RunResult]:
    """Train a model from plan on benchmark train split, predict on test; optionally use CV.

    Returns (predictions, RunResult with model_path and metrics placeholder).
    When generate_plots is True, saves train actual vs predicted, train target hist,
    test prediction hist, and optional CV results plot; sets plot_path and plot_paths.
    """
    train, test = benchmark.get_train_test_split()
    X_train = np.asarray(train.inputs)
    y_train = np.asarray(train.targets)
    X_test = np.asarray(test.inputs)

    radius = int((plan.hyperparameters or {}).get("morgan_radius", 2))
    n_bits = int((plan.hyperparameters or {}).get("morgan_n_bits", 2048))

    X_train_f = build_features(
        X_train,
        fingerprint=plan.fingerprint,
        radius=radius,
        n_bits=n_bits,
    )
    X_test_f = build_features(
        X_test,
        fingerprint=plan.fingerprint,
        radius=radius,
        n_bits=n_bits,
    )

    model_type = plan.model_type or "ridge"
    hp = _default_hyperparams(model_type, plan)
    search = None

    if plan.use_cv and plan.tune_over:
        param_grid = {}
        if "alpha" in plan.tune_over or not plan.tune_over:
            param_grid.setdefault("alpha", [0.1, 1.0, 10.0])
        if "n_estimators" in plan.tune_over:
            param_grid.setdefault("n_estimators", [50, 100])
        if param_grid:
            base = _get_regressor(
                model_type, **{k: v for k, v in hp.items() if k not in param_grid}
            )
            search = GridSearchCV(
                base, param_grid, cv=3, scoring="neg_mean_absolute_error"
            )
            search.fit(X_train_f, y_train)
            model = search.best_estimator_
        else:
            model = _get_regressor(model_type, **hp)
            model.fit(X_train_f, y_train)
    else:
        model = _get_regressor(model_type, **hp)
        model.fit(X_train_f, y_train)

    preds = model.predict(X_test_f).tolist()

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f"model_{run_id}.joblib")
    save_model(path=model_path, model=model, metadata={"plan": plan.model_dump()})

    plot_paths_list: list[str] = []
    if generate_plots:
        y_train_pred = model.predict(X_train_f)
        _plot_train_actual_vs_predicted(
            y_train=y_train,
            y_train_pred=y_train_pred,
            path=os.path.join(artifacts_dir, f"train_actual_vs_pred_{run_id}.png"),
        )
        plot_paths_list.append(os.path.join(artifacts_dir, f"train_actual_vs_pred_{run_id}.png"))
        _plot_train_target_histogram(
            y_train=y_train,
            path=os.path.join(artifacts_dir, f"train_target_hist_{run_id}.png"),
        )
        plot_paths_list.append(os.path.join(artifacts_dir, f"train_target_hist_{run_id}.png"))
        _plot_test_predictions_histogram(
            preds=preds,
            path=os.path.join(artifacts_dir, f"test_predictions_hist_{run_id}.png"),
        )
        plot_paths_list.append(os.path.join(artifacts_dir, f"test_predictions_hist_{run_id}.png"))
        if search is not None and hasattr(search, "cv_results_"):
            _plot_cv_results(
                cv_results=search.cv_results_,
                path=os.path.join(artifacts_dir, f"cv_results_{run_id}.png"),
            )
            plot_paths_list.append(os.path.join(artifacts_dir, f"cv_results_{run_id}.png"))

    return preds, RunResult(
        model_id=f"run_{run_id}",
        description=f"{plan.fingerprint} + {plan.model_type}",
        metric_name="",
        metric_value=0.0,
        metrics={},
        model_path=model_path,
        plot_path=plot_paths_list[0] if plot_paths_list else None,
        plot_paths=plot_paths_list if plot_paths_list else None,
        rationale=plan.rationale,
    )


def evaluate_predictions(benchmark: Any, predictions: list[float]) -> dict[str, float]:
    """Run benchmark.evaluate(predictions) and return metric name -> score."""
    from autonomous_polaris_agent.tools.polaris_tools import (
        evaluate_predictions as _eval,
    )

    return _eval(benchmark, predictions)


def save_model(path: str, model: Any, metadata: Optional[dict] = None) -> str:
    """Persist model (joblib) and optional metadata JSON. Returns path."""
    path = str(path)
    joblib.dump(model, path)
    if metadata is not None:
        meta_path = path.rsplit(".", 1)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    return path


def save_plot(path: str, figure: Any) -> str:
    """Save matplotlib figure to artifacts. Returns path."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(figure, "savefig"):
        figure.savefig(path)
    return path


def _plot_train_actual_vs_predicted(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    path: str,
) -> None:
    """Scatter: actual vs predicted on training set."""
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_train_pred, alpha=0.5, s=10)
    lims = [
        min(y_train.min(), y_train_pred.min()),
        max(y_train.max(), y_train_pred.max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Actual (train)")
    ax.set_ylabel("Predicted (train)")
    ax.set_title("Train: actual vs predicted")
    ax.legend()
    save_plot(path=path, figure=fig)
    plt.close(fig)


def _plot_train_target_histogram(y_train: np.ndarray, path: str) -> None:
    """Histogram of training targets."""
    fig, ax = plt.subplots()
    ax.hist(y_train.ravel(), bins=min(50, max(10, len(y_train) // 20)), edgecolor="white", alpha=0.8)
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    ax.set_title("Train target distribution")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def _plot_test_predictions_histogram(preds: list[float], path: str) -> None:
    """Histogram of test-set predictions (no true labels)."""
    fig, ax = plt.subplots()
    ax.hist(preds, bins=min(50, max(10, len(preds) // 20)), edgecolor="white", alpha=0.8)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Count")
    ax.set_title("Test prediction distribution")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def _plot_cv_results(cv_results: dict[str, Any], path: str) -> None:
    """Bar chart of CV mean score per parameter combination (e.g. alpha)."""
    mean_scores = cv_results.get("mean_test_score")
    if mean_scores is None or len(mean_scores) == 0:
        return
    # Prefer param_alpha for ridge; else use rank_test_score order
    param_keys = [k for k in cv_results if k.startswith("param_")]
    if not param_keys:
        x_labels = [str(i) for i in range(len(mean_scores))]
    else:
        # Use first param (e.g. param_alpha) for labels
        p = param_keys[0]
        x_labels = [str(cv_results[p][i]) for i in range(len(mean_scores))]
    fig, ax = plt.subplots()
    x_pos = range(len(mean_scores))
    ax.bar(x_pos, -np.asarray(mean_scores), tick_label=x_labels)
    ax.set_xlabel(param_keys[0].replace("param_", "") if param_keys else "Config")
    ax.set_ylabel("MAE (CV)")
    ax.set_title("Cross-validation: mean MAE per hyperparameter")
    plt.xticks(rotation=45, ha="right")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def save_metrics_comparison_plot(
    completed_runs: list[dict[str, Any]],
    output_dir: str,
    filename: str = "metrics_comparison.png",
) -> Optional[str]:
    """Bar chart of primary metric per run. Returns path or None if no metrics."""
    if not completed_runs:
        return None
    model_ids = []
    values = []
    metric_name = None
    for run in completed_runs:
        metrics = run.get("metrics") or {}
        if not metrics:
            continue
        if metric_name is None:
            metric_name = next(iter(metrics.keys()))
        model_ids.append(run.get("model_id", "?"))
        values.append(metrics.get(metric_name, float("nan")))
    if not model_ids or metric_name is None:
        return None
    fig, ax = plt.subplots()
    ax.bar(range(len(model_ids)), values, tick_label=model_ids)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Run")
    ax.set_title(f"Metric comparison: {metric_name}")
    plt.xticks(rotation=45, ha="right")
    path = os.path.join(output_dir, filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_plot(path=path, figure=fig)
    plt.close(fig)
    return path
