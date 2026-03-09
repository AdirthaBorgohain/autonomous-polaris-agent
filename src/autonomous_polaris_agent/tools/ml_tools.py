"""ML pipeline tools: train, predict, evaluate, save model/plot."""

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

from autonomous_polaris_agent.state import PipelinePlan, RunResult
from autonomous_polaris_agent.tools.featurizers import build_features

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

# Featurizer-only keys (used in build_features, not passed to sklearn)
FEATURIZER_HP_KEYS = frozenset({"radius", "morgan_radius", "n_bits", "morgan_n_bits"})

REGRESSORS: dict[str, Any] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "elastic_net": ElasticNet,
    "svr": SVR,
    "kernel_ridge": KernelRidge,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}
if XGB_AVAILABLE:
    REGRESSORS["xgboost"] = xgb.XGBRegressor
if LGBM_AVAILABLE:
    REGRESSORS["lightgbm"] = lgb.LGBMRegressor

CLASSIFIERS: dict[str, Any] = {
    "ridge": LogisticRegression,
    "lasso": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "elastic_net": LogisticRegression,
    "svr": SVC,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}
if XGB_AVAILABLE:
    CLASSIFIERS["xgboost"] = xgb.XGBClassifier
if LGBM_AVAILABLE:
    CLASSIFIERS["lightgbm"] = lgb.LGBMClassifier

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
EXTRA_TREES_KEYS = RF_KEYS
LASSO_KEYS = frozenset({"alpha", "fit_intercept", "max_iter", "tol", "random_state"})
KERNEL_RIDGE_KEYS = frozenset({"alpha", "gamma", "kernel"})
ELASTIC_NET_KEYS = frozenset(
    {"alpha", "l1_ratio", "fit_intercept", "max_iter", "tol", "random_state"}
)
SVR_KEYS = frozenset({"C", "epsilon", "gamma", "kernel", "max_iter"})
HIST_GB_KEYS = frozenset(
    {"max_iter", "max_depth", "learning_rate", "min_samples_leaf", "random_state"}
)
XGB_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
    }
)
LGBM_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "min_child_samples",
        "num_leaves",
        "verbosity",
    }
)

LOGISTIC_KEYS = frozenset(
    {
        "C",
        "fit_intercept",
        "max_iter",
        "penalty",
        "random_state",
        "solver",
        "tol",
        "l1_ratio",
    }
)
SVC_KEYS = frozenset(
    {"C", "gamma", "kernel", "max_iter", "probability", "random_state"}
)


# Param names that must be integers for sklearn
_INT_PARAM_NAMES = frozenset(
    {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_iter",
        "random_state",
        "min_child_samples",
        "num_leaves",
        "verbosity",
    }
)


def _coerce_param_values(param_name: str, values: list[Any]) -> list[Any]:
    """Coerce list of values for a param (e.g. n_estimators -> int). Drops invalid."""
    out: list[Any] = []
    for v in values:
        try:
            if param_name in _INT_PARAM_NAMES:
                out.append(int(v))
            else:
                out.append(v)
        except (TypeError, ValueError):
            continue
    return out


def _product_len(param_grid: dict[str, list[Any]]) -> int:
    """Number of combinations in a param grid (product of list lengths)."""
    if not param_grid:
        return 0
    n = 1
    for values in param_grid.values():
        n *= max(1, len(values))
    return n


def _model_hp_keys(model_type: str) -> frozenset[str]:
    key = (model_type or "ridge").lower().replace(" ", "_")
    if key == "random_forest":
        return RF_KEYS
    if key == "extra_trees":
        return EXTRA_TREES_KEYS
    if key == "lasso":
        return LASSO_KEYS
    if key == "elastic_net":
        return ELASTIC_NET_KEYS
    if key == "svr":
        return SVR_KEYS
    if key == "kernel_ridge":
        return KERNEL_RIDGE_KEYS
    if key == "hist_gradient_boosting":
        return HIST_GB_KEYS
    if key == "xgboost":
        return XGB_KEYS
    if key == "lightgbm":
        return LGBM_KEYS
    return RIDGE_KEYS


def _classifier_hp_keys(model_type: str) -> frozenset[str]:
    key = (model_type or "ridge").lower().replace(" ", "_")
    if key in ("ridge", "elastic_net", "lasso"):
        return LOGISTIC_KEYS
    if key == "random_forest":
        return RF_KEYS
    if key == "extra_trees":
        return EXTRA_TREES_KEYS
    if key == "svr":
        return SVC_KEYS
    if key == "hist_gradient_boosting":
        return HIST_GB_KEYS
    if key == "xgboost":
        return XGB_KEYS
    if key == "lightgbm":
        return LGBM_KEYS
    return LOGISTIC_KEYS


def _get_regressor(model_type: str, **kwargs) -> Any:
    key = (model_type or "ridge").lower().replace(" ", "_")
    cls = REGRESSORS.get(key, Ridge)
    allowed = _model_hp_keys(model_type)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _get_classifier(model_type: str, **kwargs) -> Any:
    key = (model_type or "ridge").lower().replace(" ", "_")
    cls = CLASSIFIERS.get(key, LogisticRegression)
    allowed = _classifier_hp_keys(model_type)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if key == "elastic_net":
        filtered.setdefault("penalty", "elasticnet")
        filtered.setdefault("solver", "saga")
    if key == "lasso":
        filtered.setdefault("penalty", "l1")
        filtered.setdefault("solver", "saga")
    return cls(**filtered)


def _get_estimator(task_type: str, model_type: str, **kwargs: Any) -> Any:
    if (task_type or "regression").lower() == "classification":
        return _get_classifier(model_type=model_type, **kwargs)
    return _get_regressor(model_type=model_type, **kwargs)


def _default_hyperparams(model_type: str, plan: PipelinePlan) -> dict[str, Any]:
    raw = dict(plan.hyperparameters) if plan.hyperparameters else {}
    # Drop featurizer-only and non-model keys before passing to sklearn
    allowed = _model_hp_keys(plan.model_type or "ridge")
    hp = {k: v for k, v in raw.items() if k in allowed}
    key = (plan.model_type or "ridge").lower().replace(" ", "_")
    if key == "ridge" and "alpha" not in hp:
        hp.setdefault("alpha", 1.0)
    if key == "lasso" and "alpha" not in hp:
        hp.setdefault("alpha", 1.0)
    if key == "random_forest" and "n_estimators" not in hp:
        hp.setdefault("n_estimators", 100)
    if key == "extra_trees" and "n_estimators" not in hp:
        hp.setdefault("n_estimators", 100)
    if key == "elastic_net":
        hp.setdefault("alpha", 1.0)
        hp.setdefault("l1_ratio", 0.5)
        hp.setdefault("max_iter", 5000)
        hp.setdefault("tol", 1e-3)
    if key == "svr":
        hp.setdefault("kernel", "rbf")
        hp.setdefault("C", 1.0)
        if "gamma" not in hp:
            hp.setdefault("gamma", "scale")
    if key == "kernel_ridge":
        hp.setdefault("alpha", 1.0)
        hp.setdefault("kernel", "rbf")
    if key == "hist_gradient_boosting" and "max_iter" not in hp:
        hp.setdefault("max_iter", 100)
    if key == "xgboost":
        hp.setdefault("n_estimators", 100)
        hp.setdefault("max_depth", 6)
        hp.setdefault("learning_rate", 0.1)
        hp.setdefault("random_state", 0)
    if key == "lightgbm":
        hp.setdefault("n_estimators", 100)
        hp.setdefault("max_depth", 6)
        hp.setdefault("learning_rate", 0.1)
        hp.setdefault("random_state", 0)
        hp.setdefault("min_child_samples", 5)
        hp.setdefault("num_leaves", 20)
        hp.setdefault("verbosity", -1)
    return hp


def _default_hyperparams_classifier(
    model_type: str, plan: PipelinePlan
) -> dict[str, Any]:
    raw = dict(plan.hyperparameters) if plan.hyperparameters else {}
    allowed = _classifier_hp_keys(plan.model_type or "ridge")
    hp = {k: v for k, v in raw.items() if k in allowed}
    key = (plan.model_type or "ridge").lower().replace(" ", "_")
    if key in ("ridge", "elastic_net"):
        hp.setdefault("C", 1.0)
        hp.setdefault("max_iter", 1000)
        hp.setdefault("random_state", 0)
        if key == "elastic_net":
            hp.setdefault("l1_ratio", 0.5)
    if key == "random_forest":
        hp.setdefault("n_estimators", 100)
    if key == "extra_trees":
        hp.setdefault("n_estimators", 100)
    if key == "lasso":
        hp.setdefault("C", 1.0)
        hp.setdefault("penalty", "l1")
        hp.setdefault("solver", "saga")
        hp.setdefault("max_iter", 1000)
        hp.setdefault("random_state", 0)
    if key == "svr":
        hp.setdefault("kernel", "rbf")
        hp.setdefault("C", 1.0)
        if "gamma" not in hp:
            hp.setdefault("gamma", "scale")
    if key == "hist_gradient_boosting":
        hp.setdefault("max_iter", 100)
    if key == "xgboost":
        hp.setdefault("n_estimators", 100)
        hp.setdefault("max_depth", 6)
        hp.setdefault("learning_rate", 0.1)
        hp.setdefault("random_state", 0)
    if key == "lightgbm":
        hp.setdefault("n_estimators", 100)
        hp.setdefault("max_depth", 6)
        hp.setdefault("learning_rate", 0.1)
        hp.setdefault("random_state", 0)
        hp.setdefault("min_child_samples", 5)
        hp.setdefault("num_leaves", 20)
        hp.setdefault("verbosity", -1)
    return hp


class _ToDataFrameTransformer:
    """Converts numpy array to DataFrame with f0, f1, ... so tree models get feature names."""

    def fit(self, X: Any, y: Any = None) -> "_ToDataFrameTransformer":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


def _build_pipeline_steps(
    plan: PipelinePlan,
    model_type: str,
    hp: dict[str, Any],
    n_features: int,
    task_type: str = "regression",
) -> tuple[list[tuple[str, Any]], str]:
    """Build list of (name, estimator) steps and the name of the final predictor step."""
    steps: list[tuple[str, Any]] = []
    use_scale = getattr(plan, "scale_features", False) or getattr(
        plan, "robust_scale", False
    )
    if use_scale:
        steps.append(
            (
                "scaler",
                RobustScaler()
                if getattr(plan, "robust_scale", False)
                else StandardScaler(),
            )
        )

    fs = (getattr(plan, "feature_selection", None) or "").strip().lower() or None
    n_sel = getattr(plan, "n_features_to_select", None)
    var_thresh = getattr(plan, "variance_threshold", None)
    is_classification = (task_type or "regression").lower() == "classification"
    model = _get_estimator(
        task_type=task_type or "regression", model_type=model_type, **hp
    )

    if fs == "variance":
        thresh = float(var_thresh) if var_thresh is not None else 0.0
        steps.append(("selector", VarianceThreshold(threshold=thresh)))
    elif fs == "kbest" and n_sel is not None:
        n_f = n_features
        if isinstance(n_sel, float) and 0 < n_sel < 1:
            k = max(1, int(n_f * n_sel))
        else:
            k = max(1, min(int(n_sel), n_f))
        score_fn = f_classif if is_classification else f_regression
        steps.append(("selector", SelectKBest(score_func=score_fn, k=k)))
    elif fs == "rfe" and n_sel is not None:
        k = max(1, min(int(n_sel), n_features))
        steps.append(("selector", RFE(estimator=model, n_features_to_select=k)))
        return steps, "selector"

    key = (model_type or "ridge").lower().replace(" ", "_")
    need_dataframe_step = key in ("lightgbm", "xgboost") and len(steps) > 0
    if need_dataframe_step:
        steps.append(("to_dataframe", _ToDataFrameTransformer()))
    steps.append(("model", model))

    if (
        getattr(plan, "use_pca", False)
        and getattr(plan, "n_components", None) is not None
    ):
        n_comp = plan.n_components
        if isinstance(n_comp, float) and 0 < n_comp < 1:
            steps.insert(-1, ("pca", PCA(n_components=n_comp)))
        else:
            n_comp_int = int(n_comp)
            max_comp = min(50, n_features, 1000)
            n_comp_int = max(2, min(n_comp_int, max_comp))
            steps.insert(-1, ("pca", PCA(n_components=n_comp_int)))
        steps = [s for s in steps if s[0] not in ("model", "to_dataframe")]
        if need_dataframe_step:
            steps.append(("to_dataframe", _ToDataFrameTransformer()))
        steps.append(("model", model))

    return steps, "model"


def train_and_predict(
    benchmark: Any,
    plan: PipelinePlan,
    artifacts_dir: str = "outputs/artifacts",
    run_id: int = 0,
    generate_plots: bool = True,
    task_type: Literal["regression", "classification"] = "regression",
    prediction_format: Literal["labels", "probabilities"] = "labels",
) -> tuple[list[float] | list[int], RunResult]:
    """Train a model from plan on benchmark train split, predict on test; optionally use CV.

    Pipeline order: features -> (optional) scaler -> (optional) feature selection -> (optional) PCA -> model.
    Returns (predictions, RunResult). Saves full pipeline (scaler/selector/pca/model) as joblib.
    """
    train, test = benchmark.get_train_test_split()
    X_train = np.asarray(train.inputs)
    y_train_raw = np.asarray(train.targets)
    y_train = np.ravel(y_train_raw)
    X_test = np.asarray(test.inputs)

    if (task_type or "regression").lower() == "classification":
        if not np.issubdtype(y_train.dtype, np.integer):
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
        else:
            y_train = np.asarray(y_train, dtype=np.int64)

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
    n_features = X_train_f.shape[1]
    feature_names = [f"f{i}" for i in range(n_features)]
    X_train_f = pd.DataFrame(X_train_f, columns=feature_names)
    X_test_f = pd.DataFrame(X_test_f, columns=feature_names)

    model_type = plan.model_type or "ridge"
    ttype = (task_type or "regression").lower()
    is_classification = ttype == "classification"
    if is_classification:
        hp = _default_hyperparams_classifier(model_type, plan)
        allowed_keys = _classifier_hp_keys(model_type)
        if prediction_format == "probabilities":
            key = (model_type or "ridge").lower().replace(" ", "_")
            if key == "svr":
                hp["probability"] = True
    else:
        hp = _default_hyperparams(model_type, plan)
        allowed_keys = _model_hp_keys(model_type)

    steps, final_step_name = _build_pipeline_steps(
        plan=plan,
        model_type=model_type,
        hp=hp,
        n_features=n_features,
        task_type=task_type or "regression",
    )
    pipeline = Pipeline(steps)
    search = None
    if plan.use_cv and plan.tune_param_grid and final_step_name == "model":
        param_grid: dict[str, list[Any]] = {}
        prefix = f"{final_step_name}__"
        for key, values in plan.tune_param_grid.items():
            if key not in allowed_keys or not values:
                continue
            coerced = _coerce_param_values(key, list(values))
            if coerced:
                param_grid[prefix + key] = coerced
        if param_grid:
            n_combos = _product_len(param_grid)
            n_iter = min(15, max(1, n_combos))
            scoring = (
                "balanced_accuracy" if is_classification else "neg_mean_absolute_error"
            )
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=3,
                scoring=scoring,
                random_state=0,
            )
            search.fit(X_train_f, y_train)
            pipeline = search.best_estimator_

    if search is None:
        pipeline.fit(X_train_f, y_train)

    if is_classification and prediction_format == "probabilities":
        proba = pipeline.predict_proba(X_test_f)
        preds = proba[:, -1].tolist()
    else:
        preds = pipeline.predict(X_test_f).tolist()

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f"model_{run_id}.joblib")
    save_model(path=model_path, model=pipeline, metadata={"plan": plan.model_dump()})
    preds_path = os.path.join(artifacts_dir, f"predictions_{run_id}.json")
    with open(preds_path, "w") as f:
        json.dump(preds, f)

    plot_paths_list: list[str] = []
    if generate_plots:
        if not is_classification:
            y_train_pred = pipeline.predict(X_train_f)
            _plot_train_actual_vs_predicted(
                y_train=y_train,
                y_train_pred=y_train_pred,
                path=os.path.join(artifacts_dir, f"train_actual_vs_pred_{run_id}.png"),
            )
            plot_paths_list.append(
                os.path.join(artifacts_dir, f"train_actual_vs_pred_{run_id}.png")
            )
        _plot_train_target_histogram(
            y_train=y_train,
            path=os.path.join(artifacts_dir, f"train_target_hist_{run_id}.png"),
        )
        plot_paths_list.append(
            os.path.join(artifacts_dir, f"train_target_hist_{run_id}.png")
        )
        _plot_test_predictions_histogram(
            preds=preds,
            path=os.path.join(artifacts_dir, f"test_predictions_hist_{run_id}.png"),
        )
        plot_paths_list.append(
            os.path.join(artifacts_dir, f"test_predictions_hist_{run_id}.png")
        )
        if search is not None and hasattr(search, "cv_results_"):
            cv_ylabel = "Balanced accuracy (CV)" if is_classification else "MAE (CV)"
            _plot_cv_results(
                cv_results=search.cv_results_,
                path=os.path.join(artifacts_dir, f"cv_results_{run_id}.png"),
                ylabel=cv_ylabel,
            )
            plot_paths_list.append(
                os.path.join(artifacts_dir, f"cv_results_{run_id}.png")
            )

    best_hyperparameters: Optional[dict[str, Any]] = None
    cv_score: Optional[float] = None
    if search is not None and hasattr(search, "best_params_"):
        prefix = f"{final_step_name}__"
        best_hyperparameters = {}
        for k, v in search.best_params_.items():
            if k.startswith(prefix):
                best_hyperparameters[k[len(prefix) :]] = v
            else:
                best_hyperparameters[k] = v
        if hasattr(search, "cv_results_") and "mean_test_score" in search.cv_results_:
            idx = search.best_index_
            cv_score = float(search.cv_results_["mean_test_score"][idx])

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
        best_hyperparameters=best_hyperparameters,
        cv_score=cv_score,
    )


def evaluate_predictions(
    benchmark: Any,
    predictions: list[float],
    prediction_format: str = "labels",
) -> dict[str, float]:
    """Run benchmark.evaluate and return metric name -> score."""
    from autonomous_polaris_agent.tools.polaris_tools import (
        evaluate_predictions as _eval,
    )

    return _eval(
        benchmark=benchmark,
        predictions=predictions,
        prediction_format=prediction_format,
    )


def create_ensemble_predictions(
    artifacts_dir: str,
    run_ids: list[str],
    strategy: Literal["mean", "median"] = "mean",
    task_type: Literal["regression", "classification"] = "regression",
    prediction_format: Literal["labels", "probabilities"] = "labels",
) -> tuple[list[float] | list[int], RunResult]:
    """Load saved predictions for run_ids, combine with strategy, return (predictions, RunResult)."""
    import re

    preds_list: list[list[float] | list[int]] = []
    for run_id in run_ids:
        match = re.search(r"run_(\d+)", run_id)
        num = match.group(1) if match else run_id
        path = os.path.join(artifacts_dir, f"predictions_{num}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Predictions not found for {run_id} at {path}")
        with open(path) as f:
            preds_list.append(json.load(f))
    if not preds_list:
        raise ValueError("No run_ids provided")
    n = len(preds_list[0])
    if not all(len(p) == n for p in preds_list):
        raise ValueError("Prediction length mismatch across runs")
    arr = np.array(preds_list)

    is_classification = (task_type or "regression").lower() == "classification"
    use_probabilities = (
        is_classification and prediction_format == "probabilities"
    )

    if use_probabilities:
        if strategy == "median":
            combined = np.nanmedian(arr, axis=0).tolist()
        else:
            combined = np.nanmean(arr, axis=0).tolist()
        strategy_label = strategy
    elif is_classification:
        combined = []
        for j in range(n):
            col = arr[:, j].astype(np.int64)
            unique_vals, counts = np.unique(col, return_counts=True)
            combined.append(int(unique_vals[np.argmax(counts)]))
        strategy_label = "voting"
    else:
        if strategy == "median":
            combined = np.nanmedian(arr, axis=0).tolist()
        else:
            combined = np.nanmean(arr, axis=0).tolist()
        strategy_label = strategy

    ensemble_id = "ensemble_" + "_".join(run_ids)
    result = RunResult(
        model_id=ensemble_id,
        description=f"Ensemble ({strategy_label}) of {', '.join(run_ids)}",
        metric_name="",
        metric_value=0.0,
        metrics={},
        model_path=None,
        rationale=f"Ensemble {strategy_label} of {len(run_ids)} runs",
    )
    return combined, result


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
    ax.hist(
        y_train.ravel(),
        bins=min(50, max(10, len(y_train) // 20)),
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xlabel("Target value")
    ax.set_ylabel("Count")
    ax.set_title("Train target distribution")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def _plot_test_predictions_histogram(preds: list[float], path: str) -> None:
    """Histogram of test-set predictions (no true labels)."""
    fig, ax = plt.subplots()
    ax.hist(
        preds, bins=min(50, max(10, len(preds) // 20)), edgecolor="white", alpha=0.8
    )
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Count")
    ax.set_title("Test prediction distribution")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def _plot_cv_results(
    cv_results: dict[str, Any], path: str, ylabel: str = "MAE (CV)"
) -> None:
    """Bar chart of CV mean score per parameter combination (e.g. alpha)."""
    mean_scores = cv_results.get("mean_test_score")
    if mean_scores is None or len(mean_scores) == 0:
        return
    param_keys = [k for k in cv_results if k.startswith("param_")]
    if not param_keys:
        x_labels = [str(i) for i in range(len(mean_scores))]
    else:
        p = param_keys[0]
        x_labels = [str(cv_results[p][i]) for i in range(len(mean_scores))]
    fig, ax = plt.subplots()
    x_pos = range(len(mean_scores))
    scores = np.asarray(mean_scores)
    # For neg_mean_absolute_error we plot -score (so lower bar = better); for balanced_accuracy we plot score as-is
    plot_vals = -scores if "MAE" in ylabel else scores
    ax.bar(x_pos, plot_vals, tick_label=x_labels)
    ax.set_xlabel(param_keys[0].replace("param_", "") if param_keys else "Config")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cross-validation: {ylabel}")
    plt.xticks(rotation=45, ha="right")
    save_plot(path=path, figure=fig)
    plt.close(fig)


def save_metrics_comparison_plot(
    completed_runs: list[dict[str, Any]],
    output_dir: str,
    filename: str = "metrics_comparison.png",
    primary_metric: Optional[str] = None,
) -> Optional[str]:
    """Bar chart of primary metric per run. Returns path or None if no metrics."""
    if not completed_runs:
        return None
    model_ids = []
    values = []
    chosen_metric = (primary_metric or "").strip() or None
    metric_name = chosen_metric
    for run in completed_runs:
        metrics = run.get("metrics") or {}
        if not metrics:
            continue
        if metric_name is None:
            metric_name = next(iter(metrics.keys()))
        model_ids.append(run.get("model_id", "?"))
        if metric_name in metrics:
            values.append(metrics.get(metric_name, float("nan")))
        else:
            fallback_metric = next(iter(metrics.keys()), "")
            values.append(metrics.get(fallback_metric, float("nan")))
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
