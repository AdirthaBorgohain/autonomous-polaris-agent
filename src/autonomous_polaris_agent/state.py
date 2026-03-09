"""State schema and parsed request types for the agent graph."""

from typing import Annotated, Any, Optional, Union

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ThoughtStep(BaseModel):
    """One step in the agent's thought process log."""

    stage: str = Field(description="e.g. analysis, plan, reflection")
    content: str = Field(description="Markdown or plain text")
    timestamp: Optional[str] = None


class ParsedRequest(BaseModel):
    """User intent parsed from natural language."""

    dataset_slug: str = Field(description="Benchmark slug (owner/name)")
    num_models: int = Field(ge=1, description="Number of distinct pipelines to run")
    task_type: Optional[str] = Field(
        default=None, description="regression or classification if inferred"
    )
    explicit_metric: Optional[str] = Field(
        default=None, description="User-requested metric if any"
    )
    save_models: bool = Field(default=True, description="Whether to persist models")


class PipelinePlan(BaseModel):
    """Description of one ML pipeline to run."""

    fingerprint: str = Field(
        description="One of: morgan, morgan_counts, rdkit_descriptors, maccs, atom_pair, morgan_rdkit_concat. Featurizer params (e.g. morgan_radius, morgan_n_bits) can be passed in hyperparameters."
    )
    model_type: str = Field(
        description="One of: ridge, lasso, elastic_net, svr, kernel_ridge, random_forest, extra_trees, hist_gradient_boosting, xgboost, lightgbm"
    )
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(
        description="2-4 sentences: why this representation/model/hyperparameters"
    )
    use_cv: bool = Field(
        default=False, description="Use internal CV on train to tune hyperparameters"
    )
    tune_param_grid: Optional[dict[str, list[Any]]] = Field(
        default=None,
        description="When use_cv: dict of param name -> list of values to search; keys must be in model params",
    )
    scale_features: bool = Field(
        default=False,
        description="Apply StandardScaler to features before fitting (recommended for ridge/SVR)",
    )
    use_pca: bool = Field(
        default=False,
        description="Apply PCA to features before fitting (optional dimensionality reduction)",
    )
    n_components: Optional[Union[int, float]] = Field(
        default=None,
        description="PCA n_components (int or 0-1 float for variance ratio); used only if use_pca",
    )
    feature_selection: Optional[str] = Field(
        default=None,
        description="Optional: 'variance', 'kbest', or 'rfe'; applied after scaling, before PCA",
    )
    n_features_to_select: Optional[Union[int, float]] = Field(
        default=None,
        description="For kbest/rfe: number (or fraction) of features to keep; used only if feature_selection is set",
    )
    variance_threshold: Optional[float] = Field(
        default=None,
        description="For feature_selection='variance': drop features with variance below this",
    )
    robust_scale: bool = Field(
        default=False,
        description="Use RobustScaler instead of StandardScaler when scale_features is implied (for outliers)",
    )


class RunResult(BaseModel):
    """Result of one pipeline run."""

    model_id: str = Field(description="Short identifier for this run")
    description: str = Field(default="", description="Human-readable description")
    metric_name: str = Field(default="")
    metric_value: float = Field(default=0.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    model_path: Optional[str] = None
    plot_path: Optional[str] = None
    plot_paths: Optional[list[str]] = None
    rationale: Optional[str] = None
    best_hyperparameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Best params from RandomizedSearchCV when use_cv was used",
    )
    cv_score: Optional[float] = Field(
        default=None, description="Mean CV score (e.g. neg MAE) when use_cv was used"
    )


class LogEntry(BaseModel):
    """Structured log entry for a tool call or step."""

    tool_or_step: str = ""
    args_summary: str = ""
    result_summary: str = ""
    timestamp: Optional[str] = None


class AgentState(TypedDict, total=False):
    """Graph state schema for LangGraph."""

    messages: Annotated[list, add_messages]
    user_input: str
    parsed: Optional[ParsedRequest]
    benchmark: Any
    benchmark_inspection: Optional[dict[str, Any]]
    primary_metric: Optional[str]
    task_analysis: Optional[str]
    thought_process: list[ThoughtStep]
    iteration: int
    completed_runs: list[RunResult]
    last_predictions: list[float]
    report_md: Optional[str]
    logs: list[LogEntry]
    last_error: Optional[str]
    output_dir: Optional[str]


def create_initial_state(user_input: str) -> dict[str, Any]:
    """Build initial graph state from user input."""
    return {
        "user_input": user_input,
        "parsed": None,
        "benchmark": None,
        "benchmark_inspection": None,
        "primary_metric": None,
        "task_analysis": None,
        "thought_process": [],
        "iteration": 0,
        "completed_runs": [],
        "report_md": None,
        "logs": [],
        "last_error": None,
        "last_predictions": [],
        "output_dir": "outputs",
        "messages": [],
    }
