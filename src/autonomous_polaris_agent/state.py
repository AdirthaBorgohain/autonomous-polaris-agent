"""State schema and parsed request types for the agent graph."""

from typing import Any, Optional

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

    fingerprint: str = Field(description="e.g. morgan, rdkit_descriptors")
    model_type: str = Field(description="e.g. random_forest, ridge, gradient_boosting")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(
        description="2-4 sentences: why this representation/model/hyperparameters"
    )
    use_cv: bool = Field(
        default=False, description="Use internal CV on train to tune hyperparameters"
    )
    tune_over: list[str] = Field(
        default_factory=list, description="Parameter names to tune if use_cv"
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


class LogEntry(BaseModel):
    """Structured log entry for a tool call or step."""

    tool_or_step: str = ""
    args_summary: str = ""
    result_summary: str = ""
    timestamp: Optional[str] = None


class AgentState(TypedDict, total=False):
    """Graph state schema for LangGraph."""

    user_input: str
    parsed: Optional[ParsedRequest]
    benchmark: Any
    benchmark_inspection: Optional[dict[str, Any]]
    task_analysis: Optional[str]
    thought_process: list[ThoughtStep]
    current_plan: Optional[PipelinePlan]
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
        "task_analysis": None,
        "thought_process": [],
        "current_plan": None,
        "iteration": 0,
        "completed_runs": [],
        "report_md": None,
        "logs": [],
        "last_error": None,
        "last_predictions": [],
        "output_dir": "outputs",
    }
