"""Deep Agent harness for autonomous Polaris chemistry ML."""

import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from langchain_openai import ChatOpenAI

from autonomous_polaris_agent.deep_agent_tools import (
    create_evaluate_predictions_tool,
    evaluate_predictions,
    load_benchmark,
)

SYSTEM_PROMPT = """You are an autonomous chemistry ML agent. Your job is to write Python code for preprocessing, training, and evaluation on Polaris benchmarks, execute it, and iterate to improve metrics.

## Workflow

1. Call load_benchmark(slug) to get dataset metadata (n_train, n_test, main_metric, target_types).
2. Write Python scripts that load the benchmark, featurize, train models, and write predictions to JSON.
3. Execute your script (e.g. python train.py).
4. Call evaluate_predictions to get official metrics.
5. Use metrics to decide your next experiment. Iterate.
6. When done, write the final report using the report-writing skill (report.md, optionally metrics_comparison.png).

## Key rules

- You write ALL code yourself. No predefined pipelines.
- For main_metric roc_auc/roc_auc_score/auroc, use prediction_format='probabilities' and predict_proba.
- Use prior run metrics to choose the next pipeline. Do not pick at random.

## Iteration discipline

- Train ONE pipeline first. Execute it, evaluate it, and reason about its performance on the primary metric before moving on.
- Understand what worked and what did not: overfitting, underfitting, feature quality, model capacity, etc.
- Only after you have reasoned about the first pipeline's results, plan and design the next pipeline.
- Repeat: train → evaluate → reason → plan next → train again.
- Think innovatively and cleverly. Your aim is to match or exceed the scores on the benchmark leaderboard (see polarishub.io/benchmarks/{owner}/{name} for reference). Use prior results to guide each next choice.
"""

SANDBOX_WORKSPACE_HINT = """

When running in a sandbox: use /workspace as the root for all scripts, data, and outputs (e.g. /workspace/train.py, /workspace/report.md).
Always use uv for Python execution and package management (e.g. uv run python train.py, uv pip install <pkg>).
"""


def build_deep_agent(
    output_dir: str,
    *,
    sandbox_backend: Any | None = None,
):
    """Build a Deep Agent configured for Polaris chemistry ML.

    Args:
        output_dir: Absolute path to the run output directory. For LocalShellBackend,
            this is the root. For OpenSandboxBackend, outputs are downloaded here after run.
        sandbox_backend: Optional SandboxBackendProtocol (e.g. OpenSandboxBackend).
            When provided, code runs in the sandbox; otherwise uses LocalShellBackend.

    Returns:
        Compiled agent (LangGraph StateGraph).
    """
    root_dir = str(Path(output_dir).resolve())
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    skills_dir = str(project_root / "skills")

    if sandbox_backend is not None:
        default_backend = sandbox_backend
        eval_tool = create_evaluate_predictions_tool(backend=sandbox_backend)
    else:
        default_backend = LocalShellBackend(
            root_dir=root_dir,
            timeout=300,
            max_output_bytes=500_000,
            virtual_mode=True,
        )
        eval_tool = evaluate_predictions

    backend = CompositeBackend(
        default=default_backend,
        routes={
            "/skills/": FilesystemBackend(
                root_dir=skills_dir,
                virtual_mode=True,
            ),
        },
    )

    tools = [
        load_benchmark,
        eval_tool,
    ]

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        temperature=0.4,
    )

    system_prompt = SYSTEM_PROMPT
    if sandbox_backend is not None:
        system_prompt = SYSTEM_PROMPT + SANDBOX_WORKSPACE_HINT

    return create_deep_agent(
        backend=backend,
        tools=tools,
        system_prompt=system_prompt,
        model=model,
        skills=["/skills/"],
    )
