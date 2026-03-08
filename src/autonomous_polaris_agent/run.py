"""Agent run entrypoint; builds and invokes the LangGraph workflow."""

import json
import os
from datetime import datetime, timezone

from autonomous_polaris_agent.console import run_end, run_start, set_enabled


def _run_id_from_timestamp() -> str:
    """Return a run id string based on current UTC timestamp (e.g. run_20260308_150530)."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def run_agent(user_message: str = "", verbose: bool = True) -> dict:
    """Run the autonomous agent on the given user message.

    Creates a timestamped directory under outputs/ (e.g. outputs/run_20260308_150530/)
    and writes report, artifacts, and logs there.

    Args:
        user_message: Natural language request (e.g. dataset link + number of models).
        verbose: If True, emit Rich console output for each step (default True).

    Returns:
        Final graph state (e.g. report path, completed_runs).
    """
    from autonomous_polaris_agent.graph import build_graph
    from autonomous_polaris_agent.state import create_initial_state

    set_enabled(verbose)
    run_id = _run_id_from_timestamp()
    run_dir = os.path.join("outputs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    message = user_message or "Generate 1 regression model on polaris/adme-fang-solu-1."
    run_start(user_message=message, run_dir=run_dir)

    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
    initial_state = create_initial_state(message)
    initial_state["output_dir"] = run_dir

    result = graph.invoke(initial_state, config=config)

    logs = result.get("logs") or []
    if logs:
        log_path = os.path.join(run_dir, "run.jsonl")
        with open(log_path, "w") as f:
            for entry in logs:
                f.write(json.dumps(entry) + "\n")

    out_dir = result.get("output_dir") or run_dir
    report_path = os.path.join(out_dir, "report.md") if result.get("report_md") else None
    artifact_paths = []
    for run in result.get("completed_runs") or []:
        path = run.get("model_path") if isinstance(run, dict) else getattr(run, "model_path", None)
        if path:
            artifact_paths.append(path)
    run_end(
        report_path=report_path,
        artifact_paths=artifact_paths,
        run_dir=out_dir,
    )
    return result
