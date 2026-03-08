"""LangGraph workflow: parse -> load -> analyze -> plan -> execute -> evaluate -> reflect -> report."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from autonomous_polaris_agent.console import (
    error_display,
    llm_call,
    metrics_display,
    plan_display,
    step_result,
    step_start,
    thinking,
)
from autonomous_polaris_agent.prompts import get_prompt
from autonomous_polaris_agent.state import (
    AgentState,
    LogEntry,
    ParsedRequest,
    PipelinePlan,
    ThoughtStep,
)
from autonomous_polaris_agent.tools import (
    evaluate_predictions,
    load_benchmark as load_benchmark_tool,
    save_metrics_comparison_plot,
    train_and_predict,
)

load_dotenv()


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature)


def _llm_model_and_temp() -> tuple[str, float]:
    """Return (model_name, temperature) for console display."""
    return (
        os.getenv("OPENAI_MODEL", "gpt-5.4"),
        float(os.getenv("OPENAI_TEMPERATURE", "0")),
    )


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM output (may be wrapped in markdown)."""
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def _log(
    state: AgentState, tool_or_step: str, args_summary: str, result_summary: str
) -> list[dict]:
    """Append a LogEntry and return updated logs list."""
    logs = list(state.get("logs") or [])
    ts = datetime.now(timezone.utc).isoformat()
    entry = LogEntry(
        tool_or_step=tool_or_step,
        args_summary=args_summary,
        result_summary=result_summary,
        timestamp=ts,
    )
    logs.append(entry.model_dump())
    return logs


def parse_input(state: AgentState) -> dict[str, Any]:
    """Parse user input into ParsedRequest via LLM."""
    step_start("parse_input", "Parsing user request")
    user_input = state.get("user_input") or ""
    model, temp = _llm_model_and_temp()
    llm_call(
        model=model,
        temperature=temp,
        purpose="Parse user input to dataset_slug, num_models, task_type",
    )
    llm = _get_llm()
    msg = llm.invoke(
        [
            {
                "role": "user",
                "content": get_prompt("parse_user_input")
                + "\n\nUser request: "
                + user_input,
            }
        ]
    )
    content = msg.content if hasattr(msg, "content") else str(msg)
    try:
        data = _extract_json(content)
        parsed = ParsedRequest(
            dataset_slug=data.get("dataset_slug", ""),
            num_models=int(data.get("num_models", 1)),
            task_type=data.get("task_type"),
            explicit_metric=data.get("explicit_metric"),
            save_models=data.get("save_models", True),
        )
        step_result(
            name="parse_input",
            summary=f"dataset_slug={parsed.dataset_slug!r}, num_models={parsed.num_models}, task_type={parsed.task_type}",
        )
        return {"parsed": parsed.model_dump(), "last_error": None}
    except Exception as e:
        error_display(str(e))
        parsed = ParsedRequest(
            dataset_slug="polaris/hello-world-benchmark", num_models=1
        )
        return {"parsed": parsed, "last_error": str(e)}


def load_benchmark_node(state: AgentState) -> dict[str, Any]:
    """Load benchmark and store inspection."""
    parsed = state.get("parsed")
    if isinstance(parsed, dict):
        slug = parsed.get("dataset_slug", "polaris/hello-world-benchmark")
    else:
        slug = getattr(parsed, "dataset_slug", "polaris/hello-world-benchmark")
    step_start("load_benchmark", f"Loading benchmark {slug!r}")
    try:
        benchmark, inspection = load_benchmark_tool(slug)
        train_sz = inspection.get("train_size")
        test_sz = inspection.get("test_size")
        step_result(
            name="load_benchmark",
            summary=f"train_size={train_sz}, test_size={test_sz}",
            details=json.dumps(inspection, indent=2),
        )
        logs = _log(
            state,
            tool_or_step="load_benchmark",
            args_summary=f"slug={slug!r}",
            result_summary=f"train_size={train_sz}, test_size={test_sz}",
        )
        return {
            "benchmark": benchmark,
            "benchmark_inspection": inspection,
            "logs": logs,
        }
    except Exception as e:
        error_display(str(e))
        logs = _log(state, "load_benchmark", f"slug={slug!r}", f"error={e!s}")
        return {"logs": logs, "last_error": str(e)}


def analyze_node(state: AgentState) -> dict[str, Any]:
    """Produce task analysis and append to thought_process."""
    step_start("analyze", "Task analysis from benchmark inspection")
    inspection = state.get("benchmark_inspection") or {}
    model, temp = _llm_model_and_temp()
    llm_call(model=model, temperature=temp, purpose="Analyze benchmark and task type")
    llm = _get_llm()
    prompt = get_prompt(
        "analysis_task", benchmark_inspection=json.dumps(inspection, indent=2)
    )
    msg = llm.invoke([{"role": "user", "content": prompt}])
    content = msg.content if hasattr(msg, "content") else str(msg)
    thinking(stage="analysis", content=content)
    step = ThoughtStep(stage="analysis", content=content)
    thought = list(state.get("thought_process") or [])
    thought.append(step.model_dump())
    return {"task_analysis": content, "thought_process": thought}


def planner_node(state: AgentState) -> dict[str, Any]:
    """Produce next pipeline plan and append reasoning to thought_process."""
    step_start("planner", "Planning next pipeline (fingerprint, model, hyperparams)")
    parsed = state.get("parsed") or {}
    num_models = (
        parsed.get("num_models", 1)
        if isinstance(parsed, dict)
        else getattr(parsed, "num_models", 1)
    )
    completed = state.get("completed_runs") or []
    completed_count = len(completed)
    user_input = state.get("user_input") or ""
    task_analysis = state.get("task_analysis") or ""
    inspection = state.get("benchmark_inspection") or {}
    completed_summary = json.dumps(
        [
            r.get("model_id") if isinstance(r, dict) else getattr(r, "model_id", "")
            for r in completed
        ]
    )

    prompt = get_prompt(
        "planning_pipeline",
        num_models=num_models,
        completed_count=completed_count,
        user_input=user_input,
        task_analysis=task_analysis,
        benchmark_inspection=json.dumps(inspection, indent=2),
        completed_runs_summary=completed_summary,
    )
    model, temp = _llm_model_and_temp()
    llm_call(model=model, temperature=temp, purpose="Plan next ML pipeline")
    llm = _get_llm()
    msg = llm.invoke([{"role": "user", "content": prompt}])
    content = msg.content if hasattr(msg, "content") else str(msg)
    data = _extract_json(content)
    reasoning = data.get("reasoning", "")
    plan_data = data.get("plan", {})
    plan = PipelinePlan(
        fingerprint=plan_data.get("fingerprint", "morgan"),
        model_type=plan_data.get("model_type", "ridge"),
        hyperparameters=plan_data.get("hyperparameters", {}),
        rationale=plan_data.get("rationale", ""),
        use_cv=plan_data.get("use_cv", False),
        tune_over=plan_data.get("tune_over", []),
    )
    thinking(stage="plan", content=reasoning + "\n\n" + plan.rationale)
    plan_display(
        fingerprint=plan.fingerprint,
        model_type=plan.model_type,
        hyperparameters=plan.hyperparameters,
        rationale="",
    )
    step = ThoughtStep(stage="plan", content=reasoning + "\n\n" + plan.rationale)
    thought = list(state.get("thought_process") or [])
    thought.append(step.model_dump())
    return {
        "current_plan": plan.model_dump(),
        "thought_process": thought,
    }


def executor_node(state: AgentState) -> dict[str, Any]:
    """Run train_and_predict and append RunResult, set last_predictions."""
    benchmark = state.get("benchmark")
    plan_data = state.get("current_plan")
    if not benchmark or not plan_data:
        step_start("executor", "Train and predict")
        error_display("Missing benchmark or current_plan")
        return {"last_error": "Missing benchmark or current_plan"}
    plan = PipelinePlan(**plan_data) if isinstance(plan_data, dict) else plan_data
    iteration = state.get("iteration", 0)
    step_start("executor", f"Train and predict (run {iteration})")
    try:
        out_dir = state.get("output_dir") or "outputs"
        artifacts_dir = os.path.join(out_dir, "artifacts")
        preds, result = train_and_predict(
            benchmark=benchmark,
            plan=plan,
            artifacts_dir=artifacts_dir,
            run_id=iteration,
        )
        step_result(
            name="executor",
            summary=f"preds_len={len(preds)}, model_path={result.model_path}",
        )
        completed = list(state.get("completed_runs") or [])
        completed.append(result.model_dump())
        logs = _log(
            state,
            "train_and_predict",
            args_summary=f"plan={plan.fingerprint}+{plan.model_type}",
            result_summary=f"preds_len={len(preds)}, model_path={result.model_path}",
        )
        return {
            "completed_runs": completed,
            "last_predictions": preds,
            "iteration": iteration + 1,
            "last_error": None,
            "logs": logs,
        }
    except Exception as e:
        error_display(str(e))
        from autonomous_polaris_agent.state import RunResult

        logs = _log(
            state,
            "train_and_predict",
            args_summary=f"plan={plan.fingerprint}+{plan.model_type}",
            result_summary=f"error={e!s}",
        )
        completed = list(state.get("completed_runs") or [])
        completed.append(
            RunResult(
                model_id=f"run_{iteration}_error",
                description=f"Error: {e!s}",
                metric_value=float("nan"),
            ).model_dump()
        )
        return {
            "completed_runs": completed,
            "last_predictions": [],
            "iteration": iteration + 1,
            "last_error": str(e),
            "logs": logs,
        }


def evaluator_node(state: AgentState) -> dict[str, Any]:
    """Evaluate last_predictions and attach metrics to last RunResult."""
    step_start("evaluator", "Evaluate predictions")
    benchmark = state.get("benchmark")
    preds = state.get("last_predictions") or []
    completed = list(state.get("completed_runs") or [])
    if not benchmark or not preds or not completed:
        return {}
    metrics = evaluate_predictions(benchmark, preds)
    metrics_display(metrics)
    last = completed[-1]
    if isinstance(last, dict):
        last["metrics"] = metrics
        metric_name = next(iter(metrics.keys()), "")
        last["metric_name"] = metric_name
        last["metric_value"] = metrics.get(metric_name, 0.0)
    completed[-1] = last
    logs = _log(
        state,
        "evaluate_predictions",
        args_summary=f"preds_len={len(preds)}",
        result_summary=json.dumps(metrics),
    )
    return {"completed_runs": completed, "logs": logs}


def reflect_node(state: AgentState) -> dict[str, Any]:
    """Reflect on last run and append to thought_process."""
    step_start("reflect", "Reflecting on last run")
    completed = state.get("completed_runs") or []
    parsed = state.get("parsed") or {}
    num_models = (
        parsed.get("num_models", 1)
        if isinstance(parsed, dict)
        else getattr(parsed, "num_models", 1)
    )
    remaining = max(0, num_models - len(completed))
    last_result = json.dumps(completed[-1], indent=2) if completed else "{}"
    completed_summary = json.dumps(
        [(r.get("model_id"), r.get("metrics")) for r in completed], indent=2
    )

    prompt = get_prompt(
        "reflection_after_run",
        last_run_result=last_result,
        completed_runs_summary=completed_summary,
        remaining=remaining,
    )
    model, temp = _llm_model_and_temp()
    llm_call(
        model=model, temperature=temp, purpose="Reflect on last run and remaining work"
    )
    llm = _get_llm()
    msg = llm.invoke([{"role": "user", "content": prompt}])
    content = msg.content if hasattr(msg, "content") else str(msg)
    try:
        data = _extract_json(content)
        reflection = data.get("reflection", content)
    except Exception:
        reflection = content
    thinking(stage="reflection", content=reflection)
    step = ThoughtStep(stage="reflection", content=reflection)
    thought = list(state.get("thought_process") or [])
    thought.append(step.model_dump())
    return {"thought_process": thought}


def decide_next(state: AgentState) -> Literal["planner", "report_generator"]:
    """Route to planner or report_generator."""
    parsed = state.get("parsed") or {}
    num_models = (
        parsed.get("num_models", 1)
        if isinstance(parsed, dict)
        else getattr(parsed, "num_models", 1)
    )
    completed = state.get("completed_runs") or []
    if len(completed) < num_models:
        return "planner"
    return "report_generator"


def report_generator_node(state: AgentState) -> dict[str, Any]:
    """Generate final report and write to outputs/report.md."""
    step_start("report_generator", "Generating final report")
    user_input = state.get("user_input") or ""
    parsed = state.get("parsed") or {}
    task_analysis = state.get("task_analysis") or ""
    thought_process = state.get("thought_process") or []
    completed = state.get("completed_runs") or []
    out_dir = state.get("output_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Metrics comparison plot (bar chart of primary metric per run)
    metrics_plot_path = save_metrics_comparison_plot(
        completed_runs=completed,
        output_dir=out_dir,
        filename="metrics_comparison.png",
    )

    thought_str = "\n\n".join(
        (t.get("content", "") if isinstance(t, dict) else getattr(t, "content", ""))
        for t in thought_process
    )
    prompt = get_prompt(
        "report_final",
        user_input=user_input,
        parsed=json.dumps(parsed, indent=2),
        task_analysis=task_analysis,
        thought_process=thought_str,
        completed_runs=json.dumps(completed, indent=2),
    )
    model, temp = _llm_model_and_temp()
    llm_call(model=model, temperature=temp, purpose="Generate final markdown report")
    llm = _get_llm()
    msg = llm.invoke([{"role": "user", "content": prompt}])
    report_md = msg.content if hasattr(msg, "content") else str(msg)
    if metrics_plot_path:
        report_md = report_md.rstrip() + f"\n\n### Metrics comparison\n\n![Metrics comparison]({os.path.basename(metrics_plot_path)})\n"
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    step_result(name="report_generator", summary=f"Report written to {report_path}")
    return {"report_md": report_md}


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("parse_input", parse_input)
    builder.add_node("load_benchmark", load_benchmark_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("report_generator", report_generator_node)

    builder.add_edge(START, "parse_input")
    builder.add_edge("parse_input", "load_benchmark")
    builder.add_edge("load_benchmark", "analyze")
    builder.add_edge("analyze", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_edge("evaluator", "reflect")
    builder.add_conditional_edges("reflect", decide_next)
    builder.add_edge("report_generator", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
