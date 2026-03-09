"""LangGraph workflow: parse -> load -> analyze -> ReAct modelling loop -> report."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from autonomous_polaris_agent.console import (
    error_display,
    format_args_summary,
    llm_call,
    step_result,
    step_start,
    thinking,
    tool_calls_display,
    tool_result_display,
)
from autonomous_polaris_agent.prompts import get_prompt
from autonomous_polaris_agent.state import (
    AgentState,
    LogEntry,
    ParsedRequest,
    ThoughtStep,
)
from autonomous_polaris_agent.tools import (
    load_benchmark as load_benchmark_tool,
)
from autonomous_polaris_agent.tools import (
    save_metrics_comparison_plot,
)
from autonomous_polaris_agent.tools.agent_tools import (
    dispatch_tool,
    get_modelling_tools,
)

load_dotenv()


class AnalysisResult(BaseModel):
    task_analysis: str = Field(description="1-2 paragraph analysis")
    primary_metric: str = Field(
        description="Metric name to optimize; should match benchmark.evaluate() metric name if possible"
    )


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
    return ChatOpenAI(model=model, temperature=temperature)


def _llm_model_and_temp() -> tuple[str, float]:
    """Return (model_name, temperature) for console display."""
    return (
        os.getenv("OPENAI_MODEL", "gpt-5.4"),
        float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
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


def _reflection_from_train_result(content: str) -> str | None:
    """Build a one-line reflection from train_and_predict JSON result. Returns None on parse error."""
    try:
        data = json.loads(content)
        if data.get("error"):
            return None
        model_id = data.get("model_id", "")
        description = data.get("description", "")
        metrics = data.get("metrics") or {}
        if not metrics:
            return f"{model_id}: {description}"
        metric_name = data.get("metric_name") or next(iter(metrics.keys()), "")
        value = data.get("metric_value")
        if value is None:
            value = metrics.get(metric_name, "")
        return f"{model_id}: {description} — {metric_name}={value}"
    except (json.JSONDecodeError, TypeError):
        return None


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
        logs = _log(
            state,
            tool_or_step="parse_input",
            args_summary="user_input",
            result_summary=f"dataset_slug={parsed.dataset_slug!r}, num_models={parsed.num_models}, task_type={parsed.task_type}",
        )
        return {"parsed": parsed.model_dump(), "last_error": None, "logs": logs}
    except Exception as e:
        error_display(str(e))
        parsed = ParsedRequest(
            dataset_slug="polaris/hello-world-benchmark", num_models=1
        )
        logs = _log(
            state,
            tool_or_step="parse_input",
            args_summary="user_input",
            result_summary=f"error={e!s}",
        )
        return {
            "parsed": parsed.model_dump(),
            "last_error": str(e),
            "logs": logs,
        }


def load_benchmark_node(state: AgentState) -> dict[str, Any]:
    """Load benchmark and store metadata for the agent."""
    parsed = state.get("parsed")
    if isinstance(parsed, dict):
        slug = parsed.get("dataset_slug", "polaris/hello-world-benchmark")
    else:
        slug = getattr(parsed, "dataset_slug", "polaris/hello-world-benchmark")
    step_start("load_benchmark", f"Loading benchmark {slug!r}")
    try:
        benchmark, metadata = load_benchmark_tool(slug)
        n_train = metadata.get("n_train_datapoints") or metadata.get("train_size")
        n_test = metadata.get("n_test_datapoints")
        if isinstance(n_test, dict):
            n_test_summary = sum(n_test.values())
        else:
            n_test_summary = n_test or metadata.get("test_size")
        display_metadata = {k: v for k, v in metadata.items() if k != "split"}
        step_result(
            name="load_benchmark",
            summary=f"n_train={n_train}, n_test={n_test_summary}",
            details=json.dumps(display_metadata, indent=2),
        )
        logs = _log(
            state,
            tool_or_step="load_benchmark",
            args_summary=f"slug={slug!r}",
            result_summary=f"n_train={n_train}, n_test={n_test_summary}",
        )
        return {
            "benchmark": benchmark,
            "benchmark_inspection": metadata,
            "logs": logs,
        }
    except Exception as e:
        error_display(str(e))
        logs = _log(state, "load_benchmark", f"slug={slug!r}", f"error={e!s}")
        return {"logs": logs, "last_error": str(e)}


def analyze_node(state: AgentState) -> dict[str, Any]:
    """Produce task analysis and append to thought_process."""
    step_start("analyze", "Task analysis from benchmark metadata")
    metadata = state.get("benchmark_inspection") or {}
    model, temp = _llm_model_and_temp()
    llm_call(model=model, temperature=temp, purpose="Analyze benchmark and task type")
    llm = _get_llm()
    prompt = get_prompt(
        "analysis_task", benchmark_metadata=json.dumps(metadata, indent=2)
    )
    llm_structured = llm.with_structured_output(AnalysisResult)
    result = llm_structured.invoke([{"role": "user", "content": prompt}])
    task_analysis = (result.task_analysis or "").strip()
    primary_metric = (result.primary_metric or "").strip() or None

    analysis_display = task_analysis
    if primary_metric:
        analysis_display = (
            f"{task_analysis}\n\nPrimary metric to optimize: {primary_metric}"
        )
    thinking(stage="analysis", content=analysis_display)
    step = ThoughtStep(stage="analysis", content=analysis_display)
    thought = list(state.get("thought_process") or [])
    thought.append(step.model_dump())
    logs = _log(
        state,
        tool_or_step="analyze",
        args_summary="benchmark_metadata",
        result_summary=analysis_display[:500] + "..."
        if len(analysis_display) > 500
        else analysis_display,
    )
    return {
        "task_analysis": task_analysis,
        "primary_metric": primary_metric,
        "thought_process": thought,
        "logs": logs,
    }


def seed_modelling_messages(state: AgentState) -> dict[str, Any]:
    """Build initial messages for the ReAct modelling agent from task_analysis and state."""
    task_analysis = state.get("task_analysis") or ""
    metadata = state.get("benchmark_inspection") or {}
    primary_metric = state.get("primary_metric") or ""
    parsed = state.get("parsed") or {}
    num_models = (
        parsed.get("num_models", 1)
        if isinstance(parsed, dict)
        else getattr(parsed, "num_models", 1)
    )
    primary_metric_line = (
        f"Primary metric to optimize: {primary_metric}\n\n" if primary_metric else ""
    )
    content = (
        primary_metric_line
        + f"Task analysis:\n{task_analysis}\n\n"
        + f"Benchmark metadata:\n{json.dumps(metadata, indent=2)}\n\n"
        + "Determine task_type (regression or classification) from the benchmark metadata (e.g. target_types) and pass it to train_and_predict and create_ensemble_predictions. If main_metric is roc_auc (or roc_auc_score), pass prediction_format='probabilities' to train_and_predict and create_ensemble_predictions. Otherwise use prediction_format='labels'.\n\n"
        + (
            f"Optimize the primary metric '{primary_metric}' when choosing pipelines and hyperparameters.\n\n"
            if primary_metric
            else ""
        )
        + f"The user asked for exactly {num_models} pipeline(s). You must run exactly N pipelines (call train_and_predict N times) before calling finish_and_report. Each of the N pipelines must be meaningfully different (e.g. different fingerprint and/or model type); repeating the same pipeline does not count toward N. Do not call finish_and_report until you have N completed runs. Run pipelines one at a time: after each train_and_predict result, use the returned metrics and description to decide the next pipeline. You can iterate on hyperparameters (use_cv and tune_param_grid, or different hyperparameters) using each run's test metrics before moving to a new pipeline. Use each pipeline's results to inform the next (e.g. try a different model or fingerprint if the last one underperformed, or add scale_features/PCA/feature selection if that might help). Prefer pipelines that could improve the primary metric (e.g. try feature selection or scaling when relevant), not only safe baselines. When done, call finish_and_report."
    )
    return {
        "messages": [
            SystemMessage(content=get_prompt("modelling_agent_system")),
            HumanMessage(content=content),
        ]
    }


def modelling_agent_node(state: AgentState) -> dict[str, Any]:
    """Invoke LLM with tools; return new messages."""
    step_start("modelling_agent", "Agent thinking / choosing tool")
    messages = list(state.get("messages") or [])
    llm = _get_llm()
    tools = get_modelling_tools()
    llm_with_tools = llm.bind_tools(tools)
    model, temp = _llm_model_and_temp()
    llm_call(model=model, temperature=temp, purpose="Modelling agent (ReAct)")
    response = llm_with_tools.invoke(messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    # Show any accompanying reasoning/thoughts first, then tool calls
    if response.content:
        thinking(
            stage="modelling_agent (reasoning)" if tool_calls else "modelling_agent",
            content=response.content,
        )
    elif not tool_calls:
        thinking(stage="modelling_agent", content="(no content)")
    if tool_calls:
        tool_calls_display(tool_calls)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute tool calls from the last AIMessage and return ToolMessages + state updates."""
    step_start("tools", "Executing tool(s)")
    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {}
    tool_messages: list[ToolMessage] = []
    current_state = dict(state)
    for tc in last.tool_calls:
        name = tc.get("name") or ""
        args = tc.get("args") or {}
        tid = tc.get("id", "")
        content, updates = dispatch_tool(
            state=current_state, tool_name=name, tool_args=args
        )
        args_summary = format_args_summary(args)
        result_summary = content if isinstance(content, str) else str(content)
        tool_result_display(
            tool_name=name, args_summary=args_summary, result_summary=result_summary
        )
        tool_messages.append(ToolMessage(content=content, tool_call_id=tid))
        if updates:
            current_state.update(updates)
        logs = _log(
            state=current_state,
            tool_or_step=name,
            args_summary=args_summary,
            result_summary=result_summary[:1000] + "..."
            if len(result_summary) > 1000
            else result_summary,
        )
        current_state["logs"] = logs
        if name == "train_and_predict" and updates:
            reflection = _reflection_from_train_result(content)
            if reflection:
                thought = list(current_state.get("thought_process") or [])
                thought.append(
                    ThoughtStep(stage="reflection", content=reflection).model_dump()
                )
                current_state["thought_process"] = thought
    result: dict[str, Any] = {"messages": tool_messages}
    if current_state.get("completed_runs") is not None:
        result["completed_runs"] = current_state["completed_runs"]
    if "last_predictions" in current_state:
        result["last_predictions"] = current_state["last_predictions"]
    if "iteration" in current_state:
        result["iteration"] = current_state["iteration"]
    if "logs" in current_state:
        result["logs"] = current_state["logs"]
    if "thought_process" in current_state:
        result["thought_process"] = current_state["thought_process"]
    return result


def should_continue_modelling(
    state: AgentState,
) -> Literal["tools", "report_generator"]:
    """Route to tools if last message has tool_calls, else report_generator. Cap at 30 messages."""
    messages = state.get("messages") or []
    if len(messages) >= 30:
        return "report_generator"
    if not messages:
        return "report_generator"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
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
    primary_metric = state.get("primary_metric") or None
    os.makedirs(out_dir, exist_ok=True)

    # Metrics comparison plot (bar chart of primary metric per run)
    metrics_plot_path = save_metrics_comparison_plot(
        completed_runs=completed,
        output_dir=out_dir,
        filename="metrics_comparison.png",
        primary_metric=primary_metric,
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
        report_md = (
            report_md.rstrip()
            + f"\n\n### Metrics comparison\n\n![Metrics comparison]({os.path.basename(metrics_plot_path)})\n"
        )
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
    builder.add_node("seed_modelling_messages", seed_modelling_messages)
    builder.add_node("modelling_agent", modelling_agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("report_generator", report_generator_node)

    builder.add_edge(START, "parse_input")
    builder.add_edge("parse_input", "load_benchmark")
    builder.add_edge("load_benchmark", "analyze")
    builder.add_edge("analyze", "seed_modelling_messages")
    builder.add_edge("seed_modelling_messages", "modelling_agent")
    builder.add_conditional_edges("modelling_agent", should_continue_modelling)
    builder.add_edge("tools", "modelling_agent")
    builder.add_edge("report_generator", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
