"""Rich console notifications for agent execution: steps, LLM calls, reasoning, results."""

from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

_console = Console()
_enabled = True


def set_enabled(enabled: bool) -> None:
    """Enable or disable console output (e.g. for tests)."""
    global _enabled
    _enabled = enabled


def step_start(step_name: str, description: str = "") -> None:
    """Emit start of a graph step."""
    if not _enabled:
        return
    title = f"Step: {step_name}"
    if description:
        title += f" — {description}"
    _console.rule(title, style="bold blue")


def step_result(name: str, summary: str, details: Optional[str] = None) -> None:
    """Emit result of a step (summary + optional details)."""
    if not _enabled:
        return
    body = summary
    if details:
        body += "\n\n" + details
    _console.print(Panel(body, title=f"[green]✓ {name}[/green]", border_style="green"))


def llm_call(model: str, temperature: float, purpose: str) -> None:
    """Emit LLM call info (model, params, purpose)."""
    if not _enabled:
        return
    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("model", model)
    table.add_row("temperature", str(temperature))
    table.add_row("purpose", purpose)
    _console.print(
        Panel(table, title="[yellow]LLM call[/yellow]", border_style="yellow")
    )


def thinking(stage: str, content: str) -> None:
    """Emit reasoning/thinking (e.g. analysis, plan rationale, reflection)."""
    if not _enabled:
        return
    _console.print(
        Panel(
            Markdown(content),
            title=f"[magenta]Thinking ({stage})[/magenta]",
            border_style="magenta",
        )
    )


def training_with_hyperparameter_tuning_display() -> None:
    """Emit notice that training is running with CV-based hyperparameter tuning."""
    if not _enabled:
        return
    _console.print(
        "[cyan]Training model with hyperparameter tuning (randomized search + CV)[/cyan]"
    )


def error_display(message: str) -> None:
    """Emit error message."""
    if not _enabled:
        return
    _console.print(Panel(message, title="[red]Error[/red]", border_style="red"))


def format_args_summary(args: dict[str, Any], max_len: int = 80) -> str:
    """Format tool args as a short one-line summary (for display)."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if v is None or v == "" or (isinstance(v, (list, dict)) and len(v) == 0):
            continue
        s = (
            f"{k}={v!r}"
            if not isinstance(v, (list, dict))
            else f"{k}=<{type(v).__name__}>"
        )
        if len(s) > 30:
            s = s[:27] + "..."
        parts.append(s)
    line = ", ".join(parts)
    return line[:max_len] + ("..." if len(line) > max_len else "")


def tool_calls_display(tool_calls: list[dict[str, Any]]) -> None:
    """Emit which tools the agent is invoking (name + args summary)."""
    if not _enabled or not tool_calls:
        return
    lines = []
    for tc in tool_calls:
        name = tc.get("name") or "?"
        args = tc.get("args") or {}
        summary = format_args_summary(args)
        if summary:
            lines.append(f"• {name}({summary})")
        else:
            lines.append(f"• {name}()")
    _console.print(
        Panel("\n".join(lines), title="[cyan]Tool calls[/cyan]", border_style="cyan")
    )


def tool_result_display(
    tool_name: str, args_summary: str, result_summary: str, max_result_len: int = 300
) -> None:
    """Emit one tool execution: name, args summary, result summary (truncated)."""
    if not _enabled:
        return
    result = result_summary
    if len(result) > max_result_len:
        result = result[:max_result_len] + "..."
    if "\n" in result:
        result = result.split("\n")[0] + " ..."
    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("tool", tool_name)
    if args_summary:
        table.add_row("args", args_summary)
    table.add_row("result", result)
    _console.print(
        Panel(
            table,
            title=f"[green]Tool result: {tool_name}[/green]",
            border_style="green",
        )
    )


def run_start(user_message: str, run_dir: str) -> None:
    """Emit run start (user message, output dir)."""
    if not _enabled:
        return
    _console.rule("[bold]Agent run[/bold]", style="bold")
    _console.print(
        Panel(user_message, title="[bold]User message[/bold]", border_style="white")
    )
    _console.print(f"[dim]Output directory: {run_dir}[/dim]\n")


def run_end(
    report_path: Optional[str], artifact_paths: list[str], run_dir: str
) -> None:
    """Emit run end (report, artifacts, dir)."""
    if not _enabled:
        return
    _console.rule("[bold]Run complete[/bold]", style="bold green")
    if report_path:
        _console.print(f"Report: {report_path}")
    for path in artifact_paths:
        _console.print(f"Artifact: {path}")
    _console.print(f"Run directory: {run_dir}")
