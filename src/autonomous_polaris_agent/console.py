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
    _console.print(Panel(table, title="[yellow]LLM call[/yellow]", border_style="yellow"))


def thinking(stage: str, content: str) -> None:
    """Emit reasoning/thinking (e.g. analysis, plan rationale, reflection)."""
    if not _enabled:
        return
    _console.print(Panel(Markdown(content), title=f"[magenta]Thinking ({stage})[/magenta]", border_style="magenta"))


def plan_display(
    fingerprint: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    rationale: str,
) -> None:
    """Emit current pipeline plan."""
    if not _enabled:
        return
    lines = [
        f"fingerprint: {fingerprint}",
        f"model_type: {model_type}",
        f"hyperparameters: {hyperparameters}",
    ]
    if rationale:
        lines.append(f"\nrationale: {rationale}")
    _console.print(
        Panel("\n".join(lines), title="[blue]Plan[/blue]", border_style="blue")
    )


def metrics_display(metrics: dict[str, float]) -> None:
    """Emit evaluation metrics."""
    if not _enabled:
        return
    rows = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    _console.print(Panel(rows, title="[green]Metrics[/green]", border_style="green"))


def error_display(message: str) -> None:
    """Emit error message."""
    if not _enabled:
        return
    _console.print(Panel(message, title="[red]Error[/red]", border_style="red"))


def run_start(user_message: str, run_dir: str) -> None:
    """Emit run start (user message, output dir)."""
    if not _enabled:
        return
    _console.rule("[bold]Agent run[/bold]", style="bold")
    _console.print(Panel(user_message, title="[bold]User message[/bold]", border_style="white"))
    _console.print(f"[dim]Output directory: {run_dir}[/dim]\n")


def run_end(report_path: Optional[str], artifact_paths: list[str], run_dir: str) -> None:
    """Emit run end (report, artifacts, dir)."""
    if not _enabled:
        return
    _console.rule("[bold]Run complete[/bold]", style="bold green")
    if report_path:
        _console.print(f"Report: {report_path}")
    for path in artifact_paths:
        _console.print(f"Artifact: {path}")
    _console.print(f"Run directory: {run_dir}")
