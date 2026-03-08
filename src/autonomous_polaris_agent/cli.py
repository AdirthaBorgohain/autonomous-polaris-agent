"""CLI entrypoint; delegates to the agent graph."""

from autonomous_polaris_agent.run import run_agent


def main() -> None:
    """Entrypoint for uv run python -m autonomous_polaris_agent or main.py."""
    run_agent()
