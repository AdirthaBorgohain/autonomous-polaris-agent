"""CLI entrypoint; delegates to the Deep Agent."""

import argparse
import sys

from autonomous_polaris_agent.runner import run_deep_agent


def main() -> None:
    """Entrypoint for uv run python -m autonomous_polaris_agent or main.py."""
    parser = argparse.ArgumentParser(
        description="Run the autonomous Polaris agent: plans and runs ML pipelines from natural language."
    )
    parser.add_argument(
        "message",
        nargs="?",
        default="",
        help="Natural language request (e.g. dataset URL + number of models).",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Run code on host (no OpenSandbox). Default is --sandbox.",
    )
    args = parser.parse_args()
    user_message = (args.message or "").strip()
    if not user_message:
        parser.print_help(sys.stderr)
        print(file=sys.stderr)
        print("Example:", file=sys.stderr)
        print(
            '  uv run python main.py "Generate 4 regression models on https://polarishub.io/benchmarks/polaris/adme-fang-solu-1"',
            file=sys.stderr,
        )
        sys.exit(1)
    run_deep_agent(user_message=user_message, use_sandbox=not args.no_sandbox)
