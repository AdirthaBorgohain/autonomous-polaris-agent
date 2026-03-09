"""Deep Agent run entrypoint."""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from autonomous_polaris_agent.deep_agent import build_deep_agent  # noqa: E402
from autonomous_polaris_agent.opensandbox_backend import (  # noqa: E402
    OpenSandboxBackend,
    create_opensandbox,
)


def _run_id_from_timestamp() -> str:
    """Return a run id string based on current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def _print_progress(verbose: bool, msg: str) -> None:
    """Print progress message if verbose."""
    if verbose:
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_start(verbose: bool, user_message: str, run_dir: str) -> None:
    """Print run start banner."""
    if not verbose:
        return
    print("─" * 60 + " Agent run " + "─" * 60, flush=True)
    print(f"User message: {user_message}", flush=True)
    print(f"Output directory: {run_dir}\n", flush=True)


def _run_end(
    verbose: bool, report_path: str | None, artifact_paths: list[str], run_dir: str
) -> None:
    """Print run end summary."""
    if not verbose:
        return
    print("\n" + "─" * 60 + " Run complete " + "─" * 60, flush=True)
    if report_path:
        print(f"Report: {report_path}", flush=True)
    for path in artifact_paths:
        print(f"Artifact: {path}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)


def _download_sandbox_outputs(backend: OpenSandboxBackend, dest_dir: str) -> None:
    """Download files from /workspace in the sandbox to dest_dir on the host."""
    try:
        infos = backend.glob_info(pattern="**/*", path="/workspace")
    except Exception:
        return
    file_paths = [fi["path"] for fi in infos if not fi.get("is_dir", False)]
    if not file_paths:
        return
    responses = backend.download_files(file_paths)
    for resp in responses:
        if resp.error or resp.content is None:
            continue
        rel = resp.path.removeprefix("/workspace/").lstrip("/")
        if not rel:
            continue
        out_path = os.path.join(dest_dir, rel)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(resp.content)


def _format_tool_call(tc: dict) -> str:
    """Format a single tool call for display."""
    name = tc.get("name") or "?"
    args = tc.get("args") or {}
    if args:
        parts = [
            f"{k}={str(v)[:20]}..." if len(str(v)) > 20 else f"{k}={v}"
            for k, v in list(args.items())[:3]
        ]
        return f"{name}({', '.join(parts)})"
    return f"{name}()"


def run_deep_agent(
    user_message: str = "",
    verbose: bool = True,
    *,
    use_sandbox: bool = True,
) -> dict:
    """Run the Deep Agent on the given user message.

    Creates a timestamped directory under outputs/ and streams the agent,
    printing progress (LLM responses, tool calls) as it runs.

    Args:
        user_message: Natural language request (e.g. dataset + number of models).
        verbose: If True, emit progress output (default True).
        use_sandbox: If True, run code in OpenSandbox (requires opensandbox-server).

    Returns:
        Final state with report path and any collected outputs.
    """
    run_id = _run_id_from_timestamp()
    run_dir = os.path.join("outputs", run_id)
    run_dir_abs = os.path.abspath(run_dir)
    os.makedirs(run_dir_abs, exist_ok=True)

    message = user_message or "Generate 1 regression model on polaris/adme-fang-solu-1."
    _run_start(verbose=verbose, user_message=message, run_dir=run_dir)

    sandbox = None
    sandbox_backend = None
    if use_sandbox:
        if verbose:
            _print_progress(verbose=True, msg="Creating OpenSandbox...")
        sandbox = create_opensandbox()
        sandbox_backend = OpenSandboxBackend(sandbox=sandbox)
        if verbose:
            _print_progress(verbose=True, msg=f"Sandbox ready (id={sandbox.id})")

    try:
        agent = build_deep_agent(
            output_dir=run_dir_abs,
            sandbox_backend=sandbox_backend,
        )

        config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        initial_state = {
            "messages": [{"role": "user", "content": message}],
        }

        if verbose:
            _print_progress(verbose=True, msg="Agent running (streaming progress)...")

        result = None
        for mode, chunk in agent.stream(
            initial_state,
            config=config,
            stream_mode=["values", "messages", "updates"],
        ):
            if mode == "values":
                result = chunk
            elif mode == "updates" and verbose:
                for node_name, update in chunk.items():
                    if node_name == "model" and isinstance(update, dict):
                        msgs = update.get("messages", [])
                        for m in (msgs or [])[-1:]:
                            tool_calls = getattr(m, "tool_calls", None) or []
                            if tool_calls:
                                for tc in tool_calls:
                                    name = (
                                        getattr(tc, "name", tc.get("name", "?"))
                                        if not isinstance(tc, dict)
                                        else tc.get("name", "?")
                                    )
                                    args = (
                                        getattr(tc, "args", tc.get("args", {}))
                                        if not isinstance(tc, dict)
                                        else tc.get("args", {})
                                    )
                                    _print_progress(
                                        verbose=verbose,
                                        msg=f"Tool: {_format_tool_call({'name': name, 'args': args or {}})}",
                                    )
                    elif node_name == "tools":
                        _print_progress(verbose=verbose, msg="Tools completed")
            elif mode == "messages" and verbose:
                msg_chunk, _meta = chunk
                content = getattr(msg_chunk, "content", None) or ""
                if content:
                    print(content, end="", flush=True)

        if verbose and result:
            print(flush=True)

        if sandbox_backend is not None and sandbox is not None:
            if verbose:
                _print_progress(verbose=True, msg="Downloading outputs from sandbox...")
            _download_sandbox_outputs(backend=sandbox_backend, dest_dir=run_dir_abs)
            if verbose:
                _print_progress(verbose=True, msg="Stopping sandbox...")
            try:
                sandbox.kill()
                sandbox.close()
            except Exception:
                pass

        messages = (result.get("messages") or []) if result else []
        if messages:
            log_path = os.path.join(run_dir_abs, "run.jsonl")
            with open(log_path, "w") as f:
                for msg in messages:
                    entry = {
                        "role": getattr(msg, "type", "unknown"),
                        "content": str(getattr(msg, "content", msg))[:500],
                    }
                    f.write(json.dumps(entry, default=str) + "\n")

        report_path = os.path.join(run_dir_abs, "report.md")
        artifact_paths = []
        if os.path.isfile(report_path):
            artifacts_dir = os.path.join(run_dir_abs, "artifacts")
            if os.path.isdir(artifacts_dir):
                for f in os.listdir(artifacts_dir):
                    if f.endswith(".joblib"):
                        artifact_paths.append(os.path.join(artifacts_dir, f))

        _run_end(
            verbose=verbose,
            report_path=report_path if os.path.isfile(report_path) else None,
            artifact_paths=artifact_paths,
            run_dir=run_dir_abs,
        )

        return {
            "messages": messages,
            "report_path": report_path,
            "output_dir": run_dir_abs,
        }
    except Exception:
        if sandbox is not None:
            try:
                sandbox.kill()
                sandbox.close()
            except Exception:
                pass
        raise
