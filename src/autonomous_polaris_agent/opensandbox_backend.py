"""OpenSandbox backend for deepagents — runs code in isolated containers."""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox import SandboxSync
from opensandbox.config.connection_sync import ConnectionConfigSync
from opensandbox.models.filesystem import WriteEntry

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_OUTPUT_BYTES = 500_000


DEFAULT_SANDBOX_IMAGE = "chemistry-ml:latest"

DOCKERFILE_PATH = "docker/chemistry-ml.Dockerfile"


def _project_root() -> str:
    """Return the project root (directory containing docker/ and pyproject.toml)."""
    path = os.path.abspath(__file__)
    for _ in range(4):  # src/autonomous_polaris_agent/opensandbox_backend.py -> project root
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
    return os.getcwd()


def _image_exists_locally(image: str) -> bool:
    """Return True if the Docker image exists locally."""
    name, _, tag = image.partition(":")
    repo_tag = f"{name}:{tag}" if tag else f"{name}:latest"
    result = subprocess.run(
        ["docker", "images", "-q", repo_tag],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return bool(result.returncode == 0 and result.stdout.strip())


def _build_chemistry_ml_image() -> None:
    """Build chemistry-ml:latest from docker/chemistry-ml.Dockerfile."""
    root = _project_root()
    dockerfile = os.path.join(root, DOCKERFILE_PATH)
    if not os.path.isfile(dockerfile):
        raise FileNotFoundError(
            f"Dockerfile not found: {dockerfile}. Run from project root."
        )
    subprocess.run(
        ["docker", "build", "-f", DOCKERFILE_PATH, "-t", DEFAULT_SANDBOX_IMAGE, "."],
        cwd=root,
        check=True,
        timeout=600,
    )


def ensure_sandbox_image(image: str) -> None:
    """Build chemistry-ml:latest if missing. No-op for registry images."""
    if image != DEFAULT_SANDBOX_IMAGE or "/" in image:
        return
    if _image_exists_locally(image):
        return
    _build_chemistry_ml_image()


def create_opensandbox(
    image: str | None = None,
    *,
    timeout: timedelta | None = None,
    domain: str | None = None,
    api_key: str | None = None,
) -> "SandboxSync":
    """Create an OpenSandbox instance with chemistry ML defaults.

    Uses the image built from docker/chemistry-ml.Dockerfile by default (uv, rdkit,
    polaris-lib). Build with: docker build -f docker/chemistry-ml.Dockerfile -t chemistry-ml:latest .

    Args:
        image: Docker image (default: chemistry-ml:latest from OPEN_SANDBOX_IMAGE or build).
        timeout: Sandbox lifetime (default 30 min).
        domain: OpenSandbox API domain (env: OPEN_SANDBOX_DOMAIN).
        api_key: API key (env: OPEN_SANDBOX_API_KEY).

    Returns:
        SandboxSync instance (caller must call kill() when done).
    """
    image = image or os.environ.get("OPEN_SANDBOX_IMAGE") or DEFAULT_SANDBOX_IMAGE
    ensure_sandbox_image(image)
    config = ConnectionConfigSync(
        domain=domain or os.environ.get("OPEN_SANDBOX_DOMAIN", "localhost:8080"),
        api_key=api_key or os.environ.get("OPEN_SANDBOX_API_KEY"),
        request_timeout=timedelta(seconds=60),
    )
    return SandboxSync.create(
        image,
        connection_config=config,
        timeout=timeout or timedelta(minutes=30),
        env={"PYTHONUNBUFFERED": "1"},
    )


class OpenSandboxBackend(BaseSandbox):
    """Deepagents backend that delegates to OpenSandbox for isolated execution."""

    def __init__(
        self,
        sandbox: SandboxSync,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        """Initialize backend with an existing OpenSandbox instance.

        Args:
            sandbox: SandboxSync instance (caller owns lifecycle; call kill() when done).
            timeout: Default command timeout in seconds.
            max_output_bytes: Max output to capture before truncation.
        """
        self._sandbox = sandbox
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox backend."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the OpenSandbox container."""
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            from opensandbox.models.execd import RunCommandOpts

            opts = RunCommandOpts(timeout=timedelta(seconds=effective_timeout))
            execution = self._sandbox.commands.run(command, opts=opts)
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        stdout_parts = [m.text for m in execution.logs.stdout]
        stderr_parts = [f"[stderr] {m.text}" for m in execution.logs.stderr]
        output = "\n".join(stdout_parts + stderr_parts) if (stdout_parts or stderr_parts) else "<no output>"

        exit_code = 0
        if execution.error:
            exit_code = 1
            output = f"{output.rstrip()}\n\nError: {execution.error.name}: {execution.error.value}"

        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if exit_code != 0:
            output = f"{output.rstrip()}\n\nExit code: {exit_code}"

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the sandbox."""
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._sandbox.files.write_files(
                    [WriteEntry(path=path, data=content, mode=0o644)],
                )
                results.append(FileUploadResponse(path=path, error=None))
            except Exception:
                results.append(
                    FileUploadResponse(path=path, error="permission_denied"),
                )
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        results: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                results.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception:
                results.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found"),
                )
        return results
