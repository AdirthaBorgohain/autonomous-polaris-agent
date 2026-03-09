# Chemistry ML sandbox image for OpenSandbox.
# Use with: create_opensandbox(image="your-registry/chemistry-ml:latest")
# Build: docker build -f docker/chemistry-ml.Dockerfile -t chemistry-ml:latest .

FROM ghcr.io/astral-sh/uv:python3.12-trixie

WORKDIR /app

COPY docker/pyproject.chemistry-ml.toml pyproject.toml

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev

ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /workspace
WORKDIR /workspace

# OpenSandbox default entrypoint: tail -f /dev/null
CMD ["tail", "-f", "/dev/null"]
