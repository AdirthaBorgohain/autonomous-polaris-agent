# Autonomous Polaris Agent

An autonomous agent that takes a natural language request (e.g. a Polaris benchmark URL and number of models), **writes and executes its own** preprocessing, training, and evaluation code, and produces a report and artifacts. Built with [Deep Agents](https://docs.langchain.com/oss/python/deepagents/) and OpenAI.

For a concise technical overview of the dataset, approach, key decisions, and results, see [REPORT.md](REPORT.md).

The agent runs without manual intervention. By default it uses `LocalShellBackend` (no isolation — commands run on the host). With `--sandbox`, it uses [OpenSandbox](https://github.com/alibaba/OpenSandbox) for isolated container execution. Tools provide `load_benchmark` and `evaluate_predictions`. Skills are in [skills/](skills/) (polaris-tools, chemistry-ml, libraries, report-writing).

## Environment setup

- **Python**: 3.12+
- **Package manager**: [uv](https://github.com/astral-sh/uv) (recommended)

```bash
# Clone and enter the project
cd autonomous-polaris-agent

# Install dependencies (creates venv and installs with uv)
uv sync

# Copy env example and set your OpenAI API key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=your_key_here
# Optional: OPENAI_MODEL=gpt-5.4 (default)
```

**macOS (XGBoost / LightGBM):** If you want the agent to use XGBoost or LightGBM, install the OpenMP runtime:

```bash
brew install libomp
```

## Run (single command)

```bash
uv run python main.py "Generate 4 different regression models on https://polarishub.io/benchmarks/polaris/adme-fang-solu-1. Save the models and give me a report on their performance on the validation set."
```

Each run gets a **timestamped directory** under `outputs/` (e.g. `outputs/run_20260308_150530/`). The agent writes and runs its own code in that directory.

### Run with OpenSandbox (isolated execution)

Requires [OpenSandbox](https://github.com/alibaba/OpenSandbox) server running (Docker or Kubernetes). The sandbox uses the chemistry-ml image by default (uv, rdkit, polaris-lib).

```bash
# 1. Build the chemistry-ml sandbox image (one-time)
docker build -f docker/chemistry-ml.Dockerfile -t chemistry-ml:latest .

# 2. Install and start OpenSandbox server (one-time)
uv pip install opensandbox-server
opensandbox-server init-config ~/.sandbox.toml --example docker
opensandbox-server

# 3. Set env vars (or use ~/.sandbox.toml)
export OPEN_SANDBOX_DOMAIN=localhost:8080
export OPEN_SANDBOX_API_KEY=your-key-if-required

# 4. Run agent (sandbox is default; use --no-sandbox to run on host)
uv run python main.py "Generate 1 regression model on polaris/adme-fang-solu-1"
```

Override the image with `OPEN_SANDBOX_IMAGE` if using a registry (e.g. `your-registry/chemistry-ml:latest`).

## Example datasets

- [polaris/adme-fang-solu-1](https://polarishub.io/benchmarks/polaris/adme-fang-solu-1)
- [tdcommons/caco2-wang](https://polarishub.io/benchmarks/tdcommons/caco2-wang)

## Architecture

- **Entry**: `main.py` → `cli.main()` → `run_deep_agent()` in [runner.py](src/autonomous_polaris_agent/runner.py)
- **Agent**: Deep Agent harness in [deep_agent.py](src/autonomous_polaris_agent/deep_agent.py) with `LocalShellBackend` (default) or `OpenSandboxBackend` (when `--sandbox`)
- **Tools**: [deep_agent_tools.py](src/autonomous_polaris_agent/deep_agent_tools.py) — `load_benchmark`, `evaluate_predictions`
- **Skills**: [skills/](skills/) — polaris-tools, chemistry-ml, libraries, report-writing

The agent **writes its own code** for featurization, training, and evaluation. No predefined pipelines. It uses rdkit, sklearn, xgboost, lightgbm, and polaris-lib.

## Output structure

Each run produces a timestamped directory under `outputs/`:

- **report.md** — Final report (dataset, approach, reasoning, results, artifact links)
- **run.jsonl** — Message log
- **artifacts/** — Models and plots produced by the agent's code
