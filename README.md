# Autonomous Polaris Agent

An autonomous agent that takes a natural language request (e.g. a Polaris benchmark URL and number of models), plans and runs multiple ML pipelines end-to-end, evaluates them, and produces a short report and artifacts. Built with LangGraph and OpenAI.

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
# Optional: OPENAI_MODEL=5.4 (default)
```

## Run (single command)

The entire workflow runs without manual intervention after you invoke:

```bash
uv run python main.py "Generate 4 different regression models on https://polarishub.io/benchmarks/polaris/adme-fang-solu-1. Save the models and give me a report on their performance on the validation set."
```

Each run gets a **timestamped directory** under `outputs/` (e.g. `outputs/run_20260308_150530/`). Everything for that run lives there:

- `outputs/run_<timestamp>/report.md` – final report (dataset, approach, reasoning, results, artifact links)
- `outputs/run_<timestamp>/artifacts/` – saved models (e.g. `model_0.joblib`)
- `outputs/run_<timestamp>/run.jsonl` – structured log of tool calls and results

## How it works

1. **Parse** – The agent extracts benchmark slug and number of models from your message.
2. **Load** – It loads the Polaris benchmark and inspects train/test size, target, and task type.
3. **Analyze** – It produces a short task analysis (strategy, representation and model choices).
4. **Loop** – For each of the requested pipelines: plan (with rationale) → train/predict → evaluate → reflect. Plans differ (e.g. Morgan + Ridge, then Random Forest, then RDKit descriptors).
5. **Report** – It writes a ≤3-page Markdown report with reasoning, decisions, and results.

No manual steps are required beyond the initial configuration and the single run command.
