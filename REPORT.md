# Autonomous Polaris Agent: Technical Report

A concise technical overview of the dataset, approach, key decisions, and results for the autonomous chemistry ML agent.

---

## 1. Dataset

The agent uses **Polaris Hub** benchmarks—standardized chemistry ML datasets with fixed train/test splits and evaluation metrics. Benchmarks are identified by slugs (e.g. `polaris/adme-fang-solu-1`, `tdcommons/caco2-wang`).

### Data source and format

- **Source:** [Polaris Hub](https://polarishub.io) via `polaris-lib`
- **Input:** SMILES strings and target values from `get_train_test_split()`
- **Evaluation:** Held-out test set from the benchmark (referred to as “validation set” in user-facing reports)

### Example benchmarks

| Benchmark | Domain | Target | Task | Train / Test | Primary metric |
|-----------|--------|--------|------|--------------|----------------|
| `polaris/adme-fang-solu-1` | ADME | LOG_SOLUBILITY | Regression | 1,578 / 400 | Pearson r |
| `tdcommons/caco2-wang` | ADME | Caco-2 permeability | Regression | 728 / 182 | MAE |
| `tdcommons/pgp-broccatelli` | ADME | P-gp inhibition | Binary classification | 973 / 245 | ROC AUC |

The agent resolves benchmark slugs from URLs (e.g. `https://polarishub.io/benchmarks/polaris/adme-fang-solu-1`) or short names (e.g. `adme-fang-solu-1` → `polaris/adme-fang-solu-1`).

---

## 2. Overall Approach

### Architecture

The agent is a **LangGraph** workflow that runs end-to-end without manual intervention:

```
Parse → Load → Analyze → ReAct (modelling loop) → Report
```

1. **Parse** — LLM extracts `dataset_slug`, `num_models`, `task_type`, `save_models` from natural language.
2. **Load** — `polaris.load_benchmark(slug)` loads data and metadata (n_train, n_test, target types, main metric).
3. **Analyze** — LLM produces a task analysis (strategy, fingerprint/model choices, prediction format) from metadata.
4. **ReAct loop** — Modelling agent runs N distinct pipelines sequentially, using prior run metrics to choose the next pipeline.
5. **Report** — LLM generates a Markdown report from `task_analysis`, `thought_process`, and `completed_runs`.

### How the agent executes pipelines

The modelling agent operates in a **ReAct loop**: it reasons, calls a tool, receives the result, then reasons again before the next call. Each pipeline is chosen *after* seeing the previous run’s metrics.

**Initial setup.** Before the loop, the agent receives:
- **Task analysis** — LLM-generated strategy (which fingerprints/models fit the task, data size, prediction format).
- **Benchmark metadata** — n_train, n_test, target types, `main_metric`.
- **Catalog** — `list_available_models` returns fingerprints (Morgan, RDKit descriptors, MACCS, atom_pair, morgan_rdkit_concat), model types (ridge, XGBoost, LightGBM, SVR, etc.), and pipeline options (scale_features, use_cv, tune_param_grid).

**Choosing each pipeline.** The agent must run exactly N *distinct* pipelines (different fingerprint and/or model type). For each next pipeline it:
1. **Reasons from prior results** — The system prompt instructs it to “reason briefly from the latest result: what improved or worsened, and why you are choosing this next pipeline.” Example: “ridge had high MAE, try random_forest with the same fingerprint.”
2. **Uses returned metrics** — Each `train_and_predict` returns `model_id`, `description`, `metrics`, `metric_name`, `metric_value`, and optionally `best_hyperparameters` and `cv_score`. The agent uses these to decide the next fingerprint, model type, and hyperparameters.
3. **Iterates within a pipeline** — It can refine hyperparameters (e.g. via `use_cv=True` and `tune_param_grid`) before switching to a new pipeline; only distinct pipelines count toward N.

**Reflection and state.** After each `train_and_predict`, the graph appends a short reflection to `thought_process` (e.g. “run_0: morgan_rdkit_concat + ridge — pearsonr=0.447”). The full conversation (messages, tool results) is kept in state, so the LLM sees all prior runs when choosing the next one. `finish_and_report` is rejected until exactly N pipelines are completed.

**Example flow (adme-fang-solu-1).** Run 0: Morgan+RDKit + Ridge → pearsonr 0.447. Agent: “linear baseline underperformed, try XGBoost on same features.” Run 1: Morgan+RDKit + XGBoost → 0.609. Agent: “nonlinear helped; test whether descriptors alone suffice.” Run 2: RDKit + LightGBM → 0.414. Agent: “descriptors alone weaker; try Morgan counts + SVR.” Run 3: Morgan counts + SVR → 0.555. Agent calls `finish_and_report`.

### Modelling pipeline

Each pipeline is: **featurize → (optional CV) → fit → predict on test → evaluate → save model/plots**.

- **Featurizers:** Morgan (ECFP), Morgan counts, RDKit descriptors, MACCS, atom_pair, or Morgan+RDKit concatenation.
- **Models:** Ridge, Random Forest, XGBoost, LightGBM, SVR, Extra Trees, Kernel Ridge.
- **Metrics:** Benchmark-specific (e.g. Pearson r, MAE, ROC AUC); `prediction_format` is inferred from the main metric (e.g. ROC AUC → probabilities).

### State across iterations

The agent keeps `completed_runs`, `thought_process`, and `messages` in `AgentState`. Each new pipeline is chosen based on prior results (e.g. “linear baseline underperformed → try XGBoost”).

---

## 3. Key Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph** | Stateful multi-step workflows, conditional edges, checkpointing. |
| **ReAct-style modelling agent** | Reasons before each tool call; uses prior run results to choose the next pipeline. |
| **Polaris integration** | Standard benchmarks, fixed splits, and evaluation via `polaris-lib`. |
| **Domain-appropriate choices** | Fingerprints and models chosen by LLM from task analysis; `prediction_format` inferred from `main_metric`. |
| **Classical ML (sklearn, XGBoost, LightGBM)** | Suited to small–medium datasets; avoids overfitting and heavy compute. |
| **Structured logs (run.jsonl)** | One JSON object per step: `tool_or_step`, `args_summary`, `result_summary`, `timestamp`. |
| **Timestamped output dirs** | Each run isolated under `outputs/run_YYYYMMDD_HHMMSS/`. |
| **Single command** | `uv run python main.py "..."` runs the full workflow. |

---

## 4. Results

Results below are from sample runs on the held-out evaluation set.

### polaris/adme-fang-solu-1 (regression, Pearson r)

| Pipeline | Pearson r | MAE | R² |
|----------|-----------|-----|-----|
| morgan_rdkit_concat + xgboost | **0.609** | **0.403** | **0.367** |
| morgan_counts + svr | 0.555 | 0.462 | 0.305 |
| morgan_rdkit_concat + ridge | 0.447 | 0.590 | -0.116 |
| rdkit_descriptors + lightgbm | 0.414 | 0.461 | 0.168 |

**Takeaway:** Hybrid features (Morgan+RDKit) with nonlinear models outperformed linear and descriptor-only baselines.

### tdcommons/caco2-wang (regression, MAE)

| Pipeline | Validation MAE |
|----------|----------------|
| rdkit_descriptors + xgboost | **0.326** |
| morgan_rdkit_concat + ridge | 0.336 |
| rdkit_descriptors + lightgbm | 0.348 |
| morgan_counts + kernel_ridge | 2.173 |

**Takeaway:** Descriptor-based gradient boosting worked best; kernel ridge on Morgan counts underperformed.

### tdcommons/pgp-broccatelli (classification, ROC AUC)

| Pipeline | Validation ROC AUC |
|----------|--------------------|
| morgan_counts + extra_trees | **0.924** |
| morgan_rdkit_concat + lightgbm | 0.917 |
| morgan_counts + xgboost | 0.914 |
| rdkit_descriptors + ridge | 0.896 |

**Takeaway:** Substructure-focused tree models (Morgan fingerprints) outperformed descriptor-only baselines.

### Summary

- **Regression:** Hybrid or descriptor-based features with tree/boosting models (XGBoost, LightGBM) or SVR typically beat linear baselines.
- **Classification:** Morgan fingerprints with tree ensembles (XGBoost, LightGBM, Extra Trees) gave the best ROC AUC.
- **Iterative planning:** The agent used prior run metrics to guide the next pipeline (e.g. “ridge underperformed → try XGBoost”).

---

## 5. Outputs

Each run produces:

- **report.md** — Final report (≤3 pages) with dataset, approach, reasoning, results table, and artifact links.
- **run.jsonl** — Structured log of plans, tool calls, and results.
- **artifacts/** — Saved models (`model_0.joblib`, …), train/test histograms, CV results, and `metrics_comparison.png`.
