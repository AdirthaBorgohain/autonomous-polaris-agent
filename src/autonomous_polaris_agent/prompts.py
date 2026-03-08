"""Centralized prompts for parsing, analysis, planning, reflection, and report."""

PARSE_USER_INPUT = """You are a parser for a chemistry ML agent. Extract from the user's natural language request the following, and respond with valid JSON only (no markdown, no explanation).

Required:
- dataset_slug: the Polaris benchmark slug. If the user gives a URL like https://polarishub.io/benchmarks/polaris/adme-fang-solu-1, use "polaris/adme-fang-solu-1". If they give a short name like "adme-fang-solu-1", use "polaris/adme-fang-solu-1".
- num_models: integer number of different models/pipelines to build (e.g. "4 different models" -> 4).

Optional (infer if possible):
- task_type: "regression" or "classification" if the user says so or it's clear from context.
- explicit_metric: only if the user explicitly asks for a specific metric.
- save_models: true unless the user says not to save.

Output format (JSON only):
{"dataset_slug": "...", "num_models": N, "task_type": "regression"|"classification"|null, "explicit_metric": null|"...", "save_models": true|false}
"""

ANALYSIS_TASK = """You are an expert in chemistry ML and ADME modeling. Given the following benchmark inspection (train/test sizes, target info, input modalities, task type), write a short task analysis (1-2 paragraphs) that covers:

1. What the task is: what the target represents (e.g. solubility, permeability) and the task type (regression/classification).
2. Data characteristics: size, what might make this easy or hard, any red flags (e.g. small n, skewed target).
3. High-level strategy: which molecular representations (e.g. Morgan fingerprints, RDKit descriptors) and model families (e.g. ridge, random forest, gradient boosting) are appropriate and why.

Do not set performance targets or mention leaderboards. Focus on understanding and domain-appropriate strategy.

Benchmark inspection:
{benchmark_inspection}
"""

PLANNING_PIPELINE = """You are planning the next ML pipeline for a chemistry benchmark. The user asked for {num_models} different models. So far {completed_count} pipeline(s) have been run. You must choose a pipeline that is DIFFERENT from previous runs (different fingerprint, model type, or hyperparameters).

Context:
- User request: {user_input}
- Task analysis: {task_analysis}
- Benchmark inspection: {benchmark_inspection}
- Previous runs (do not repeat): {completed_runs_summary}

Domain guidance (ADME/chemistry):
- Morgan fingerprints (ECFP) are standard for structure-activity; use radius 2-3, 2048 bits typically.
- RDKit descriptors are smaller and interpretable; good for smaller datasets.
- For small n, prefer regularized models (ridge) or simple baselines; avoid overfitting.
- Each new pipeline should vary: e.g. try morgan+ridge, then morgan+random_forest, then rdkit_descriptors+ridge, or different hyperparameters.

Output valid JSON only with two keys:
- "reasoning": 1-2 sentences on why you chose this pipeline now.
- "plan": object with keys: fingerprint (string, e.g. "morgan" or "rdkit_descriptors"), model_type (string, e.g. "ridge", "random_forest", "gradient_boosting"), hyperparameters (object, e.g. {{"alpha": 1.0}} or {{"n_estimators": 100}}), rationale (2-4 sentences: why this representation/model/hyperparameters), use_cv (boolean), tune_over (array of param names to tune if use_cv, e.g. ["alpha"]).
"""

REFLECTION_AFTER_RUN = """You just ran one pipeline and got these results. Reflect briefly, then say whether to continue or finish.

Last run result: {last_run_result}
All completed runs so far: {completed_runs_summary}
Remaining iterations to reach user's requested count: {remaining}

Output valid JSON only:
- "reflection": 2-4 sentences: what worked or didn't, how this compares to prior runs, what you might try differently next.
- "next_action": "continue" or "finish"
- "next_direction" (optional): one sentence hint for the next plan if next_action is "continue".
"""

REPORT_FINAL = """Generate a concise final report (max 3 pages when rendered) for the chemistry ML agent run.

The metrics in completed_runs are computed on the benchmark's held-out evaluation set (the test split from get_train_test_split). When the user asks for "performance on the validation set", that is this same set—report it clearly.

Sections to include:
1. Dataset and task: which benchmark was used, what the target is, task type.
2. Approach: how many pipelines were run and high-level strategy.
3. Reasoning and decisions: weave together the initial task analysis and per-iteration reasoning/reflections (from thought_process). Keep this section concise (bullets or short paragraphs).
4. Results: table or list of each run (model id, description, metrics), and comparison. State that metrics are on the held-out evaluation (validation) set.
5. Artifacts: paths to saved models and plots so the user can open them.

User request: {user_input}
Parsed request: {parsed}
Task analysis: {task_analysis}
Thought process (reasoning and reflections): {thought_process}
Completed runs: {completed_runs}
Report in Markdown.
"""


def get_prompt(name: str, **kwargs) -> str:
    """Return the prompt named `name`, with format kwargs applied."""
    prompts = {
        "parse_user_input": PARSE_USER_INPUT,
        "analysis_task": ANALYSIS_TASK,
        "planning_pipeline": PLANNING_PIPELINE,
        "reflection_after_run": REFLECTION_AFTER_RUN,
        "report_final": REPORT_FINAL,
    }
    template = prompts.get(name)
    if template is None:
        raise KeyError(f"Unknown prompt: {name}")
    return template.format(**kwargs) if kwargs else template
