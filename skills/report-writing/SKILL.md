---
name: report-writing
description: Write final reports for chemistry ML agent runs. Use when summarizing completed experiments, producing report.md, or when the user asks for a report on validation/evaluation performance.
---

# Report Writing

## When to Use

- After completing all training runs on a Polaris benchmark
- When the user asks for "performance on the validation set" or "report on results"
- When finishing an autonomous chemistry ML run

## Report Structure

Generate a concise final report (max 3 pages when rendered) in Markdown.

**Important:** Metrics in `completed_runs` are computed on the benchmark's held-out evaluation set (the test split from `get_train_test_split`). When the user asks for "performance on the validation set", that is this same set—report it clearly.

### Required Sections

1. **Dataset and task** — Which benchmark was used, what the target is, task type.
2. **Approach** — How many pipelines were run and high-level strategy.
3. **Reasoning and decisions** — Weave together initial task analysis and per-iteration reasoning/reflections. Keep concise (bullets or short paragraphs).
4. **Results** — Table or list of each run (model id, description, metrics), and comparison. State that metrics are on the held-out evaluation (validation) set.
5. **Artifacts** — Paths to saved models and plots so the user can open them.

## Output Format

- Write the report as Markdown.
- Save to `report.md` in the run output directory.
- Optionally add a metrics comparison plot: generate `metrics_comparison.png` and include `![Metrics comparison](metrics_comparison.png)` in the report.

## Metrics Comparison Plot

To generate a bar chart of the primary metric per run, write and execute a script like:

```python
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# completed_runs: list of dicts with model_id, metrics
with open("completed_runs.json") as f:
    runs = json.load(f)

metric_name = next(iter(runs[0]["metrics"].keys())) if runs else "metric"
model_ids = [r.get("model_id", "?") for r in runs]
values = [r["metrics"].get(metric_name, r["metrics"].get(next(iter(r["metrics"].keys())))) for r in runs]

fig, ax = plt.subplots()
ax.bar(range(len(model_ids)), values, tick_label=model_ids)
ax.set_ylabel(metric_name)
ax.set_xlabel("Run")
ax.set_title(f"Metric comparison: {metric_name}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.close()
```

Save the plot in the output directory and reference it in the report.
