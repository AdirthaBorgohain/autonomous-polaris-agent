---
name: polaris-tools
description: Use this skill when working with Polaris benchmarks. Documents load_benchmark and evaluate_predictions tools — inputs, outputs, and when to call each.
allowed-tools: load_benchmark, evaluate_predictions
---

# polaris-tools

## Overview

This skill documents the Polaris-specific tools available for autonomous chemistry ML on Polaris benchmarks. Use these tools to load benchmark metadata and evaluate predictions. For report writing, use the report-writing skill.

## Instructions

### load_benchmark

**Purpose:** Load a Polaris benchmark and get metadata before writing code.

**Inputs:**
- `slug` (str): Benchmark slug (e.g. `polaris/adme-fang-solu-1`) or URL

**Outputs:** JSON with `n_train_datapoints`, `n_test_datapoints`, `target_types`, `main_metric`, `resolved_slug`, etc.

**Usage:**
- Call first to understand the dataset size, task type (regression/classification), and primary metric
- Use `main_metric` to decide `prediction_format`: if it's `roc_auc`/`roc_auc_score`/`auroc`, use probabilities; otherwise use labels
- Use `resolved_slug` when calling `evaluate_predictions` or in your code with `polaris.load_benchmark(slug)`

### evaluate_predictions

**Purpose:** Get official metrics for predictions your code produced.

**Inputs:**
- `predictions_path` (str): Path to JSON file containing a list of predictions (floats or ints)
- `benchmark_slug` (str): e.g. `polaris/adme-fang-solu-1`
- `prediction_format` (str): `"labels"` or `"probabilities"` — use probabilities for ROC AUC benchmarks

**Outputs:** JSON with metric names and scores (e.g. `{"pearsonr": 0.61, "mean_absolute_error": 0.40}`)

**Usage:**
- Your training script should write predictions to a JSON file (e.g. `predictions_0.json`)
- Call this tool with the file path to get official benchmark metrics
- Use results to decide your next experiment (e.g. try different features or model)

