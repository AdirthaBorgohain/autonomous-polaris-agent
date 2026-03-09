---
name: chemistry-ml
description: Use this skill for chemistry ML pipelines on molecular datasets. Covers featurization (Morgan, RDKit descriptors, MACCS), model selection (Ridge, XGBoost, LightGBM), pipeline design, and iteration best practices.
---

# chemistry-ml

## Overview

This skill provides best practices for building chemistry ML pipelines on molecular datasets. You design and implement featurization, training, and evaluation yourself — no predefined pipelines.

## Instructions

### Featurization

- **Morgan (ECFP):** Use `rdkit.Chem.rdFingerprintGenerator` for Morgan fingerprints
- **Other options:** Morgan counts, RDKit descriptors, MACCS keys, atom pair, or concatenations
- **Source:** All from `rdkit.Chem`

### Models

- **Regression:** Ridge, SVR, RandomForest, XGBoost, LightGBM
- **Classification:** LogisticRegression, RandomForest, XGBoost, LightGBM
- **ROC AUC benchmarks:** Use `predict_proba` when main_metric is roc_auc/roc_auc_score/auroc

### Pipeline

1. Featurize molecules
2. Optional: StandardScaler
3. Optional: feature selection
4. Train model

### Evaluation

- Always use the benchmark's held-out test set
- Call `evaluate_predictions` with your predictions file to get official metrics

### Iteration

- Use prior run metrics to choose the next pipeline
- Try diverse fingerprint–model combinations
- Do not pick at random — let metrics guide your next experiment
