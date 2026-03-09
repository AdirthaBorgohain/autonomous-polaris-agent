---
name: libraries
description: Use this skill to understand available libraries in the execution environment — polaris, sklearn, rdkit, xgboost, lightgbm, pandas, numpy — and their key APIs for chemistry ML.
---

# libraries

## Overview

Your code runs in an environment with these libraries pre-installed. Use this skill to understand their APIs and how to combine them for chemistry ML workflows.

## Instructions

### polaris

- `polaris.load_benchmark(slug)` — load a benchmark
- `benchmark.get_train_test_split()` — get train/test splits
- `benchmark.evaluate(y_pred=...)` or `benchmark.evaluate(y_prob=...)` — evaluate predictions

### sklearn

- **Models:** Ridge, SVR, RandomForest, XGBoost-style APIs
- **Preprocessing:** StandardScaler, PCA
- Other standard scikit-learn components

### pandas, numpy

- Data handling and numerical operations

### rdkit

- Morgan fingerprints, RDKit descriptors, MACCS keys (from `rdkit.Chem`)
- Use `rdkit.Chem.rdFingerprintGenerator` for Morgan fingerprints

### xgboost, lightgbm

- Gradient boosting libraries with sklearn-style APIs
