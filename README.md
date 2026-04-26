# EE-559 Course Project

This repository contains the broader project plus a lean, modular hate-speech baseline component.

## Development install (recommended)

Install in editable mode once so imports work cleanly everywhere (scripts, notebooks, tests):

```bash
python -m pip install -e .
```

After that, use namespaced imports:

```python
from ee559_project.hate_baseline import ExperimentConfig, run_experiment
```

## What is implemented

- One unified data pipeline for all CSV files in `datasets/`
- One configurable model wrapper for:
	- `bert-base-uncased`
	- `roberta-base`
	- `microsoft/deberta-v3-base`
- Optional dataset filtering by dataset name before splitting/training
- Swappable pooling (`auto`, `cls`, `mean`)
- Swappable classifier head (`linear`, `mlp`)
- One shared train/eval flow with Macro F1 as primary metric
- Prediction export per run to CSV for later analysis
- Split-by-dataset count report per run

## Structure

- `src/ee559_project/hate_baseline/data.py`: loading, normalization, stratified split, dataloaders
- `src/ee559_project/hate_baseline/model.py`: transformer backbone + pooling + head
- `src/ee559_project/hate_baseline/train_eval.py`: training/evaluation/prediction utilities
- `src/ee559_project/hate_baseline/evaluation.py`: CSV-based metric reporting and confusion matrix
- `src/ee559_project/hate_baseline/runner.py`: end-to-end experiment runner
- `simple_classifier.ipynb`: lightweight experiment orchestration

## Typical usage (from notebook)

```python
from ee559_project.hate_baseline import ExperimentConfig, run_experiment

config = ExperimentConfig(
		epochs=3,
		batch_size=16,
		max_length=128,
		learning_rate=2e-5,
		pooling="auto",
		head="linear",
		device="auto",
		datasets=["ETHOS", "HateXplain"],
)

summary = run_experiment(
		model_name="bert-base-uncased",
		datasets_dir="datasets",
		output_dir="outputs",
		config=config,
)

print(summary)
```

For a quick CPU smoke test, lower `max_length`, `batch_size`, and set `max_train_samples`, `max_val_samples`, and `max_test_samples` to small values.

## Evaluation workflow

The training run only writes prediction CSVs. Use the standalone evaluator on those CSVs to compute the report you want to extend later.

```python
from ee559_project.hate_baseline import (
	evaluate_prediction_csv,
	evaluate_prediction_csv_overall_and_per_dataset,
)

report = evaluate_prediction_csv("outputs/predictions_bert-base-uncased.csv")
print(report["accuracy"])
print(report["macro_f1"])
print(report["confusion_matrix"])

tables = evaluate_prediction_csv_overall_and_per_dataset("outputs/predictions_bert-base-uncased.csv")
print(tables["overall"])
print(tables["per_dataset"])
```

This keeps evaluation independent from training and makes it easy to add more metrics later.

## Output artifacts

- `outputs/predictions_<model>.csv`
	- `text`, `dataset`, `true_label`, `pred_label`, `pred_prob`
- `outputs/split_report_<model>.csv`
	- one row per dataset with train / val / test / total counts
- `outputs/run_<model>.json`
	- run configuration + split sizes + split report path
