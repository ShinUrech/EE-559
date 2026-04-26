from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def _binary_class_metrics(y_true: list[int], y_pred: list[int], positive_label: int) -> dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp == positive_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp == positive_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp != positive_label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(1 for yt in y_true if yt == positive_label),
    }


def confusion_matrix_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    """Return the binary confusion matrix as a simple dictionary."""

    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _evaluate_prediction_frame_core(frame: pd.DataFrame) -> dict[str, object]:
    """Compute metrics for a single prediction frame without nested grouping."""

    required = {"true_label", "pred_label"}
    if not required.issubset(frame.columns):
        missing = required.difference(frame.columns)
        raise ValueError(f"Missing columns {missing} in prediction frame")

    y_true = frame["true_label"].astype(int).tolist()
    y_pred = frame["pred_label"].astype(int).tolist()

    total = len(y_true)
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total if total else 0.0

    non_hate = _binary_class_metrics(y_true, y_pred, positive_label=0)
    hate = _binary_class_metrics(y_true, y_pred, positive_label=1)
    macro_f1 = (non_hate["f1"] + hate["f1"]) / 2.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "hateful_f1": hate["f1"],
        "non_hate_f1": non_hate["f1"],
        "non_hate_precision": non_hate["precision"],
        "non_hate_recall": non_hate["recall"],
        "hateful_precision": hate["precision"],
        "hateful_recall": hate["recall"],
        "confusion_matrix": confusion_matrix_from_predictions(y_true, y_pred),
    }


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate a prediction frame with true labels and model predictions.

    Parameters
    ----------
    frame:
        DataFrame containing at least ``true_label`` and ``pred_label``.
        If ``dataset`` is present, per-dataset reports are included too.
    """

    report = _evaluate_prediction_frame_core(frame)

    if "dataset" in frame.columns:
        per_dataset = {}
        for dataset_name, group in frame.groupby("dataset"):
            dataset_report = _evaluate_prediction_frame_core(group)
            per_dataset[str(dataset_name)] = dataset_report
        report["per_dataset"] = per_dataset

    return report


def evaluate_prediction_csv(csv_path: str | Path) -> dict[str, object]:
    """Load a saved prediction CSV and evaluate it.

    Parameters
    ----------
    csv_path:
        Path to a CSV produced by ``run_experiment``.
    """

    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    report = evaluate_prediction_frame(frame)
    report["csv_path"] = str(csv_path)
    report["sample_count"] = len(frame)
    return report


def evaluate_prediction_csv_overall_and_per_dataset(csv_path: str | Path) -> dict[str, pd.DataFrame]:
    """Return overall and per-dataset metric tables for a saved prediction CSV."""

    report = cast(dict[str, object], evaluate_prediction_csv(csv_path))
    confusion_matrix = cast(dict[str, int], report["confusion_matrix"])
    overall = pd.DataFrame(
        [
            {
                "sample_count": report["sample_count"],
                "accuracy": report["accuracy"],
                "macro_f1": report["macro_f1"],
                "hateful_f1": report["hateful_f1"],
                "non_hate_f1": report["non_hate_f1"],
                "non_hate_precision": report["non_hate_precision"],
                "non_hate_recall": report["non_hate_recall"],
                "hateful_precision": report["hateful_precision"],
                "hateful_recall": report["hateful_recall"],
                "tn": confusion_matrix["tn"],
                "fp": confusion_matrix["fp"],
                "fn": confusion_matrix["fn"],
                "tp": confusion_matrix["tp"],
            }
        ]
    )

    per_dataset_rows = []
    per_dataset_reports = cast(dict[str, dict[str, object]], report.get("per_dataset", {}))
    for dataset_name, dataset_report in per_dataset_reports.items():
        dataset_confusion_matrix = cast(dict[str, int], dataset_report["confusion_matrix"])
        per_dataset_rows.append(
            {
                "dataset": dataset_name,
                "sample_count": dataset_confusion_matrix["tn"]
                + dataset_confusion_matrix["fp"]
                + dataset_confusion_matrix["fn"]
                + dataset_confusion_matrix["tp"],
                "accuracy": dataset_report["accuracy"],
                "macro_f1": dataset_report["macro_f1"],
                "hateful_f1": dataset_report["hateful_f1"],
                "non_hate_f1": dataset_report["non_hate_f1"],
                "non_hate_precision": dataset_report["non_hate_precision"],
                "non_hate_recall": dataset_report["non_hate_recall"],
                "hateful_precision": dataset_report["hateful_precision"],
                "hateful_recall": dataset_report["hateful_recall"],
                "tn": dataset_confusion_matrix["tn"],
                "fp": dataset_confusion_matrix["fp"],
                "fn": dataset_confusion_matrix["fn"],
                "tp": dataset_confusion_matrix["tp"],
            }
        )

    per_dataset = pd.DataFrame(per_dataset_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return {"overall": overall, "per_dataset": per_dataset}
