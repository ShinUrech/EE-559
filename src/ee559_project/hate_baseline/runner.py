from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .data import filter_datasets, load_merged_dataframe, make_dataloader, split_dataset_report, stratified_split
from .model import TransformerClassifier
from .train_eval import evaluate, predict, set_seed, train_one_epoch


def _limit_frame(frame, limit: int | None, seed: int):
    if limit is None or limit <= 0 or len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def run_experiment(
    model_name: str,
    datasets_dir: str | Path = "datasets",
    output_dir: str | Path = "outputs",
    config: ExperimentConfig | None = None,
) -> dict[str, object]:
    config = config or ExperimentConfig()
    set_seed(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_merged_dataframe(datasets_dir)
    frame = filter_datasets(frame, config.datasets)
    splits = stratified_split(
        frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    splits = {
        "train": _limit_frame(splits["train"], config.max_train_samples, config.seed),
        "val": _limit_frame(splits["val"], config.max_val_samples, config.seed + 1),
        "test": _limit_frame(splits["test"], config.max_test_samples, config.seed + 2),
    }

    split_report = split_dataset_report(splits)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = make_dataloader(
        splits["train"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = make_dataloader(
        splits["val"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = make_dataloader(
        splits["test"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = TransformerClassifier(
        model_name=model_name,
        pooling=config.pooling,
        head=config.head,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_macro_f1 = -1.0

    for _ in range(config.epochs):
        train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics = evaluate(model, val_loader, device=device)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device=device)
    y_true, y_pred, y_prob = predict(model, test_loader, device=device)

    prediction_frame = splits["test"][["text", "dataset"]].copy().reset_index(drop=True)
    prediction_frame["true_label"] = y_true
    prediction_frame["pred_label"] = y_pred
    prediction_frame["pred_prob"] = y_prob

    safe_name = model_name.replace("/", "_")
    pred_path = output_dir / f"predictions_{safe_name}.csv"
    prediction_frame.to_csv(pred_path, index=False)

    split_report_path = output_dir / f"split_report_{safe_name}.csv"
    split_report.to_csv(split_report_path, index=False)

    summary = {
        "model_name": model_name,
        "pooling": config.pooling,
        "head": config.head,
        "seed": config.seed,
        "datasets": config.datasets,
        "prediction_path": str(pred_path),
        "split_report_path": str(split_report_path),
        "split_sizes": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "split_dataset_report": split_report.to_dict(orient="records"),
    }

    metadata_path = output_dir / f"run_{safe_name}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
