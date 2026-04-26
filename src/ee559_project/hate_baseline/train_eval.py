from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")

    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total else 0.0

    class_f1 = []
    for cls in (0, 1):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        class_f1.append(f1)

    macro_f1 = sum(class_f1) / len(class_f1)
    hateful_f1 = class_f1[1]

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "hateful_f1": hateful_f1,
    }


def train_one_epoch(model, dataloader, optimizer, device: str) -> float:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=-1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


@torch.no_grad()
def predict(model, dataloader, device: str) -> tuple[list[int], list[int], list[float]]:
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    return y_true, y_pred, y_prob
