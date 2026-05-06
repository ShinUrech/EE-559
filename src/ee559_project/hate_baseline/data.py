from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class HateSpeechDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def load_merged_dataframe(datasets_dir: str | Path) -> pd.DataFrame:
    datasets_dir = Path(datasets_dir)
    frames = []

    for csv_file in sorted(datasets_dir.glob("*.csv")):
        frame = pd.read_csv(csv_file)
        required = {"text", "label", "dataset"}
        if not required.issubset(frame.columns):
            missing = required.difference(set(frame.columns))
            raise ValueError(f"Missing columns {missing} in {csv_file.name}")

        normalized = frame[["text", "label", "dataset"]].copy()
        normalized["text"] = normalized["text"].astype(str)
        normalized["label"] = normalized["label"].astype(int)
        normalized["dataset"] = normalized["dataset"].astype(str)
        frames.append(normalized)

    if not frames:
        raise ValueError(f"No CSV files found in {datasets_dir}")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return merged


def filter_datasets(frame: pd.DataFrame, dataset_names: list[str] | None) -> pd.DataFrame:
    if not dataset_names:
        return frame.reset_index(drop=True)

    selected = {str(name) for name in dataset_names}
    filtered = frame[frame["dataset"].isin(selected)].copy().reset_index(drop=True)
    if filtered.empty:
        available = sorted(frame["dataset"].astype(str).unique().tolist())
        raise ValueError(f"No rows matched datasets {sorted(selected)}. Available datasets: {available}")
    return filtered


def stratified_split(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for label_value, group in frame.groupby("label"):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n_test < 0:
            raise ValueError(f"Invalid split sizing for label {label_value}")

        train_parts.append(frame.loc[idx[:n_train]])
        val_parts.append(frame.loc[idx[n_train : n_train + n_val]])
        test_parts.append(frame.loc[idx[n_train + n_val :]])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def split_dataset_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in splits.items():
        if "dataset" not in split_frame.columns:
            raise ValueError(f"Missing dataset column in {split_name} split")

        counts = split_frame["dataset"].astype(str).value_counts().sort_index()
        for dataset_name, count in counts.items():
            rows.append({"dataset": dataset_name, "split": split_name, "count": int(count)})

    if not rows:
        return pd.DataFrame(columns=["dataset", "train", "val", "test", "total"])

    report = pd.DataFrame(rows).pivot(index="dataset", columns="split", values="count").fillna(0).astype(int)
    for split_name in ("train", "val", "test"):
        if split_name not in report.columns:
            report[split_name] = 0

    report = report[["train", "val", "test"]]
    report["total"] = report.sum(axis=1)
    return report.reset_index().sort_values("dataset").reset_index(drop=True)


def make_weighted_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    """Returns a WeightedRandomSampler that up-samples the minority class.

    Each sample gets weight = 1 / count(its class), so minority samples are
    drawn more often and majority samples less often, giving an approximately
    balanced effective training distribution without discarding any data.
    """
    labels = frame["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels)
    # Guard against a class with zero samples
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = torch.tensor(1.0 / class_counts[labels], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloader(
    frame: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    oversample_minority: bool = False,
) -> DataLoader:
    """Build a DataLoader for a split.

    Args:
        oversample_minority: when True (recommended for training on imbalanced
            datasets), uses a WeightedRandomSampler so every class is seen
            roughly equally often per epoch.  Mutually exclusive with shuffle;
            the sampler controls ordering.  Has no effect on val/test loaders
            (pass False there).
    """
    dataset = HateSpeechDataset(frame, tokenizer=tokenizer, max_length=max_length)
    if oversample_minority:
        sampler = make_weighted_sampler(frame)
        # shuffle must be False when a sampler is provided
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)