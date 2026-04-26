from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for the transformer hate-speech baseline.

    Tune these fields most often:
    - epochs: number of training passes over the train split.
    - batch_size: samples per optimizer step.
    - max_length: tokenizer truncation length for the text field.
    - learning_rate: optimizer step size.
    - pooling: how token embeddings become a single vector.
    - head: classifier stack on top of pooled embeddings.
    - device: "cpu", "cuda", or "auto".
    - datasets: optional list of dataset names to keep from datasets/.
    - max_train_samples / max_val_samples / max_test_samples: optional
      caps for quick CPU smoke tests.
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    pooling: str = "auto"  # auto, cls, mean
    head: str = "linear"  # linear, mlp
    hidden_dim: int = 256
    dropout: float = 0.1

    device: str = "cpu"
    datasets: list[str] | None = None

    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
