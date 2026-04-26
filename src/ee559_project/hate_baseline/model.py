from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "auto",
        head: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        feature_dim = int(self.backbone.config.hidden_size)

        if head == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_labels),
            )
        elif head == "mlp":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return summed / count

        if self.pooling == "cls":
            return last_hidden[:, 0, :]

        if self.pooling == "auto":
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return last_hidden[:, 0, :]

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        logits = self.classifier(pooled)
        return logits
