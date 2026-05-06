from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# MoE components
# ---------------------------------------------------------------------------

class _Expert(nn.Module):
    """Single feed-forward expert (2-layer MLP with GELU)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TopKRouter(nn.Module):
    """Soft Top-K gating network."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)                          # (B, E)
        probs = F.softmax(logits, dim=-1)              # (B, E)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return weights, logits


class MoEHead(nn.Module):
    """
    Mixture-of-Experts classification head.

    Replaces the standard linear/MLP head with N parallel expert FFNs selected
    per-sample via a Top-K soft router.  A load-balancing auxiliary loss
    (Switch Transformer, Fedus et al. 2022) is stored in ``_last_lb_loss``
    after each forward pass and must be added to the task loss during training.

    Architecture:  x → Router + Experts (residual) → LayerNorm → Dropout → Linear
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        lb_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.lb_coeff = lb_coeff

        self.router = _TopKRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            _Expert(input_dim, input_dim * 2, dropout) for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(input_dim, num_classes)

    def _load_balance_loss(self, weights: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """Switch Transformer auxiliary loss: num_experts * sum(f_i * P_i)."""
        f = (weights > 0).float().mean(dim=0)           # fraction routed per expert
        P = F.softmax(router_logits, dim=-1).mean(dim=0)  # mean router prob per expert
        return self.lb_coeff * self.num_experts * (f * P).sum()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, router_logits = self.router(x)                          # (B,E)
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)   # (B,E,D)
        moe_out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)        # (B,D)
        out = self.norm(x + moe_out)  # residual connection
        out = self.drop(out)
        logits = self.out(out)
        lb_loss = self._load_balance_loss(weights, router_logits)
        return logits, lb_loss


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "auto",
        head: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_labels: int = 2,
        # MoE-specific (ignored when head != "moe")
        num_experts: int = 8,
        top_k: int = 2,
        lb_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.head_type = head

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
        elif head == "moe":
            self.classifier = MoEHead(
                input_dim=feature_dim,
                num_classes=num_labels,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                lb_coeff=lb_coeff,
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

        # Stores the MoE auxiliary loss after each forward pass (None for non-MoE).
        # train_one_epoch adds this to the task loss automatically.
        self._last_lb_loss: torch.Tensor | None = None

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

        if self.head_type == "moe":
            logits, lb_loss = self.classifier(pooled)
            self._last_lb_loss = lb_loss
        else:
            logits = self.classifier(pooled)
            self._last_lb_loss = None

        return logits
