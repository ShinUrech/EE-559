# Model Architecture — Hate Speech Detection with MoE

## Overview

The model is a **BERT encoder + classification head** pipeline. Three head variants are supported: `linear`, `mlp`, and `moe` (Mixture-of-Experts). The MoE head is the main contribution of this work.

---

## Full Architecture Diagram

```
Input text (raw string)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Tokenizer  (bert-base-uncased, max_length=128)     │
│  → input_ids [B, L]  +  attention_mask [B, L]       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            BERT Encoder  (bert-base-uncased)         │
│  12 transformer layers, hidden_size = 768            │
│  → last_hidden_state  [B, L, 768]                    │
│  → pooler_output      [B, 768]  (CLS token, linear) │
└───────────────────────┬─────────────────────────────┘
                        │
                   Pooling (auto)
                  uses pooler_output
                        │
                        ▼
              pooled  [B, 768]
                        │
          ┌─────────────┴──────────────┐
          │   Classification Head      │
          │   (one of three variants)  │
          └──────────────┬─────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    [linear]          [mlp]          [moe]  ← used in experiments
    Dropout         Dropout       see below
    Linear(768,2)   Linear(768,256)
                    ReLU
                    Dropout
                    Linear(256,2)
                         │
                         ▼
                  logits  [B, 2]
                         │
                         ▼
               CrossEntropyLoss
               (+ lb_loss for MoE)
```

---

## MoE Head — Detailed Diagram

```
pooled  [B, 768]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌──────────────┐                    ┌──────────────────────┐
│   Router     │                    │  8 Expert FFNs (×8)  │
│  Linear      │                    │                      │
│  (768 → 8)   │                    │  Linear(768 → 1536)  │
│  (no bias)   │                    │  GELU                │
│  Softmax     │                    │  Dropout             │
│              │                    │  Linear(1536 → 768)  │
│  Top-2       │                    │                      │
│  selection   │                    └──────────┬───────────┘
└──────┬───────┘                               │
       │                                       │
  weights [B, 8]                    expert_out [B, 8, 768]
  (sparse: only                               │
   top-2 non-zero)                            │
       │                                       │
       └──────────────┬────────────────────────┘
                      │
                      ▼
          weighted sum over experts
          (weights.unsqueeze(-1) * expert_out).sum(dim=1)
                      │
                  moe_out [B, 768]
                      │
                      ▼
             Residual:  x + moe_out
                      │
                  LayerNorm [B, 768]
                      │
                  Dropout
                      │
               Linear(768 → 2)
                      │
                 logits [B, 2]
                      │
               ┌──────┴──────┐
               ▼             ▼
          task loss      lb_loss
        CrossEntropy   (auxiliary,
                      added during
                       training)
```

### Router Detail

The router selects **top-2 out of 8 experts** per sample:

```
gate_logits = x @ W_gate.T           [B, 8]
probs       = softmax(gate_logits)   [B, 8]  — all 8 probabilities
weights     = sparse: keep top-2, zero out the rest
weights     = renormalize (sum to 1)
```

### Load-Balancing Loss (Switch Transformer)

To prevent all samples routing to the same expert ("expert collapse"):

$$\mathcal{L}_{lb} = \lambda \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $N = 8$ = number of experts  
- $f_i$ = fraction of samples routed to expert $i$ in the batch  
- $P_i$ = mean router probability assigned to expert $i$  
- $\lambda = 0.01$ (`lb_coeff`)

Total loss: $\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{lb}$

---

## Class Imbalance Handling

Hate speech datasets are heavily skewed (~10% hate, ~90% non-hate). Two strategies are implemented:

### WeightedRandomSampler (oversampling)

When `oversample_minority=True`, each training sample is assigned a weight inversely proportional to its class frequency:

```
weight[i] = 1 / count(class of sample i)
```

PyTorch's `WeightedRandomSampler` then samples with replacement so that both classes are seen roughly equally per epoch.

### Class weights in loss (not used here)

An alternative approach (not implemented) would be to pass `weight` to `CrossEntropyLoss`. Oversampling was chosen as it interacts better with the load-balancing objective.

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Model | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 16 |
| Epochs | 5 |
| Max sequence length | 128 |
| Pooling | `auto` (BERT pooler_output) |
| MoE experts | 8 |
| MoE top-k | 2 |
| lb_coeff | 0.01 |

Best model (by val macro-F1) is checkpointed and used for final test evaluation.

---

## File Map

```
src/ee559_project/hate_baseline/
├── config.py       — ExperimentConfig dataclass (all hyperparameters)
├── model.py        — _Expert, _TopKRouter, MoEHead, TransformerClassifier
├── data.py         — dataset loading, splitting, WeightedRandomSampler
├── train_eval.py   — training loop (with lb_loss), evaluate(), predict()
├── runner.py       — run_experiment() end-to-end function
└── evaluation.py   — metric computation (teammate's code, unchanged)

scripts/
├── preprocess_white_supremacy.py    — raw annotations → text/label CSV
└── preprocess_wikipedia_comments.py — raw train/test CSVs → text/label CSV
```

---

## Key Design Choices

1. **Residual connection in MoE head** — `x + moe_out` ensures the head degrades gracefully to identity if experts output zero, stabilising early training.

2. **Expert hidden_dim = 2× input_dim** — each expert expands from 768 → 1536 → 768, following the FFN ratio common in transformers.

3. **Soft routing, sparse weights** — top-2 are kept and renormalised rather than using a hard one-hot. This gives gradient flow to the router from both selected experts.

4. **lb_loss stored on model** — `self._last_lb_loss` is set during `forward()` and read by `train_one_epoch()`. This avoids changing the `forward()` signature used by the rest of the codebase.
