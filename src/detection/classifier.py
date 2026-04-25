from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

LABEL_NAMES = ["Low", "Medium", "High"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABEL_NAMES)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABEL_NAMES)}


class CognitiveLoadModel(nn.Module):
    """DeBERTa-v3-base with two output heads:
    - regression_head: outputs load score (0–100), single float
    - classification_head: outputs 3-class logits (Low/Medium/High)

    Input: tokenized text + optionally concatenated feature vector (num_features-dim projected to 64-dim)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_features: int = 20,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden: int = self.encoder.config.hidden_size  # 768 for deberta-v3-base

        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        combined = hidden + 64

        self.regression_head = nn.Sequential(
            nn.Linear(combined, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # outputs 0–1; multiply by 100 for score
        )

        self.classification_head = nn.Sequential(
            nn.Linear(combined, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            score: Tensor of shape (batch,) with load scores 0–100
            logits: Tensor of shape (batch, 3) with class logits
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        if feature_vector is not None:
            feat_proj = self.feature_proj(feature_vector)  # (batch, 64)
            combined = torch.cat([cls_repr, feat_proj], dim=-1)  # (batch, hidden+64)
        else:
            # Pad features with zeros if not provided
            batch_size = cls_repr.size(0)
            zero_feat = torch.zeros(batch_size, 64, device=cls_repr.device, dtype=cls_repr.dtype)
            combined = torch.cat([cls_repr, zero_feat], dim=-1)

        score = self.regression_head(combined).squeeze(-1) * 100.0  # (batch,)
        logits = self.classification_head(combined)  # (batch, 3)
        return score, logits


def compute_loss(
    score_pred: torch.Tensor,
    score_true: torch.Tensor,
    logits: torch.Tensor,
    label_true: torch.Tensor,
) -> torch.Tensor:
    """Combined regression + classification loss.

    Args:
        score_pred: predicted scores (batch,) in range 0–100
        score_true: ground-truth scores (batch,) in range 0–100
        logits: class logits (batch, 3)
        label_true: integer class labels (batch,)
    """
    # Normalise to 0-1 for MSE stability
    mse = nn.MSELoss()(score_pred / 100.0, score_true / 100.0)
    ce = nn.CrossEntropyLoss()(logits, label_true)
    return 0.6 * mse + 0.4 * ce


def load_tokenizer(model_name: str = "microsoft/deberta-v3-base") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)
