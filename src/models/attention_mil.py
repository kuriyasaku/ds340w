from typing import Dict

import torch
import torch.nn as nn

from src.models.mil_backbone import GatedAttention, MILProjection


class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, attention_dim: int = 256, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.proj = MILProjection(input_dim, hidden_dim, dropout)
        self.attn = GatedAttention(hidden_dim, attention_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.instance_head = nn.Linear(hidden_dim, num_classes)

    def aggregate(self, h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(h * weights.unsqueeze(-1), dim=0)

    def forward(self, bag: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.proj(bag)
        weights, raw_scores = self.attn(h)
        slide_repr = self.aggregate(h, weights)
        logits = self.classifier(slide_repr.unsqueeze(0))
        prob = torch.softmax(logits, dim=-1)
        instance_logits = self.instance_head(h)
        instance_prob = torch.softmax(instance_logits, dim=-1)
        return {
            "logits": logits,
            "prob": prob,
            "attention": weights,
            "attention_raw": raw_scores,
            "instance_logits": instance_logits,
            "instance_prob": instance_prob,
            "slide_repr": slide_repr,
            "embeddings": h,
        }
