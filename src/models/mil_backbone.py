import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.v = nn.Linear(input_dim, attention_dim)
        self.u = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor):
        a = torch.tanh(self.v(x))
        b = torch.sigmoid(self.u(x))
        scores = self.w(a * b)
        weights = torch.softmax(scores.squeeze(-1), dim=0)
        return weights, scores.squeeze(-1)


class MILProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
