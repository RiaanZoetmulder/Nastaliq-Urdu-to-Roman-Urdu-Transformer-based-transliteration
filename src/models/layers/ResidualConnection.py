import torch
import torch.nn as nn
from .LayerNorm import LayerNorm


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float ):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(features, 1e-6)

    def forward(self, x: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:

        return x + self.dropout(self.norm1(transformed_x))




