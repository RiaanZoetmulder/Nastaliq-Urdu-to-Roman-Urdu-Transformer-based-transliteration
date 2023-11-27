import torch.nn as nn
from ..layers import LayerNorm
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features, 1e-6)

    def forward(self, x, encoder_output, src_mask, tgt_mask) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
