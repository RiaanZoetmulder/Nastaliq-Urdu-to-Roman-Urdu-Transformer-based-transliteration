import torch.nn as nn
from ..blocks import EncoderBlock
from ..embeddings import InputEmbedding, PositionalEncoding
from ..layers import LayerNorm


class Encoder(nn.Module):
    def __init__(self, features: int,  layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()

        self.layers = layers

        self.layer_norm = LayerNorm(features, 1e-6)


    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)

