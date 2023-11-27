import torch.nn as nn
from ..attention import MultiHeadAttention
from ..layers import ResidualConnection, SimpleFeedForward


class EncoderBlock(nn.Module):
    def __init__(self,  features: int, self_attention_block: MultiHeadAttention, ff_block: SimpleFeedForward,  dropout:float =0.1) -> None:
        super().__init__()

        # Multi Head attention block
        self.attn = self_attention_block

        # initialize skip connections
        self.res_one = ResidualConnection(features, dropout)
        self.res_two = ResidualConnection(features, dropout)

        # feed forward block
        self.ffn = ff_block


    def forward(self, x, mask=None):

        # Multi Head attention block
        x_attention = self.attn(x, x, x, mask=mask)
        x = self.res_one(x, x_attention)

        # feed forward block
        x_after_ff = self.ffn(x)
        x = self.res_two(x, x_after_ff)

        return x


