import torch.nn as nn
from ..attention import MultiHeadAttention
from ..layers import ResidualConnection, SimpleFeedForward

class DecoderBlock(nn.Module):
    def __init__(self,  features: int, self_attention_block, cross_attention_block, feed_forward_block, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = self_attention_block
        self.mha2 = cross_attention_block
        self.ffn = feed_forward_block

        self.residual_one = ResidualConnection(features, dropout)
        self.residual_two = ResidualConnection(features, dropout)
        self.residual_three = ResidualConnection(features, dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):

        x_att_one = self.mha1(x, x, x, tgt_mask)
        x = self.residual_one(x, x_att_one)

        x_att_two = self.mha2(x, enc_output, enc_output, src_mask)
        x = self.residual_two(x, x_att_two)

        x_ff = self.ffn(x)
        x = self.residual_three(x, x_ff)

        return x
