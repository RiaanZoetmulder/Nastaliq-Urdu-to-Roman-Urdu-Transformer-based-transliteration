import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_len: int)-> None:

        super(InputEmbedding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(self.seq_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)




