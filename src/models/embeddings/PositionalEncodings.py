import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float =0.5) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.seq_len = seq_len

        # calculate the positional encodings divisor term
        encodings = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        # Apply evenly and unevenly
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)

        # ensure batch dimension
        encodings = encodings.unsqueeze(0 )

        self.register_buffer('positional_encodings', encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.dropout(x + self.positional_encodings[:, :x.shape[1], :].requires_grad_(False))

