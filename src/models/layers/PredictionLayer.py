import torch
import torch.nn as nn


class TransformerPredictionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        # convert features into a logits for the vocabulary
        self.ff_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input) -> torch.Tensor:

        # go from: [batch, sequence length, hidden_dims] -> [batch, sequence_length, vocabulary size]
        return torch.log_softmax(self.ff_layer(input), dim = -1)





