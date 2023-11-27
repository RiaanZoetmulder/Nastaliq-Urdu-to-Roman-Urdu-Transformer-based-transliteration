
import torch.nn as nn
import torch


class SimpleFeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, dropout: float = 0.5):
        """
        Simple Feedforward Network. To be used for the implementation of "Attention is all you Need"
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param dropout:
        """
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

