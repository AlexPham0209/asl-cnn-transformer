import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(in_features=d_model, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(in_features=hidden_size, out_features=d_model)

    def forward(self, x: Tensor):
        """
        Feeds input tensor into the first linear layer to get a tensor of shape (batch_size, sequence_size, hidden_size)
        Apply ReLU activation
        Then, feed it into a second linear layer to retrieve a final output tensor of shape (batch_size, sequence_size, d_model)

        Args:
            x (Tensor): Input tensor (batch_size, sequence_size, d_model)

        Returns:
            (Tensor): Output tensor (batch_size, sequence_size, d_model)

        """
        x = self.w1(x)
        x = self.relu(x)
        return self.w2(x)
