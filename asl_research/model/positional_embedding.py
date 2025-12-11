import math
import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[: x.size(dim=1)]
        return self.dropout(x)


# d_model = 1000
# dim = torch.arange(0, d_model, 2)
# a = torch.exp(dim * (-math.log(10000.0) / d_model))
# b = 1 / (10000 ** (dim / d_model))
# c = torch.exp(-math.log(10000.) * (dim / d_model))
# print(a)
# print(b)

# print(torch.allclose(a, b))
