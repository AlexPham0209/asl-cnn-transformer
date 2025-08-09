import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        """
        Create a positional encoding matrix.

        Args:
            max_len: The maximum length a sequence of tokens can be
            d_model: The dimensions of the encoding vector
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        dim = torch.arange(0, d_model, 2)

        # For even position indices, we utilize the sine function
        # For odd indices, we utilize the cosine function
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / d_model)))
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Adds the positional encoding to the embedding matrix
        Helps describes the position of the embedding vector within a sequence.

        Args:
            x (Tensor): Embedding matrix (Batch, Sequence Size, Embedding Size)

        Returns:
            Tensor: Embedding matrix with positions encoded into them
        """

        res = x + self.pe[: x.size(dim=1), :]
        return self.dropout(res)
