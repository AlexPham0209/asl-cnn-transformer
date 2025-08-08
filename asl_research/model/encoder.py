import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from asl_research.model.positional_embedding import PositionalEncoding
from asl_research.model.attention import MultiHeadAttention
from asl_research.model.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int = 512, num_heads: int = 8, hidden_size: int = 1024, dropout: float = 0.1
    ):
        super(EncoderLayer, self).__init__()

        # Self Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)

        # Position-Wise Feed Forward
        self.ff = PositionWiseFeedForward(d_model, hidden_size, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self Attention
        # Shape: (batch_size, sequence_size, d_model)
        x = x + self.attention(q=x, k=x, v=x, mask=mask)
        x = self.layer_norm_1(x)
        x = self.dropout_1(x)

        # Position-Wise Feed Forward
        # Shape: (batch_size, sequence_size, d_model)
        x = x + self.ff(x)
        x = self.layer_norm_2(x)
        x = self.dropout_2(x)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        hidden_size: int = 512,
        dropout: float = 0.1,
    ):
        super(TransformerEncoder, self).__init__()

        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """
        Feeds input tensor through multiple layers of encoders which encodes the
        input vector into a fixed representation that has the context of the other tokens in the sequence.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
            mask (Optional[Tensor]): A tensor used to mask certain values such as padding tokens during attention (batch_size, 1, sequence_length, sequence_length)

        Returns:
            Tensor: Encoded tensor of shape (batch_size, sequence_length, d_model)
        """

        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
