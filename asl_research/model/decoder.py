import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from asl_research.model.positional_embedding import PositionalEncoding
from asl_research.model.attention import MultiHeadAttention
from asl_research.model.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 8, hidden_size: int = 512, dropout: float = 0.1
    ):
        super(DecoderLayer, self).__init__()

        # Self Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)

        # Cross Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

        # Position-Wise Feed Forward
        self.ff = PositionWiseFeedForward(d_model, hidden_size)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x: Tensor, encoded: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        # Masked Self Attention
        # Shape: (batch_size, target_sequence_size, d_model)
        x = x + self.self_attention(q=x, k=x, v=x, mask=mask)
        x = self.layer_norm_1(x)
        x = self.dropout_1(x)

        if encoded is not None:
            # Cross Attention
            # Shape: (batch_size, target_sequence_size, d_model)
            x = x + self.cross_attention(q=x, k=encoded, v=encoded)
            x = self.layer_norm_2(x)
            x = self.dropout_2(x)

        # Position-Wise Feed Forward
        # Shape: (batch_size, target_sequence_size, d_model)
        x = x + self.ff(x)
        x = self.layer_norm_3(x)
        x = self.dropout_3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        hidden_size: int = 512,
        dropout: float = 0.1,
    ):
        super(TransformerDecoder, self).__init__()

        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, encoded: Tensor, mask: Optional[Tensor] = None):
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, encoded, mask)

        return x
