import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from asl_research.utils.utils import concat, split


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: float, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        """
        Splits the query, key, and value tensors into a number of heads
        Calculates the attention scores using Scaled Dot Attention
        Then,

        Args:
            q (Tensor): Query tensor (batch_size, target_sequence_size, d_model)
            k (Tensor): Key tensor (batch_size, src_sequence_size, d_model)
            v (Tensor): Value tensor (batch_size, src_sequence_size, d_model)

            mask (Optional[Tensor]): Used to mask out elements in the attention score matrix

        Returns:
            Tensor: The attention matrix (batch_size, target_sequence_size, d_model)
        """

        # Shape: (batch_size, sequence_length, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Split tensor into heads
        # Shape: (batch_size, num_heads, sequence_size, d_model // num_heads)
        q = split(q, self.num_heads)
        k = split(k, self.num_heads)
        v = split(v, self.num_heads)

        # Calculate the attention score which is used to gauge which tokens are important to each token
        # Shape: (batch_size, num_heads, sequence_size, d_model // num_heads)
        out = self.attention(q, k, v, mask)

        # Concatenate heads together
        # Shape: (batch_size, target_sequence_length, d_model)
        out = concat(out)

        # Determines which token/word it should attend to?
        # Shape: (batch_size, target_sequence_length, d_model)
        return self.w_o(out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        """
        Calculates the scaled dot product attention between q, k, v tensors

        Args:
            q (Tensor): Query tensor (batch_size, num_heads, target_sequence_size, d_model // num_heads)
            k (Tensor): Key tensor (batch_size, num_heads, src_sequence_size, d_model // num_heads)
            v (Tensor): Value tensor (batch_size, num_heads, src_sequence_size, d_model // num_heads)

            mask (Optional[Tensor]): Used to mask out elements in the attention score matrix (target_sequence)

        Returns:
            Tensor: Attention matrix (batch_size, target_sequence_size, d_model)
        """

        # QK and Scores' Shape: (batch_size, num_heads, target_sequence_size, src_sequence_size)
        qk = q @ k.transpose(-2, -1)
        scores = qk / math.sqrt(k.shape[-1])

        # Filled all elements that are either padding tokens or in the future with -torch.inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -torch.inf)

        # Calculate a probability distribution with the current token to all other tokens in the sequence
        scores = self.softmax(scores)

        # Shape: (batch_size, num_heads, target_sequence_size, d_v)
        return scores @ v
