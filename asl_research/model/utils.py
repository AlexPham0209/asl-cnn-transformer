import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

def split(x: Tensor, num_heads: int):
    """
    Splits the tensor into num_heads

    Args:
         x (Tensor): Original tensor (batch_size, sequence_size, d_model)

    Returns:
        Tensor: Tensor that is split into n heads 
        (batch_size, num_heads, sequence_size, d_model // num_heads)
    """
        # Shape: (batch_size, sequence_length, d_model)
    N, length, _ = x.shape

    # Reshape into (batch_size, num_heads, sequence_length, d_models // num_heads)
    return x.reshape(N, length, num_heads, -1).transpose(1, 2)
    
def concat(x: Tensor):
    """
    Concatenate the tensor's heads together

    Args:
        x (Tensor): Original tensor (batch_size, num_heads, sequence_size, d_model // num_heads)

    Returns:
        Tensor: Tensor that is split into n heads (batch_size, sequence_size, d_model)
    """

    N, _, length, _ = x.shape

    # Transpose into (batch_size, sequence_length, num_heads, d_model)
    # Then, reshape into (batch_size, sequence_length, d_model)
    return x.transpose(1, 2).reshape(N, length, -1)

def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones((size, size))) == 1
    mask = mask.float().masked_fill(mask == 1, 0).masked_fill(mask == 0, -torch.inf)
    return mask

def generate_subsequent_mask(height, width):
    mask = torch.tril(torch.ones((height, width))) == 1
    mask = mask.float().masked_fill(mask == 1, 0).masked_fill(mask == 0, -torch.inf)
    return mask