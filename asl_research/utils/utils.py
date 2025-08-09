from typing import Optional

import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_square_subsequent_mask(x: Tensor, pad_token: int):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token or is in the future
    as False.

    Args:
        x (Tensor): Original tensor (batch_size, sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, sequence_size, sequence_size)
    """

    N, sequence_length = x.shape
    causal_mask = (
        torch.tril(torch.ones((N, 1, sequence_length, sequence_length))).bool().to(DEVICE)
    )
    padding_mask = generate_padding_mask(x, pad_token).bool().to(DEVICE)

    mask = causal_mask & padding_mask
    return mask


def generate_padding_mask(x: Tensor, pad_token: int):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token as False.

    Args:
        x (Tensor): Original tensor (batch_size, sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, 1, sequence_size)
    """

    N, sequence_length = x.shape
    return (x != pad_token).unsqueeze(1).unsqueeze(2).bool().to(DEVICE)


def generate_video_padding_mask(lengths: Tensor, max_length: Optional[int] = None):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token as False.

    Args:
        x (Tensor): Original tensor (sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, 1, sequence_size)
    """
    max_length = torch.max(lengths, dim=-1)[0].item() if not max_length else max_length

    lengths = lengths.unsqueeze(0).transpose(0, 1)
    indices = torch.arange(0, max_length).unsqueeze(0)

    out = indices <= lengths - 1
    return out.unsqueeze(1).unsqueeze(2).bool().to(DEVICE)


def pad_video_with_value(x: Tensor, length: int = 10, padding: float = 0):
    T, C, H, W = x.shape
    out = torch.zeros(length, C, H, W)
    out[:T] = x
    return out


def pad_video_with_last_frame(x: Tensor, length: int = 10):
    T = x.shape[0]
    out = x[-1].repeat(length, 1, 1, 1)
    out[:T] = x
    return out


# T, C, H, W = 5, 3, 4, 4
# tensor = torch.arange(0, T * C * H * W).reshape(T, C, H, W)
# print(pad_video_with_last_frame(tensor, 10))
