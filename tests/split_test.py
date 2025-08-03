import pytest
import torch 
from asl_research.model.utils import split

def split_2(x, num_heads):
    # Shape: (Batch Size, Sequence Length, d_model)
    N, length, d_model = x.shape

    l = d_model // num_heads
    # Reshape into length: (Batch Size, Sequence Length, d_models / num_heads, )
    return x.reshape(N, length, num_heads, l).transpose(1, 2)

def test_split_shape():
    tensor = torch.arange(0, 5 * 6 * 32).reshape(5, 6, 32)
    tensor = split(tensor, num_heads=8)

    assert tensor.shape == (5, 8, 6, 4)

def test_split_equal():
    tensor = torch.arange(0, 5 * 6 * 32).reshape(5, 6, 32)
    a = split(tensor, num_heads=8)
    b = split_2(tensor, num_heads=8)

    assert torch.equal(a, b)