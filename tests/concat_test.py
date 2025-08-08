import pytest
import torch 
from asl_research.utils.utils import concat

def concat_2(tensor):
    """
    inverse function of self.split(tensor : torch.Tensor)

    :param tensor: [batch_size, head, length, d_tensor]
    :return: [batch_size, length, d_model]
    """
    batch_size, head, length, d_tensor = tensor.size()
    d_model = head * d_tensor

    tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
    return tensor

def test_split_shape():
    tensor = torch.arange(0, 5 * 8 * 6 * 64).reshape(5, 8, 6, 64)
    tensor = concat(tensor)

    assert tensor.shape == (5, 6, 512)

def test_split_equal():
    tensor = torch.arange(0, 5 * 8 * 6 * 64).reshape(5, 8, 6, 64)
    a = concat(tensor)
    b = concat_2(tensor)

    assert torch.equal(a, b)