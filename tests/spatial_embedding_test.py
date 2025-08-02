import pytest
import torch
from asl_research.model import Spatial3DEmbedding

def test_spatial_embedding_shape_1():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 120, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert torch.equal(torch.tensor([x.shape[0], x.shape[-1]]), torch.tensor([batch_size, 512]))

def test_spatial_embedding_shape_2():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 170, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert torch.equal(torch.tensor([x.shape[0], x.shape[-1]]), torch.tensor([batch_size, 512]))
    
def test_spatial_embedding_shape_3():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 16, 3, 24, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert torch.equal(torch.tensor([x.shape[0], x.shape[-1]]), torch.tensor([batch_size, 512]))

def test_spatial_embedding_shape_4():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 16, 3, 120, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)
    
    assert torch.equal(torch.tensor([x.shape[0], x.shape[-1]]), torch.tensor([batch_size, 512]))