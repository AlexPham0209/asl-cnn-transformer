import pytest
import torch
from asl_research.model.spatial_embedding import Spatial3DEmbedding, Spatial2DEmbedding, Conv2DBlock

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

def test_conv2d_block():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 16, 3, 120, 224, 224
    embedding = Conv2DBlock(channels, 16).to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    print(x.shape)
    
    assert torch.equal(torch.tensor([x.shape[0], x.shape[2]]), torch.tensor([batch_size, depth]))

def test_spatial_2d_embedding():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 24, 224, 224
    embedding = Spatial2DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    print(x.shape)
    
    assert torch.equal(torch.tensor([x.shape[0], x.shape[1]]), torch.tensor([batch_size, depth]))
