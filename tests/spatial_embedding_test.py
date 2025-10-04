import pytest
import torch
from asl_research.model.spatial_embedding import Spatial3DEmbedding, SpatialEmbedding, Conv2DBlock

def test_spatial_embedding_shape_1():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 120, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert (x.shape[0], x.shape[-1]) == (batch_size, 512)

def test_spatial_embedding_shape_2():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 170, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert (x.shape[0], x.shape[-1]) == (batch_size, 512)
    
def test_spatial_embedding_shape_3():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 16, 3, 24, 224, 224
    embedding = Spatial3DEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert (x.shape[0], x.shape[-1]) == (batch_size, 512)


def test_conv2d_block():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, time, channels, height, width = 3, 120, 3, 224, 224
    embedding = Conv2DBlock(channels, 16).to(DEVICE)
    x = torch.rand(batch_size, channels, time, height, width).to(DEVICE)
    x = embedding(x)

    assert (x.shape[0], x.shape[1]) == (batch_size, time)

def test_spatial_2d_embedding():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, time, channels, height, width = 8, 60, 3, 224, 224
    embedding = SpatialEmbedding().to(DEVICE)
    x = torch.rand(batch_size, time, channels, height, width).to(DEVICE)
    x = embedding(x)

    assert x.shape == (batch_size, time, 512)