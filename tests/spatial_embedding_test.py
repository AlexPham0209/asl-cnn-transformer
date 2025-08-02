import pytest
import torch
from asl_research.model import SpatialEmbedding

def test_spatial_embedding_shape():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, depth, height, width = 1, 3, 120, 224, 224
    embedding = SpatialEmbedding().to(DEVICE)
    x = torch.rand(batch_size, channels, depth, height, width).to(DEVICE)
    x = embedding(x)

    assert torch.equal(torch.tensor(x.shape), torch.tensor([batch_size, 9, 512]))
    