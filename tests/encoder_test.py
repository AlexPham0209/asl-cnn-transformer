import pytest
import torch
from asl_research.model.encoder import TransformerEncoder
from asl_research.utils.utils import generate_square_subsequent_mask

def test_encoder_shape():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, sequence_length, d_model = 32, 10, 512
    encoder = TransformerEncoder(num_layers=2, d_model=d_model).to(DEVICE)
    x = torch.rand(batch_size, sequence_length, d_model).to(DEVICE)

    out = encoder(x).to(DEVICE)
    assert x.shape == (batch_size, sequence_length, d_model)

