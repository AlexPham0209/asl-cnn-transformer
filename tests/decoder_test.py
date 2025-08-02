import pytest
import torch
from asl_research.model import TransformerDecoder

def test_decoder_shape():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, src_sequence_length, trg_sequence_length, d_model = 32, 10, 12, 512
    decoder = TransformerDecoder(num_layers=2, d_model=d_model).to(DEVICE)

    x = torch.rand(batch_size, trg_sequence_length, d_model).to(DEVICE)
    encoded = torch.rand(batch_size, src_sequence_length, d_model).to(DEVICE)
    
    out = decoder(x, encoded).to(DEVICE)
    assert torch.equal(torch.tensor(x.shape), torch.tensor([batch_size, trg_sequence_length, d_model]))

