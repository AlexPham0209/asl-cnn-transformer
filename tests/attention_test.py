import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import MultiheadAttention
from asl_research.transformer import TransformerEncoder, generate_square_subsequent_mask, generate_subsequent_mask, ScaledDotProductAttention, MultiHeadAttention

def test_self_attention_equal():
    batch_size, n_head, sequence_size, d_model = 16, 8, 10, 64
    q = torch.rand(batch_size, n_head, sequence_size, d_model)
    k = torch.rand(batch_size, n_head, sequence_size, d_model)
    v = torch.rand(batch_size, n_head, sequence_size, d_model)

    mask = generate_square_subsequent_mask(sequence_size)
    attention = ScaledDotProductAttention()
    
    a = scaled_dot_product_attention(q, k, v, mask)
    b = attention(q, k, v, mask)
    
    assert torch.allclose(a, b)

def test_self_attention_shape():
    batch_size, n_head = 16, 8
    trg_sequence_size, src_sequence_size = 10, 12
    d_v, d_model = 128, 512

    q = torch.rand(batch_size, n_head, trg_sequence_size, d_model)
    k = torch.rand(batch_size, n_head, src_sequence_size, d_model)
    v = torch.rand(batch_size, n_head, src_sequence_size, d_v)

    mask = generate_subsequent_mask(trg_sequence_size, src_sequence_size)
    attention = ScaledDotProductAttention()
    out = attention(q, k, v, mask)
    
    assert torch.equal(torch.tensor(out.shape), torch.tensor([batch_size, n_head, trg_sequence_size, d_v]))