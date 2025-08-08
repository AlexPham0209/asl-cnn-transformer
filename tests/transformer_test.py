import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import MultiheadAttention
from asl_research.model.attention import ScaledDotProductAttention, MultiHeadAttention
from asl_research.model.transformer import BaseTransformer
from asl_research.utils.utils import generate_square_subsequent_mask, generate_padding_mask
from asl_research.model.model import ASLModel

def make_trg_mask(trg, pad_token):
    trg_pad_mask = (trg != pad_token).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

def test_square_subsequent_mask():
    x = torch.tensor([
        [1, 2, 0, 0],
        [3, 2, 4, 0]
    ])

    a = generate_square_subsequent_mask(x, 0)
    b = make_trg_mask(x, 0) == 1

    assert torch.equal(a, b)

def test_target_mask_creation():
    """Test target mask combines causal and padding correctly"""
    target = torch.tensor([[1, 2, 3, 0]])
        
    trg_mask = generate_square_subsequent_mask(target, 0)
        
    assert trg_mask.shape == (1, 1, 4, 4)
        
    # Check causal pattern
    assert trg_mask[0, 0, 0, 0] == True   # Position 0 can see position 0
    assert trg_mask[0, 0, 1, 0] == True   # Position 1 can see position 0
    assert trg_mask[0, 0, 1, 1] == True   # Position 1 can see position 1
    assert trg_mask[0, 0, 0, 1] == False  # Position 0 cannot see position 1
    
        # Check padding masking
    assert trg_mask[0, 0, 0, 3] == False  # Cannot attend to padding
    assert trg_mask[0, 0, 3, 3] == False  # Padding position masked

def test_source_mask_creation():
    """Test source mask is created correctly"""
    source = torch.tensor([[1, 2, 3, 0, 0, 0]])
        
    src_mask = generate_padding_mask(source, 0)
        
    assert src_mask.shape == (1, 1, 1, 6)
    expected_mask = torch.tensor([[[[True, True, True, False, False, False]]]])
    assert torch.equal(src_mask, expected_mask)

def test_transformer_input_test(): 
    model = BaseTransformer()
    src = torch.tensor([
        [1, 5, 10, 2],
        [23, 1, 11, 0],
    ])

    trg = torch.tensor([
        [1, 5, 10, 2, 12],
        [23, 3, 11, 32, 0],
    ])

    out = model(src, trg)
    assert out.shape == (2, 5, 1000)

def test_asl_model_input_test(): 
    model = ASLModel()
    src = torch.randn(1, 20, 3, 224, 224)
    trg = torch.tensor([[1, 2, 3, 4, 5]])

    src, trg = model(src, trg)
    assert src.shape == (1, 20, 1000)
    assert trg.shape == (1, 5, 1000)

def test_transformer_greedy_decode(): 
    model = BaseTransformer()
    src = torch.tensor([1, 5, 10, 2])
    out = model.greedy_decode(src)
    print(out)

