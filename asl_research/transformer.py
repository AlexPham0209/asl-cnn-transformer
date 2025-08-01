import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        """
        Create a positional encoding matrix.

        Args:
            max_len: The maximum length a sequence of tokens can be
            d_model: The dimensions of the encoding vector
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        dim = torch.arange(0, d_model, 2)

        # For even position indices, we utilize the sine function
        # For odd indices, we utilize the cosine function
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / d_model)))
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Adds the positional encoding to the embedding matrix
        Helps describes the position of the embedding vector within a sequence.

        Args:
            x (Tensor): Embedding matrix (Batch, Sequence Size, Embedding Size)

        Returns:
            Tensor: Embedding matrix with positions encoded into them
        """

        res = x + self.pe[:x.size(dim=1), :]
        return self.dropout(res)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: float, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        """
        Splits the query, key, and value tensors into a number of heads 
        Calculates the attention scores using Scaled Dot Attention
        Then, 

        Args:
            q (Tensor): Query tensor (batch_size, target_sequence_size, d_model)
            k (Tensor): Key tensor (batch_size, src_sequence_size, d_model)
            v (Tensor): Value tensor (batch_size, src_sequence_size, d_model)

            mask (Optional[Tensor]): Used to mask out elements in the attention score matrix

        Returns:
            Tensor: The attention matrix (batch_size, target_sequence_size, d_model)
        """

        # Shape: (batch_size, sequence_length, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Split tensor into heads
        # Shape: (batch_size, sequence_length, d_model)
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        # Calculate the attention score which is used to gauge which tokens are important to each token
        # Shape: (batch_size, num_heads, sequence_size, d_model // num_heads)
        out = self.attention(q, k, v, mask)
        
        # Concatenate heads together
        # Shape: (batch_size, target_sequence_length, d_model)
        out = self.concat(out)

        # Determines which token/word it should attend to?
        # Shape: (batch_size, target_sequence_length, d_model)
        return self.w_o(out)


    def split(self, x: Tensor):
        """
        Splits the tensor into num_heads

        Args:
            x (Tensor): Original tensor (batch_size, sequence_size, d_model)

        Returns:
            Tensor: Tensor that is split into n heads 
            (batch_size, num_heads, sequence_size, d_model // num_heads)
        """
        # Shape: (batch_size, sequence_length, d_model)
        N, length, _ = x.shape

        # Reshape into (batch_size, num_heads, sequence_length, d_models // num_heads)
        return x.reshape(N, length, self.num_heads, -1).transpose(1, 2)
    
    def concat(self, x: Tensor):
        """
        Concatenate the tensor's heads together

        Args:
            x (Tensor): Original tensor (batch_size, num_heads, sequence_size, d_model // num_heads)

        Returns:
            Tensor: Tensor that is split into n heads (batch_size, sequence_size, d_model)
        """

        N, _, length, _ = x.shape

        # Transpose into (batch_size, sequence_length, num_heads, d_model)
        # Then, reshape into (batch_size, sequence_length, d_model)
        return x.transpose(1, 2).reshape(N, length, -1)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        """
        Calculates the scaled dot product attention between q, k, v tensors

        Args:
            q (Tensor): Query tensor (batch_size, num_heads, target_sequence_size, d_model // num_heads)
            k (Tensor): Key tensor (batch_size, num_heads, src_sequence_size, d_model // num_heads)
            v (Tensor): Value tensor (batch_size, num_heads, src_sequence_size, d_model // num_heads)

            mask (Optional[Tensor]): Used to mask out elements in the attention score matrix (target_sequence)
        
        Returns:
            Tensor: Attention matrix (batch_size, target_sequence_size, d_model)
        """

        # QK and Scores' Shape: (batch_size, num_heads, target_sequence_size, src_sequence_size)
        qk = q @ k.transpose(-2, -1)
        scores = qk / math.sqrt(k.shape[-1])

        # Filled all elements where the with -torch.inf
        if mask is not None:
            scores = scores + mask

        # Calculate a probability distribution with the current token to all other tokens in the sequence
        scores = self.softmax(scores)

        # Shape: (batch_size, num_heads, target_sequence_size, d_v)
        return scores @ v 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(in_features=d_model, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(in_features=hidden_size, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.w1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.w2(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        # Self Attention
        self.attention = MultiHeadAttention(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        # Position-Wise Feed Forward
        self.ff = PositionWiseFeedForward(d_model, hidden_size, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        # Self Attention
        # Shape: (batch_size, sequence_size, d_model)
        _x = x 
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout(x)
        x = self.layer_norm_1(_x + x)
        
        # Position-Wise Feed Forward
        # Shape: (batch_size, sequence_size, d_model)
        _x = x 
        x = self.ff(x)
        x = self.layer_norm_2(_x + x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        # Self Attention
        self.self_attention = MultiHeadAttention(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        # Cross Attention
        self.cross_attention = MultiHeadAttention(d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # Position-Wise Feed Forward
        self.ff = PositionWiseFeedForward(d_model, hidden_size)
        self.dropout_3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, encoded: Tensor, src_mask: Optional[Tensor] = None, trg_mask: Optional[Tensor] = None):
        # Masked Self Attention
        # Shape: (batch_size, target_sequence_size, d_model)
        _x = x
        x = self.self_attention(q=x, k=x, v=x, mask=trg_mask)
        x = self.layer_norm_1(_x + x)
        x = self.dropout_1(x)

        # Cross Attention
        # Shape: (batch_size, target_sequence_size, d_model)
        _x = x 
        x = self.cross_attention(q=x, k=encoded, v=encoded)
        x = self.dropout_2(x)
        x = self.layer_norm_2(_x + x)
        
        # Position-Wise Feed Forward
        # Shape: (batch_size, target_sequence_size, d_model)
        _x = x
        x = self.ff(x)
        x = self.dropout_3(x)
        x = self.layer_norm_3(_x + x)

        return x 


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int = 512, hidden_size: int = 512, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__() 
        
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, hidden_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None):
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x 
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int = 512, hidden_size: int = 512, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__() 
        
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, hidden_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x: Tensor, encoded: Tensor, src_mask: Optional[Tensor] = None, trg_mask: Optional[Tensor] = None):
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, encoded, src_mask, trg_mask)
        
        return x 

def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones((size, size))) == 1
    mask = mask.float().masked_fill(mask == True, 0).masked_fill(mask == False, -torch.inf)
    return mask

def generate_subsequent_mask(height, width):
    mask = torch.tril(torch.ones((height, width))) == 1
    mask = mask.float().masked_fill(mask == True, 0).masked_fill(mask == False, -torch.inf)
    return mask
