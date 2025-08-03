import math
from torch import Tensor
import torch.nn as nn
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.decoder import TransformerDecoder


class BaseTransformer(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        src_vocab_size: int = 1000,
        trg_vocab_size: int = 1000,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(BaseTransformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim=d_model)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim=d_model)
        self.decoder = TransformerEncoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        self.ff = nn.Linear(d_model, trg_vocab_size)
