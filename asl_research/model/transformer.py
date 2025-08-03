import math
from torch import Tensor
import torch
import torch.nn as nn
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.decoder import TransformerDecoder
from asl_research.model.utils import generate_padding_mask, generate_square_subsequent_mask


class BaseTransformer(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        src_vocab_size: int = 1000,
        trg_vocab_size: int = 1000,
        pad_token: int = 0,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(BaseTransformer, self).__init__()
        self.pad_token = pad_token

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim=d_model)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # Decoder
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim=d_model)
        self.decoder = TransformerEncoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # Classification
        self.ff = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src: Tensor, trg: Tensor):
        src_mask: Tensor = generate_padding_mask(src, self.pad_token)
        src_mask = (
            src_mask.float().masked_fill(src_mask == 1, 0).masked_fill(src_mask == 0, -torch.inf)
        )

        trg_mask: Tensor = generate_square_subsequent_mask(trg, self.pad_token)
        trg_mask: Tensor = (
            trg_mask.float().masked_fill(trg_mask == 1, 0).masked_fill(trg_mask == 0, -torch.inf)
        )

        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        src = self.encoder(src, src_mask)
        trg = self.decoder(trg, src, trg_mask, src_mask)

        return self.ff(trg)
