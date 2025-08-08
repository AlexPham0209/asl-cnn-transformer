import math
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.decoder import TransformerDecoder
from asl_research.model.spatial_embedding import Spatial2DEmbedding
from asl_research.model.utils import generate_padding_mask, generate_square_subsequent_mask


class ASLModel(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        src_vocab_size: int = 1000,
        trg_vocab_size: int = 1000,
        gloss_pad_token: int = 1,
        word_pad_token: int = 2,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(ASLModel, self).__init__()
        self.gloss_pad_token = gloss_pad_token
        self.word_pad_token = word_pad_token

        # Encoder
        self.src_embedding = Spatial2DEmbedding(d_model=d_model)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_1 = nn.Linear(d_model, trg_vocab_size)

        # Decoder
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim=d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_2 = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src: Tensor, trg: Tensor):
        trg_mask: Tensor = generate_square_subsequent_mask(trg, self.word_pad_token).to(trg.device)
        
        # Project src image and trg sequences to the embedding space 
        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        src = self.encoder(src)
        trg = self.decoder(trg, src, trg_mask, None)

        src = self.ff_1(src)
        trg = self.ff_2(trg)

        return src, trg

