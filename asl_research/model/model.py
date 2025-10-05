import itertools
import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax

from asl_research.model.decoder import TransformerDecoder
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.spatial_embedding import SpatialEmbedding
from asl_research.utils.utils import generate_square_subsequent_mask


class ASLModel(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        gloss_to_idx: dict = {"-": 0, "<pad>": 1},
        idx_to_gloss: dict = {0: "-", 1: "<pad>"},
        word_to_idx: dict = {"<sos>": 0, "<eos>": 1, "<pad>": 2},
        idx_to_word: dict = {0: "<sos>", 1: "<eos>", 2: "<pad>"},
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(ASLModel, self).__init__()
        
        # Vocab
        self.gloss_to_idx = gloss_to_idx
        self.idx_to_gloss = idx_to_gloss
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        # Padding tokens
        self.gloss_pad_token = gloss_to_idx["<pad>"]
        self.word_pad_token = word_to_idx["<pad>"]

        self.d_model = d_model

        # Encoder
        self.src_embedding = SpatialEmbedding(d_model=d_model, dropout=dropout)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_1 = nn.Linear(d_model, len(self.gloss_to_idx))

        # Decoder
        self.trg_embedding = nn.Embedding(len(self.word_to_idx), embedding_dim=d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_2 = nn.Linear(d_model, len(self.word_to_idx))

    def forward(self, src: Tensor, trg: Tensor):
        trg_mask: Tensor = generate_square_subsequent_mask(trg, self.word_pad_token).to(trg.device)

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        trg = self.trg_embedding(trg) * math.sqrt(self.d_model)
        
        src = self.encoder(src)
        trg = self.decoder(trg, src, trg_mask)
        
        src = self.ff_1(src)
        trg = self.ff_2(trg)

        return src, trg

    def greedy_decode(
        self,
        src: Tensor,
        max_len: int = 100,
    ):
        self.eval()

        # Convert the sequences from (sequence_size) to (batch, sequence_size)
        src = src.unsqueeze(0) if src.dim() <= 1 else src

        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encoder(self.src_embedding(src) * math.sqrt(self.d_model))

        # Get the gloss sequence
        encoded = self.ff_1(memory)
        encoded = softmax(encoded, dim=-1)
        encoded = torch.argmax(encoded, dim=-1).tolist()
        encoded = [[gloss for gloss, _ in itertools.groupby(sample)] for sample in encoded]
        encoded = [
            list(filter(lambda gloss: gloss != self.gloss_to_idx["-"], sample))
            for sample in encoded
        ]

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = (
            torch.ones(src.shape[0], max_len)
            .fill_(self.word_to_idx["<pad>"])
            .type(torch.long)
            .to(src.device)
        )
        # Fill first column (or the beginning of the sequences) with <SOS> tokens
        sequence[:, 0] = self.word_to_idx["<sos>"]

        for t in range(1, max_len):
            out = sequence[:, :t]
            trg_mask = generate_square_subsequent_mask(out, self.word_pad_token).to(src.device)

            # Feeds the target and retrieves a vector (batch_size, sequence_size, trg_vocab_size)
            out = self.trg_embedding(out) * math.sqrt(self.d_model)
            out = self.decoder(out, memory, trg_mask)
            out = softmax(self.ff_2(out), dim=-1)
            
            next_word = torch.argmax(out[:, -1], dim=-1).to(src.device)
            sequence[:, t] = next_word

        return encoded, sequence
