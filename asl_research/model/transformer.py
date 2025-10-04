import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from asl_research.model.decoder import TransformerDecoder
from asl_research.model.encoder import TransformerEncoder
from asl_research.utils.utils import generate_padding_mask, generate_square_subsequent_mask


class BaseTransformer(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        src_vocab_size: int = 1000,
        trg_vocab_size: int = 1000,
        pad_token: int = 2,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(BaseTransformer, self).__init__()
        self.pad_token = pad_token
        self.d_model = d_model

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim=d_model)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # Decoder
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim=d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # Classification
        self.ff = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: Tensor, trg: Tensor):
        src_mask: Tensor = generate_padding_mask(src, self.pad_token).to(src.device)
        trg_mask: Tensor = generate_square_subsequent_mask(trg, self.pad_token).to(trg.device)

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        trg = self.trg_embedding(trg) * math.sqrt(self.d_model)

        src = self.encoder(src, src_mask)
        trg = self.decoder(trg, src, trg_mask, src_mask)

        return self.ff(trg)

    def greedy_decode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        trg_vocab: dict = {"<sos>": 0, "<eos>": 1, "<pad>": 2},
        max_len: int = 100,
    ):
        self.eval()

        # # Convert the sequences from (sequence_size) to (batch, sequence_size)
        # src = src.unsqueeze(0)
        
        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encoder(self.src_embedding(src) * math.sqrt(self.d_model), src_mask)

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = (
            torch.ones(src.shape[0], max_len)
            .fill_(trg_vocab["<pad>"])
            .type(torch.long)
            .to(src.device)
        )
        sequence[:, 0] = trg_vocab["<sos>"]

        for t in range(1, max_len):
            out = sequence[:, :t]
            trg_mask = generate_square_subsequent_mask(out, self.pad_token).to(src.device)

            # Feeds the target and retrieves a vector (batch_size, sequence_size, trg_vocab_size)
            out = self.trg_embedding(out) * math.sqrt(self.d_model)
            out = self.decoder(out, memory, trg_mask, src_mask)
            out = self.softmax(self.ff(out))

            next_word = torch.argmax(out[:, -1], dim=-1).to(src.device)

            # Concatenate the predicted token to the output sequence
            sequence[:, t] = next_word

        return sequence
