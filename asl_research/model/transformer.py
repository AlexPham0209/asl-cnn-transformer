import math
from typing import Optional
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
        self.decoder = TransformerDecoder(
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

    def greedy_decode(
        self, src: Tensor, src_mask: Optional[Tensor], src_vocab, trg_vocab, max_len=100
    ):
        self.eval()

        # Convert the sequences from (sequence_size) to (batch, sequence_size)
        src = src.unsqueeze(0)

        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encoder(self.src_embedding(src), src_mask)

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = torch.ones(1, 1).fill_(trg_vocab["<sos>"]).type(torch.long)

        for _ in range(max_len):
            mask = generate_square_subsequent_mask(sequence.shape[-1], pad_token=0)
            mask = (
                mask.float().masked_fill(mask == 1, 0).masked_fill(mask == 0, -torch.inf)
            )
            
            # Feeds the target and retrieves a vector (batch_size, sequence_size, trg_vocab_size)
            out = self.decoder(self.trg_embedding(sequence), src, mask, src_mask)
            _, next_word = torch.max(out[:, :, -1], dim=-1)
            next_word = next_word.unsqueeze(dim=0)

            # Concatenate the predicted token to the output sequence
            sequence = torch.cat((sequence, next_word), dim=-1)

            if next_word == trg_vocab["<eos>"]:
                break

        return sequence.squeeze(0)
