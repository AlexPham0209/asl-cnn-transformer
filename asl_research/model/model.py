import itertools
import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax

from asl_research.model.decoder import TransformerDecoder
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.spatial_embedding import PoseEmbedding, SpatialEmbedding
from asl_research.utils.utils import (
    generate_square_subsequent_mask,
    generate_padding_mask_from_lengths,
)
from asl_research.vocab import GlossVocabulary, TextVocabulary


class ASLModel(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        gloss_vocab: GlossVocabulary = None,
        text_vocab: TextVocabulary = None,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(ASLModel, self).__init__()

        # Vocab
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

        self.d_model = d_model

        # Encoder
        # self.src_embedding = SpatialEmbedding(
        #     d_model=d_model,
        #     hidden_size=1024,
        #     dropout=dropout,
        #     pretrained_model=pretrained_embedding,
        # )

        self.src_embedding = PoseEmbedding(
            in_channels=225,
            d_model=d_model,
            hidden_size=1024,
            dropout=dropout,
        )

        self.encoder = TransformerEncoder(
            num_layers=num_encoders,
            d_model=d_model,
            num_heads=num_heads,
            hidden_size=2048,
            dropout=dropout,
        )
        self.ff_1 = nn.Linear(d_model, self.gloss_vocab.get_size())

        # Decoder
        self.trg_embedding = nn.Embedding(self.text_vocab.get_size(), embedding_dim=d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoders,
            d_model=d_model,
            num_heads=num_heads,
            hidden_size=2048,
            dropout=dropout,
        )
        self.ff_2 = nn.Linear(d_model, self.text_vocab.get_size())
        self._init_weights()

    def forward(self, src: Tensor, trg: Tensor, src_lengths: Optional[Tensor] = None):
        src, src_mask, src_lengths = self.src_embedding(src, src_lengths)
        trg_mask = generate_square_subsequent_mask(trg, self.text_vocab.pad_token)

        src = src * math.sqrt(self.d_model)
        trg = self.trg_embedding(trg) * math.sqrt(self.d_model)

        src = self.encoder(src, src_mask)
        trg = self.decoder(trg, src, trg_mask, src_mask)

        src = self.ff_1(src)
        trg = self.ff_2(trg)

        # Should output the encoder output
        # src: (batch_size, gloss_sequence_length, gloss_vocab_size)
        # trg: (batch_size, video_length, word_vocab_size)
        return src, trg, src_lengths

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def greedy_decode(
        self,
        src: Tensor,
        src_lengths: Optional[Tensor] = None,
        max_len: int = 30,
    ):
        self.eval()

        # Convert the sequences from (sequence_size) to (batch, sequence_size)
        src = src.unsqueeze(0) if src.dim() <= 1 else src
        src, src_mask, _ = self.src_embedding(src, src_lengths)

        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encoder(src * math.sqrt(self.d_model), src_mask)

        # Get the gloss sequence
        encoded = self.ff_1(memory)
        encoded = softmax(encoded, dim=-1)
        encoded = torch.argmax(encoded, dim=-1).tolist()
        encoded = [[gloss for gloss, _ in itertools.groupby(sample)] for sample in encoded]
        encoded = [
            list(filter(lambda gloss: gloss != self.gloss_vocab.blank_token, sample))
            for sample in encoded
        ]

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = (
            torch.ones(src.shape[0], max_len)
            .fill_(self.text_vocab.pad_token)
            .type(torch.long)
            .to(src.device)
        )
        # Fill first column (or the beginning of the sequences) with <SOS> tokens
        sequence[:, 0] = self.text_vocab.sos_token

        for t in range(1, max_len):
            out = sequence[:, :t]
            trg_mask = generate_square_subsequent_mask(out, self.text_vocab.pad_token).to(
                src.device
            )

            # Feeds the target and retrieves a vector (batch_size, sequence_size, trg_vocab_size)
            out = self.trg_embedding(out) * math.sqrt(self.d_model)
            out = self.decoder(out, memory, trg_mask, src_mask)
            out = softmax(self.ff_2(out), dim=-1)

            next_word = torch.argmax(out[:, -1], dim=-1).to(src.device)
            sequence[:, t] = next_word

        return encoded, sequence
