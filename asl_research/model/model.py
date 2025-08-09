import itertools
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax

from asl_research.model.decoder import TransformerDecoder
from asl_research.model.encoder import TransformerEncoder
from asl_research.model.spatial_embedding import Spatial2DEmbedding
from asl_research.utils.utils import generate_square_subsequent_mask


class ASLModel(nn.Module):
    def __init__(
        self,
        num_encoders: int = 2,
        num_decoders: int = 2,
        gloss_vocab_size: int = 1000,
        word_vocab_size: int = 1000,
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
        self.src_embedding = Spatial2DEmbedding(d_model=d_model, dropout=dropout)
        self.encoder = TransformerEncoder(
            num_layers=num_encoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_1 = nn.Linear(d_model, gloss_vocab_size)

        # Decoder
        self.trg_embedding = nn.Embedding(word_vocab_size, embedding_dim=d_model)
        self.decoder = TransformerDecoder(
            num_layers=num_decoders, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff_2 = nn.Linear(d_model, word_vocab_size)

    def forward(self, src: Tensor, trg: Tensor):
        trg_mask: Tensor = generate_square_subsequent_mask(trg, self.word_pad_token).to(trg.device)

        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        src = self.encoder(src)
        trg = self.decoder(trg, src, trg_mask)

        src = self.ff_1(src)
        trg = self.ff_2(trg)

        return src, trg

    def greedy_decode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_vocab: dict = {"-": 0, "<pad>": 1},
        trg_vocab: dict = {"<sos>": 0, "<eos>": 1, "<pad>": 2},
        max_len: int = 100,
    ):
        self.eval()

        # Convert the sequences from (sequence_size) to (batch, sequence_size)
        src = src.unsqueeze(0) if src.dim() <= 1 else src
        
        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encoder(self.src_embedding(src), src_mask)

        # Get the gloss sequence
        encoded = self.ff_1(memory)
        encoded = softmax(encoded)
        encoded = torch.argmax(encoded, dim=-1).tolist()
        encoded = [[gloss for gloss, _ in itertools.groupby(sample)] for sample in encoded]
        encoded = [[gloss for gloss in sample if gloss != src_vocab["-"]] for sample in encoded]

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = (
            torch.ones(src.shape[0], max_len)
            .fill_(trg_vocab["<pad>"])
            .type(torch.long)
            .to(src.device)
        )
        sequence[:, 0] = trg_vocab["<sos>"]

        for t in range(1, max_len):
            trg_mask = generate_square_subsequent_mask(sequence, self.word_pad_token).to(
                src.device
            )

            # Feeds the target and retrieves a vector (batch_size, sequence_size, trg_vocab_size)
            out = self.trg_embedding(sequence)
            out = self.decoder(out, memory, trg_mask, src_mask)
            out = softmax(self.ff_2(out))

            next_word = torch.argmax(out[:, t-1], dim=-1).to(src.device)
            sequence[:, t] = next_word
            
        return sequence.squeeze(0), encoded
