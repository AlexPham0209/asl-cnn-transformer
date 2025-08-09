import os
from typing import Counter
import warnings
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from asl_research.model.transformer import BaseTransformer
import random
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pad_sequence
from torcheval.metrics.functional import word_error_rate

from asl_research.utils.utils import generate_padding_mask

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def train_epoch(
    model: BaseTransformer,
    data: Dataset,
    optimizer: Optimizer,
    criterion: _Loss,
    epoch: int,
):
    # Set model to training mode
    model.train()
    losses = 0

    # Go through batches in the epoch
    for src, trg in tqdm(data, desc=f"Epoch {epoch}"):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
        # instead of the model's output in the prior timestep
        out = model(
            src,
            trg_input,
        )

        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)

        loss = criterion(actual, expected)
        losses += loss.item()
        loss.backward()

        # Apply the gradient vector on the trainable parameters in the model and reset the gradients
        optimizer.step()
        optimizer.zero_grad()

    losses /= len(data)
    return losses

class TestDataset(Dataset):
    def __init__(self, src, trg, src_vocab, trg_vocab):
        self.src = src
        self.trg = trg

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = self.src[index]
        trg = self.trg[index]

        # Split sentence of ASL gloss and English text to a list of words (Adds the SOS and EOS also)
        src_words = ["<sos>"] + src.split() + ["<eos>"]
        trg_words = ["<sos>"] + trg.split() + ["<eos>"]

        # Convert those list of words to tokens/indices in the vocab
        src_tokens = torch.tensor([self.src_vocab[word] for word in src_words])
        trg_tokens = torch.tensor([self.trg_vocab[word] for word in trg_words])

        return src_tokens, trg_tokens

def collate_fn(batch):
    """
    Processes the list of samples in the batch so that all sample sentences are the same length.

    Parameter:
        batch: A batch in the dataloader

    Returns:
        The batch with both sequences padded
    """
    x, y = zip(*batch)
    x = [torch.tensor(val) for val in x]
    y = [torch.tensor(val) for val in y]

    return pad_sequence(x, batch_first=True, padding_value=2), pad_sequence(
        y, batch_first=True, padding_value=2
    )


def test_transformer_training():
    LENGTH = 200
    EPOCHS = 250
    EXAMPLES = 100

    # Creating a synthetic corpus using words in the words string
    max_sentence_length = 15
    words = "the of and to a home words where apple orange minecraft penis hello world alex who what when damn"
    count = Counter(words.split())

    vocab = ['<sos>', '<eos>', '<pad>'] + sorted(count.keys(), key=lambda key: count[key])
    idx_to_word = {id:word for id, word in enumerate(vocab)}
    word_to_idx = {word:id for id, word in enumerate(vocab)}

    key = [
        " ".join(random.choices(population=vocab[3:], k=random.randint(5, max_sentence_length)))
        for _ in range(LENGTH)
    ]

    value = [
        " ".join(random.choices(population=vocab[3:], k=random.randint(5, max_sentence_length)))
        for _ in range(LENGTH)
    ]

    # Creating dataloader
    dataset = TestDataset(key, value, word_to_idx, word_to_idx)
    data = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    transformer = BaseTransformer(
        num_encoders=2,
        num_decoders=2,
        src_vocab_size=len(word_to_idx),
        trg_vocab_size=len(word_to_idx),
        pad_token=word_to_idx['<pad>'],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training model
    for epoch in range(EPOCHS + 1):
        loss = train_epoch(transformer, data, optimizer, criterion, epoch)
        print(f'Total Loss: {loss}\n')
    
    encoded = []
    inputs = []
    actual = []
    
    # Testing greedy decoding
    for i in range(EXAMPLES):
        index = random.randint(0, len(value) - 1)
        src = key[index]
        trg = value[index]
        
        e = torch.tensor([word_to_idx[word] for word in ["<sos>"] + src.split() + ["<eos>"]]).to(DEVICE)
        inputs.append(src)
        encoded.append(e)
        actual.append(trg)
    
    encoded = pad_sequence(encoded, batch_first=True, padding_value=word_to_idx["<pad>"])
    src_mask = generate_padding_mask(encoded, word_to_idx["<pad>"]).to(encoded.device)
    out = transformer.greedy_decode(encoded, src_mask, trg_vocab=word_to_idx, max_len=20)
    
    preds = []
    targets = []

    remove_special_tokens = (
        lambda token: token != word_to_idx["<pad>"]
        and token != word_to_idx["<eos>"]
        and token != word_to_idx["<sos>"]
    )

    predicted_sentences = [
        " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        for sample in out.tolist()
    ]

    for i in range(EXAMPLES):
        src = inputs[i]
        trg = actual[i]
        res = predicted_sentences[i]

        print(f'Input Sentence: {src}')
        print(f'Output Sentence: {res}')
        print(f'Actual Sentence: {trg}\n')
        
        preds.append(res)
        targets.append(trg)

    wer = word_error_rate(preds, targets)
    print(f"WER: {wer}")
    
    assert wer <= 0.05
