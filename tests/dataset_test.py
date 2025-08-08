import math
import pytest
import torch 
from asl_research.dataloader import PhoenixDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def test_dataset():
    dataset = PhoenixDataset(root_dir="data\\processed\\phoenixweather2014t")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PhoenixDataset.collate_fn)

    videos, gloss_sequences, gloss_lengths, sentences = next(iter(dataloader))
    assert videos.shape == (2, 120, 3, 224, 224)
    assert gloss_lengths.shape[0] == 2
    assert gloss_sequences.shape[0] == 2
    assert sentences.shape[0] == 2


def test_dataset_split():
    # Creating dataset and getting gloss and word vocabulary dictionaries
    dataset = PhoenixDataset(
        root_dir="data\\processed\\phoenixweather2014t", num_frames=60, target_size=(224, 224)
    )

    gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = dataset.get_vocab()

    # Splitting dataset into training, validation, and testing sets
    generator = torch.Generator().manual_seed(29)
    train_set, valid_set, test_set = random_split(
        dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator
    )

    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=16, shuffle=True)

    assert len(train_set) == math.floor(0.8 * len(dataset))
    assert len(valid_set) == math.floor(0.1 * len(dataset))
    assert len(test_set) == math.floor(0.1 * len(dataset))
