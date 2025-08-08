import pytest
import torch 
from asl_research.dataloader import PhoenixDataset
from torch.utils.data import DataLoader


def test_dataset():
    dataset = PhoenixDataset(root_dir="data\\processed\\phoenixweather2014t")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PhoenixDataset.collate_fn)

    videos, gloss_sequences, gloss_lengths, sentences = next(iter(dataloader))
    assert videos.shape == (2, 120, 3, 224, 224)
    assert gloss_lengths.shape[0] == 2
    assert gloss_sequences.shape[0] == 2
    assert sentences.shape[0] == 2