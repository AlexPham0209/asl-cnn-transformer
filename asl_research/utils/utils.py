from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk


def generate_square_subsequent_mask(x: Tensor, pad_token: int):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token or is in the future
    as False.

    Args:
        x (Tensor): Original tensor (batch_size, sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, sequence_size, sequence_size)
    """

    N, sequence_length = x.shape
    # Causal mask: (1, 1, sequence_size, sequence_size)
    causal_mask = (
        torch.tril(torch.ones(sequence_length, sequence_length))
        .unsqueeze(0)
        .unsqueeze(1)
        .bool()
        .to(x.device)
    )

    # Padding mask: (batch_size, 1, 1, sequence_size)
    padding_mask = generate_padding_mask(x, pad_token).to(x.device)

    # Uses Bitwise And operation to combine the causal and padding masks
    # For an entry, ij, if it is not a padding mask AND if it is not a future token,
    # then we don't mask this entry and we allow the attention module to pay attention to it
    mask = causal_mask & padding_mask
    return mask


def generate_padding_mask(x: Tensor, pad_token: int):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token as False.

    Args:
        x (Tensor): Original tensor (batch_size, sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, 1, sequence_size)
    """

    N, sequence_length = x.shape
    return (x != pad_token).unsqueeze(1).unsqueeze(2).bool().to(x.device)


def generate_padding_mask_from_lengths(
    lengths: Optional[Tensor] = None, max_length: Optional[int] = None
):
    """
    Generates a tensor that has the locations in the original tensor where there is a padding token as False.

    Args:
        x (Tensor): Original tensor (sequence_size)

    Returns:
        Tensor: Masking boolean tensor (batch_size, 1, 1, sequence_size)
    """
    if lengths is None:
        return None

    max_length = torch.max(lengths, dim=-1)[0].item() if not max_length else max_length

    lengths = lengths.unsqueeze(1)
    indices = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)

    out = indices < lengths
    return out.unsqueeze(1).unsqueeze(2).to(lengths.device)


def pad_video_with_value(x: Tensor, length: int = 100, padding: float = 0):
    """
    Given a tensor representing a video, pad the video to a specific length with frames containing only
    the padding token value.

    Args:
        x (Tensor): Original tensor (T, C, H, W)
        length (int): Number of frames in returning video
        padding (float): The padding token

    Returns:
        Tensor: (length, C, H, W)
    """

    T, C, H, W = x.shape
    out = torch.zeros(length, C, H, W)
    out[:T] = x
    return out


def pad_video_with_last_frame(x: Tensor, length: int = 100):
    """
    Given a tensor representing a video, pad the video to a specific length with the last frame.

    Args:
        x (Tensor): Original tensor (T, C, H, W)
        length (int): Number of frames in returning video

    Returns:
        Tensor: (length, C, H, W)
    """

    T = x.shape[0]
    out = x[-1].repeat(length, 1, 1, 1)
    out[:T] = x
    return out


def pad_video_with_first_frame(x: Tensor, length: int = 100):
    """
    Given a tensor representing a video, pad the video to a specific length with the last frame.

    Args:
        x (Tensor): Original tensor (T, C, H, W)
        length (int): Number of frames in returning video

    Returns:
        Tensor: (length, C, H, W)
    """

    T = x.shape[0]
    out = x[0].repeat(length, 1, 1, 1)
    out[length - T :] = x
    return out


def pad_landmarks(batch: Tensor):
    T = max([landmarks.size(dim=0) for landmarks in batch])
    _, features = batch[0].shape
    res = torch.zeros(len(batch), T, features)
    mask = torch.zeros(len(batch), T)

    for i, landmarks in enumerate(batch):
        res[i, :landmarks.size(dim=0), :] = landmarks
        mask[i, :landmarks.size(dim=0)] = 1
        
    return res, mask.unsqueeze(1).unsqueeze(2)


def decode_sentences(sequence: list, word_to_idx: dict, idx_to_word: dict):
    assert "<pad>" in word_to_idx
    assert "<eos>" in word_to_idx
    assert "<sos>" in word_to_idx

    remove_special_tokens = (
        lambda token: token != word_to_idx["<pad>"]
        and token != word_to_idx["<eos>"]
        and token != word_to_idx["<sos>"]
    )

    sentences = [
        " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        for sample in sequence
    ]

    return sentences


def decode_glosses(sequence: list, gloss_to_idx: dict, idx_to_gloss: dict):
    assert "<pad>" in gloss_to_idx

    remove_padding = lambda x: x != gloss_to_idx["<pad>"]

    sequence = [
        " ".join([idx_to_gloss[token] for token in list(filter(remove_padding, sample))])
        for sample in sequence
    ]
    return sequence


def calculate_bleu_scores(predicted: list, actual: list):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = []

    for reference, hypothesis in zip(predicted, actual):
        score = sentence_bleu([reference.split()], hypothesis.split(), weights=[1])
        scores.append(score)

    return torch.tensor(scores)


def calculate_rouge_scores(predicted: list, actual: list):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    precisions = []
    recalls = []
    fmeasures = []

    for a, b in zip(predicted, actual):
        score = scorer.score(a, b)
        precision, recall, fmeasure = score["rouge1"]

        precisions.append(precision)
        recalls.append(recall)
        fmeasures.append(fmeasure)

    return torch.tensor(precisions), torch.tensor(recalls), torch.tensor(fmeasures)


if __name__ == "__main__":
    n_features = 12

    a = torch.arange(1, 6 * 4 + 1).reshape(6, 4)

    batch, mask = pad_landmarks([a[:1], a[:4], a])
    print(mask)
    print(generate_padding_mask(batch[:, :, 0], 0))
