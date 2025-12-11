from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


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


if __name__ == "__main__":
    n_features = 12

    a = torch.arange(1, 3 * n_features + 1).reshape(3, n_features)
    b = torch.arange(1, 6 * n_features + 1).reshape(6, n_features)
    c = torch.arange(1, 3 * n_features + 1).reshape(3, n_features)

    test = torch.zeros(3, max([a.shape[0], b.shape[0], c.shape[0]]), n_features)
    test[0, : a.shape[0]] = a
    test[1, : b.shape[0]] = b
    test[2, : c.shape[0]] = c
    print(test.shape)
    a = (test != torch.zeros(n_features))[..., 0].unsqueeze(1).unsqueeze(2)
    b = generate_video_padding_mask(torch.tensor([a.shape[0], b.shape[0], c.shape[0]]))
    print(torch.equal(a, b))

    print(
        pad_sequence(
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 3])],
            batch_first=True,
        )
    )
