import json
import os

import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Lambda,
    RandomCrop,
    RandomRotation,
    Resize,
    Normalize,
    GaussianBlur,
    CenterCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms.v2 import UniformTemporalSubsample
from torchvision.io import decode_image, read_file, decode_jpeg
from asl_research.utils.utils import (
    generate_padding_mask_from_lengths,
    pad_video_with_last_frame,
    pad_video_with_value,
)
import random

from asl_research.vocab import GlossVocabulary, TextVocabulary

# mean = (0.53724027, 0.5272855, 0.51954997)
# std = (1, 1, 1)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class PhoenixDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        gloss_vocab: GlossVocabulary,
        text_vocab: TextVocabulary,
        sampling_ratio: int = 2,
        masking_ratio: float = 0.8,
        random_sampling: bool = True,
        random_masking: bool = True,
        is_train: bool = True,
    ):
        super().__init__()
        self.df = df
        self.text_vocab = text_vocab
        self.gloss_vocab = gloss_vocab

        self.sampling_ratio = sampling_ratio
        self.random_sampling = random_sampling
        self.masking_ratio = masking_ratio
        self.random_masking = random_masking

        self.is_train = is_train

        # Data augmentation settings
        self.train_transform = Compose(
            [
                Resize((256, 256)),
                RandomCrop((224, 224)),
                Lambda(self.normalize_color),
                Normalize(mean, std),
            ]
        )

        self.valid_transform = Compose(
            [
                Resize((256, 256)),
                CenterCrop((224, 224)),
                Lambda(self.normalize_color),
                Normalize(mean, std),
            ]
        )

    def normalize_color(self, x: torch.Tensor):
        return x / 255.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Reading information from row entry in the dataframe
        item = self.df.iloc[index]
        name = item["id"]
        video_path = item["video_path"]
        landmark_path = item["landmark_path"]
        gloss = item["gloss"]
        sentence = item["text"]

        # Convert strings into token sequences
        gloss_tokens = self.gloss_vocab.tokenize(gloss)
        word_tokens = self.text_vocab.tokenize(sentence)

        # Getting landmarks data (time, 225) and standardizing it
        # landmarks = torch.tensor(np.load(landmark_path))
        # if self.random_sampling and self.is_train:
        #     landmarks = landmarks[:: self.sampling_ratio]

        video = self.read_video(video_path)
        video = self.train_transform(video) if self.is_train else self.valid_transform(video)

        return (
            video,
            gloss_tokens,
            word_tokens,
            self.gloss_vocab.pad_token,
            self.text_vocab.pad_token,
        )

    def read_video(self, path: str):
        frames = []
        frame_files = sorted(
            os.listdir(path), key=lambda p: int(p.split("_")[1].replace(".jpg", ""))
        )

        start = (
            random.randint(0, self.sampling_ratio - 1)
            if self.is_train and self.random_sampling
            else 0
        )
        frame_positions = torch.arange(
            start=start, end=len(frame_files) - 1, step=self.sampling_ratio
        )

        if self.random_masking and self.is_train:
            size = int(frame_positions.shape[-1] * self.masking_ratio)
            masking_idx, _ = torch.randperm(frame_positions.shape[-1])[:size].sort()
            frame_positions = frame_positions[masking_idx]

        for pos in frame_positions:
            frame = os.path.join(path, frame_files[pos.item()])

            if not frame.endswith(".jpg"):
                continue

            frames.append(read_file(frame))

        return torch.stack(decode_jpeg(frames), dim=0)

    @staticmethod
    def collate_fn(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Padding videos with 0
        video_lengths = torch.tensor([video.shape[0] for video in videos])
        videos = pad_sequence(videos, batch_first=True, padding_value=0)

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return {videos, video_lengths, gloss_sequences, gloss_lengths, sentences, sentence_lengths}

    @staticmethod
    def collate_fn_landmarks(batch: list):
        landmarks, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Padding landmarks with 0
        landmark_lengths = torch.tensor([landmark.shape[0] for landmark in landmarks])
        landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0)

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return (
            landmarks,
            landmark_lengths,
            gloss_sequences,
            gloss_lengths,
            sentences,
            sentence_lengths,
        )

    def standardize_points(self, x: torch.Tensor):
        return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-4)

    @staticmethod
    def collate_fn_no_padding(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Assumes videos are equal length
        video_lengths = torch.tensor([video.shape[0] for video in videos])
        max_video_length = video_lengths.max().item()
        videos = torch.stack(videos, dim=0)

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, video_lengths, gloss_sequences, gloss_lengths, sentences, sentence_lengths

    @staticmethod
    def collate_fn_last_frame_padding(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Padding videos with its last frame
        video_lengths = torch.tensor([video.shape[0] for video in videos])
        max_video_length = video_lengths.max().item()
        videos = list(
            map(lambda video: pad_video_with_last_frame(video, max_video_length), videos)
        )
        videos = torch.stack()

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, video_lengths, gloss_sequences, gloss_lengths, sentences, sentence_lengths
