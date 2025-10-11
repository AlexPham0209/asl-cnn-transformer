import json
import os

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
)
from torchvision.transforms.v2 import UniformTemporalSubsample
from torchvision.io import decode_image, read_file, decode_jpeg
from asl_research.utils.utils import pad_video_with_last_frame, pad_video_with_value
import random

# mean = (0.53724027, 0.5272855, 0.51954997)
# std = (1, 1, 1)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class PhoenixDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        target_size: tuple = (224, 224),
        num_frames: int | list = 120,
        max_start_frame: int = 10,
        min_end_frame: int = 10,
        random_sampling: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        self.dataset_path = os.path.join(root_dir, "dataset.csv")
        self.vocab_path = os.path.join(root_dir, "vocab.json")
        self.video_dir = os.path.join(root_dir, "videos_phoenix", "videos")
        self.processed_video_dir = os.path.join(root_dir, "processed_videos")

        self.num_frames = num_frames
        self.max_start_frame = max_start_frame
        self.min_end_frame = min_end_frame
        self.random_sampling = random_sampling

        self.is_train = is_train

        assert os.path.exists(self.dataset_path), (
            "Dataset directory doesn't exists (try running the download script)"
        )
        assert os.path.exists(self.vocab_path), (
            "Vocab.json doesn't exist (try running the download script)"
        )
        assert os.path.exists(self.video_dir), (
            "Video directory doesn't exist (try running the download script)"
        )
        assert os.path.exists(self.processed_video_dir), (
            "Processed video directory doesn't exist (try running the preprocessing script)"
        )

        self.df = df
        self.vocab = json.load(open(self.vocab_path))

        self.glosses = self.vocab["glosses"]
        self.words = self.vocab["words"]

        # Create dictionaries to convert string tokens into their ids and vice versa
        self.gloss_to_idx = {gloss: i for i, gloss in enumerate(self.glosses)}
        self.idx_to_gloss = {i: gloss for i, gloss in enumerate(self.glosses)}

        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}

        # Data augmentation settings
        self.normalize = Compose(
            [
                Lambda(self.normalize_color),
                Normalize(mean, std),
                Resize((256, 256)),
                RandomCrop(target_size)
            ]
        )

        self.augment = Compose(
            [
                RandomRotation(10),
                ColorJitter(brightness=(0.5, 1.0), hue=0.1),
            ]
        )

    def normalize_color(self, x: torch.Tensor):
        return x / 255.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Reading information from row entry in the dataframe
        item = self.df.iloc[index]
        path = os.path.join(self.video_dir, item["paths"])
        processed_path = os.path.join(self.processed_video_dir, item["processed_paths"])
        glosses = item["glosses"]
        sentence = item["texts"]

        # Convert strings into token sequences
        gloss_tokens = torch.tensor([self.gloss_to_idx[gloss] for gloss in glosses.split()])
        word_tokens = torch.tensor(
            [self.word_to_idx["<sos>"]]
            + [self.word_to_idx[word] for word in sentence.split()]
            + [self.word_to_idx["<eos>"]]
        )

        # Get video and apply augmentations on it
        assert os.path.exists(processed_path), "Processed path doesn't exists"
        video_data = self.read_video(processed_path)
        video_data = self.normalize(video_data)

        if self.is_train:
            video_data = self.augment(video_data)

        return (
            video_data,
            gloss_tokens,
            word_tokens,
            self.gloss_to_idx["<pad>"],
            self.word_to_idx["<pad>"],
        )
    
    def get_vocab(self):
        return self.gloss_to_idx, self.idx_to_gloss, self.word_to_idx, self.idx_to_word

    def read_video(self, path: str):
        frames = []
        frame_files = sorted(
            os.listdir(path), key=lambda p: int(p.split("_")[1].replace(".jpg", ""))
        )

        frame_positions = (
            self.random_frame_subsampling(frame_files)
            if self.random_sampling
            else self.uniform_frame_subsampling(frame_files)
        )
        
        for pos in frame_positions:
            frame = os.path.join(path, frame_files[pos.item()])

            if not frame.endswith(".jpg"):
                continue

            frames.append(read_file(frame))

        return torch.stack(decode_jpeg(frames), dim=0)

    def uniform_frame_subsampling(self, frames):
        start = random.randint(0, self.max_start_frame)
        end = random.randint(len(frames) - self.min_end_frame - 1, len(frames) - 1)
        steps = (
            self.num_frames
            if isinstance(self.num_frames, list)
            else random.randint(self.num_frames[0], self.num_frames[1])
        )

        return torch.linspace(start=start, end=end, steps=steps, dtype=int)

    def random_frame_subsampling(self, frames):
        start = random.randint(0, self.max_start_frame)
        end = random.randint(len(frames) - self.min_end_frame - 1, len(frames) - 1)
        steps = (
            self.num_frames
            if isinstance(self.num_frames, tuple[int, int])
            else random.randint(self.num_frames[0], self.num_frames[1])
        )

        frame_positions, _ = torch.randint(low=start, high=end, size=(steps,), dtype=int).sort()

        return frame_positions

    @staticmethod
    def collate_fn(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Assumes videos are equal length
        videos = torch.stack(videos, dim=0)
        
        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, gloss_sequences, gloss_lengths, sentences, sentence_lengths

    @staticmethod
    def collate_fn_last_frame_padding(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Padding videos with its last frame
        max_video_length = max([video.shape[0] for video in videos])
        videos = list(
            map(lambda video: pad_video_with_last_frame(video, max_video_length), videos)
        )
        videos = videos.stack()

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, gloss_sequences, gloss_lengths, sentences, sentence_lengths

    @staticmethod
    def collate_fn_zero_padding(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        # Padding videos with 0
        max_video_length = max([video.shape[0] for video in videos])
        videos = list(map(lambda video: pad_video_with_value(video, max_video_length, 0), videos))
        videos = videos.stack()

        # Padding gloss sequences
        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        # Padding sentences
        sentence_lengths = torch.tensor([sentence.shape[0] for sentence in sentences])
        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, gloss_sequences, gloss_lengths, sentences, sentence_lengths
