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

# mean = (0.53724027, 0.5272855, 0.51954997)
# std = (1, 1, 1)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class PhoenixDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame, 
        root_dir: str,
        device,
        num_frames: int = 120,
        target_size: tuple = (224, 224),
    ):
        super().__init__()
        self.dataset_path = os.path.join(root_dir, "dataset.csv")
        self.vocab_path = os.path.join(root_dir, "vocab.json")
        self.video_dir = os.path.join(root_dir, "videos_phoenix", "videos")
        self.processed_video_dir = os.path.join(root_dir, "processed_videos")
        self.device = device
        
        assert os.path.exists(self.dataset_path)
        assert os.path.exists(self.vocab_path)
        assert os.path.exists(self.video_dir)
        
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
        self.transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(self.normalize_color),
                Normalize(mean, std),
                Resize((256, 256)),
                RandomCrop(target_size),
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
        
        # Load video from path and transpose time and batch dimensions
        # assert os.path.exists(path)
        # video: EncodedVideo = EncodedVideo.from_path(path)
        # clip_duration = video.duration
        # video_data = video.get_clip(start_sec=0, end_sec=clip_duration)["video"].transpose(0, 1)
        # video_data = self.transform(video_data)
        
        assert os.path.exists(processed_path)
        video_data = self.read_video(processed_path)
        video_data = self.transform(video_data)
        
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
        for frame in sorted(
            os.listdir(path), key=lambda p: int(p.split("_")[1].replace(".jpg", ""))
        ):
            frame = os.path.join(path, frame)

            if not frame.endswith(".jpg"):
                continue

            frames.append(read_file(frame))
        
        return torch.stack(decode_jpeg(frames), dim=0)
        
    @staticmethod
    def collate_fn(batch: list):
        videos, gloss_sequences, sentences, gloss_pad_token, word_pad_token = zip(*batch)
        gloss_pad_token = gloss_pad_token[0]
        word_pad_token = word_pad_token[0]

        videos = torch.stack(videos, dim=0)

        gloss_lengths = torch.tensor([glosses.shape[0] for glosses in gloss_sequences])
        gloss_sequences = pad_sequence(
            gloss_sequences, batch_first=True, padding_value=gloss_pad_token
        )

        sentences = pad_sequence(sentences, batch_first=True, padding_value=word_pad_token)

        return videos, gloss_sequences, gloss_lengths, sentences
