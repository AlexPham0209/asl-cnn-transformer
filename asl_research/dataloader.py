import os
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision.transforms as transforms
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
    Normalize,
    RandomRotation,
    RandomCrop,
    ColorJitter,
)

from torchvision.transforms.v2 import (
    UniformTemporalSubsample
)

import cv2
import pandas as pd
from tqdm import tqdm

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]


class PhoenixDataset(Dataset):
    def __init__(
        self,
        paths: list,
        glosses: list,
        texts: list,
        gloss_vocab: dict,
        text_vocab: dict,
        target_size: tuple = (224, 224),
    ):
        super().__init__()
        self.paths = paths
        self.glosses = glosses
        self.texts = texts

        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

        transform = Compose([
            UniformTemporalSubsample(30),
            Resize((256, 256)),
            RandomCrop((224, 224)),
            Lambda(lambda x: x/255.0),
            Normalize(mean, std),
        ])


    def __len__(self):
        return len(self.glosses)

    def __getitem__(self, index):
        path = self.paths[index]

        video: EncodedVideo = EncodedVideo.from_path(path)
        clip_duration = video.duration
        video_data = video.get_clip(start_sec=0, end_sec=clip_duration)["video"].transpose(0, 1)
        video_data = self.transform(video_data)
        
        return video_data
        


videos = []
train = pd.read_csv("data\\processed\\phoenixweather2014t\\train.csv")
paths = list(train["paths"])


