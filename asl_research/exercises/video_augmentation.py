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

from torchvision.transforms.v2 import UniformTemporalSubsample

import cv2
import pandas as pd
from tqdm import tqdm

FRAMES = 30

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
videos = []
train = pd.read_csv("data\\processed\\phoenixweather2014t\\train.csv")
paths = list(train["paths"])

transform = Compose(
    [
        Lambda(lambda x: x / 255.0),
        # Normalize(mean, std),
        UniformTemporalSubsample(FRAMES),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        RandomRotation(5),
        ColorJitter(0.5, 0.5, 0.1, 0.1),
        Lambda(lambda x: x * 255.0),
    ]
)

i = 0
for path in tqdm(paths[:5], desc="Loading Videos"):
    i += 1
    video: EncodedVideo = EncodedVideo.from_path(path)
    clip_duration = video.duration
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)["video"].transpose(0, 1)
    video_data = transform(video_data)

    video_data = video_data.permute(0, 2, 3, 1)
    torchvision.io.write_video(
        os.path.join("data", "test_video", f"test_video{i}.mp4"), video_data, 24
    )
