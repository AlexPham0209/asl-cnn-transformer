import os

import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Lambda,
    RandomCrop,
    RandomRotation,
    Resize,
    ToTensor,
    Normalize,
    ToPILImage
)
from torchvision.transforms.v2 import UniformTemporalSubsample
from tqdm import tqdm
import matplotlib.pyplot as plt

FRAMES = 30

mean = [0.53724027, 0.5272855, 0.51954997]
std = [1, 1, 1]
videos = []
train = pd.read_csv("data\\processed\\phoenixweather2014t\\dataset.csv")
paths = list(train["paths"])

transform = Compose(
    [
        UniformTemporalSubsample(FRAMES),
        Lambda(lambda x: x / 255.),
        Normalize(mean, std),
        Resize((256, 256)),
        RandomCrop((224, 224)),
    ]
)

i = 0
for path in tqdm(paths[:5], desc="Loading Videos"):
    i += 1
    video: EncodedVideo = EncodedVideo.from_path(os.path.join("data\\processed\\phoenixweather2014t\\videos_phoenix\\videos", path))
    clip_duration = video.duration
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)["video"].transpose(0, 1)
    video_data = transform(video_data)

    video_data = video_data.permute(0, 2, 3, 1)
    print(video_data)
    for t in range(video_data.shape[0]):
        plt.imshow(video_data[t])
        plt.show()

    # torchvision.io.write_video(
    #     os.path.join("data", "test_video", f"test_video{i}.mp4"), video_data, 24
    # )
