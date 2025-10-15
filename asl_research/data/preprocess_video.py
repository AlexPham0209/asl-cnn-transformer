import os
from matplotlib import rc_file
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.io import read_image, read_file, decode_jpeg
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DATA_PATH = os.path.join("data", "processed")
PROCESSED_VIDEO_PATH = os.path.join(PROCESSED_DATA_PATH, "phoenixweather2014t", "processed_videos")
FEATURES_PATH = os.path.join(PROCESSED_DATA_PATH, "features")

# Using pretrained model
conv = nn.Sequential(*list(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).children())[:-1]).to(DEVICE)
for param in conv.parameters():
    param.requires_grad = False
conv.eval()

def process_features(name):
    path = os.path.join(PROCESSED_VIDEO_PATH, name)
    frames = []
    frame_names = sorted(
        os.listdir(path), key=lambda p: int(p.split("_")[1].replace(".jpg", ""))
    )

    for frame in frame_names:
        frame = os.path.join(path, frame)
        frames.append(read_file(frame))

    x = torch.stack(decode_jpeg(frames, device=DEVICE), dim=0)

    T, C, H, W = x.shape
    x = conv(x).to(x.device)

    # Reshaping the output of the convolution
    return x


try:
    os.mkdir(FEATURES_PATH)
    print(f"Directory '{os.path.basename(FEATURES_PATH)}' created successfully.")
except FileExistsError:
    print(f"Directory '{os.path.basename(FEATURES_PATH)}' already exists.")


for video in os.listdir(PROCESSED_VIDEO_PATH):
    print(process_features(video).shape)