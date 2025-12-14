import os
from matplotlib import rc_file
import numpy as np
import torch
import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
# from torchvision.io import read_image, read_file, decode_jpeg
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import os
import contextlib
import sys

# Silence MediaPipe / TensorFlow / Abseil logs
os.environ["GLOG_minloglevel"] = "2"        # 0=INFO, 1=WARNING, 2=ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # TensorFlow C++ logs
os.environ["GRPC_VERBOSITY"] = "ERROR"

import mediapipe as mp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DATA_PATH = os.path.join("data", "processed")
EXTERNAL_DATA_PATH = os.path.join("data", "external")
FEATURES_PATH = os.path.join(PROCESSED_DATA_PATH, "phoenixweather2014t", "features")

PROCESSED_VIDEO_PATH = os.path.join(PROCESSED_DATA_PATH, "phoenixweather2014t", "processed_videos")
EXTERNAL_VIDEO_PATH = os.path.join(
    PROCESSED_DATA_PATH, "phoenixweather2014t", "videos_phoenix", "videos"
)

I3D_PATH = os.path.join(FEATURES_PATH, "i3d")
LANDMARKS_PATH = os.path.join(FEATURES_PATH, "landmarks")

# Using pretrained model



def extract_landmarks(landmarks):
    return torch.tensor(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    ).flatten()


def get_features(results):
    left_hand = (
        extract_landmarks(results.left_hand_landmarks)
        if results.left_hand_landmarks
        else torch.zeros(63)
    )
    right_hand = (
        extract_landmarks(results.right_hand_landmarks)
        if results.right_hand_landmarks
        else torch.zeros(63)
    )
    pose = extract_landmarks(results.pose_landmarks) if results.pose_landmarks else torch.zeros(99)
    feature = torch.cat((left_hand, right_hand, pose))

    return feature


def process_features(path):
    # Filter all warnings
    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0
    landmarks = []

    mp_holistic = mp.solutions.holistic
    model = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2
    )
    
    while success:
        success, image = video.read()

        if not success:
            break

        # Process model using Mediapipe's Hollistic model
        results = model.process(image)
        features = get_features(results)
        landmarks.append(features)
        count += 1

    landmarks = torch.stack(landmarks, dim=0)
    video.release()

    return landmarks, os.path.basename(path)


def process_videos(folder):
    PATH = os.path.join(LANDMARKS_PATH)

    # if not os.path.exists(PATH):
    #     os.mkdir(PATH)

    videos_path = os.path.join(EXTERNAL_VIDEO_PATH, folder)
    videos_list = [os.path.join(videos_path, video) for video in os.listdir(videos_path)]
    
    with Pool(processes=cpu_count()) as p:
        for features, name in tqdm(
            p.imap_unordered(process_features, videos_list),
            total=len(videos_list),
            desc=f"Processing {folder} folder",
        ):
            features = features.cpu().detach().numpy()
            np.save(os.path.join(PATH, f"{name}.npy"), features)


if __name__ == "__main__":
    # Creating features folder
    if not os.path.exists(FEATURES_PATH):
        os.mkdir(FEATURES_PATH)

    if not os.path.exists(LANDMARKS_PATH):
        os.mkdir(FEATURES_PATH)

    # Process train, dev, and test videos so they are matrices of landmark data
    process_videos("train")
    process_videos("dev")
    process_videos("test")

# print(
#     np.load(
#         "data\\processed\\phoenixweather2014t\\features\\landmarks\\train\\01April_2010_Thursday_heute-6694.mp4.npy"
#     ).shape
# )
