import os
from matplotlib import rc_file
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.io import read_image, read_file, decode_jpeg
import cv2
import mediapipe as mp
from tqdm import tqdm

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
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2
)


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


def draw_connections(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def process_features(path):
    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0
    landmarks = []

    while success:
        success, image = video.read()

        if not success:
            break

        # Process model using Mediapipe's Hollistic model
        results = holistic_model.process(image)
        features = get_features(results)
        landmarks.append(features)
        count += 1

    landmarks = torch.stack(landmarks, dim=0)
    video.release()

    return landmarks


def process_videos(folder):
    PATH = os.path.join(LANDMARKS_PATH)
    try:
        os.mkdir(PATH)
        print(f"Directory '{os.path.basename(os.mkdir(PATH))}' created successfully.")
    except FileExistsError:
        print(f"Directory '{os.path.basename(PATH)}' already exists.")
    

    videos = os.path.join(EXTERNAL_VIDEO_PATH, folder)
    for video in tqdm(os.listdir(videos), desc=f"Processing {folder} folder"):
        video_path = os.path.join(videos, video)
        features = process_features(video_path)
        features = features.cpu().detach().numpy()
        np.save(os.path.join(PATH, f"{os.path.basename(video)}.npy"), features)


if __name__ == "__main__":
    # Creating features folder
    try:
        os.mkdir(FEATURES_PATH)
        print(f"Directory '{os.path.basename(os.mkdir(FEATURES_PATH))}' created successfully.")
    except FileExistsError:
        print(f"Directory '{os.path.basename(FEATURES_PATH)}' already exists.")

    # Creating landmark folder inside of features folder
    try:
        os.mkdir(LANDMARKS_PATH)
        print(f"Directory '{os.path.basename(os.mkdir(LANDMARKS_PATH))}' created successfully.")
    except FileExistsError:
        print(f"Directory '{os.path.basename(LANDMARKS_PATH)}' already exists.")

    # Process train, dev, and test videos so they are matrices of landmark data
    process_videos("train")
    process_videos("dev")
    process_videos("test")

# print(
#     np.load(
#         "data\\processed\\phoenixweather2014t\\features\\landmarks\\train\\01April_2010_Thursday_heute-6694.mp4.npy"
#     ).shape
# )
