import os
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision.transforms as transforms
import cv2

class PhoenixDataset(Dataset):
    def __init__(
        self,
        paths: list,
        glosses: list,
        texts: list,
        gloss_vocab: dict,
        text_vocab: dict,
        target_size: tuple = (224, 224)
    ):
        super().__init__()
        self.paths = paths
        self.glosses = glosses
        self.texts = texts
        
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

        self.video_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(target_size)]
        )

        self.augmentation = transforms.Compose([])

    def load_video(self, file_path: str = ""):
        cap = cv2.VideoCapture(file_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.video_transform(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        cap.release()

        return frames
    
    def __len__(self):
        return len(self.glosses)
    
    def __getitem__(self, index):
        pass
