from collections import Counter
import gzip
import itertools
import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from scipy import stats

EXTERNAL_PATH = os.path.join("data", "external", "phoenixweather2014t")
PROCESSED_PATH = os.path.join("data", "processed", "phoenixweather2014t")
VIDEO_PATH = os.path.join(EXTERNAL_PATH, "videos_phoenix", "videos")
PROCESSED_VIDEO_PATH = os.path.join(PROCESSED_PATH, "features", "videos")
LANDMARKS_PATH = os.path.join(PROCESSED_PATH, "features", "landmarks")

def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count


def convert_to_frames(video_folder, path):
    name = os.path.basename(path).split(".")[0]
    folder_path = os.path.join(video_folder, name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        return os.path.basename(folder_path), folder_path
        
    
    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0
    
    while success:
        cv2.imwrite(os.path.join(folder_path, f"frame_{count}.jpg"), image)
        # save frame as JPEG file
        success, image = video.read()
        count += 1

    return os.path.basename(folder_path), folder_path
    

def video_path(set):
    return list(
        map(
            lambda x: os.path.join(*x.split("/")) + ".mp4",
            [key["name"] for key in set],
        )
    )


def create_dataset(paths, glosses, texts, name):
    df = pd.DataFrame({"path": paths, "gloss": glosses, "text": texts})
        
    # Removing duplicate rows
    print("Removing duplicates and dropping missing information...")
    df = df.drop_duplicates()
    
    # Removing rows with missing information
    df = df.dropna()

    # Make text lowercase and glosses uppercase
    print("Preprocessing text and glosses...")
    df["text"] = df["text"].str.lower()
    df["gloss"] = df["gloss"].str.upper()

    # Removing periods
    df["text"] = df["text"].str.replace(".", "")
    df["text"] = df["text"].str.strip()

    # Removing sequences with plus in it
    df = df.loc[~df["gloss"].str.contains(r"[\+]+")]
    
    # Removing numbers
    df["text"] = df["text"].str.replace(r"\d+", "")
    df["gloss"] = df["gloss"].str.replace(r"\d+", "")

    # Removing invalid video paths
    df = df.loc[
        df["path"].astype(str).map(lambda file: os.path.exists(os.path.join(VIDEO_PATH, file)))
    ]

    # Creating new columns for the number of frames
    print("Create frame count column...")
    frames = list(
        map(
            lambda x: with_opencv(x),
            list([os.path.join(VIDEO_PATH, path) for path in list(df["path"])]),
        )
    )
    df["frames"] = frames
    
    video_folder = os.path.join(PROCESSED_VIDEO_PATH, name)
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    
    print("Splitting .mp4 into JPEG frames...")
    video_paths = list(
        map(
            lambda x: convert_to_frames(video_folder, x),
            list([os.path.join(VIDEO_PATH, path) for path in list(df["path"])]),
        )
    )

    ids, video_paths = zip(*video_paths)

    df["id"] = ids
    df["video_path"] = video_paths

    landmarks_folder = os.path.join(LANDMARKS_PATH, name)
    if not os.path.exists(landmarks_folder):
        os.mkdir(landmarks_folder)
    df["landmark_path"] = df["id"].map(lambda id: os.path.join(landmarks_folder, f"{id}.npy"))
    
    # Filter outliers outside of 3 standard deviations
    # df = df[np.abs(stats.zscore(df["frames"])) < 3]

    df.to_csv(os.path.join(PROCESSED_PATH, f"{name}.csv"), index=False)

def create_vocab(glosses, texts):
    df = pd.DataFrame({"gloss": glosses, "text": texts})
        
    # Removing duplicate rows
    print("Removing duplicates and dropping missing information...")
    df = df.drop_duplicates()
    
    # Removing rows with missing information
    df = df.dropna()

    # Make text lowercase and glosses uppercase
    print("Preprocessing text and glosses...")
    df["text"] = df["text"].str.lower()
    df["gloss"] = df["gloss"].str.upper()

    # Removing periods
    df["text"] = df["text"].str.replace(".", "")
    df["text"] = df["text"].str.strip()

    # Removing sequences with plus in it
    df = df.loc[~df["gloss"].str.contains(r"[\+]+")]
    
    # Removing numbers
    df["text"] = df["text"].str.replace(r"\d+", "")
    df["gloss"] = df["gloss"].str.replace(r"\d+", "")
    
    # Filter outliers outside of 3 standard deviations
    # df = df[np.abs(stats.zscore(df["frames"])) < 3]
    
    print("Creating vocabulary...")
    # Create vocabulary for gloss sequences and words
    gloss_count = Counter(
        itertools.chain.from_iterable([sequence.split() for sequence in df["gloss"]])
    )
    word_count = Counter(
        itertools.chain.from_iterable([sentence.split() for sentence in df["text"]])
    )

    gloss_list = ["-", "<pad>"] + sorted(
        gloss_count.keys(), key=lambda x: gloss_count[x], reverse=True
    )
    word_list = ["<sos>", "<eos>", "<pad>"] + sorted(
        word_count.keys(), key=lambda x: word_count[x], reverse=True
    )

    vocab = {"glosses": gloss_list, "words": word_list}

    print("Saving vocabulary and data frame...")
    # Saving vocab and dataset
    with open(os.path.join(PROCESSED_PATH, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=4)

    
def main():
    try:
        os.mkdir(PROCESSED_VIDEO_PATH)
        print(f"Directory '{os.path.basename(PROCESSED_VIDEO_PATH)}' created successfully.")
    except FileExistsError:
        print(f"Directory '{os.path.basename(PROCESSED_VIDEO_PATH)}' already exists.")

    # Load dataset
    print("Loading dataset...")
    with gzip.open(
        os.path.join(EXTERNAL_PATH, "phoenix14t.pami0.train.annotations_only.gzip"), "rb"
    ) as f:
        train = pickle.load(f)

    with gzip.open(
        os.path.join(EXTERNAL_PATH, "phoenix14t.pami0.dev.annotations_only.gzip"), "rb"
    ) as f:
        dev = pickle.load(f)

    with gzip.open(
        os.path.join(EXTERNAL_PATH, "phoenix14t.pami0.test.annotations_only.gzip"), "rb"
    ) as f:
        test = pickle.load(f)

    # Getting gloss sequences and sentences from all samples
    glosses = (
        [key["gloss"].upper().strip() for key in train]
        + [key["gloss"].upper().strip() for key in test]
        + [key["gloss"].upper().strip() for key in dev]
    )
    texts = (
        [key["text"].lower().replace(".", "").strip() for key in train]
        + [key["text"].lower().replace(".", "").strip() for key in test]
        + [key["text"].lower().replace(".", "").strip() for key in dev]
    )

    paths = video_path(train) + video_path(test) + video_path(dev)
    df = pd.DataFrame({"path": paths, "gloss": glosses, "text": texts})
    
    if not os.path.exists(PROCESSED_VIDEO_PATH):
        os.mkdir(PROCESSED_VIDEO_PATH)

    create_dataset(
        video_path(train),
        [key["gloss"].upper().strip() for key in train],
        [key["text"].lower().replace(".", "").strip() for key in train],
        "train",
    )
    
    create_dataset(
        video_path(dev),
        [key["gloss"].upper().strip() for key in dev],
        [key["text"].lower().replace(".", "").strip() for key in dev],
        "dev",
    )

    create_dataset(
        video_path(test),
        [key["gloss"].upper().strip() for key in test],
        [key["text"].lower().replace(".", "").strip() for key in test],
        "test",
    )

    create_vocab(glosses, texts)


if __name__ == "__main__":
    main()
