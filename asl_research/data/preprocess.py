from collections import Counter
import gzip
import os
import pickle
import pandas as pd
import cv2
import itertools

DATA_PATH = os.path.join("data", "processed", "phoenixweather2014t")
VIDEO_PATH = os.path.join(DATA_PATH, "videos_phoenix", "videos")

def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count

def video_path(set):
    return list(
        map(
            lambda x: os.path.join(*x.split("/")) + ".mp4",
            [key["name"] for key in set],
        )
    )

def preprocess_data(file_name: str, name: str):
    try:
        with gzip.open(os.path.join(DATA_PATH, file_name), "rb") as f:
            annotations = pickle.load(f)
    except:
        print("Error: Invalid path")
        return

    
    paths = list(
        map(
            lambda x: os.path.join(*x.split("/")) + ".mp4",
            [key["name"] for key in annotations],
        )
    )

    glosses = [key["gloss"] for key in annotations]
    texts = [key["text"] for key in annotations]

    data = {"paths": paths, "glosses": glosses, "texts": texts}

    df = pd.DataFrame(data)

    # Removing duplicate rows
    df = df.drop_duplicates()

    # Removing rows with missing information
    df = df.dropna()

    # Make text lowercase and glosses uppercase
    df["texts"] = df["texts"].str.lower()
    df["glosses"] = df["glosses"].str.upper()

    # Removing periods
    df["texts"] = df["texts"].str.replace(".", "")
    df["texts"] = df["texts"].str.strip()

    # Removing numbers
    df["texts"] = df["texts"].str.replace(r"\d+", "")
    df["glosses"] = df["glosses"].str.replace(r"\d+", "")

    # Filter by valid video path
    existing = (
        df["paths"].astype(str).map(lambda file: os.path.exists(os.path.join(VIDEO_PATH, file)))
    )
    df = df[existing]

    # Saving data frame as csv
    df.to_csv(os.path.join(DATA_PATH, name), index=False)

# Load dataset
with gzip.open(os.path.join(DATA_PATH, "phoenix14t.pami0.train.annotations_only.gzip"), "rb") as f:
    train = pickle.load(f)

with gzip.open(os.path.join(DATA_PATH, "phoenix14t.pami0.dev.annotations_only.gzip"), "rb") as f:
    dev = pickle.load(f)

with gzip.open(os.path.join(DATA_PATH, "phoenix14t.pami0.test.annotations_only.gzip"), "rb") as f:
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

paths = (video_path(train) + video_path(test) + video_path(dev))
df = pd.DataFrame({"paths": paths, "glosses": glosses, "texts": texts})

# Creating new columns for the number of frames
glosses_length = list(map(lambda x: len(x.split()), list(df['glosses'])))
sentences_length = list(map(lambda x: len(x.split()), list(df['texts'])))
frames = list(map(lambda x: with_opencv(x), list([os.path.join(VIDEO_PATH, path) for path in list(df['paths'])])))

df['sentences_length'] = sentences_length
df['glosses_length'] = glosses_length
df['frames'] = frames

# Removing duplicate rows
df = df.drop_duplicates()

# Removing rows with missing information
df = df.dropna()

# Make text lowercase and glosses uppercase
df["texts"] = df["texts"].str.lower()
df["glosses"] = df["glosses"].str.upper()

# Removing periods
df["texts"] = df["texts"].str.replace(".", "")
df["texts"] = df["texts"].str.strip()

# Removing sequences with plus in it
df = df.loc[~df["glosses"].str.contains(r'[\+]+')]

# Removing numbers
df["texts"] = df["texts"].str.replace(r"\d+", "")
df["glosses"] = df["glosses"].str.replace(r"\d+", "")

# Removing invalid video paths
df = df.loc[df["paths"].astype(str).map(lambda file: os.path.exists(os.path.join(VIDEO_PATH, file)))]
gloss_count = Counter(itertools.chain.from_iterable([sequence.split() for sequence in glosses]))

df.to_csv(os.path.join(DATA_PATH, 'dataset.csv'), index=False)

# Preprocess the validation, training and testing sets
# preprocess_data("phoenix14t.pami0.train.annotations_only.gzip", "train.csv")
# preprocess_data("phoenix14t.pami0.test.annotations_only.gzip", "test.csv")
# preprocess_data("phoenix14t.pami0.dev.annotations_only.gzip", "dev.csv")
