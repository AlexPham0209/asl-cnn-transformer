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


# Load dataset
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
df = pd.DataFrame({"paths": paths, "glosses": glosses, "texts": texts})

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
df = df.loc[~df["glosses"].str.contains(r"[\+]+")]

# Removing numbers
df["texts"] = df["texts"].str.replace(r"\d+", "")
df["glosses"] = df["glosses"].str.replace(r"\d+", "")

# Removing invalid video paths
df = df.loc[
    df["paths"].astype(str).map(lambda file: os.path.exists(os.path.join(VIDEO_PATH, file)))
]

# Creating new columns for the number of frames
frames = list(
    map(
        lambda x: with_opencv(x),
        list([os.path.join(VIDEO_PATH, path) for path in list(df["paths"])]),
    )
)
df["frames"] = frames

# Filter outliers outside of 3 standard deviations
df = df[np.abs(stats.zscore(df["frames"])) < 3]

# Create vocabulary for gloss sequences and words
gloss_count = Counter(
    itertools.chain.from_iterable([sequence.split() for sequence in df["glosses"]])
)
word_count = Counter(itertools.chain.from_iterable([sentence.split() for sentence in df["texts"]]))

gloss_list = ["-", "<pad>"] + sorted(
    gloss_count.keys(), key=lambda x: gloss_count[x], reverse=True
)
word_list = ["<sos>", "<eos>", "<pad>"] + sorted(
    word_count.keys(), key=lambda x: word_count[x], reverse=True
)

vocab = {"glosses": gloss_list, "words": word_list}

# Saving vocab and dataset
with open(os.path.join(PROCESSED_PATH, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=4)

df.to_csv(os.path.join(PROCESSED_PATH, "dataset.csv"), index=False)
