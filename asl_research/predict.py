import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from asl_research.dataloader import PhoenixDataset
from asl_research.model.model import ASLModel
from asl_research.utils.early_stopping import EarlyStopping
from torcheval.metrics.functional import word_error_rate
from torch.nn.functional import log_softmax, softmax

from asl_research.utils.utils import decode_glosses, decode_sentences, generate_padding_mask
import pandas as pd
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

PROCESSED_PATH = os.path.join("data", "processed", "phoenixweather2014t")

with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
    config = yaml.safe_load(file)

model_config = config["model"]
training_config = config["training"]

# df = pd.read_csv(os.path.join(PROCESSED_PATH, "dataset.csv"))
# train_size, valid_size, test_size = training_config["split"]
# size = valid_size + test_size
# test_size /= size
# train, test = train_test_split(df, train_size=train_size, random_state=training_config["seed"])
# test, valid = train_test_split(df, test_size=test_size, random_state=training_config["seed"])

train = pd.read_csv(os.path.join(PROCESSED_PATH, "train.csv"))
valid = pd.read_csv(os.path.join(PROCESSED_PATH, "dev.csv"))
test = pd.read_csv(os.path.join(PROCESSED_PATH, "test.csv"))

# Creating datasSet and getting gloss and word vocabulary dictionaries
dataset = PhoenixDataset(
    df=train,
    root_dir=PROCESSED_PATH,
    num_frames=training_config["num_frames"],
    target_size=(224, 224),
    is_train=False,
    random_subsampling=2,
)

gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = dataset.get_vocab()

model = ASLModel(
    num_encoders=model_config["num_encoders"],
    num_decoders=model_config["num_decoders"],
    gloss_to_idx=gloss_to_idx,
    idx_to_gloss=idx_to_gloss,
    word_to_idx=word_to_idx,
    idx_to_word=idx_to_word,
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    dropout=model_config["dropout"],
).to(DEVICE)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=True,
    collate_fn=PhoenixDataset.collate_fn,
)

load_path = training_config["load_path"]
ctc_loss = nn.CTCLoss(blank=gloss_to_idx["-"]).to(DEVICE)
cross_entropy_loss = nn.CrossEntropyLoss().to(DEVICE)


if len(load_path) > 0:
    print("Loading checkpoint...")
    checkpoint = torch.load(load_path, weights_only=False)
    curr_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    best_metric = checkpoint["best_metric"]
    train_loss_history = checkpoint["train_loss_history"]
    valid_loss_history = checkpoint["valid_loss_history"]

model.eval()
remove_special_tokens = (
    lambda token: token != word_to_idx["<pad>"]
    and token != word_to_idx["<eos>"]
    and token != word_to_idx["<sos>"]
)

for i in range(50):
    videos, video_lengths, glosses, gloss_lengths, sentences, sentence_lengths = next(
        iter(dataloader)
    )
    videos = videos.to(DEVICE)
    glosses = glosses.to(DEVICE)
    gloss_lengths = gloss_lengths.to(DEVICE)
    sentences = sentences.to(DEVICE)
    video_lengths = video_lengths.to(DEVICE)

    encoder_out, decoder_out = model.greedy_decode(videos, video_lengths, max_len=30)

    actual_gloss = decode_glosses(glosses.tolist(), gloss_to_idx, idx_to_gloss)
    predicted_gloss = decode_glosses(encoder_out, gloss_to_idx, idx_to_gloss)

    actual_sentence = decode_sentences(sentences.tolist(), word_to_idx, idx_to_word)
    predicted_sentence = decode_sentences(decoder_out.tolist(), word_to_idx, idx_to_word)

    print(f"Actual Sentence: {actual_sentence[0]}")
    print(f"Predicted Sentence: {predicted_sentence[0]}")
    print(f"Actual Gloss: {actual_gloss[0]}")
    print(f"Predicted Gloss: {predicted_gloss[0]}")
    print()
