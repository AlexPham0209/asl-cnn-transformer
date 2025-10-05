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

from asl_research.utils.utils import generate_padding_mask

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
    config = yaml.safe_load(file)

model_config = config["model"]
training_config = config["training"]

# Creating dataset and getting gloss and word vocabulary dictionaries
print("tmp_ondemand_ocean_cis250077p_symlink/apham8/asl-cnn-transformer/data/processed/phoenixweather2014t")
dataset = PhoenixDataset(
    root_dir="tmp_ondemand_ocean_cis250077p_symlink/apham8/asl-cnn-transformer/data/processed/phoenixweather2014t",
    num_frames=training_config["num_frames"],
    target_size=(224, 224),
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
    criterion = checkpoint["criterion"]
    best_loss = checkpoint["best_loss"]
    train_loss_history = checkpoint["train_loss_history"]
    valid_loss_history = checkpoint["valid_loss_history"]

model.eval()
remove_special_tokens = (
    lambda token: token != word_to_idx["<pad>"]
    and token != word_to_idx["<eos>"]
    and token != word_to_idx["<sos>"]
)

for i in range(10):
    videos, glosses, gloss_lengths, sentences = next(iter(dataloader))
    videos = videos.to(DEVICE)
    glosses = glosses.to(DEVICE)
    gloss_lengths = gloss_lengths.to(DEVICE)
    sentences = sentences.to(DEVICE)

    encoder_out, decoder_out = model.greedy_decode(videos, max_len=30)
    # plt.imshow(videos[0, 15].permute(1, 2, 0).cpu())
    # plt.show()

    actual_sentence = [
        " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        for sample in sentences.tolist()
    ]

    predicted_sentence = [
        " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        for sample in decoder_out.tolist()
    ]

    actual_gloss_sequence = [
        " ".join([idx_to_gloss[token] for token in sample]) for sample in glosses.tolist()
    ]

    gloss_sequence = [
        " ".join([idx_to_gloss[token] for token in sample]) for sample in encoder_out
    ]

    print(actual_sentence[0])
    print(predicted_sentence[0])
    print(actual_gloss_sequence[0])
    print(gloss_sequence[0])
    print()
