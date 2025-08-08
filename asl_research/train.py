import os
import time
from asl_research.dataloader import PhoenixDataset
from asl_research.model.model import ASLModel
import matplotlib.pyplot as plt
from torch.nn.functional import log_softmax, softmax
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import yaml

from asl_research.utils.early_stopping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"


def train(config):
    model_config = config["model"]
    training_config = config["train"]

    # Creating dataset and getting gloss and word vocabulary dictionaries
    dataset = PhoenixDataset(
        root_dir="data\\processed\\phoenixweather2014t", num_frames=60, target_size=(224, 224)
    )

    gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = dataset.get_vocab()

    # Splitting dataset into training, validation, and testing sets
    generator = torch.Generator().manual_seed(29)
    train_set, valid_set, test_set = random_split(
        dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator
    )

    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=16, shuffle=True)

    # Creating the model
    assert "<pad>" in gloss_to_idx
    assert "<pad>" in word_to_idx
    model = ASLModel(
        num_encoders=2,
        num_decoders=2,
        trg_vocab_size=len(word_to_idx),
        gloss_pad_token=gloss_to_idx["<pad>"],
        word_pad_token=word_to_idx["<pad>"],
        d_model=512,
        num_heads=8,
        dropout=0.1,
    ).to(DEVICE)

    ctc_loss = nn.CTCLoss().to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    early_stopping = EarlyStopping(patience=3, delta=0.1)

    best_loss = torch.inf
    train_loss_history = []
    valid_loss_history = []

    epochs = config["epochs"]
    save_path = config["save_path"]
    load_path = config["load_path"]

    curr_epoch = 1

    if len(load_path) > 0:
        print("Loading checkpoint...")
        checkpoint = torch.load(load_path, weights_only=False)
        curr_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        best_loss = checkpoint["best_loss"]
        train_loss_history = checkpoint["train_loss_history"]
        valid_loss_history = checkpoint["valid_loss_history"]

    valid_loss = validate(model, valid_dl, criterion)
    print(f"Valid Average loss: {valid_loss:>8f}\n")

    for epoch in range(curr_epoch, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, ctc_loss, cross_entropy_loss, epoch)
        valid_loss = validate(model, valid_dl, criterion)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            print("New best model, saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion": criterion,
                    "best_loss": best_loss,
                    "train_loss_history": train_loss_history,
                    "valid_loss_history": valid_loss_history,
                },
                os.path.join(save_path, "best.pt"),
            )

        total_time = time.time() - start_time
        print(f"\nEpoch Time: {total_time:.1f} seconds")
        print(f"Training Average loss: {train_loss:>8f}")
        print(f"Valid Average loss: {valid_loss:>8f}\n")

        scheduler.step(valid_loss)
        if early_stopping.early_stop(valid_loss):
            print("Early stopping")
            break

    # Plot model's loss over epochs
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.locator_params(axis="x", integer=True, tight=True)
    plt.plot(train_loss_history, label="train")
    plt.plot(valid_loss_history, label="valid")
    plt.legend(["train", "valid"], loc="upper left")

    plt.show()


def train_epoch(
    model: ASLModel,
    data: DataLoader,
    optimizer: Optimizer,
    ctc_loss: nn.CTCLoss,
    cross_entropy_loss: nn.CrossEntropyLoss,
    epoch=10,
):
    model.train()
    losses = 0.0

    for videos, glosses, gloss_lengths, sentences in tqdm(data, desc=f"Epoch {epoch}"):
        videos = videos.to(DEVICE)
        glosses = glosses.to(DEVICE)
        gloss_lengths = gloss_lengths.to(DEVICE)
        sentences = sentences.to(DEVICE)

        optimizer.zero_grad()

        # Should output two tensors 
        encoder_out, decoder_out = model(videos).to(DEVICE)

        encoder_out = log_softmax(out, dim=-1)
        T, N, C = out.shape

        input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
        target_lengths = torch.full(size=(N,), fill_value=label.shape[-1]).to(DEVICE)

        loss = ctc_loss(out, label, input_lengths, target_lengths) + cross_entropy_loss()
        losses += loss.item()

        loss.backward()
        optimizer.step()

    return losses / len(data)


def validate(model, data, criterion):
    model.eval()
    losses = 0.0

    for image, label in tqdm(data, desc="Validating"):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        # Should output a 3d tensor of size: (T, N, num_classes)
        out = model(image).to(DEVICE)
        out = log_softmax(out, dim=-1)
        T, N, C = out.shape

        input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
        target_lengths = torch.full(size=(N,), fill_value=label.shape[-1]).to(DEVICE)

        loss = criterion(out, label, input_lengths, target_lengths)
        losses += loss.item()

    return losses / len(data)


if __name__ == "__main__":
    with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
        config = yaml.safe_load(file)

    train(config)
