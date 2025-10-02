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

torch.cuda.empty_cache()


def train(config: dict):
    model_config = config["model"]
    training_config = config["training"]

    # Creating dataset and getting gloss and word vocabulary dictionaries
    dataset = PhoenixDataset(
        root_dir="data\\processed\\phoenixweather2014t",
        num_frames=training_config["num_frames"],
        target_size=(224, 224),
    )

    gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = dataset.get_vocab()

    # Splitting dataset into training, validation, and testing sets
    generator = torch.Generator().manual_seed(training_config["seed"])
    train_set, valid_set, test_set = random_split(
        dataset=dataset, lengths=training_config["split"], generator=generator
    )

    train_dl = DataLoader(
        train_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=True,
        collate_fn=PhoenixDataset.collate_fn,
    )
    valid_dl = DataLoader(
        valid_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=True,
        collate_fn=PhoenixDataset.collate_fn,
    )
    test_dl = DataLoader(
        test_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=True,
        collate_fn=PhoenixDataset.collate_fn,
    )

    # Creating the model
    assert "<pad>" in gloss_to_idx
    assert "<pad>" in word_to_idx

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

    ctc_loss = nn.CTCLoss(blank=gloss_to_idx["-"]).to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    early_stopping = EarlyStopping(patience=3, delta=0.05)

    best_loss = torch.inf
    train_loss_history = []
    valid_loss_history = []

    epochs = training_config["epochs"]
    save_path = training_config["save_path"]
    load_path = training_config["load_path"]
    file_name = training_config["file_name"]

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

    valid_loss, valid_wer = validate(
        model,
        valid_dl,
        ctc_loss,
        cross_entropy_loss,
        gloss_to_idx,
        idx_to_gloss,
        word_to_idx,
        idx_to_word,
        training_config["train_recognition"],
        training_config["train_translation"],
    )
    print(f"Valid Average loss: {valid_loss:>8f}")
    print(f"Valid Word Error Rate: {valid_wer:>8f}\n")
    torch.cuda.empty_cache()

    for epoch in range(curr_epoch, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(
            model,
            train_dl,
            optimizer,
            ctc_loss,
            cross_entropy_loss,
            epoch,
            training_config["train_recognition"],
            training_config["train_translation"],
        )

        valid_loss, valid_wer = validate(
            model,
            valid_dl,
            ctc_loss,
            cross_entropy_loss,
            gloss_to_idx,
            idx_to_gloss,
            word_to_idx,
            idx_to_word,
            training_config["train_recognition"],
            training_config["train_translation"],
        )

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
                    "best_loss": best_loss,
                    "train_loss_history": train_loss_history,
                    "valid_loss_history": valid_loss_history,
                },
                os.path.join(save_path, f"{file_name}.pt"),
            )

        total_time = time.time() - start_time
        print(f"\nEpoch Time: {total_time:.1f} seconds")
        print(f"Training Average loss: {train_loss:>8f}")
        print(f"Valid Average loss: {valid_loss:>8f}")
        print(f"Valid Word Error Rate: {valid_wer:>8f}\n")

        scheduler.step(valid_loss)
        if early_stopping.early_stop(valid_loss):
            print("Early stopping")
            break

        torch.cuda.empty_cache()

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
    epoch: int = 1,
    train_recognition: bool = True,
    train_translation: bool = True,
):
    model.train()
    losses = 0.0

    for videos, glosses, gloss_lengths, sentences in tqdm(data, desc=f"Epoch {epoch}"):
        videos = videos.to(DEVICE)
        glosses = glosses.to(DEVICE)
        gloss_lengths = gloss_lengths.to(DEVICE)
        sentences = sentences.to(DEVICE)

        optimizer.zero_grad()

        # Should outpust the encoder output
        # encoder_out: (batch_size, gloss_sequence_length, gloss_vocab_size)
        # decoder_out: (batch_size, sentence_length, word_vocab_size)
        encoder_out, decoder_out = model(videos, sentences[:, :-1])

        # Encoder loss
        recognition_loss = 0.0
        if train_recognition:
            encoder_out = log_softmax(encoder_out.permute(1, 0, 2), dim=-1)
            T, N, _ = encoder_out.shape
            input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
            recognition_loss = ctc_loss(encoder_out, glosses, input_lengths, gloss_lengths)
        
        # Decoder loss
        translation_loss = 0.0
        if train_translation:
            actual = decoder_out.reshape(-1, decoder_out.shape[-1])
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = cross_entropy_loss(actual, expected)

        # Calculating the joint loss
        loss = recognition_loss + translation_loss
        losses += loss.item()

        loss.backward()
        optimizer.step()

    return losses / len(data)


def validate(
    model: ASLModel,
    data: DataLoader,
    ctc_loss: nn.CTCLoss,
    cross_entropy_loss: nn.CrossEntropyLoss,
    gloss_to_idx: dict,
    idx_to_gloss: dict,
    word_to_idx: dict,
    idx_to_word: dict,
    validate_recognition: bool = True,
    validate_translation: bool = True,
):
    model.eval()

    remove_special_tokens = (
        lambda token: token != word_to_idx["<pad>"]
        and token != word_to_idx["<eos>"]
        and token != word_to_idx["<sos>"]
    )
    losses = 0.0

    actual_sentences = list()
    predicted_sentences = list()

    actual_glosses = list()
    predicted_glosses = list()

    for videos, glosses, gloss_lengths, sentences in tqdm(data, desc=f"Validating"):
        videos = videos.to(DEVICE)
        glosses = glosses.to(DEVICE)
        gloss_lengths = gloss_lengths.to(DEVICE)
        sentences = sentences.to(DEVICE)

        with torch.no_grad():
            encoder_out, decoder_out = model.greedy_decode(videos, max_len=30)

        # actual_sentence = [
        #     " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        #     for sample in sentences.tolist()
        # ]
        # predicted_sentence = [
        #     " ".join([idx_to_word[token] for token in list(filter(remove_special_tokens, sample))])
        #     for sample in decoder_out.tolist()
        # ]

        actual_gloss = [
            " ".join(
                [
                    idx_to_gloss[token]
                    for token in list(filter(lambda x: x != gloss_to_idx["<pad>"], sample))
                ]
            )
            for sample in glosses.tolist()
        ]

        predicted_gloss = [
            " ".join(
                [
                    idx_to_gloss[token]
                    for token in list(filter(lambda x: x != gloss_to_idx["<pad>"], sample))
                ]
            )
            for sample in encoder_out
        ]

        # actual_sentences.extend(actual_sentence)
        # predicted_sentences.extend(predicted_sentence)

        actual_glosses.extend(actual_gloss)
        predicted_glosses.extend(predicted_gloss)

        # Should outpust the encoder output
        # encoder_out: (batch_size, gloss_sequence_length, gloss_vocab_size)
        # decoder_out: (batch_size, sentence_length, word_vocab_size)
        with torch.no_grad():
            encoder_out, decoder_out = model(videos, sentences[:, :-1])

        # Encoder loss
        recognition_loss = 0.0
        if validate_recognition:
            encoder_out = log_softmax(encoder_out.transpose(0, 1), dim=-1)
            T, N, _ = encoder_out.shape
            input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
            recognition_loss = ctc_loss(encoder_out, glosses, input_lengths, gloss_lengths)

        # Decoder loss
        translation_loss = 0.0
        if validate_translation:
            actual = decoder_out.reshape(-1, decoder_out.shape[-1])
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = cross_entropy_loss(actual, expected)

        # Calculating the joint loss
        loss = recognition_loss + translation_loss
        losses += loss.item()

    return losses / len(data), word_error_rate(actual_glosses, predicted_glosses)


if __name__ == "__main__":
    with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
        config = yaml.safe_load(file)

    train(config)
