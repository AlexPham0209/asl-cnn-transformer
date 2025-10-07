import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.functional import log_softmax, softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from asl_research.dataloader import PhoenixDataset
from asl_research.model.model import ASLModel
from asl_research.utils.early_stopping import EarlyStopping
from torcheval.metrics.functional import word_error_rate

from asl_research.utils.utils import decode_glosses, decode_sentences, generate_padding_mask
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

PROCESSED_PATH = os.path.join("data", "processed", "phoenixweather2014t")


def train(config: dict):
    model_config = config["model"]
    training_config = config["training"]
    # Creating dataset and getting gloss and word vocabulary dictionaries
    dataset = PhoenixDataset(
        root_dir=PROCESSED_PATH,
        num_frames=training_config["num_frames"],
        target_size=(224, 224),
        device=DEVICE,
    )

    gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = dataset.get_vocab()

    # Splitting dataset into training, validation, and testing sets
    generator = torch.Generator().manual_seed(training_config["seed"])
    train_set, valid_set, test_set = random_split(
        dataset=dataset, lengths=training_config["split"], generator=generator
    )

    # Creating dataloaders for each subset
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

    # Checking that the special tokens exists inside of the vocabularies
    assert "-" in gloss_to_idx
    assert "<sos>" in word_to_idx
    assert "<eos>" in word_to_idx

    assert "<pad>" in gloss_to_idx
    assert "<pad>" in word_to_idx

    # Creating the model
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

    # Creating the losses used for recognition and translation
    ctc_loss = nn.CTCLoss(blank=gloss_to_idx["-"]).to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss().to(DEVICE)

    # Creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    early_stopping = EarlyStopping(
        patience=training_config["patience"], delta=training_config["delta"]
    )

    # Loading hyperparameters
    best_loss = torch.inf
    train_loss_history = []
    valid_loss_history = []

    epochs = training_config["epochs"]
    save_path = training_config["save_path"]
    load_path = training_config["load_path"]
    file_name = training_config["file_name"]

    curr_epoch = 1

    if len(load_path) > 0:
        assert os.path.exists(load_path)
        print("Loading checkpoint...")
        checkpoint = torch.load(load_path, weights_only=False)
        curr_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["best_loss"]
        train_loss_history = checkpoint["train_loss_history"]
        valid_loss_history = checkpoint["valid_loss_history"]
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(DEVICE)

    # Start training
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
    # print(f"Valid Word Error Rate: {valid_wer:>8f}\n")

    for epoch in range(curr_epoch, epochs + 1):
        start_time = time.time()
        train_recognition_loss, train_translation_loss, train_loss = train_epoch(
            model,
            train_dl,
            optimizer,
            ctc_loss,
            cross_entropy_loss,
            epoch,
            training_config["train_recognition"],
            training_config["train_translation"],
        )

        (
            valid_recognition_loss,
            valid_translation_loss,
            valid_loss,
            valid_gloss_wer,
            valid_sentence_wer,
        ) = validate(
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
            print("\nNew best model, saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "train_loss_history": train_loss_history,
                    "valid_loss_history": valid_loss_history,
                },
                os.path.join(save_path, f"{file_name}.pt"),
            )

        total_time = time.time() - start_time
        print(f"\nEpoch Time: {total_time:.1f} seconds")
        print(f"Training Average Recognition Loss: {train_recognition_loss:>8f}")
        print(f"Training Average Translation Loss: {train_translation_loss:>8f}")
        print(f"Training Average Joint Loss: {train_loss:>8f}\n")

        print(f"Valid Average Recognition Loss: {valid_recognition_loss:>8f}")
        print(f"Training Average Translation Loss: {valid_translation_loss:>8f}")
        print(f"Valid Average Loss: {valid_loss:>8f}")
        print(f"Valid Gloss Word Error Rate: {valid_gloss_wer:>8f}")
        print(f"Valid Sentence Word Error Rate: {valid_sentence_wer:>8f}\n")

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
    epoch: int = 1,
    train_recognition: bool = True,
    train_translation: bool = True,
):
    model.train()
    losses = 0.0
    recognition_losses = 0.0
    translation_losses = 0.0

    for videos, glosses, gloss_lengths, sentences in tqdm(data, desc=f"Epoch {epoch}"):
        videos = videos.to(DEVICE)
        glosses = glosses.to(DEVICE)
        gloss_lengths = gloss_lengths.to(DEVICE)
        sentences = sentences.to(DEVICE)

        optimizer.zero_grad()

        # Should output the encoder output
        # encoder_out: (batch_size, gloss_sequence_length, gloss_vocab_size)
        # decoder_out: (batch_size, video_length, word_vocab_size)
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
            actual = softmax(decoder_out.reshape(-1, decoder_out.shape[-1]), dim=-1)
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = cross_entropy_loss(actual, expected)

        # Calculating the joint loss
        recognition_losses += recognition_loss
        translation_losses += translation_loss
        loss = recognition_loss + translation_loss
        losses += loss.item()

        loss.backward()
        optimizer.step()

    return recognition_losses / len(data), translation_losses / len(data), losses / len(data)


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
    recognition_losses = 0.0
    translation_losses = 0.0

    actual_sentences = []
    predicted_sentences = []

    actual_glosses = []
    predicted_glosses = []

    for videos, glosses, gloss_lengths, sentences in tqdm(data, desc=f"Validating"):
        videos = videos.to(DEVICE)
        glosses = glosses.to(DEVICE)
        gloss_lengths = gloss_lengths.to(DEVICE)
        sentences = sentences.to(DEVICE)

        encoder_out, decoder_out = model.module.greedy_decode(videos)

        # # Convert output tensors into strings
        actual_gloss = decode_glosses(glosses.tolist(), gloss_to_idx, idx_to_gloss)
        predicted_gloss = decode_glosses(encoder_out, gloss_to_idx, idx_to_gloss)

        actual_sentence = decode_sentences(sentences.tolist(), word_to_idx, idx_to_word)
        predicted_sentence = decode_sentences(decoder_out.tolist(), word_to_idx, idx_to_word)

        # Add to collection of sentences and glosses for WER calculation
        actual_glosses.extend(actual_gloss)
        predicted_glosses.extend(predicted_gloss)

        actual_sentences.extend(actual_sentence)
        predicted_sentences.extend(predicted_sentence)

        # Should outpust the encoder output
        # encoder_out: (batch_size, gloss_sequence_length, gloss_vocab_size)
        # decoder_out: (batch_size, sentence_length, word_vocab_size)
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
            actual = softmax(decoder_out.reshape(-1, decoder_out.shape[-1]))
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = cross_entropy_loss(actual, expected)

        # Calculating the joint loss
        recognition_losses += recognition_loss
        translation_losses += translation_loss
        loss = recognition_loss + translation_loss
        losses += loss.item()

    return (
        recognition_losses / len(data),
        translation_losses / len(data),
        losses / len(data),
        word_error_rate(predicted_sentences, actual_sentences),
        word_error_rate(actual_glosses, actual_glosses),
    )


def main():
    with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
        config = yaml.safe_load(file)

    train(config)


if __name__ == "__main__":
    main()
