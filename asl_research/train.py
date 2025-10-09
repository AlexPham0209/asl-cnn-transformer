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
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

PROCESSED_PATH = os.path.join("data", "processed", "phoenixweather2014t")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: ASLModel,
        vocab: tuple[dict, dict, dict, dict],
        train_dl: DataLoader,
        valid_dl: DataLoader,
        test_dl: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        early_stopping: EarlyStopping,
        training_config: dict,
        gpu_id: int,
    ):
        self.model = model.to(gpu_id)
        self.gpu_id = gpu_id
        self.training_config = training_config

        # Vocab
        self.gloss_to_idx, self.idx_to_gloss, self.word_to_idx, self.idx_to_word = vocab

        # Saving dataloaders
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

        # Config and training settings
        self.best_loss = torch.inf
        self.train_loss_history = []
        self.valid_loss_history = []

        self.epochs = training_config["epochs"]
        self.curr_epoch = 1

        self.save_path = training_config["save_path"]
        self.load_path = training_config["load_path"]
        self.file_name = training_config["file_name"]
        self.diagram_path = training_config["diagram_path"]

        # Set up loss weights
        self.recognition_weight = training_config["recognition_weight"]
        self.translation_weight = training_config["translation_weight"]

        self.max_len = training_config["max_len"]

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        # Load checkpoint
        self._load_checkpoint()

        # Convert model into DistributedDataParallel model using GPU {gpu_id}
        self.model = DistributedDataParallel(model, device_ids=[gpu_id])

        # Creating the losses used for recognition and translation
        self.ctc_loss = nn.CTCLoss(blank=self.gloss_to_idx["-"]).to(gpu_id)
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(gpu_id)
        
    def train(self):
        valid_recognition_loss, valid_translation_loss, valid_loss, valid_gloss_wer, valid_sentence_wer = self._validate()
        if self.gpu_id == 0:
            print(f"Starting Average Gloss Loss: {valid_recognition_loss:>8f}", end=" - ")
            print(f"Starting Average Sentence Loss: {valid_translation_loss:>8f}", end=" - ")
            print(f"Starting Average Loss: {valid_loss:>8f}", end = " - ")
            print(f"Starting Gloss WER: {valid_gloss_wer:>8f}", end = " - ")
            print(f"Starting Sentence WER: {valid_sentence_wer:>8f}\n")

        for epoch in range(self.curr_epoch, self.epochs + 1):
            start_time = time.time()
            train_recognition_loss, train_translation_loss, train_loss = self._train_epoch(epoch)
            valid_recognition_loss, valid_translation_loss, valid_loss, valid_gloss_wer, valid_sentence_wer  = self._validate()

            # Saving model
            self._save_checkpoint(epoch, valid_loss)

            # Only print out diagnostic messages
            if self.gpu_id == 0:
                total_time = time.time() - start_time

                # Adding to training and validation history
                self.train_loss_history.append(train_loss)
                self.valid_loss_history.append(valid_loss)

                # Showing metrics
                print(f"\nEpoch Time: {total_time:.1f} seconds")
                print(f"Training Average Gloss Loss: {train_recognition_loss:>8f}", end=" - ")
                print(f"Training Average Sentence Loss: {train_translation_loss:>8f}", end=" - ")
                print(f"Training Average Loss: {train_loss:>8f}")

                print(f"Valid Average Gloss Loss: {valid_recognition_loss:>8f}", end=" - ")
                print(f"Valid Average Sentence Loss: {valid_translation_loss:>8f}", end=" - ")
                print(f"Valid Average Loss: {valid_loss:>8f}", end = " - ")
                print(f"Valid Gloss WER: {valid_gloss_wer:>8f}", end = " - ")
                print(f"Valid Sentence WER: {valid_sentence_wer:>8f}\n")
                
            # Step scheduler and early stopping
            self.scheduler.step(valid_loss)
            if self.early_stopping.early_stop(valid_loss):
                print("Early stopping")
                break

    def _train_epoch(self, epoch: int):
        self.model.train()
        losses = 0.0
        recognition_losses = 0.0
        translation_losses = 0.0
        dl = self.train_dl if self.gpu_id != 0 else tqdm(self.train_dl, desc=f"Epoch {epoch}")

        for videos, glosses, gloss_lengths, sentences in dl:
            videos = videos.to(self.gpu_id)
            glosses = glosses.to(self.gpu_id)
            gloss_lengths = gloss_lengths.to(self.gpu_id)
            sentences = sentences.to(self.gpu_id)

            self.optimizer.zero_grad()

            encoder_out, decoder_out = self.model(videos, sentences[:, :-1])
            
            # Encoder loss
            encoder_out = log_softmax(encoder_out.permute(1, 0, 2), dim=-1)
            T, N, _ = encoder_out.shape
            input_lengths = torch.full(size=(N,), fill_value=T)
            recognition_loss = self.ctc_loss(encoder_out, glosses, input_lengths, gloss_lengths)

            # Decoder loss
            actual = decoder_out.reshape(-1, decoder_out.shape[-1])
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = self.cross_entropy_loss(actual, expected)
            
            # Calculating the joint loss
            recognition_losses += recognition_loss.item()
            translation_losses += translation_loss.item()
            loss = (
                self.recognition_weight * recognition_loss
                + self.translation_weight * translation_loss
            )
            losses += loss.item()
            
            loss.backward()
            self.optimizer.step()

        return (
            recognition_losses / len(self.train_dl),
            translation_losses / len(self.train_dl),
            losses / len(self.train_dl),
        )

    def _validate(self):
        self.model.eval()

        losses = 0.0
        recognition_losses = 0.0
        translation_losses = 0.0

        actual_sentences = []
        predicted_sentences = []

        actual_glosses = []
        predicted_glosses = []

        dl = (
            self.valid_dl
            if self.gpu_id != 0
            else tqdm(self.valid_dl, desc=f"Validating")
        )

        for videos, glosses, gloss_lengths, sentences, sentence_lengths in dl:
            videos = videos.to(self.gpu_id)
            glosses = glosses.to(self.gpu_id)
            gloss_lengths = gloss_lengths.to(self.gpu_id)
            sentences = sentences.to(self.gpu_id)
            sentence_lengths = sentence_lengths.to(self.gpu_id)

            with torch.no_grad():
                encoder_out, decoder_out = self.model.module.greedy_decode(videos, max_len=torch.max(sentence_lengths).item())
                
            # # Convert output tensors into strings
            actual_gloss = decode_glosses(glosses.tolist(), self.gloss_to_idx, self.idx_to_gloss)
            predicted_gloss = decode_glosses(encoder_out, self.gloss_to_idx, self.idx_to_gloss)
            
            actual_sentence = decode_sentences(
                sentences.tolist(), self.word_to_idx, self.idx_to_word
            )
            predicted_sentence = decode_sentences(
                decoder_out.tolist(), self.word_to_idx, self.idx_to_word
            )
            
            # Add to collection of sentences and glosses for WER calculation
            actual_glosses.extend(actual_gloss)
            predicted_glosses.extend(predicted_gloss)

            actual_sentences.extend(actual_sentence)
            predicted_sentences.extend(predicted_sentence)

            with torch.no_grad():
                encoder_out, decoder_out = self.model(videos, sentences[:, :-1])

            # Encoder loss
            encoder_out = log_softmax(encoder_out.permute(1, 0, 2), dim=-1)
            T, N, _ = encoder_out.shape
            input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
            recognition_loss = self.ctc_loss(encoder_out, glosses, input_lengths, gloss_lengths)
            
            # Decoder loss
            actual = decoder_out.reshape(-1, decoder_out.shape[-1])
            expected = sentences[:, 1:].reshape(-1)
            translation_loss = self.cross_entropy_loss(actual, expected)
            
            # Calculating the joint loss
            recognition_losses += recognition_loss.item()
            translation_losses += translation_loss.item()
            loss = (
                self.recognition_weight * recognition_loss
                + self.translation_weight * translation_loss
            )
            losses += loss.item()
        
        return (
            recognition_losses / len(self.valid_dl),
            translation_losses / len(self.valid_dl),
            losses / len(self.valid_dl),
            word_error_rate(predicted_glosses, actual_glosses),
            word_error_rate(predicted_sentences, actual_sentences),
        )

    def _load_checkpoint(self):
        if len(self.load_path) <= 0:
            return

        assert os.path.exists(self.load_path), "Load path doesn't exist"
        print("Loading checkpoint...")
        checkpoint = torch.load(self.load_path, weights_only=False)
        self.curr_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint["best_loss"]
        self.train_loss_history = checkpoint["train_loss_history"]
        self.valid_loss_history = checkpoint["valid_loss_history"]

    def _save_checkpoint(self, epoch: int, valid_loss: float):
        if valid_loss > self.best_loss or self.gpu_id != 0:
            return
        
        self.best_loss = valid_loss
        print("\nNew best model, saving...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "train_loss_history": self.train_loss_history,
                "valid_loss_history": self.valid_loss_history,
            },
            os.path.join(self.save_path, f"{self.file_name}.pt"),
        )
    
    def _save_diagrams(self):
        if self.gpu_id != 0:
            return

        assert os.path.exists(self.diagram_path), "Diagram path doesn't exist"
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

        plt.locator_params(axis="x", integer=True, tight=True)
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.valid_loss_history, label="valid")
        plt.legend(["train", "valid"], loc="upper left")

        plt.savefig(os.path.join(self.diagram_path, "figure.png"))


def create_dataloaders(path: str, training_config: dict):
    # Splitting dataset into training, validation, and testing sets
    df = pd.read_csv(os.path.join(path, "dataset.csv"))
    train, test = train_test_split(df, test_size=0.2)
    test, valid = train_test_split(df, test_size=0.5)

    train_set = PhoenixDataset(
        df=train,
        root_dir=PROCESSED_PATH,
        num_frames=training_config["num_frames"],
        target_size=(224, 224),
    )

    valid_set = PhoenixDataset(
        df=valid,
        root_dir=PROCESSED_PATH,
        num_frames=training_config["num_frames"],
        target_size=(224, 224),
    )

    test_set = PhoenixDataset(
        df=test,
        root_dir=PROCESSED_PATH,
        num_frames=training_config["num_frames"],
        target_size=(224, 224),
    )

    # Creating dataloaders for each subset
    train_dl = DataLoader(
        train_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        collate_fn=PhoenixDataset.collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(train_set),
    )
    valid_dl = DataLoader(
        valid_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        collate_fn=PhoenixDataset.collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(valid_set),
    )
    test_dl = DataLoader(
        test_set,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        collate_fn=PhoenixDataset.collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(test_set),
    )

    return train_set.get_vocab(), train_dl, valid_dl, test_dl


def start_training(rank: int, world_size: int, config: dict):
    ddp_setup(rank, world_size)
    model_config = config["model"]
    training_config = config["training"]

    vocab, train_dl, valid_dl, test_dl = create_dataloaders(PROCESSED_PATH, training_config)
    gloss_to_idx, idx_to_gloss, word_to_idx, idx_to_word = vocab
    
    assert "-" in gloss_to_idx
    assert "<sos>" in word_to_idx
    assert "<eos>" in word_to_idx

    assert "<pad>" in gloss_to_idx
    assert "<pad>" in word_to_idx

    # Creating the model
    model = ASLModel(
        num_encoders=model_config["num_encoders"],
        num_decoders=model_config["num_decoders"],
        pretrained_embedding=model_config["pretrained_embedding"],
        gloss_to_idx=gloss_to_idx,
        idx_to_gloss=idx_to_gloss,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
    )

    # Creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    early_stopping = EarlyStopping(
        patience=training_config["patience"], delta=training_config["delta"]
    )

    trainer = Trainer(
        model=model,
        vocab=vocab,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        training_config=training_config,
        gpu_id=rank,
    )

    trainer.train()
    destroy_process_group()


def main():
    with open(os.path.join(CONFIG_PATH, "model.yaml"), "r") as file:
        config = yaml.safe_load(file)

    world_size = torch.cuda.device_count()
    print(f"GPU count: {world_size}")

    assert world_size > 0, "Not enough GPUs (Need more than 1)"
    mp.spawn(start_training, args=(world_size, config), nprocs=world_size)


if __name__ == "__main__":
    main()
