import os
import time
from models.dataloader import load_data
import matplotlib.pyplot as plt
from torch.nn.functional import log_softmax, softmax
import torch
from torch import nn
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

def train(train_dl, test_dl, valid_dl, char_to_idx, idx_to_char, config):
    model = CaptchaDetectionModel(len(char_to_idx), device=DEVICE).to(DEVICE)
    criterion = nn.CTCLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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
        train_loss = train_epoch(model, train_dl, optimizer, criterion, epoch)
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
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.locator_params(axis="x", integer=True, tight=True)
    plt.plot(train_loss_history, label='train')
    plt.plot(valid_loss_history, label='valid')
    plt.legend(['train', 'valid'], loc='upper left')
    
    plt.show()
    
            
def train_epoch(model, data, optimizer, criterion, epoch=10):
    model.train()
    losses = 0.0

    for image, label in tqdm(data, desc=f"Epoch {epoch}"):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        
        # Should output a 3d tensor of size: (T, N, num_chars)
        out = model(image).to(DEVICE)
        out = log_softmax(out, dim=-1)
        T, N, C = out.shape

        input_lengths = torch.full(size=(N,), fill_value=T).to(DEVICE)
        target_lengths = torch.full(size=(N,), fill_value=label.shape[-1]).to(DEVICE)

        loss = criterion(out, label, input_lengths, target_lengths)
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
    with open(os.path.join(CONFIG_PATH, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    training, test, valid, idx_to_char, char_to_idx = load_data(batch_size=config["training"]["batch_size"])
    train(training, test, valid, char_to_idx, idx_to_char, config["training"])