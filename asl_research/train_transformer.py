import warnings
import torch
import tqdm
from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer
from asl_research.model.transformer import BaseTransformer
from torch.nn.modules.loss import _Loss

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def train_epoch(
    model: BaseTransformer,
    data: Dataset,
    optimizer: Optimizer,
    criterion: _Loss,
    src_vocab: dict,
    trg_vocab: dict,
    epoch: int,
):
    # Set model to training mode
    model.train()
    losses = 0

    # Go through batches in the epoch
    for src, trg in tqdm(data, desc=f"Epoch {epoch}"):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
        # instead of the model's output in the prior timestep
        out = model(
            src,
            trg_input,
        )

        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size)
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)

        # We zero the gradients of the model, cadlculate the total loss of the sample
        # Then compute the gradient vector for the model over the loss

        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence
        loss = criterion(actual, expected)
        losses += loss.item()
        loss.backward()

        # Apply the gradient vector on the trainable parameters in the model and reset the gradients
        optimizer.step()
        optimizer.zero_grad()

    losses /= len(data)
    return losses


def validate(model, data, criterion, src_vocab, trg_vocab):
    losses, correct, wrong = 0, 0, 0
    model.eval()
    # Go through batches in the epoch
    for src, trg in tqdm(data, desc="Validating"):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
        # instead of the model's output in the prior timestep
        out = model(src, trg_input)

        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size)
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)

        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence
        loss = criterion(actual, expected)
        losses += loss.item()

    losses /= len(data)
    return losses


transformer = BaseTransformer(
    num_encoders=2,
    num_decoders=2,
    src_vocab_size=1000,
    trg_vocab_size=1000,
    pad_token=3,
).to(DEVICE)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
