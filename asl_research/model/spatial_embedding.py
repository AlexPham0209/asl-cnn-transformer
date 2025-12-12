from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b4,
    EfficientNet_B4_Weights,
)
from torch.nn.utils.rnn import pad_sequence
from pytorchvideo.models.hub import i3d_r50


from asl_research.utils.utils import generate_padding_mask_from_lengths


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = MaskedBatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None):
        x = x.permute(0, 2, 1)
        x = self.conv(x)

        mask = None
        if lengths is not None and lengths.dim() == 1:
            lengths = self.calculate_new_lengths(lengths)
            mask = generate_padding_mask_from_lengths(lengths).to(lengths.device)
            x = x * mask.squeeze(1)

        x = x.permute(0, 2, 1)
        x = self.bn(x, mask)
        x = self.relu(x)

        return x, lengths

    def calculate_new_lengths(self, lengths: Tensor):
        return (
            lengths
            + 2 * self.conv.padding[0]
            - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            - 1
        ) // self.conv.stride[0] + 1


class MaskedBatchNorm(nn.Module):
    def __init__(self, num_features: int):
        super(MaskedBatchNorm, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor = None):
        """x is the input tensor of shape [batch_size, n_channels, time_length]
        mask is of shape [batch_size, 1, time_length]
        bn is a BatchNorm1d object
        """

        if mask is None:
            x = self.bn(x.permute(0, 2, 1))
            return x.permute(0, 2, 1)

        N, T, features = x.shape
        reshaped = x.reshape(-1, features)
        reshaped_mask = mask.reshape(-1, 1) > 0
        selected = torch.masked_select(reshaped, reshaped_mask).reshape(-1, features)
        batchnormed = self.bn(selected)
        scattered = reshaped.masked_scatter(reshaped_mask, batchnormed)
        backshaped = scattered.reshape(-1, T, features)
        return backshaped


class SpatialEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        hidden_size: int = 512,
        dropout: float = 0.1,
        pretrained_model: str = "efficientnet_b0",
    ):
        super(SpatialEmbedding, self).__init__()

        # Initializing the model based on our model parameter and freezing all the weights
        self.extractor = None
        match pretrained_model:
            case "efficientnet_b0":
                self.extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            case "efficientnet_b4":
                self.extractor = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            case "resnet50":
                self.extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in self.extractor.parameters():
            param.requires_grad = False

        # Replacing final classification layer with our own depending on what model we choose
        match pretrained_model:
            case "efficientnet_b0":
                self.extractor.classifier[1] = nn.Linear(
                    self.extractor.classifier[1].in_features, hidden_size
                )
            case "efficientnet_b4":
                self.extractor.classifier[1] = nn.Linear(
                    self.extractor.classifier[1].in_features, hidden_size
                )
            case "resnet50":
                self.extractor.fc = nn.Linear(self.extractor.fc.in_features, hidden_size)

        self.conv_1 = Conv1DBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5)
        # self.conv_2 = Conv1DBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.ff = nn.Linear(hidden_size, d_model)
        self.bn = MaskedBatchNorm(num_features=d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, lengths: Tensor = None):
        """
        Convert T frames of a 224x224 video into a 2d embedding matrix of size (time_out, d_model)

        Args:
        x: Batch of videos (batch_size, time, in_channels, 224, 224)

        Returns:
            (Tensor): Tensor of shape (batch_size, time_out, depth_out * height_out * width_out)
        """
        # Merge batches and time into the first dimension
        # Allows for the CNN to be applied to every temporal slice
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)

        # Using pretrained weights
        x = self.extractor(x).to(x.device)
        x = x.reshape(N, T, -1)
        x, lengths = self.conv_1(x, lengths)
        # x, lengths = self.conv_2(x, lengths)

        mask = generate_padding_mask_from_lengths(lengths).to(lengths.device)
        x = self.ff(x)
        x = self.bn(x, mask)
        x = self.relu(x)

        # Reshaping the output of the Resnet
        return x, mask, lengths

class PoseEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 225,
        d_model: int = 512,
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ):
        super(SpatialEmbedding, self).__init__()

        self.conv_1 = Conv1DBlock(in_channels=in_channels, out_channels=hidden_size, kernel_size=5)
        self.conv_2 = Conv1DBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.ff = nn.Linear(in_channels, d_model)
        self.bn = MaskedBatchNorm(num_features=d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, lengths: Tensor = None):
        """
        Convert T frames of a 224x224 video into a 2d embedding matrix of size (time_out, d_model)

        Args:
        x: Batch of videos (batch_size, time, 225)

        Returns:
            (Tensor): Tensor of shape (batch_size, time_out, depth_out * height_out * width_out)
        """
        # Merge batches and time into the first dimension
        # Allows for the CNN to be applied to every temporal slice
        # x, lengths = self.conv_1(x, lengths)
        # x, lengths = self.conv_2(x, lengths)

        mask = generate_padding_mask_from_lengths(lengths).to(lengths.device)
        x = self.ff(x)
        x = self.bn(x, mask)
        x = self.relu(x)

        # Reshaping the output of the Resnet
        return x, mask, lengths