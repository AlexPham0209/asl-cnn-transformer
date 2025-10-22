import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b4, EfficientNet_B4_Weights
from torch.nn.utils.rnn import pad_sequence


from asl_research.utils.utils import generate_video_padding_mask

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3, 3)):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        """
        Convolution Block for the spatial embedding layer

        Args:
            x: Batch of videos (batch_size, in_channels, depth, height, width)

        Returns:
            (Tensor): Tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
        """

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Spatial3DEmbedding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super(Spatial3DEmbedding, self).__init__()
        self.conv = nn.Sequential(
            Conv3DBlock(in_channels=3, out_channels=32),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            Conv3DBlock(in_channels=32, out_channels=64),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            Conv3DBlock(in_channels=64, out_channels=32, kernel_size=(2, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            Conv3DBlock(in_channels=32, out_channels=16, kernel_size=(2, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.ff_1 = nn.Sequential(
            nn.Linear(in_features=2304, out_features=512), nn.ReLU(), nn.Dropout(p=dropout)
        )

        self.ff_2 = nn.Linear(512, d_model)

    def forward(self, x: Tensor):
        """
        Convert T frames of a 224x224 video into a 2d embedding matrix of size (time_out, d_model)

        Args:
        x: Batch of videos (batch_size, in_channels, time, 224, 224)

        Returns:
            (Tensor): Tensor of shape (batch_size, time_out, depth_out * height_out * width_out)
        """
        x = self.conv(x)
        N, _, T, _, _ = x.shape

        # Reshapes the tensor from (batch_size, height)
        x = x.transpose(1, 2).reshape(N, T, -1)
        x = self.ff_1(x)
        x = self.ff_2(x)

        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3)):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: Tensor):
        """
        Convolution Block for the spatial embedding layer

        Args:
            x: Batch of videos (batch_size, time, in_channels, height, width)

        Returns:
            (Tensor): Tensor of shape (batch_size, time, out_channels, height_out, width_out)
        """

        # Transpose the time and the channel dimensions
        # Then, combine the batch and
        N, T, C, W, H = x.shape
        x = x.reshape(N * T, C, W, H)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Restore the original dimensions
        return x.reshape(N, T, x.shape[-3], x.shape[-2], x.shape[-1])


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        pooling_kernel_size: int = 2,
    ):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return self.pooling(x)



class MaskedBatchNorm(nn.Module):
    def __init__(self, num_features: int):
        super(MaskedBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor = None):
        """x is the input tensor of shape [batch_size, n_channels, time_length]
            mask is of shape [batch_size, 1, time_length]
            bn is a BatchNorm1d object
        """
        if mask.dim() >= 4:
            mask = mask.squeeze(dim=1)

        if mask is None:
            x = self.bn(x.permute(0, 2, 1))
            return x.permute(0, 2, 1)

        N, T, features = x.shape
        reshaped = x.reshape(-1, features, 1)
        reshaped_mask = mask.reshape(-1, 1, 1) > 0
        selected = torch.masked_select(reshaped, reshaped_mask).reshape(-1, features, 1)
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
        self.conv = None
        match pretrained_model:
            case "efficientnet_b0":
                self.conv = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            case "efficientnet_b4":
                self.conv = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            case "resnet50":
                self.conv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in self.conv.parameters():
            param.requires_grad = False

        # Replacing final classification layer with our own depending on what model we choose
        match pretrained_model:
            case "efficientnet_b0":
                self.conv.classifier[1] = nn.Linear(
                    self.conv.classifier[1].in_features, hidden_size
                )
            case "efficientnet_b4":
                self.conv.classifier[1] = nn.Linear(
                    self.conv.classifier[1].in_features, hidden_size
                )
            case "resnet50":
                self.conv.fc = nn.Linear(self.conv.fc.in_features, hidden_size)
        
        self.ff = nn.Linear(hidden_size, d_model)
        self.bn = MaskedBatchNorm(num_features=d_model)
        self.relu = nn.ReLU()
        

    def forward(self, x: Tensor, mask: Tensor = None):
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
        x = self.conv(x).to(x.device)
        x = x.reshape(N, T, -1)
        x = self.ff(x)
        x = self.bn(x, mask)
        x = self.relu(x)
        
        # Reshaping the output of the Resnet
        return x
