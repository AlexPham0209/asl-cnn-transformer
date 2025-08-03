import math
from torch import Tensor
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3, 3)):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
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
            nn.Linear(in_features=2304, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout)
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
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: Tensor):
        """
        Convolution Block for the spatial embedding layer

        Args:
            x: Batch of videos (batch_size, in_channels, depth, height, width)

        Returns:
            (Tensor): Tensor of shape (batch_size, out_channels, depth_out, height_out, width_out)
        """

        # Transpose the time and the channel dimensions
        # Then, combine the batch and 
        N, C, T, W, H = x.shape
        x = x.transpose(1, 2).reshape(-1, C, W, H)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        # Restore the original dimensions
        return x.reshape(N, T, x.shape[-3], x.shape[-2], x.shape[-1]).transpose(1, 2)
    
class Spatial2DEmbedding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super(Spatial2DEmbedding, self).__init__()
        self.conv = nn.Sequential(
            Conv2DBlock(in_channels=3, out_channels=32),
            Conv2DBlock(in_channels=32, out_channels=64),
            Conv2DBlock(in_channels=64, out_channels=32, kernel_size=(2, 2)),
            Conv2DBlock(in_channels=32, out_channels=16, kernel_size=(2, 2)),
        )
        
        self.ff_1 = nn.Sequential(
            nn.Linear(in_features=2304, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout)
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