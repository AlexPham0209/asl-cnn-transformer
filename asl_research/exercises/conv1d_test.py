import torch
import torch.nn as nn

from asl_research.model.spatial_embedding import MaskedBatchNorm
from asl_research.utils.utils import generate_video_padding_mask
from torch.nn.utils.rnn import pad_sequence

CHANNELS = 6
a = torch.arange(0, 5 * CHANNELS * 3, 3).reshape(5, CHANNELS).float()
b = torch.arange(1, 8 * CHANNELS * 2 + 1, 2).reshape(8, CHANNELS).float()
c = torch.arange(3, 12 * CHANNELS * 5 + 3, 5).reshape(12, CHANNELS).float()

size = [a.shape[0], b.shape[0], c.shape[0]]
batch = pad_sequence([a, b, c], batch_first=True)
conv = nn.Conv1d(CHANNELS, CHANNELS, kernel_size=3, stride=3)

batch = batch.permute(0, 2, 1)
N, C, T = batch.shape
new_length = (
    lambda T: (T + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1)
    // conv.stride[0]
    + 1
)
batch = conv(batch)

new_sizes = torch.tensor(list(map(new_length, size)))
padding_mask = generate_video_padding_mask(new_sizes)

batch = (batch * padding_mask.squeeze(1)).permute(0, 2, 1)
a = conv(a.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)
b = conv(b.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)
c = conv(c.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)

print(torch.allclose(batch[0, : a.shape[1]], a.squeeze(0)))
print(torch.allclose(batch[1, : b.shape[1]], b.squeeze(0)))
print(torch.allclose(batch[2, : c.shape[1]], c.squeeze(0)))

print(batch.float())
print(a)
print(b)
print(c)

bn = MaskedBatchNorm(CHANNELS)
bn2 = nn.BatchNorm1d(CHANNELS)
print(bn(batch, padding_mask))
