import math
from torch import Tensor
import torch.nn as nn

class BaseTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)