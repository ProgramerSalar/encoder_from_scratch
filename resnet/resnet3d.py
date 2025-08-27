import torch 
from torch import nn 

from typing import Union

class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: Union[int, tuple[int, int, int]] = 3,
                 stride: Union[int, tuple[int, int, int]] = 1,
                 padding = 0,
                 
                 )