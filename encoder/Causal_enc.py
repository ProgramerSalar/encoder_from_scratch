import torch 
from torch import nn 
from typing import Tuple, Union

from conv import CausalConv3d




class CausalEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: Union[int, Tuple[int, ...]] = (2,),
                 encoder_num_layers: int = 4,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 scale_factor: float = 1.0):
        super().__init__()

        self.conv_in = 


    def forward(self, 
                sample: torch.FloatTensor) -> torch.FloatTensor:
        

        return sample