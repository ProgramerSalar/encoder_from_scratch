import torch 
from torch import nn 

from conv import CausalConv3d


class CausalEncoder3D(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 ):
        

        super().__init__()
        
        self.conv_in = CausalConv3d()
