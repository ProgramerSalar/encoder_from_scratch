import torch
from torch import nn 

from resnets.Causal_resnet import CausalResnet3d

class CausalBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32):
        super().__init__()

        # [128] -> [128]    -> loop 1st
        # [128] -> [128]    -> loop 2nd
        ## ---------------
        # [128] -> [256]
        # [256] -> [256]
        ## ---------------
        # [256] -> [512]
        # [512] -> [512]
        ## ----------------
        # [512] -> [512]
        # [512] -> [512]
        self.block_layers = nn.ModuleList([])
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            out_channels = out_channels

            self.block_layers.append(
                CausalResnet3d(in_channels=input_channels,
                           out_channels=out_channels,
                           dropout=dropout,
                           eps=eps,
                           scale_factor=scale_factor,
                           norm_num_groups=norm_num_groups)
            )






    def forward(self, x):
        return x 