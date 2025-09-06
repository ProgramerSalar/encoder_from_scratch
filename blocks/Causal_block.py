# Copyright 2025 The savitri-AI Team. All rights reserved.
#
#    company: https://github.com/savitri-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn 

from resnets.Causal_resnet import (
    CausalResnet3d,
    CausalHeightWidth2x,
    CausalFrame2x
)

class CausalBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 add_height_width_2x: bool = True,
                 add_frame_2x: bool = True):
        super().__init__()
        self.add_height_width_2x = add_height_width_2x
        self.add_frame_2x = add_frame_2x

        
        
        ## <-------- increase the channels dim -----------> ##
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
            # out_channels = out_channels

            self.block_layers.append(
                CausalResnet3d(in_channels=input_channels,
                           out_channels=out_channels,
                           dropout=dropout,
                           eps=eps,
                           scale_factor=scale_factor,
                           norm_num_groups=norm_num_groups)
            )

        ## <---------- decrease the (height, width) dim ---------> 
        if self.add_height_width_2x:
            self.height_width_dims = nn.ModuleList([
                CausalHeightWidth2x(
                    in_channels=in_channels,
                    out_channels=out_channels
                )
            ])

        ## <-------- decrease the (frame) dim -----------> ##
        if self.add_frame_2x:
            self.frame_dims = nn.ModuleList([
                CausalFrame2x(in_channels=in_channels,
                            out_channels=out_channels)
            ])



    def forward(self, x):
        
        for block_layer in self.block_layers:
            x = block_layer(x)
        print(f"what is the shape of x: {x.shape}")

        if self.add_height_width_2x:
            for height_width_dim in self.height_width_dims:
                x = height_width_dim(x)

        if self.add_frame_2x:
            for frame_dim in self.frame_dims:
                x = frame_dim(x)

        return x 
    

if __name__ == "__main__":

    causal_block3d = CausalBlock3d(in_channels=128,
                                   out_channels=256,
                                   num_layers=2,
                                   norm_num_groups=2,
                                   add_height_width_2x=True,
                                   add_frame_2x=True)
    
    x = torch.randn(2, 128, 8, 256, 256)

    output = causal_block3d(x)
    print(output.shape)