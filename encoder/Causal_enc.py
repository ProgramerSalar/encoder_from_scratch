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
from typing import List, Union, Tuple

from conv import CausalConv3d
from blocks.Causal_block import CausalBlock3d




class CausalEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: List = [128, 256, 512, 512],
                 num_layers: int = 2,
                 encoder_num_layers: int = 4,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 add_height_width_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False),
                 add_frame_2x: Tuple[bool, bool, bool, bool] = (True, True, True, False)
                 ):
        super().__init__()

        # [2, 3, 8, 256, 256] -> [2, 3, 128, 256, 256]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=channels[0],
                                    )
        
        
        
        self.encoder_block_layers = nn.ModuleList([])
        output_channels = channels[0]
        for i in range(encoder_num_layers):
            input_channels = output_channels
            output_channels = channels[i]
            # [128] -> [128]
            # [128] -> [256]
            # [256] -> [512]
            # [512] -> [512]
            
            self.encoder_block_layers.append(
                CausalBlock3d(in_channels=input_channels,
                            out_channels=output_channels,
                            num_layers=num_layers,
                            dropout=dropout,
                            eps=eps,
                            scale_factor=scale_factor,
                            norm_num_groups=norm_num_groups,
                            add_height_width_2x=add_height_width_2x[i],
                            add_frame_2x=add_frame_2x[i])
            )


    def forward(self, 
                sample: torch.FloatTensor) -> torch.FloatTensor:
        
        # [2, 3, 8, 256, 256] -> [2, 128, 8, 256, 256]
        sample = self.conv_in(sample)
        
        
        for encoder_block_layer in self.encoder_block_layers:
            # [2, 128, 8, 256, 256] -> [2, 128, 4, 128, 128]
            # [2, 128, 4, 128, 128] -> [2, 256, 2, 64, 64]
            # [2, 256, 2, 64, 64] -> [2, 512, 1, 32, 32]
            # [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
            sample = encoder_block_layer(sample)
            

        

        return sample
    

if __name__ == "__main__":

    causal_encoder = CausalEncoder(in_channels=3,
                                   out_channels=4,
                                   channels=[128, 256, 512, 512],
                                   num_layers=2,
                                   encoder_num_layers=4,
                                   dropout=0.0,
                                   eps=1e-6,
                                   scale_factor=1.0,
                                   norm_num_groups=2
                                   )
    print(causal_encoder)

    x = torch.randn(2, 3, 8, 256, 256)
    output = causal_encoder(x)
    print(output.shape)
