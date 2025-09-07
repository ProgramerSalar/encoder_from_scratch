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

from conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 use_in_shortcut=None
                 ):
        
        """ 
            in_channels (`int`): input channels 
            out_channels (`int`): output channels 
            dropout (`float`): what is the dropout to put on the `nn.GroupNorm()`
            eps (`float`): what is the eps to put on the `nn.GroupNorm()`
            scale_factor (`float`): the float value to scale the video tensor 
            norm_num_groups (`int`): the value to use in the `nn.GroupNorm()`
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        self.norm1 = CausalGroupNorm(in_channels=in_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels)
        
        self.norm2 = CausalGroupNorm(in_channels=out_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = out_channels
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels)
        self.act_fn = nn.SiLU()


        # [128] != [256]
        if use_in_shortcut is None:
            self.use_in_shortcut = in_channels != conv_2d_out_channels

        else:
            self.use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels=in_channels,
                                              out_channels=conv_2d_out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=True
                                              )
            
        else:
            self.use_in_shortcut = False

        
       

        
            
    def forward(self, x):

        # feed-forward layer
        input_tensor = x
       

        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)


        # [128] != [256]
        # [256] != [512]
        if self.conv_shortcut is not None:
            # [2, 128, 8, 256, 256] -> [2, 256, 8, 256, 256]
            # [2, 256, 8, 256, 256] -> [2, 256, 8, 512, 512]
            input_tensor = self.conv_shortcut(input_tensor)

        # [2, 128, 8, 256, 256] + [2, 128, 8, 256, 256]
        # [2, 128, 8, 256, 256] + [2, 128, 8, 256, 256]
        # [2, 256, 8, 256, 256] + [2, 256, 8, 256, 256]
        # [2, 256, 8, 256, 256] + [2, 256, 8, 256, 256]
        # [2, 512, 8, 256, 256] + [2, 512, 8, 256, 256]
        x = (input_tensor + x) / self.scale_factor

        return x 
    
class CausalHeightWidth2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        
        super().__init__()
        
        # frame/2, height/2, width/2
        stride = (1, 2, 2)

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 bias=True)

    def forward(self, x): 
        x = self.conv(x)
        return x 


class CausalFrame2x(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        
        super().__init__()

        # frame/2, height, width
        stride = (2, 1, 1)

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 bias=True)
        

    def forward(self, x):
        x = self.conv(x)
        return x 




    


if __name__ == "__main__":

    # causal_resnet3d = CausalResnet3d(in_channels=128,
    #                                  out_channels=256,
    #                                  norm_num_groups=2)
    # print(causal_resnet3d)

    # x = torch.randn(2, 128, 8, 256, 256)
    # output = causal_resnet3d(x)

    # print(output.shape)
    # -------------------------------------------------------------------------

    # causal_height_width = CausalHeightWidth2x(in_channels=128,
    #                                           out_channels=128)
    
    # x = torch.randn(2, 128, 8, 256, 256)
    # output = causal_height_width(x)
    # print(output.shape) # [2, 3, 8, 128, 128]
    # ----------------------------------------------------------------------------

    causal_frame = CausalFrame2x(in_channels=128,
                                 out_channels=128)
    
    x = torch.randn(2, 128, 8, 256, 256)

    output = causal_frame(x)
    print(output.shape)