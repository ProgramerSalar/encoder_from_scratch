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
from typing import Union, Tuple
from torch.nn import functional as F

class CausalConv3d(nn.Module):

    """
        in_channels (`int`): input channels of conv3d
        out_channels (`int`): output_channels of conv3d
        kernel_size (`int, Tuple(int, int, int)`): kernel size of conv3d 
        stride (`int, Tuple(int, int, int)`): stride of conv3d
        padding (`int, Tuple(int, int, int)`): padding of conv3d
        **kwargs: key words argument like (`dilation`, `groups`, `bias`, `padding_mode`)
    """

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 **kwargs):
        
        
        
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size = (self.kernel_size,)*3 if isinstance(kernel_size, int) else self.kernel_size
        self.stride = stride
        self.stride = (self.stride)*3 if isinstance(self.stride, int) else self.stride
        self.padding = padding
        self.padding = (self.padding)*3 if isinstance(self.padding, int) else self.padding
        self.padding_mode = 'constant'

        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = self.kernel_size
        self.dilation = 1
        
        # padding 
        """
            the calcultion of time_padding is realted to paper: https://arxiv.org/abs/1511.07122
            we had understanding this padding in notebook section (question-8): https://github.com/ProgramerSalar/encoder_from_scratch/blob/main/Notebook/notebook.ipynb
        """
        self.time_padding = self.dilation * (self.time_kernel_size - 1)  # 1*(3-1)=>2


        self.height_padding = self.height_kernel_size // 2      # ≈1
        self.width_padding = self.width_kernel_size // 2        # ≈1
        # (1, 1, 1, 1, 2, 0)
        self.causal_time_padding = (self.width_padding, self.width_padding, self.height_padding, self.height_padding, self.time_padding, 0)

        self.conv = nn.Conv3d(in_channels=in_channels,      
                              out_channels=out_channels,
                              kernel_size=kernel_size,      # default=3
                              stride=stride,                # default=1
                              padding=self.padding,         # default=0
                              dilation=self.dilation,       # default=0
                              **kwargs)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        """ 
            x (`floatTensor`): video input tensor 

            calculate the width, height and frame padding
                padding (1, 1, 1, 1, 2, 0) -> (width, height, frame) 
                (8) + (2, 0) => 10
                256, 256) + (1, 1, 1, 1) => (258, 258)
                ([1, 2, 8, 256, 256])  ->  ([2, 3, 10, 258, 258])
                for more info: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        """
                
        # [2, 3, 8, 256, 256] -> [2, 3, 10, 258, 258]
        x = F.pad(input=x,
                  pad=self.causal_time_padding,
                  mode=self.padding_mode)
        
        # [2, 3, 10, 258, 258] -> [batch_size, out_channels, frame, height, width]
        x = self.conv(x)
        return x
    


    


if __name__ ==  "__main__":
    
    causal_conv = CausalConv3d(in_channels=3,
                               out_channels=3,
                               kernel_size=3)
    
    print(causal_conv)

    x = torch.randn(2, 3, 8, 256, 256)
    output = causal_conv(x)
    print(output.shape)