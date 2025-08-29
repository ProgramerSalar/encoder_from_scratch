import torch 
from torch import nn 
from typing import Union

from conv import CausalConv3d
from utils import get_activation

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 64,
                 num_groups=32,
                 eps=1e-6,
                 act_fn: str = "swish",
                 dropout: float = 0.1,
                 output_scale_factor: float = 1.0,
                 use_in_shortcut: bool = None
                 ):
        
        super().__init__()
        self.output_scale_factor = output_scale_factor
        
        # print(f"what is the num_groups: {num_groups} and num_channels: {in_channels}")
        self.norm1 = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=in_channels,
                                  eps=eps,
                                  affine=True)
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1)
        

        self.norm2 = nn.GroupNorm(num_groups=num_groups,
                                  num_channels=out_channels,
                                  eps=eps,
                                  affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1)
        self.activation_fn = get_activation(act_fn=act_fn)

        # [128] != [64]
        if in_channels != out_channels:
            self.use_in_shortcut = True 
            self.conv_shortcut = CausalConv3d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=True)
        else:
            self.use_in_shortcut = False 

            
            



        
    def forward(self,
                x: torch.FloatTensor,
                temb: torch.FloatTensor = None,
                is_init_image = True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        hidden_states = x
        # print(f"what is the norm1: {self.norm1} and data shape: {hidden_states.shape}")
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv1(hidden_states, is_init_image, temporal_chunk)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, is_init_image, temporal_chunk)

        if self.use_in_shortcut:
            print('working...')
            x = self.conv_shortcut(x,
                            is_init_image=is_init_image,
                            temporal_chunk=temporal_chunk)

        output_tensor = (x + hidden_states) / self.output_scale_factor

        return output_tensor
        


class CausalDownsample3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 use_conv: bool = True,
                 out_channels: int = None,
                 kernel_size: int = 3,
                 bias=True):
        
        super().__init__()
        self.in_channels = in_channels
        self.use_conv = use_conv
        
        stride = (1, 2, 2)
        if self.use_conv:
            self.conv = CausalConv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                bias=bias)
            
        else:
            assert in_channels == out_channels, 'make sure `in_channels` and `out_channels` are equal!'
            self.conv = nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=stride,    # (1, 1, 1)
                             stride=stride  # (1, 1, 1)
                             )
            
        

    
    def forward(self, 
                x: torch.FloatTensor,
                is_init_image: bool = True,
                temporal_chunk: bool = False):
        
        assert x.shape[1] == self.in_channels, 'make sure `video_channels` are equal to `channels`!'

        x = self.conv(x,
                      is_init_image=is_init_image,
                      temporal_chunk=temporal_chunk)
        
        
        if self.use_conv:
            x = self.conv(x,
                          is_init_image=is_init_image,
                          temporal_chunk=temporal_chunk)
            
        
        return x 
        


class CausalTemporalDownsample3D(nn.ModuleList):

    pass 



if __name__ == "__main__":

            ## CausalResnet3d ##
    # causal_resnet_3d = CausalResnet3d(in_channels=64,
    #                                   out_channels=64)
    # print(causal_resnet_3d)

    # x = torch.randn(2, 64, 8, 256, 256)
    
    # output = causal_resnet_3d(x)
    # print(output.shape)
    # ---------------------------------------------------------

            ## CausalDownsampe ## 
    causal_down_sample = CausalDownsample3D(in_channels=3,
                                            out_channels=3,
                                            use_conv=True)
    
    print(causal_down_sample)
    
    x = torch.randn(2, 3, 8, 256, 256)
    output = causal_down_sample(x)

    print(output.shape)


      

