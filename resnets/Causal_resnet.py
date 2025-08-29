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
        
        print(f"what is the num_groups: {num_groups} and num_channels: {in_channels}")
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
        







if __name__ == "__main__":

    causal_resnet_3d = CausalResnet3d(in_channels=64,
                                      out_channels=64)
    print(causal_resnet_3d)

    x = torch.randn(2, 64, 8, 256, 256)
    
    output = causal_resnet_3d(x)
    print(output.shape)


      

